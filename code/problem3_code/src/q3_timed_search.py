'有时间限制的第三问 BSPg、粗粒度、HC、多步及 MGR 搜索，并执行内部复核。'
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from statistics import median
from time import perf_counter

from q2_bspg import BSPG_RULES, bspg_greedy
from q2_coarse import coarse_candidates
from q2_core_pool import CoreSolutionPool
from q2_greedy import RULES, greedy
from q2_hc import run_hc
from q2_io import digest, load_json
from q2_mgr import run_mgr
from q2_model import Model, State, score_key
from q2_search import Context, MGR_PROFILES, PROFILES, Settings, multi_search, hill_climb
from q3_evaluator import Problem3Evaluator
from q3_time import TimePolicy, REVISION, STAGE_SHARES, Stagnation, check, deadline, commit_section
from scene_b_internal import SceneBEvaluator


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def brief(result):
    return canonical(CoreSolutionPool._summary(result))


def entry_of(state, result, source):
    plan, metrics = state.plan(), brief(result)
    return dict(plan=plan, result=metrics, plan_sha256=digest(plan),
                content_sha256=digest(dict(plan=plan, result=metrics)), source=source)


def validate_entry(entry):
    if digest(entry['plan']) != entry.get('plan_sha256') or digest(
            dict(plan=entry['plan'], result=entry['result'])) != entry.get('content_sha256'):
        raise ValueError('saved plan/metrics checksum changed; resume refused')
    if entry['result'].get('problem') != 3:
        raise ValueError('Q3 checkpoint contains a different evaluator objective')


class TimedContext(Context):
    def __init__(self, model, hardware, settings, policy, entries, references,
                 counts, save, emit, end):
        super().__init__(model, hardware, settings, backend='problem3')
        self.policy, self.entries, self.references, self.counts = policy, entries, references, counts
        self.save, self.output = save, emit
        self.deadline = end
        self.stage_name = 'initial'
        self.stage_started = perf_counter()
        self.last_improvement = self.stage_started
        self.stage_valid = self.stage_macros = self.stage_timeouts = 0
        self.eval_times = []
        self.gains = []
        self.conversion_seconds = 0.0
        self.conversion_limit = 0.0
        self.pending = {}
        self.conversion_seen = set()
        self.native = None
        self.native_result = None
        self.pre_mgr_T = None
        self.initial_T = None
        self.factor = None
        self.nominal = None
        self.stopped = False
        self.current_core = None

    def emit(self, event):
        # 超时监督不需要保存大型移动列表或成员列表。
        event = dict(event, elapsed_seconds=perf_counter()-self.started)
        if self.output:
            self.output(event)

    def check_work(self):
        check(self.deadline)
        if self.stopped:
            raise Stagnation('stage stagnated')
        enough = self.stage_valid >= 3 and (self.stage_name != 'mgr' or self.stage_macros >= 1)
        if self.stage_name != 'initial' and enough and perf_counter()-self.last_improvement >= self.policy.stagnation_seconds:
            self.stopped = True
            raise Stagnation('no improvement after completed initial probes')

    def operation(self, end=None):
        self.check_work()
        return deadline(min(self.deadline, end) if end is not None else self.deadline)

    def protected_commit(self):
        with commit_section():
            self.save(self.entries, self.references)

    def install(self, state, result, source):
        key = str(len(state.cores))
        old = self.entries.get(key)
        if old is not None and score_key(result) >= score_key(old['result']):
            return False
        with commit_section():
            self.entries[key] = entry_of(state, result, source)
            self.save(self.entries, self.references)
        return True

    def evaluate(self, state, stage, move=None, *, diagnostics=False, guide_against=None):
        self.check_work()
        tick = perf_counter()
        mandatory = stage in ('baseline', 'warm_start', 'inherited', 'reference', 'required_seed')
        limit = self.policy.macro_candidate_seconds if stage.startswith(('mgr','greedy_')) else self.policy.candidate_seconds
        estimate = median(self.eval_times[-7:]) if self.eval_times else .5
        allowance = max(limit, (3 if stage.startswith('mgr') else 2)*self.progress.get('evaluation_seconds', estimate)) if hasattr(self, 'progress') else min(limit, max(.5, 3*estimate))
        end = self.deadline if mandatory else min(self.deadline, tick+allowance)
        self.evaluations += 1
        result = None
        try:
            with deadline(end):
                result = self.evaluator.evaluate(state.plan(), deadline=end,
                    diagnostics=diagnostics,
                    hc_guidance=self.settings.hc_mode == 'targeted' and self.settings.profile != 'initial')
        except TimeoutError:
            # 异步中断可能发生在 LRU 缓存修改过程中。
            # 清空软件缓存，不复用只完成部分提交的模板。
            with commit_section():
                self.evaluator.clear_cache()
            self.stage_timeouts += 1
            if hasattr(self, 'progress'):
                self.progress['evaluation_seconds'] = max(self.progress.get('evaluation_seconds', 0),
                    1.5*(perf_counter()-tick))
                self.protected_commit()
            self.emit(dict(event='candidate_timeout', stage=stage, seconds=perf_counter()-tick))
            check(self.deadline)
            return None
        except (ValueError, RuntimeError) as exc:
            self.emit(dict(event='candidate_rejected', stage=stage, error=str(exc)[:400]))
            return None
        self.eval_times.append(perf_counter()-tick)
        if hasattr(self, 'progress'):
            self.progress['evaluation_seconds'] = max(self.progress.get('evaluation_seconds', 0), perf_counter()-tick)
        if 'hc_guidance' in result:
            result['hc_guidance']['source_fingerprint'] = state.fingerprint()
        self.stage_valid += 1
        self.stage_macros += int(stage == 'mgr_macro')
        previous = self.best_result['makespan'] if self.best_result is not None else None
        improved = self.best_result is None or score_key(result) < score_key(self.best_result)
        # 先提交已经完成的来源分数，再执行可选的跨核回放。
        with commit_section():
            if improved:
                self.best, self.best_result = state, result
                self.last_improvement = perf_counter()
                if previous is not None and result['makespan'] < previous:
                    self.gains.append((perf_counter(), previous, result['makespan']))
            self.install(state, result, stage)
        self.emit(dict(event='candidate', stage=stage, evaluation=self.evaluations,
                       makespan=result['makespan'], new_best=improved,
                       seconds=perf_counter()-tick, fingerprint=state.fingerprint()))
        if self.settings.core_sharing:
            active = sum(bool(row) for row in state.cores)
            for target in self.counts:
                if target != len(state.cores) and active <= target:
                    token = (target, state.fingerprint())
                    if token not in self.conversion_seen:
                        # 优先处理改进方案，同时保留数量受限的其他合法候选，
                        # 因为第三问的跨核转换可能改变分数排序。
                        self.pending[token] = (0 if improved else 1, state, stage)
            while len(self.pending) > 24:
                worst = max(self.pending, key=lambda k: self.pending[k][0])
                self.pending.pop(worst)
            self.drain_conversions()
        return result

    def drain_conversions(self):
        while self.pending and self.conversion_seconds < self.conversion_limit:
            check(self.deadline)
            token = min(self.pending, key=lambda k: (self.pending[k][0], k[0], k[1]))
            _, state, stage = self.pending.pop(token)
            target = token[0]
            tick = perf_counter()
            end = min(self.deadline, tick+self.conversion_limit-self.conversion_seconds)
            if end-tick < .01:
                break
            try:
                with deadline(end):
                    converted = state.resized(target)
                    value = self.evaluator.evaluate(converted.plan(), deadline=end)
            except TimeoutError:
                with commit_section():
                    self.evaluator.clear_cache()
                # 超时的转换不标记为已经评价。
                break
            except (ValueError, RuntimeError):
                self.conversion_seen.add(token)
                continue
            else:
                self.conversion_seen.add(token)
                self.install(converted, value, 'shared_'+stage)
            finally:
                self.conversion_seconds += perf_counter()-tick

    def run_stage(self, name, end, action):
        self.stage_name, self.stage_started = name, perf_counter()
        self.last_improvement = self.stage_started
        self.stage_valid = self.stage_macros = self.stage_timeouts = 0
        self.stopped = False
        self.deadline = end
        before, count = self.best_result['makespan'], self.evaluations
        info, status = {}, 'completed'
        try:
            with deadline(end):
                info = action() or {}
        except Stagnation:
            status = 'stagnated'
        except TimeoutError:
            status = 'time_budget_exhausted'
        row = dict(stage=name, before=before, after=self.best_result['makespan'],
                   seconds=perf_counter()-self.stage_started, status=status,
                   evaluations=self.evaluations-count, details=info)
        self.stages.append(row)
        if hasattr(self, 'progress'):
            self.progress.setdefault('stages', []).append(row)
            if name == 'mgr':
                covered = self.stage_macros > 0 or info.get('disabled', False) or any(
                    r.get('rounds') and all(not x['candidates'] for x in r['rounds'])
                    for r in info.get('starts', []))
            else:
                covered = self.stage_valid > 0 or (status != 'time_budget_exhausted'
                    and not self.stage_timeouts and info.get('status') != 'time_budget_exhausted')
            row.update(valid_evaluations=self.stage_valid,macro_evaluations=self.stage_macros,
                       candidate_timeouts=self.stage_timeouts,coverage_completed=bool(covered))
            if covered and name not in self.progress.setdefault('completed_stages', []):
                self.progress['completed_stages'].append(name)
        self.emit(dict(event='stage_complete', **{k:v for k,v in row.items() if k != 'details'}))
        self.protected_commit()


def make_evaluator(graph, hardware, cache_mb, *, l2=True):
    if l2:
        return Problem3Evaluator(graph, **hardware, cache_bytes=cache_mb*1024**2)
    hw = {k:hardware[k] for k in ('bandwidth', 'capacity', 'cross_core_copy_delay')}
    return SceneBEvaluator(graph, **hw, cache_bytes=cache_mb*1024**2)


def recheck(model, hardware, settings, state, expected, references, end, *, observed=None, save=lambda: None):
    '每次完整回放和参考值计算独立提交，并以方案为索引。'
    key = digest(state.plan())
    cache = references.setdefault('recheck_items', {}).setdefault(key, {})
    with deadline(end):
        evaluator = make_evaluator(model.graph, hardware, settings.cache_mb)
        if 'observed' not in cache:
            observed = evaluator.evaluate(state.plan(), deadline=end) if observed is None else observed
            if brief(observed) != brief(expected):
                raise AssertionError('saved/internal replay metrics disagree')
            cache['observed'] = brief(observed)
            save()
        observed = cache['observed']
        if brief(observed) != brief(expected):
            raise AssertionError('cached replay metrics disagree')
        plain = make_evaluator(model.graph, hardware, settings.cache_mb, l2=False)
        if 'reference_T1' not in references:
            references['reference_T1'] = plain.evaluate(model.fallback(1).plan(), deadline=end)['makespan']
            save()
        if 'same_plan_Q2_T' not in cache:
            cache['same_plan_Q2_T'] = plain.evaluate(state.plan(), deadline=end)['makespan']
            save()
        if 'reference_singlecore_Q3_T' not in references:
            references['reference_singlecore_Q3_T'] = evaluator.evaluate(model.fallback(1).plan(), deadline=end)['makespan']
            save()
        t1, t2, t3 = references['reference_T1'], cache['same_plan_Q2_T'], observed['makespan']
        return dict(problem=3, backend='problem3', algorithm=REVISION, cores=len(state.cores),
            final_T=t3, reference_T1=t1, reference_T1_kind='original_singlecore_without_L2',
            reference_singlecore_Q3_T=references['reference_singlecore_Q3_T'],
            same_plan_Q2_T=t2, cache_gain_cycles=t2-t3,
            cache_gain_fraction=(t2-t3)/t2 if t2 else None,
            k=t1/t3 if t3 else None, k_per_core=t1/t3/len(state.cores) if t3 else None,
            active_cores=observed['active_cores'], subgraphs=len(state.groups),
            data_movement_bytes=observed['data_movement_bytes'],
            memory_peak_by_core=observed['memory_peak_by_core'],
            cache_stats=observed['cache_stats'], cache_miss_reasons=observed['cache_miss_reasons'],
            physical_transfer_bytes=observed['physical_transfer_bytes'],
            plan_sha256=key, internally_validated=True,
            internal_recheck_passed=True, evaluation_kind='internal_only', officially_validated=False)


def execute(request, save, emit):
    '按已完成阶段恢复，不根据存在已保存方案就推断搜索完成。'
    policy = TimePolicy(**request['policy']).check()
    settings = Settings(**request['settings']); settings.check()
    if settings.search_seconds <= 0:
        raise ValueError('Q3 search_seconds must be positive')
    end, started = request['deadline'], request['started']
    with deadline(end):
        graph = load_json(request['graph_path']); model = Model(graph)
        entries = deepcopy(request.get('entries', {}))
        for key, entry in entries.items():
            validate_entry(entry)
            if len(entry['plan']['core_schedules']) != int(key):
                raise ValueError('checkpoint core count mismatch')
        references = deepcopy(request.get('references', {}))
        ctx = TimedContext(model, request['hardware'], settings, policy, entries,
            references, request['counts'], save, emit, end)
    ctx.started = started
    core, mode = request['cores'], request.get('mode', 'search')
    ctx.current_core = core
    progress = references.setdefault('search_progress', {}).setdefault(str(core), {})
    ctx.progress = progress
    if progress.get('evaluation_seconds'):
        ctx.eval_times = [progress['evaluation_seconds']]
    if str(core) in entries:
        # 身份与校验和匹配的完整分数可作为恢复优化的起点。
        # 最终报告产生前仍必须执行一次独立的新回放。
        ctx.best = State.from_plan(model, entries[str(core)]['plan'])
        ctx.best_result = entries[str(core)]['result']
    if progress.get('native'):
        validate_entry(progress['native'])
        ctx.native = State.from_plan(model, progress['native']['plan'])
        ctx.native_result = progress['native']['result']
    report = None; status = 'time_budget_exhausted'
    try:
        with deadline(end):
            if mode != 'recheck':
                if ctx.best is None:
                    ctx.evaluate(model.fallback(core), 'baseline')
                if not progress.get('seed_scored'):
                    if not progress.get('pending_seed'):
                        rules = BSPG_RULES if settings.constructor == 'bspg' else RULES
                        rule = next(iter(rules))
                        try:
                            with deadline(min(end, perf_counter()+policy.constructor_seconds)):
                                if settings.constructor == 'bspg':
                                    candidate, _ = bspg_greedy(model, core, request['hardware'], rule,
                                        escape_weight=settings.escape_weight)
                                else:
                                    candidate, _ = greedy(model, core, request['hardware'], rule,
                                        group_size=settings.group_size, ready_window=settings.ready_window)
                        except TimeoutError:
                            check(end)
                            ctx.emit(dict(event='constructor_timeout', rule=rule))
                            with deadline(min(end, perf_counter()+max(policy.constructor_seconds, 15))):
                                candidate, _ = greedy(model, core, request['hardware'], 'critical',
                                    group_size=settings.group_size, ready_window=settings.ready_window)
                        progress['pending_seed'] = candidate.plan()
                        progress['seed_constructed'] = True
                        ctx.protected_commit()
                    candidate = State.from_plan(model, progress['pending_seed'])
                    observed = ctx.evaluate(candidate, 'required_seed')
                    if observed is None:
                        # 非法方案不能标记为已经完成初解阶段。
                        progress.pop('pending_seed', None); ctx.protected_commit()
                        raise RuntimeError('required seed rejected by internal evaluator')
                    ctx.native, ctx.native_result = candidate, observed
                    progress.update(seed_scored=True, native=entry_of(candidate, observed, 'required_seed'))
                    progress.pop('pending_seed', None)
                    ctx.protected_commit()
                # 报告参考值的计算放在第一个非备用初解之后。
                if 'reference_T1' not in references:
                    plain = make_evaluator(graph, request['hardware'], settings.cache_mb, l2=False)
                    references['reference_T1'] = plain.evaluate(model.fallback(1).plan(), deadline=end)['makespan']
                    ctx.protected_commit()
                if not progress.get('competition_done'):
                    competition_end = min(end, perf_counter()+min(30, max(5, 2*progress.get('evaluation_seconds', 1))))
                    ctx.deadline = competition_end
                    try:
                        with deadline(competition_end):
                            seeds = [e for k,e in entries.items() if int(k)<core] if settings.core_sharing else []
                            for e in sorted(seeds, key=lambda e:e['result']['makespan'])[:1]:
                                ctx.evaluate(State.from_plan(model,e['plan']).resized(core), 'inherited')
                            for plan in request.get('seed_plans', []):
                                state = State.from_plan(model, plan)
                                if sum(bool(row) for row in state.cores)<=core:
                                    ctx.evaluate(state.resized(core), 'inherited')
                            rules = BSPG_RULES if settings.constructor == 'bspg' else RULES
                            for rule in list(rules)[1:]:
                                if competition_end-perf_counter()<1.5*progress.get('evaluation_seconds', .5):break
                                try:
                                    with deadline(min(competition_end, perf_counter()+policy.constructor_seconds)):
                                        if settings.constructor == 'bspg':
                                            candidate,_=bspg_greedy(model,core,request['hardware'],rule,escape_weight=settings.escape_weight)
                                        else:
                                            candidate,_=greedy(model,core,request['hardware'],rule,
                                                group_size=settings.group_size,ready_window=settings.ready_window)
                                    value=ctx.evaluate(candidate,'greedy_'+rule)
                                    if value is not None and score_key(value)<score_key(ctx.native_result):
                                        ctx.native,ctx.native_result=candidate,value
                                        progress['native']=entry_of(candidate,value,'greedy_'+rule)
                                        ctx.protected_commit()
                                except TimeoutError:check(competition_end)
                            if settings.coarse_candidates:
                                with deadline(min(competition_end,perf_counter()+policy.coarse_seconds)):
                                    proposals,_=coarse_candidates(model,[('incumbent',ctx.best),('native_bspg',ctx.native)],
                                        core,request['hardware'],settings.coarse_candidates)
                                    for candidate,move in proposals:ctx.evaluate(candidate,'coarse',move)
                    except TimeoutError:pass
                    progress['competition_done']=True;ctx.deadline=end;ctx.protected_commit()
                if 'nominal_seconds' not in progress:
                    nominal,factor=policy.nominal_seconds(core,references['reference_T1'],ctx.best_result['makespan'])
                    progress.update(nominal_seconds=nominal,dynamic_factor=factor,initial_T=ctx.best_result['makespan'])
                    ctx.protected_commit()
                ctx.nominal,ctx.factor=progress['nominal_seconds'],progress['dynamic_factor']
                ctx.initial_T=progress['initial_T']
                if request.get('extension'):
                    completed=set()
                else:completed=set(progress.get('completed_stages', []))
                needed=[(name,share) for name,share in STAGE_SHARES if (name!='mgr' or settings.mgr) and name not in completed]
                # 初始化耗时由外层租约计入，
                # 不再从局部搜索时钟中重复扣除。
                cost=progress.get('evaluation_seconds', .5)
                search_budget=min(180, max(ctx.nominal, cost*sum(8 if n=='mgr' else 3 for n,_ in needed)))
                search_end=min(end,perf_counter()+search_budget)
                available=max(0,search_end-perf_counter())
                ctx.emit(dict(event='greedy_complete',initial_T=ctx.initial_T,dynamic_factor=ctx.factor,
                    nominal_seconds=ctx.nominal,remaining_search_seconds=available))
                cfg,mc=PROFILES[settings.profile],MGR_PROFILES[settings.profile]
                for index,(name,share) in enumerate(needed):
                    check(search_end)
                    remaining=search_end-perf_counter()
                    if remaining < min(1.5*cost, 30):break
                    remaining_shares=sum(s for _,s in needed[index:])
                    allotted=max(remaining*share/remaining_shares, min(remaining,2*cost))
                    stage_end=min(search_end,perf_counter()+allotted)
                    ctx.conversion_limit=ctx.conversion_seconds+allotted*policy.conversion_fraction
                    if name in ('hc','final_hc'):
                        def action():
                            if settings.hc_mode=='targeted':
                                return run_hc(ctx,ctx.best,ctx.best_result,{k:cfg[k] for k in ('order','migrate','swap')},label=name)[2]
                            return {k:hill_climb(ctx,k,cfg[k]) for k in ('order','migrate','swap')}
                    elif name=='multi':
                        def action():return multi_search(ctx,cfg['multi'],cfg['beam'],cfg['depth'])
                    else:
                        ctx.pre_mgr_T=ctx.best_result['makespan']
                        def action():
                            starts=[('incumbent',ctx.best,ctx.best_result),('native_m_core',ctx.native,ctx.native_result)]
                            return run_mgr(ctx,starts,request['hardware'],
                                budget=mc['budget'] if settings.mgr_budget is None else settings.mgr_budget,
                                rounds=mc['rounds'],repair_budget=mc['repair'],max_nodes=settings.mgr_max_nodes)
                    ctx.run_stage(name,stage_end,action)
                required={n for n,_ in STAGE_SHARES if n!='mgr' or settings.mgr}
                progress['search_finished']=progress.get('seed_scored',False) and required.issubset(progress.get('completed_stages',[]))
                ctx.deadline=end;ctx.protected_commit()
                status='search_finished' if progress['search_finished'] else 'search_pending'
            if ctx.best is not None and progress.get('search_finished'):
                report=recheck(model,request['hardware'],settings,ctx.best,ctx.best_result,references,end,save=ctx.protected_commit)
                status='complete'
    except TimeoutError:
        status='recheck_pending' if progress.get('search_finished') else 'search_pending'
    ctx.protected_commit()
    now=perf_counter()
    recent=[x for x in ctx.gains if now-x[0]<=policy.improvement_window]
    recent_gain=((recent[0][1]-ctx.best_result['makespan'])/recent[0][1] if recent and recent[0][1] else 0)
    details=dict(settings=asdict(settings),time_policy=asdict(policy),dynamic_factor=progress.get('dynamic_factor'),
        nominal_seconds=progress.get('nominal_seconds'),greedy_best_T=progress.get('initial_T'),pre_mgr_T=ctx.pre_mgr_T,
        stages=progress.get('stages',[]),evaluations=ctx.evaluations,elapsed_seconds=now-started,
        recent_gain_fraction=recent_gain,extension_eligible=bool(progress.get('search_finished') and recent_gain>=policy.extension_gain),
        search_finished=bool(progress.get('search_finished')),seed_scored=bool(progress.get('seed_scored')),
        stop_reason=status,internally_validated=ctx.best is not None,internal_recheck_passed=report is not None,
        officially_validated=False)
    if report is not None:report.update(details,status='complete')
    return dict(entries=entries,references=references,report=report,details=details,
        status='complete' if report is not None else status)
