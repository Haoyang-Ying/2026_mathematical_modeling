"""使用精确评分的顺序与片段邻域，以及有界的最优前缀搜索。"""
from __future__ import annotations

import random
import statistics
from dataclasses import asdict, dataclass
from time import perf_counter

from q2_config import (DEFAULT_SEED, DEFAULT_ESCAPE_WEIGHT, DEFAULT_SEARCH_SECONDS,
                       CORE_TIMEOUT_SECONDS, MAX_SEARCH_SECONDS,
                       INITIAL_MAX_SECONDS, COARSE_MAX_SECONDS, CANDIDATE_MAX_SECONDS)
from q2_limits import mark_core, validate_limits, remaining_seconds, elapsed_seconds
from q2_budget import CaseBudget, timed_call
from q2_model import Model, State, View, SceneBEvaluator, score_key
from q2_greedy import greedy, RULES
from q2_bspg import bspg_greedy, BSPG_RULES
from q2_coarse import coarse_candidates
from q2_mgr import run_mgr
from q2_hc import run_hc, mixed_neighborhoods, semantic_digest
from q2_core_pool import CoreSolutionPool, checked_counts
from q2_io import digest


# 提交版仅保留标准搜索配置；时间预算由 q2_config 统一提供。
PROFILES = {
    'standard': dict(order=32, migrate=32, swap=24, multi=64, beam=4, depth=6),
}

MGR_PROFILES = {
    'standard': dict(budget=96, rounds=3, repair=6),
}


@dataclass
class Settings:
    profile: str = 'standard'
    seed: int = DEFAULT_SEED
    group_size: int = 8
    ready_window: int = 8
    cache_mb: int = 128
    search_seconds: float = DEFAULT_SEARCH_SECONDS
    constructor: str = 'bspg'
    escape_weight: float = DEFAULT_ESCAPE_WEIGHT
    coarse_candidates: int = 3
    mgr: bool = True
    mgr_max_nodes: int = 512
    mgr_budget: int | None = None
    hc_mode: str = 'targeted'
    core_sharing: bool = True
    core_timeout: float = CORE_TIMEOUT_SECONDS
    search_max_seconds: float = MAX_SEARCH_SECONDS

    def check(self):
        validate_limits(self.core_timeout)
        if (isinstance(self.search_max_seconds,bool) or not isinstance(self.search_max_seconds,(int,float))
                or self.search_max_seconds != MAX_SEARCH_SECONDS):
            raise ValueError(f'固定阶段上限：search_max_seconds 必须为 {MAX_SEARCH_SECONDS}')
        if type(self.core_sharing) is not bool:
            raise ValueError('core_sharing must be a boolean')
        if self.hc_mode not in ('targeted','legacy'):
            raise ValueError('hc_mode must be targeted or legacy')
        if type(self.mgr) is not bool:
            raise ValueError('mgr must be a boolean')
        if type(self.mgr_max_nodes) is not int or self.mgr_max_nodes < 1:
            raise ValueError('mgr_max_nodes must be a positive integer')
        if self.mgr_budget is not None and (type(self.mgr_budget) is not int or self.mgr_budget < 0):
            raise ValueError('mgr_budget must be a nonnegative integer or None')
        if self.constructor not in ('bspg', 'legacy'):
            raise ValueError('constructor must be bspg or legacy')
        if type(self.coarse_candidates) is not int or self.coarse_candidates < 0:
            raise ValueError('coarse_candidates must be a nonnegative integer')
        if not isinstance(self.escape_weight, (int, float)) or not 0 <= self.escape_weight < float('inf'):
            raise ValueError('escape_weight must be finite and nonnegative')
        if self.profile not in PROFILES:
            raise ValueError(f'unknown profile: {self.profile}')
        for field in ('group_size', 'ready_window', 'cache_mb'):
            value = getattr(self, field)
            if type(value) is not int or value < (0 if field == 'cache_mb' else 1):
                raise ValueError(f'invalid {field}: {value!r}')
        if type(self.seed) is not int:
            raise ValueError('seed must be an integer')
        if isinstance(self.search_seconds, bool) or not isinstance(self.search_seconds, (int, float)) or self.search_seconds != DEFAULT_SEARCH_SECONDS:
            raise ValueError(f'固定阶段基数：search_seconds 必须为 {DEFAULT_SEARCH_SECONDS}')


def anchors(state, result, rng, limit=12):
    groups = sorted(result.get('anchor_groups', state.groups))
    if not groups:
        return []
    ends, spans = result.get('ends', {}), result.get('durations', {})
    hot = sorted(groups, key=lambda g: (spans.get(g, 0) + .2 * ends.get(g, 0), g), reverse=True)
    chosen = hot[:max(1, limit//2)]
    remaining = [g for g in groups if g not in set(chosen)]
    chosen.extend(rng.sample(remaining, min(len(remaining), limit-len(chosen))))
    rng.shuffle(chosen)
    return chosen


def order_neighbors(model, state, result, rng):
    view = View(model, state)
    for g in anchors(state, result, rng):
        c = view.core[g]
        old = state.cores[c]
        i = old.index(g)
        rest = old[:i] + old[i+1:]
        lo, hi = view.insertion_bounds(g, rest)
        positions = [i-1, i+1, i-3, i+3, lo, hi]
        for j in dict.fromkeys(positions):
            if not (lo <= j <= hi) or j == i:
                continue
            cores = list(state.cores)
            cores[c] = rest[:j] + (g,) + rest[j:]
            yield State(state.groups, tuple(cores)), {'kind': 'order', 'group': g, 'from': i, 'to': j}
        # 拆分优先级桶，然后在同一核心上移动后缀。
        # 调整后的优先级是否合法，由官方编译流程判断。
        nodes = state.groups[g]
        if len(nodes) > 1:
            cut = len(nodes)//2
            h = max(state.groups, default=-1)+1
            groups = dict(state.groups)
            groups[g], groups[h] = nodes[:cut], nodes[cut:]
            base = old[:i+1] + (h,) + old[i+1:]
            for j in dict.fromkeys((i+1, max(0, i-1), min(len(old), i+3), i)):
                without = tuple(x for x in base if x != h)
                cores = list(state.cores)
                cores[c] = without[:j] + (h,) + without[j:]
                yield State(groups, tuple(cores)), {'kind': 'split_order', 'group': g, 'new_group': h}


def detach(state, g, fragment, source):
    """从所在核心移除一个片段，保留非空的原优先级桶。"""
    groups = dict(state.groups)
    cores = [list(x) for x in state.cores]
    selected = set(fragment)
    remaining = tuple(v for v in groups[g] if v not in selected)
    if remaining:
        groups[g] = remaining
        h = max(groups, default=-1)+1
        groups[h] = tuple(fragment)
    else:
        h = g
        cores[source].remove(g)
    return groups, cores, h


def migrate_neighbors(model, state, result, rng):
    old_view = View(model, state)
    for g in anchors(state, result, rng):
        source = old_view.core[g]
        nodes = state.groups[g]
        fragments = [nodes]
        if len(nodes) > 1:
            size = min(4, max(1, len(nodes)//2))
            fragments = [nodes[:size], nodes[-size:], nodes]
        for fragment in dict.fromkeys(fragments):
            groups, cores, h = detach(state, g, fragment, source)
            # 暂时追加到来源核心，以便构造完整依赖映射。
            tmp = [list(x) for x in cores]
            tmp[source].append(h)
            view = View(model, State(groups, tuple(tuple(x) for x in tmp)))
            targets = [c for c in range(len(cores)) if c != source]
            targets.sort(key=lambda c: (result.get('core_ends', [0]*len(cores))[c], c))
            for c in targets:
                target = tuple(cores[c])
                lo, hi = view.insertion_bounds(h, target)
                if lo > hi:
                    continue
                center = sum(model.position[v] for v in fragment)/len(fragment)
                natural = next((j for j, x in enumerate(target)
                                if sum(model.position[v] for v in groups[x])/len(groups[x]) > center),
                               len(target))
                natural = max(lo, min(hi, natural))
                for j in dict.fromkeys((natural, lo, hi)):
                    new_cores = [tuple(x) for x in cores]
                    new_cores[c] = target[:j] + (h,) + target[j:]
                    yield State(groups, tuple(new_cores)), {
                        'kind': 'migrate', 'source': source, 'target': c,
                        'nodes': list(fragment), 'position': j}


def swap_neighbors(model, state, result, rng):
    view = View(model, state)
    selected = anchors(state, result, rng)
    for g in selected:
        a = view.core[g]
        partners = [h for h in state.groups if view.core[h] != a]
        partners.sort(key=lambda h: (abs(len(state.groups[h])-len(state.groups[g])), h))
        chosen = partners[:3]
        if len(partners) > 3:
            chosen += rng.sample(partners[3:], min(3, len(partners)-3))
        for h in chosen:
            b = view.core[h]
            cores = [list(x) for x in state.cores]
            i, j = cores[a].index(g), cores[b].index(h)
            cores[a][i], cores[b][j] = h, g
            yield State(state.groups, tuple(tuple(x) for x in cores)), {
                'kind': 'swap', 'groups': [g, h], 'cores': [a, b]}


NEIGHBORS = {'order': order_neighbors, 'migrate': migrate_neighbors, 'swap': swap_neighbors}


def mixed_neighbors(model, state, result, rng):
    """轮转邻域，避免将多步搜索的全部探测机会用在同一种邻域。"""
    kinds = list(NEIGHBORS)
    rng.shuffle(kinds)
    generators = [iter(NEIGHBORS[k](model, state, result, rng)) for k in kinds]
    while generators:
        live = []
        for gen in generators:
            try:
                yield next(gen)
                live.append(gen)
            except StopIteration:
                pass
        generators = live


class Context:
    def __init__(self, model, hardware, settings, on_event=None, core_pool=None):
        self.model, self.settings = model, settings
        self.hardware = hardware
        self.core_pool = core_pool
        self.evaluator = core_pool.evaluator if core_pool is not None else SceneBEvaluator(
            model.graph, **hardware, cache_bytes=settings.cache_mb*1024**2)
        # 每次搜索使用独立随机数流，由 HC、多步搜索和 MGR 共享。
        # 所有默认入口统一使用 DEFAULT_SEED = 1234。
        self.rng = random.Random(settings.seed)
        self.best, self.best_result = None, None
        self.events, self.stages = [], []
        self.evaluations = 0
        self.started = perf_counter()
        self.deadline = None
        self.on_event = on_event
        self.completed_evaluations = 0
        self.evaluation_seconds = []
        self.current_phase = None
        self.hc_seen = {}
        self.hc_attempts = {}
        self.cost_floor = .001

    def evaluation_estimate(self):
        measured = statistics.median(self.evaluation_seconds[-8:]) if self.evaluation_seconds else .001
        return max(self.cost_floor,measured)

    def emit(self, event):
        event = dict(event, elapsed_seconds=perf_counter()-self.started)
        self.events.append(event)
        if self.on_event:
            self.on_event(event, self.best, self.best_result)

    def evaluate(self, state, stage, move=None, *, diagnostics=False, guide_against=None):
        if self.deadline is not None and perf_counter() >= self.deadline:
            raise TimeoutError('local search time budget exhausted')
        t0 = perf_counter()
        self.evaluations += 1
        event = {'event': 'candidate', 'stage': stage, 'evaluation': self.evaluations,
                 'move': move, 'fingerprint': state.fingerprint()}
        try:
            guide = self.settings.hc_mode == 'targeted'
            cutoff = score_key(guide_against) if guide_against is not None else None
            # 初始或原生起点、宏观候选及多步试探，
            # 即使暂时劣于全局最优，也可能需要继续局部修复。
            if cutoff is None and stage not in ('baseline','inherited','warm_start','coarse','mgr_macro','multi') \
                    and not stage.startswith('greedy_') and self.best_result is not None:
                cutoff = score_key(self.best_result)
            candidate_deadline = (None if self.deadline is None else
                                  min(self.deadline,t0+CANDIDATE_MAX_SECONDS))
            result = self.evaluator.evaluate(state.plan(), deadline=candidate_deadline,
                                             diagnostics=diagnostics,
                                             hc_guidance=guide, guidance_below=cutoff)
        except TimeoutError:
            self.cost_floor = min(CANDIDATE_MAX_SECONDS,
                                  max(self.cost_floor,1.25*(perf_counter()-t0)))
            event.update(valid=None,status='timed_out',seconds=perf_counter()-t0)
            self.emit(event)
            raise
        except (ValueError, RuntimeError) as exc:
            event.update(valid=False, error_type=type(exc).__name__, error=str(exc)[:800],
                         seconds=perf_counter()-t0)
            self.emit(event)
            return None
        self.completed_evaluations += 1
        self.cost_floor = max(.001,.8*self.cost_floor)
        self.evaluation_seconds.append(perf_counter()-t0)
        if 'hc_guidance' in result:
            result['hc_guidance']['source_fingerprint'] = state.fingerprint()
        improved = self.best is None or score_key(result) < score_key(self.best_result)
        if improved:
            self.best, self.best_result = state, result
        event.update(valid=True, makespan=result['makespan'], score_key=score_key(result),
                     new_best=improved, seconds=perf_counter()-t0,
                     cache_stats=result['cache_stats'])
        self.emit(event)
        # 所有可行候选都参与跨核数择优，包括局部拒绝的 HC 移动、
        # 多步搜索中间状态，以及当前核数未接受的 MGR 候选。
        if self.core_pool is not None:
            if stage == 'baseline':
                self.core_pool.set_reference(result['makespan'])
            self.core_pool.offer(state, result, stage, emit=self.emit, deadline=self.deadline)
        return result


def hill_climb(ctx, kind, budget):
    state, result = ctx.best, ctx.best_result
    attempted, accepted = 0, 0
    seen = {state.fingerprint()}
    while attempted < budget:
        improved = False
        for candidate, move in NEIGHBORS[kind](ctx.model, state, result, ctx.rng):
            fingerprint = candidate.fingerprint()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            attempted += 1
            trial = ctx.evaluate(candidate, kind, move)
            if trial is not None and score_key(trial) < score_key(result):
                state, result = candidate, trial
                accepted += 1
                improved = True
                break
            if attempted >= budget:
                break
        if not improved:
            break
    return {'attempted': attempted, 'accepted_moves': accepted}


def best_prefix_walk(initial, initial_result, neighbors, evaluate, *, depth, beam, budget,
                     key=score_key, fingerprint=lambda state: state.fingerprint()):
    """有界的 FM 风格搜索，不具有 FM 的割增益或近似保证。
    
    每步最多评价 beam 个候选，选择此前未访问的最优可行候选，即使其劣于当前状态。
    返回包含第0步在内的最优前缀；候选进入路径前必须先通过可行性评价。
    不能仅因预算到期就输出路径末尾状态。"""
    if depth < 0 or beam < 1 or budget < 0:
        raise ValueError('depth/budget must be nonnegative and beam positive')
    current, current_result = initial, initial_result
    best, best_result = initial, initial_result
    seen = {fingerprint(initial)}
    attempts, chosen, worsening = 0, [], 0
    for step in range(depth):
        if attempts >= budget:
            break
        options, probes = [], 0
        for candidate, move in neighbors(current, current_result):
            ident = fingerprint(candidate)
            if ident in seen:
                continue
            seen.add(ident)
            trial = evaluate(candidate, move)
            attempts += 1
            probes += 1
            if trial is not None:
                options.append((key(trial), candidate, trial, move))
            if probes >= beam or attempts >= budget:
                break
        if not options:
            break
        _, candidate, trial, move = min(options, key=lambda x: x[0])
        is_worse = key(trial) > key(current_result)
        worsening += int(is_worse)
        chosen.append({'step': step+1, 'move': move, 'key': list(key(trial)),
                       'temporarily_worse': is_worse})
        current, current_result = candidate, trial
        if key(trial) < key(best_result):
            best, best_result = candidate, trial
        if attempts >= budget:
            break
    return best, best_result, {'attempted': attempts, 'walk': chosen,
                               'worsening_steps': worsening,
                               'returned_best_prefix': True,
                               'rolled_back': fingerprint(best) != fingerprint(current)}


def multi_search(ctx, budget, beam, depth):
    attempts, passes = 0, []
    while attempts < budget:
        before = ctx.best_result
        targeted = ctx.settings.hc_mode == 'targeted'
        def neighbors(state,result):
            stream = iter(mixed_neighborhoods(ctx.model,state,result,ctx.rng,ctx.hardware)
                if targeted else mixed_neighbors(ctx.model,state,result,ctx.rng))
            while True:
                try:
                    item = timed_call(ctx.deadline,next,stream)
                except StopIteration:
                    return
                yield item
        _, _, info = best_prefix_walk(
            ctx.best, ctx.best_result,
            neighbors,
            lambda state, move: ctx.evaluate(state, 'multi', move),
            depth=depth, beam=beam, budget=budget-attempts,
            fingerprint=semantic_digest if targeted else lambda state:state.fingerprint())
        attempts += info['attempted']
        passes.append(info)
        if info['attempted'] == 0 or score_key(ctx.best_result) >= score_key(before):
            break
    return {'attempted': attempts, 'passes': passes,
            'worsening_steps': sum(x['worsening_steps'] for x in passes)}


def refine_small(ctx, budgets, deadline=None):
    """供正式求解与受控 HC 实验共用的小邻域改进入口。"""
    ctx.deadline = deadline
    names = ('hc','multi') if ctx.settings.hc_mode == 'targeted' else ('order','migrate','swap','multi')
    hc_info = None
    for kind in names:
        before,start,count = ctx.best_result['makespan'],perf_counter(),ctx.evaluations
        status,info = 'completed',{}
        try:
            if kind == 'hc':
                if deadline is not None and budgets['multi']:
                    now = perf_counter(); ctx.deadline = now+.75*max(0,deadline-now)
                _,_,info = run_hc(ctx,ctx.best,ctx.best_result,
                    {k:budgets[k] for k in ('order','migrate','swap')})
                status,hc_info = info['status'],info
            elif kind == 'multi':
                ctx.deadline = deadline
                info = multi_search(ctx,budgets[kind],budgets['beam'],budgets['depth'])
            else:
                info = hill_climb(ctx,kind,budgets[kind])
        except TimeoutError:
            status = 'time_budget_exhausted'
        stage = dict(stage=kind,status=status,before=before,after=ctx.best_result['makespan'],
                     evaluations=ctx.evaluations-count,seconds=perf_counter()-start,details=info)
        ctx.stages.append(stage); ctx.emit(dict(event='stage_complete',**stage))
    ctx.deadline = deadline
    return hc_info


def optimize(model, cores, hardware, settings=None, on_event=None, warm_start=None,
             inherited=(), *, core_pool=None, on_pool_update=None, pool_seeds=(),
             case_budget=None, remaining_counts=1):
    mark_core(cores)
    settings = settings or Settings()
    settings.check()
    case_budget = case_budget or CaseBudget(spent=elapsed_seconds())
    checked_counts((cores,))
    owns_pool = core_pool is None
    if owns_pool:
        counts = tuple(range(2, cores+1)) if settings.core_sharing and cores > 1 else (cores,)
        core_pool = CoreSolutionPool(model, hardware, counts, cache_mb=settings.cache_mb,
                                     on_update=on_pool_update, sharing=settings.core_sharing)
    elif cores not in core_pool.counts:
        raise ValueError('requested count is missing from the shared solution pool')
    ctx = Context(model, hardware, settings, on_event, core_pool=core_pool)
    # 先完整评价可执行的兜底方案，并固定单核参考时间；后续可选搜索即使超时，也有已评分方案可保留。
    baseline = ctx.evaluate(model.fallback(cores), 'baseline')
    if baseline is None:
        raise ValueError('the one-core fallback is not executable; inspect graph/hardware')
    reference_t1 = baseline['makespan']
    # 在对可选工作设置时限之前，已准备好经过完整评分的兜底方案。
    initial_allowance = min(INITIAL_MAX_SECONDS,max(30.0,36*ctx.evaluation_estimate()),
                            remaining_seconds(),case_budget.search_allowance(remaining_counts))
    ctx.deadline = perf_counter()+max(0.0,initial_allowance)
    initial_status = 'completed'
    def optional_score(state,stage,move=None):
        nonlocal initial_status
        try:
            return ctx.evaluate(state,stage,move)
        except TimeoutError:
            initial_status = 'time_budget_exhausted'
            return None
    for label, state in pool_seeds:
        optional_score(state,str(label))
    pooled = core_pool.entries.get(cores)
    if pooled is not None and pooled['state'].fingerprint() != ctx.best.fingerprint():
        optional_score(pooled['state'], 'core_pool_seed', pooled['origin'])
    if warm_start is not None:
        if len(warm_start.cores) != cores:
            raise ValueError('warm start core count differs')
        optional_score(warm_start, 'warm_start')
    inherited_records = []
    for label, lower in inherited:
        state = lower.resized(cores)
        observed = optional_score(state, 'inherited', {'source': str(label),
                                'source_cores': len(lower.cores), 'target_cores': cores})
        inherited_records.append({'source': str(label), 'source_cores': len(lower.cores),
                                  'source_active_cores': sum(bool(row) for row in lower.cores),
                                  'target_cores': cores, 'valid': observed is not None,
                                  'makespan': observed['makespan'] if observed else None,
                                  'source_fingerprint': lower.fingerprint(),
                                  'padded_fingerprint': state.fingerprint()})
    seeds = []
    native, native_result = None, None
    # 分别构造各规则的初始候选，用完整内部评价器比较；启发式构造分数不能直接替代完工时间。
    for rule in (BSPG_RULES if settings.constructor == 'bspg' else RULES):
        t0 = perf_counter()
        try:
            if settings.constructor == 'bspg':
                state, info = timed_call(ctx.deadline,bspg_greedy,model,cores,hardware,rule,
                                        escape_weight=settings.escape_weight)
            else:
                state, info = timed_call(ctx.deadline,greedy,model,cores,hardware,rule,
                                     group_size=settings.group_size,ready_window=settings.ready_window)
        except TimeoutError:
            initial_status = 'time_budget_exhausted'
            break
        info['construction_seconds'] = perf_counter()-t0
        result = optional_score(state, 'greedy_'+rule)
        info.update(valid=result is not None, groups=len(state.groups),
                    makespan=result['makespan'] if result else None,
                    data_movement_bytes=result['data_movement_bytes'] if result else None,
                    fingerprint=state.fingerprint())
        seeds.append(info)
        if result is not None and (native_result is None or score_key(result) < score_key(native_result)):
            native, native_result = state, result
    bspg_best = ctx.best_result['makespan']
    native_seed_T = native_result['makespan'] if native_result else None
    ctx.emit({'event': 'bspg_complete', 'best_makespan': bspg_best})
    # 粗优化只生成和筛选候选；接受方案前仍需精确评价，并保留之前的最优解。
    coarse_started = perf_counter()
    ctx.deadline = perf_counter()+max(0.0,min(COARSE_MAX_SECONDS,
        max(3.0,9*ctx.evaluation_estimate()),remaining_seconds(),
        case_budget.search_allowance(remaining_counts)))
    bases = [('incumbent', ctx.best)]
    if native is not None and native.fingerprint() != ctx.best.fingerprint():
        bases.append(('native_bspg', native))
    try:
        proposals, coarse_info = timed_call(ctx.deadline,coarse_candidates,model,bases,cores,
                                            hardware,settings.coarse_candidates)
    except TimeoutError:
        proposals, coarse_info = [], {'status':'time_budget_exhausted'}
    coarse_records = []
    for state, info in proposals:
        trial = optional_score(state, 'coarse', info)
        coarse_records.append({**info, 'valid': trial is not None,
                               'makespan': trial['makespan'] if trial else None})
        if info['source'] == 'native_bspg' and trial is not None and (
                native_result is None or score_key(trial) < score_key(native_result)):
            native, native_result = state, trial
    coarse_info.update(before=bspg_best, after=ctx.best_result['makespan'],
                       seconds=perf_counter()-coarse_started, candidates=coarse_records)
    ctx.emit({'event': 'coarse_complete', **coarse_info})
    initial = ctx.best_result['makespan']
    initial_fingerprint = ctx.best.fingerprint()
    initial_rule = next((s['rule'] for s in seeds
                         if s['fingerprint'] == initial_fingerprint),
                        'warm_start_or_singlecore_fallback')
    for s in inherited_records:
        if s['padded_fingerprint'] == initial_fingerprint:
            initial_rule = 'inherited_' + str(s['source_cores'])
    if any(s['fingerprint'] == initial_fingerprint for s in coarse_records):
        initial_rule = 'scene_b_coarse'
    ctx.emit({'event': 'greedy_complete', 'best_makespan': initial})
    ctx.deadline = None
    mgr_config = MGR_PROFILES[settings.profile]
    budgets = PROFILES[settings.profile]
    from q2_refine import refine
    # 进入带早停的自适应搜索：依次分配局部爬山、多步搜索、宏观重构及末尾局部改进的时间。
    hc_info,mgr_info,before,timing = refine(ctx,cores,reference_t1,native,native_result,
        hardware,budgets,mgr_config,case_budget,remaining_counts)
    # 输出搜索全过程中已通过评价的最优方案，不直接采用最后访问的候选。
    result = ctx.best_result
    report = {'algorithm': 'problem2_latest_corepool_v1', 'revision': 'scene_b_core_pool_1',
              'settings': asdict(settings), 'budgets': budgets,
              'cores': cores, 'reference_T1': reference_t1, 'greedy_best_T': initial,
              'bspg_best_T': bspg_best, 'native_bspg_best_T': native_seed_T,
              'coarse': coarse_info, 'inherited_candidates': inherited_records,
              'mgr': mgr_info, 'pre_mgr_T': before, 'hc':hc_info,
              'time_budget':timing, 'initial_status':initial_status,
              'greedy_best_rule': initial_rule, 'greedy_fingerprint': initial_fingerprint,
              'final_T': result['makespan'], 'k': reference_t1/result['makespan'] if result['makespan'] else None,
              'k_per_core': reference_t1/result['makespan']/cores if result['makespan'] else None,
              'active_cores': result['active_cores'], 'subgraphs': len(ctx.best.groups),
              'seeds': seeds, 'stages': ctx.stages, 'evaluations': ctx.evaluations,
              'invalid_candidates': sum(e.get('valid') is False for e in ctx.events),
              'data_movement_bytes': result['data_movement_bytes'],
              'memory_peak_by_core': result['memory_peak_by_core'],
              'elapsed_seconds': perf_counter()-ctx.started,
              'final_fingerprint': ctx.best.fingerprint(),
              'internally_validated': True, 'officially_validated': False}
    ctx.emit({'event': 'complete', 'final_T': report['final_T']})
    plan, report = core_pool.final_result(cores, report)
    if owns_pool:
        report['core_solutions'] = core_pool.snapshot()
    return plan, report


def optimize_many(model, counts, hardware, settings=None, *, on_event=None,
                  on_pool_update=None, on_checkpoint=None, checkpoint=None, seeds=()):
    """在一次联合求解会话的所有阶段中共享精确评分最优解。
    
    本核心求解器独立于命令行、批量任务及文件存储。
    解池更新或阶段完成时，回调收到可原子保存的完整断点内容。
    恢复时重新回放已存方案，仅从保存的最优解继续未完成阶段，不逐条恢复随机数指令。"""
    settings = settings or Settings()
    settings.check()
    counts = checked_counts(counts)
    mark_core(counts[0])
    seeds = tuple(seeds)
    identity = digest({'graph': model.graph, 'hardware': hardware, 'counts': counts,
                       'settings': asdict(settings),
                       'seeds': [(str(label), state.fingerprint()) for label, state in seeds]})
    phase_reports, completed = {}, []
    current = None
    started = perf_counter()
    case_budget = CaseBudget(spent=elapsed_seconds()+
        (checkpoint.get('budget_elapsed_seconds',0.0) if checkpoint else 0.0))
    def save(pool):
        if on_checkpoint:
            on_checkpoint({'format_version': 1, 'identity': identity, 'counts': list(counts),
                           'settings': asdict(settings), 'completed_phases': list(completed),
                           'current_phase': current, 'phase_reports': dict(phase_reports),
                           'budget_elapsed_seconds':case_budget.elapsed(),
                           'pool': pool.snapshot()})
    def changed(event, pool):
        save(pool)
        if on_pool_update:
            on_pool_update(event, pool)
    pool = CoreSolutionPool(model, hardware, counts, cache_mb=settings.cache_mb,
                            on_update=changed, sharing=settings.core_sharing)
    if checkpoint is not None:
        if checkpoint.get('format_version') != 1 or checkpoint.get('identity') != identity:
            raise ValueError('resume refused: solver checkpoint inputs/settings/seeds changed')
        completed = list(checkpoint['completed_phases'])
        if completed != list(counts[:len(completed)]):
            raise ValueError('invalid completed phase prefix in checkpoint')
        phase_reports = dict(checkpoint['phase_reports'])
        if set(phase_reports) != {str(m) for m in completed}:
            raise ValueError('checkpoint phase reports differ from completed phases')
        pool.restore(checkpoint['pool'])
        if any(m not in pool.entries for m in completed):
            raise ValueError('checkpoint has a completed phase without its plan')
    # 即使优化后的一核方案进入解池，单核参考基准也保持固定。
    # 包括完整恢复在内，每次都重新计算同一参考定义下的基准。
    baseline = model.fallback(counts[0])
    value = pool.evaluator.evaluate(baseline.plan())
    pool.set_reference(value['makespan'])
    pool.offer(baseline, value, 'baseline')
    for label, state in seeds:
        pool.import_state(state, str(label))
    for m in counts:
        if m in completed:
            continue
        current = m
        inherited = [(f'pool_m{lower}', pool.entries[lower]['state'])
                     for lower in completed if settings.core_sharing]
        def phase_event(event, best, result):
            if on_event:
                on_event(dict(event, search_cores=m), best, result)
        _, report = optimize(model, m, hardware, settings, phase_event,
                             inherited=inherited, core_pool=pool,case_budget=case_budget,
                             remaining_counts=len(counts)-len(completed))
        phase_reports[str(m)] = report
        completed.append(m)
        current = None
        save(pool)
    results = {}
    for m in counts:
        plan, report = pool.final_result(m, phase_reports[str(m)])
        report['joint_core_counts'] = list(counts)
        report['session_elapsed_seconds'] = perf_counter()-started
        results[m] = plan, report
    save(pool)
    return results
