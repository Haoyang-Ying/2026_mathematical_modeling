'由第三问实际 COPY_IN 驱动的固定核分配优先级搜索。不增加等待、预取、依赖或修改输入图；移动只改变优先级分组及其顺序。每个候选从空 L2 开始回放全部核，保持初始节点到物理核的映射不变。'
from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import islice
from time import perf_counter

from q2_config import DEFAULT_SEED
from q2_model import Model, State, View, score_key
from q2_hc import Neighborhood, semantic_digest as _semantic_digest, static_feasible
from q2_macro import placement_of
from q3_evaluator import Problem3Evaluator
from q3_io import summary, digest
from q3_time import deadline as guard, commit_section


def semantic_digest(state):
    return _semantic_digest(state).hex()


@dataclass(frozen=True)
class OrderSettings:
    seed: int = DEFAULT_SEED
    max_evaluations: int = 96
    compile_cache_mb: int = 128
    search_seconds: float = 30
    max_targets: int = 24
    cache_guided: bool = True
    candidate_seconds: float = 8.0
    stagnation_seconds: float = 12.0

    def check(self):
        for name in ('max_evaluations', 'compile_cache_mb'):
            v = getattr(self, name)
            if type(v) is not int or v < 0:
                raise ValueError(name+' must be a nonnegative integer')
        if type(self.seed) is not int or type(self.max_targets) is not int or self.max_targets < 1:
            raise ValueError('seed must be an integer; max_targets must be a positive integer')
        if isinstance(self.search_seconds,bool) or not isinstance(self.search_seconds, (int, float)) or not 0 < self.search_seconds <= 600:
            raise ValueError('search_seconds must be finite and in (0,600]')
        if type(self.cache_guided) is not bool:
            raise ValueError('cache_guided must be boolean')
        for key in ('candidate_seconds','stagnation_seconds'):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not 0<value<=600:
                raise ValueError(key+' must be finite and in (0,600]')


class OrderCandidates:
    def __init__(self, model, state, result, hardware, settings):
        self.model, self.state, self.result = model, state, result
        self.view = View(model, state)
        self.placement = placement_of(state)
        self.settings = settings
        # 将确定性随机流绑定到当前最优解，使快照恢复
        # 不依赖已用时间或暂停的生成器状态。
        seed = settings.seed ^ int(semantic_digest(state)[:16], 16)
        self.rng = random.Random(seed)
        self.space = Neighborhood(model, state, result, self.rng, hardware)
        misses = [r for r in result['cache_read_records']
                  if r['miss_reason'] in ('inflight_first_load', 'evicted')]
        misses.sort(key=lambda r: (-r['size_bytes']*(1+r['end']/max(1, result['makespan'])),
                                  r['start'], r['core_id'], r['op_id']))
        selected, used = [], set()
        for r in misses:
            key = r['tensor_id'], r['core_id'], r['miss_reason']
            if key not in used:
                used.add(key); selected.append(r)
            if len(selected) >= settings.max_targets:
                break
        self.targets = selected

    def move_group(self, c, g, direction='both', target_time=None):
        old = self.state.cores[c]
        if g not in old:
            return
        i = old.index(g); rest = old[:i]+old[i+1:]
        lo, hi = self.view.insertion_bounds(g, rest)
        positions = []
        if target_time is not None:
            j = next((j for j, h in enumerate(rest)
                      if self.result['starts'].get(h, 0) >= target_time), len(rest))
            positions.extend((j, j-1))
        if direction in ('early', 'both'):
            positions.extend((lo, i-1, i-4, i-16))
        if direction in ('late', 'both'):
            positions.extend((i+1, i+4, i+16, hi))
        for j in dict.fromkeys(positions):
            if not lo <= j <= hi or j == i:
                continue
            orders = list(self.state.cores)
            orders[c] = rest[:j]+(g,)+rest[j:]
            yield State(self.state.groups, tuple(orders)), dict(core=c, group=g, old=i, new=j)

    def consumer_block(self, r):
        c, tid = r['core_id'], r['tensor_id']
        consumers = self.model.consumers.get(tid, ())
        groups = {self.view.mapping[v] for v in consumers if self.placement[v] == c}
        if not groups:
            return
        old = self.state.cores[c]
        block = tuple(g for g in old if g in groups)
        rest = tuple(g for g in old if g not in groups)
        if not block or not rest:
            return
        positions = {g: i for i, g in enumerate(rest)}
        lo = max((positions[p]+1 for g in block for p in self.view.pred[g]
                  if p in positions), default=0)
        hi = min((positions[p] for g in block for p in self.view.succ[g]
                  if p in positions), default=len(rest))
        old_first = old.index(block[0])
        target_time = r.get('first_insertion', {}).get('time', r['start'])
        natural = next((j for j, g in enumerate(rest)
                        if self.result['starts'].get(g, 0) >= target_time), len(rest))
        tries = (natural, old_first+1, old_first+4, hi) if r['miss_reason'] == 'inflight_first_load' else (lo, natural)
        for j in dict.fromkeys(tries):
            if not lo <= j <= hi:
                continue
            order = rest[:j]+block+rest[j:]
            if order == old:
                continue
            orders = list(self.state.cores); orders[c] = order
            yield State(self.state.groups, tuple(orders)), dict(core=c, groups=list(block), new=j)

    def cache(self):
        for r in self.targets:
            c, g = r['core_id'], r['subgraph_id']
            kind = r['miss_reason']
            direction = 'late' if kind == 'inflight_first_load' else 'early'
            target = r.get('first_insertion', {}).get('time')
            for candidate, move in self.move_group(c, g, direction, target):
                yield candidate, dict(move, family='cache', reason=kind, tensor=r['tensor_id'])
            for candidate, move in self.consumer_block(r):
                yield candidate, dict(move, family='cache', reason='consumer_block', tensor=r['tensor_id'])
            source = r.get('first_insertion') if kind == 'inflight_first_load' else r.get('last_eviction')
            if source:
                for candidate, move in self.move_group(source['core_id'], source['subgraph_id'],
                                                       'early' if kind == 'inflight_first_load' else 'late'):
                    yield candidate, dict(move, family='cache', reason='move_fill_or_evictor', tensor=r['tensor_id'])

    def paired(self):
        for r in self.targets:
            first = r.get('first_insertion')
            if r['miss_reason'] != 'inflight_first_load' or not first or first['core_id'] == r['core_id']:
                continue
            leader = list(islice(self.move_group(first['core_id'], first['subgraph_id'], 'early'), 3))
            follower = list(islice(self.move_group(r['core_id'], r['subgraph_id'], 'late', first['time']), 4))
            for a, ma in leader:
                for b, mb in follower:
                    orders = list(self.state.cores)
                    orders[ma['core']] = a.cores[ma['core']]
                    orders[mb['core']] = b.cores[mb['core']]
                    yield State(self.state.groups, tuple(orders)), dict(
                        family='paired', reason='earlier_fill_later_read', tensor=r['tensor_id'],
                        moves=[ma, mb])

    def fragment(self):
        for r in self.targets:
            c, tid = r['core_id'], r['tensor_id']
            nodes = tuple(v for v in self.model.consumers.get(tid, ()) if self.placement[v] == c)
            if not nodes:
                continue
            # 只抽取数量受限的原始计算节点，不移动自动生成的 COPY。
            nodes = tuple(sorted(nodes, key=self.model.position.__getitem__)[:16])
            modes = ('late', 'natural') if r['miss_reason'] == 'inflight_first_load' else ('early', 'natural')
            for mode in modes:
                candidate = self.space.splice([(nodes, c, mode)])
                if candidate is not None:
                    yield candidate, dict(family='fragment', reason='read_priority_regroup',
                                          tensor=tid, core=c, nodes=list(nodes), mode=mode)

    def basic(self):
        # 保留普通流水线、换出及尾部顺序移动，
        # 避免缓存命中收益独占搜索或替代原优化目标。
        groups = list(self.state.groups)
        self.rng.shuffle(groups)
        groups.sort(key=lambda g: -self.result['ends'].get(g, 0))
        for g in groups[:self.settings.max_targets]:
            c = self.view.core[g]
            for candidate, move in self.move_group(c, g):
                yield candidate, dict(move, family='basic', reason='pipe_tail_order')
        for candidate, move in islice(self.space.targeted_order(), 96):
            yield candidate, dict(move, family='basic', reason='existing_order_neighborhood')

    def generate(self):
        names = ('cache', 'paired', 'basic', 'fragment') if self.settings.cache_guided else ('basic',)
        generators = [iter(getattr(self, name)()) for name in names]
        while generators:
            live = []
            for gen in generators:
                try:
                    yield next(gen)
                    live.append(gen)
                except StopIteration:
                    pass
            generators = live


def _optimize_fixed(graph, plan, hardware, settings=None, *, checkpoint=None,
                   on_checkpoint=None, on_event=None, evaluator=None):
    settings = settings or OrderSettings()
    settings.check()
    started = perf_counter()
    previous_seconds = checkpoint.get('elapsed_seconds',0.0) if checkpoint else 0.0
    deadline = started + settings.search_seconds - previous_seconds
    model = Model(graph)
    initial = State.from_plan(model, plan)
    frozen = placement_of(initial)
    identity = digest(dict(graph=graph, plan=initial.plan(), hardware=hardware,
                           settings=asdict(settings)))
    if checkpoint is not None and checkpoint.get('identity') != identity:
        raise ValueError('checkpoint input/hardware/settings mismatch')
    evaluator = evaluator or Problem3Evaluator(graph, **hardware,
                                               cache_bytes=settings.compile_cache_mb*1024**2)
    baseline = evaluator.evaluate(initial.plan(), hc_guidance=True, deadline=deadline)
    best = State.from_plan(model, checkpoint['best_plan']) if checkpoint else initial
    if len(best.cores) != len(initial.cores) or placement_of(best) != frozen:
        raise ValueError('checkpoint changes the fixed node-to-core mapping')
    result = evaluator.evaluate(best.plan(), hc_guidance=True, deadline=deadline) if checkpoint else baseline
    if score_key(result) > score_key(baseline):
        raise ValueError('checkpoint is worse than the required initial incumbent')
    result['hc_guidance']['source_fingerprint'] = best.fingerprint()
    seen = set(checkpoint.get('seen', [])) if checkpoint else set()
    seen.add(semantic_digest(best))
    events = list(checkpoint.get('events', [])) if checkpoint else []
    attempted = checkpoint.get('attempted', 0) if checkpoint else 0
    accepted = checkpoint.get('accepted', 0) if checkpoint else 0
    valid = checkpoint.get('valid', 0) if checkpoint else 0
    filtered = checkpoint.get('filtered', 0) if checkpoint else 0
    family_counts = Counter(checkpoint.get('family_counts', {})) if checkpoint else Counter()
    status = 'running'
    last_improvement=perf_counter()

    def save():
        snap = dict(version=1, identity=identity, best_plan=best.plan(),
                    baseline=summary(baseline), best=summary(result), seen=sorted(seen),
                    attempted=attempted, accepted=accepted, valid=valid, filtered=filtered,
                    family_counts=dict(family_counts), events=events, status=status,
                    elapsed_seconds=previous_seconds+perf_counter()-started)
        if on_checkpoint:
            on_checkpoint(snap)
        return snap

    save()
    while attempted < settings.max_evaluations:
        if deadline is not None and perf_counter() >= deadline:
            status = 'time_budget_exhausted'; break
        if valid>=3 and perf_counter()-last_improvement>=settings.stagnation_seconds:
            status = 'stagnated'; break
        space = OrderCandidates(model, best, result, hardware, settings)
        changed, tried = False, False
        for candidate, move in space.generate():
            if attempted >= settings.max_evaluations:
                break
            if deadline is not None and perf_counter() >= deadline:
                status = 'time_budget_exhausted'; break
            key = semantic_digest(candidate)
            if key in seen:
                continue
            if len(candidate.cores) != len(initial.cores) or placement_of(candidate) != frozen:
                raise AssertionError('fixed-core candidate invariant violated')
            if not static_feasible(model, candidate):
                seen.add(key); filtered += 1
                continue
            tick = perf_counter()
            observed, error = None, None
            try:
                candidate_end=min(deadline,perf_counter()+settings.candidate_seconds)
                with guard(candidate_end):
                    observed = evaluator.evaluate(candidate.plan(), hc_guidance=True, deadline=candidate_end)
            except TimeoutError:
                evaluator.clear_cache()
                if perf_counter()>=deadline:
                    status = 'time_budget_exhausted'; break
                error='candidate_time_budget_exhausted'
            except (ValueError, RuntimeError) as exc:
                error = str(exc)[:600]
            attempted += 1; tried = True; seen.add(key)
            family_counts[move['family']] += 1
            entry = dict(index=attempted, move=move, fingerprint=key,
                         seconds=perf_counter()-tick, valid=observed is not None, accepted=False)
            if observed is None:
                entry['error'] = error
            else:
                valid += 1
                entry.update(makespan=observed['makespan'], hit_rate=observed['cache_stats']['hit_rate'])
                if score_key(observed) < score_key(result):
                    best, result = candidate, observed
                    result['hc_guidance']['source_fingerprint'] = best.fingerprint()
                    accepted += 1; changed = True; entry['accepted'] = True
                    last_improvement=perf_counter()
            events.append(entry)
            if on_event:
                on_event(entry)
            save()
            if changed:
                break
        if status == 'time_budget_exhausted':
            break
        if not changed:
            status = 'evaluation_budget_exhausted' if attempted >= settings.max_evaluations else 'neighborhood_exhausted'
            break
    if status == 'running':
        status = 'evaluation_budget_exhausted'
    final_checkpoint = save()
    assert placement_of(best) == frozen and len(best.cores) == len(initial.cores)
    assert score_key(result) <= score_key(baseline)
    report = dict(problem=3, optimization='fixed_core_read_order', settings=asdict(settings),
                  status=status, baseline=summary(baseline), final=summary(result),
                  time_reduction_cycles=baseline['makespan']-result['makespan'],
                  time_reduction_fraction=1-result['makespan']/baseline['makespan'] if baseline['makespan'] else 0,
                  fixed_core_assignment_verified=True, assignment_sha256=digest(sorted(frozen.items())),
                  attempted=attempted, valid=valid, accepted=accepted, filtered=filtered,
                  family_counts=dict(family_counts), seconds_this_invocation=perf_counter()-started,
                  officially_validated=False, events=events)
    return best.plan(), report, final_checkpoint


def optimize_fixed(graph,plan,hardware,settings=None,*,checkpoint=None,on_checkpoint=None,on_event=None,evaluator=None):
    settings=settings or OrderSettings();settings.check()
    previous=checkpoint.get('elapsed_seconds',0.0) if checkpoint else 0.0
    end=perf_counter()+max(0.0,settings.search_seconds-previous)
    saved=[]
    def persist(snapshot):
        with commit_section():
            saved[:]=[snapshot]
            if on_checkpoint:on_checkpoint(snapshot)
    try:
        with guard(end):
            return _optimize_fixed(graph,plan,hardware,settings,checkpoint=checkpoint,
                on_checkpoint=persist,on_event=on_event,evaluator=evaluator)
    except TimeoutError:
        if not saved:raise
        snap=saved[-1]
        report=dict(problem=3,optimization='fixed_core_read_order',status='time_budget_exhausted',
            baseline=snap['baseline'],final=snap['best'],attempted=snap['attempted'],
            accepted=snap['accepted'],fixed_core_assignment_verified=True,
            internally_validated=True,internal_recheck_passed=False,officially_validated=False)
        return snap['best_plan'],report,snap
