'对已完成的 BSPg 初解进行通信驱动的整 任务 聚合。\n\n这是一种与 HC 相关的贪心局部改进，不保证全局最优。仅合并完整 任务，同时考虑剩余算子、核内顺序及下游依赖。先用估计代价筛选，再以完整内部回放决定是否替换当前最优方案。'
from collections import defaultdict
from time import perf_counter

from dagp_scheduler import evaluate_plan
from hybrid_search import copy_plan, score_summary, plan_key
from dagp_execution import _preparation_deadline

ALGORITHM_REVISION = 'bspg_task_aggregation_v1'

def _blocks(plan):
    groups = defaultdict(set)
    for u, t in plan['node_to_subgraph'].items():
        groups[t].add(int(u))
    return {t: frozenset(ops) for t, ops in groups.items()}


def merge_tasks(plan, blocks, source, target):
    '将源 任务 的全部算子移入目标 任务，保留目标位置和编号。'
    result = copy_plan(plan)
    for u in blocks[source]:
        result['node_to_subgraph'][str(u)] = target
    for order in result['core_schedules']:
        if source in order:
            order.remove(source)
    new_blocks = dict(blocks)
    new_blocks[target] = new_blocks[target] | new_blocks.pop(source)
    return result, new_blocks


def proposals(model, plan, metrics, *, pool=96, shortlist=12):
    blocks = _blocks(plan)
    _, succ, _ = model.validate_partition(blocks)
    owner = {t: c for c, order in enumerate(plan['core_schedules']) for t in order}
    costs = {t: model.cost(ops) for t, ops in blocks.items()}
    pairs = set()
    for order in plan['core_schedules']:
        for a, b in zip(order, order[1:]):
            pairs.update(((a, b), (b, a)))
    for a, children in succ.items():
        for b in children:
            pairs.update(((a, b), (b, a)))
    # 即使不存在直接边，共享边界输入也可形成有价值的候选对。
    task_of = {u: t for t, ops in blocks.items() for u in ops}
    shared_pairs = defaultdict(int)
    for tensor in model.tensors.values():
        consumers = sorted({task_of[u] for u in tensor.consumers})
        # 只考察有限个相邻消费者，避免高扇出节点产生二次规模的候选完全图。
        for a, b in zip(consumers, consumers[1:]):
            shared_pairs[a, b] += tensor.size
    for (a, b), _ in sorted(shared_pairs.items(), key=lambda x: (-x[1], x[0]))[:pool]:
        pairs.update(((a, b), (b, a)))
    estimates = []
    for source, target in sorted(pairs):
        combined = model.cost(blocks[source] | blocks[target])
        if any(combined.peak_bytes[mem] > capacity
               for mem, capacity in model.hardware.capacity.items()):
            continue
        saved = costs[source].copy_bytes + costs[target].copy_bytes - combined.copy_bytes
        related = target in succ[source] or source in succ[target]
        wait = (model.hardware.same_wait if owner[source] == owner[target]
                else model.hardware.cross_wait if related else 0)
        # 全局排序优先选择合并规模较小、通信节省较大的块；后续代理评价显式考虑下游消费者的延迟。
        priority = (saved / model.hardware.bandwidth + wait) / max(1.0, combined.compute)
        estimates.append((-priority, -saved, source, target, saved))
    candidates = []
    for _, _, source, target, saved in sorted(estimates)[:pool]:
        candidate, candidate_blocks = merge_tasks(plan, blocks, source, target)
        try:
            predicted = evaluate_plan(model, candidate_blocks, candidate)
        except ValueError:
            continue
        candidates.append((predicted['makespan'], -saved, source, target, candidate,
                           {'source': source, 'target': target, 'saved_boundary_bytes': saved,
                            'same_core': owner[source] == owner[target],
                            'proxy_makespan': predicted['makespan']}))
    # 同时保留同核和跨核合并，避免代理校准在短评价预算中排除某一类候选。
    candidates.sort(key=lambda x: x[:4])
    chosen = []
    used = set()
    for same in (True, False):
        for row in candidates:
            if row[5]['same_core'] == same:
                chosen.append(row)
                used.add((row[2], row[3]))
                break
    for row in candidates:
        if len(chosen) >= shortlist:
            break
        if (row[2], row[3]) not in used:
            chosen.append(row)
            used.add((row[2], row[3]))
    return chosen


def aggregate_bspg_plan(model, evaluator, plan, metrics, *, seconds=30.0,
                        max_rounds=16, on_improvement=None):
    started = perf_counter()
    deadline = started + max(0.0, seconds)
    best_plan, best_metrics = copy_plan(plan), metrics
    original = metrics['makespan']
    logs, accepted_moves, seen = [], [], {plan_key(plan)}
    calls = accepted = 0
    stop = 'round_limit'
    def quality(observed):
        return observed['makespan'], observed['copy_bytes'], observed['task_count']
    for round_index in range(max_rounds):
        if perf_counter() >= deadline:
            stop = 'time_limit'
            break
        round_best = None
        # 仅当首批候选没有可接受合并时扩大候选范围，以修正代理排序误差，避免每次成功迭代都承担大候选池的成本。
        for pool, shortlist in ((96, 12), (256, 32)):
            try:
                with _preparation_deadline(deadline):
                    candidates = proposals(model, best_plan, best_metrics, pool=pool, shortlist=shortlist)
            except TimeoutError:
                stop = 'proposal_time_limit'
                break
            for _, _, _, _, candidate, info in candidates:
                key = plan_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                remaining = deadline - perf_counter()
                if remaining <= .005:
                    break
                try:
                    observed = evaluator.evaluate(candidate, deadline=perf_counter() + min(8.0, remaining))
                    calls += 1
                except TimeoutError:
                    logs.append(dict(info, round=round_index, status='timeout'))
                    continue
                logs.append(dict(info, round=round_index, status='evaluated',
                                 actual_makespan=observed['makespan'],
                                 actual_copy_bytes=observed['copy_bytes']))
                if (observed['makespan'] <= best_metrics['makespan']
                        and observed['copy_bytes'] <= best_metrics['copy_bytes']
                        and quality(observed) < quality(best_metrics)):
                    if round_best is None or quality(observed) < quality(round_best[1]):
                        round_best = candidate, observed, info
                        if on_improvement:
                            on_improvement(candidate, observed)
            if round_best is not None or perf_counter() >= deadline:
                break
        if round_best is None:
            stop = 'time_limit' if perf_counter() >= deadline else 'no_improvement_in_shortlist'
            break
        accepted_moves.append(dict(round_best[2], round=round_index,
            before=score_summary(best_metrics), after=score_summary(round_best[1])))
        best_plan, best_metrics, chosen = round_best
        accepted += 1
    return best_plan, best_metrics, dict(
        algorithm='BSPg + whole-Task communication aggregation',
        evaluation_kind='internal_full_shared_ddr', seconds=perf_counter() - started,
        budget_seconds=seconds, max_rounds=max_rounds, stop_reason=stop,
        initial_makespan=original, final_makespan=best_metrics['makespan'],
        detailed_evaluations=calls, accepted_merges=accepted,
        non_regression=(best_metrics['makespan'] <= original
                        and best_metrics['copy_bytes'] <= metrics['copy_bytes']),
        best_metrics=score_summary(best_metrics), accepted_moves=accepted_moves, log=logs)
