'在预算内进行多起点宏重构、受保护的局部修复及回退。'
from __future__ import annotations

from time import perf_counter
from contextlib import nullcontext

from q2_model import score_key
from q2_macro import macro_candidates, placement_of, semantic_key


def _outside(state, selected):
    return tuple(tuple(v for g in order for v in sorted(state.groups[g]) if v not in selected)
                 for order in state.cores)


def _repair(ctx, state, result, selected, budget, *, protect_assignment, label):
    '局部下降从当前宏候选出发，即使它劣于全局最优解。先优化区域内顺序，第二阶段才允许改变核分配；两个阶段均保护区域外的算子放置及顺序。'
    if getattr(getattr(ctx,'settings',None),'hc_mode','legacy') == 'targeted':
        from q2_hc import run_hc
        kinds = ('order',) if protect_assignment else ('migrate','swap','order')
        state,result,info = run_hc(ctx,state,result,{k:budget for k in kinds},
            total_budget=budget,selected=selected,protect_assignment=protect_assignment,
            label=label,family_order=kinds)
        info['accepted'] = info['accepted_moves']
        return state,result,info
    from q2_search import NEIGHBORS
    selected = set(selected)
    exterior, placement = _outside(state, selected), placement_of(state)
    seen, attempted, accepted = {semantic_key(state)}, 0, 0
    while attempted < budget:
        kinds = ('order',) if protect_assignment else ('order', 'migrate', 'swap')
        guidance = dict(result, anchor_groups=[g for g, vs in state.groups.items()
                                              if any(v in selected for v in vs)])
        generators = [iter(NEIGHBORS[k](ctx.model, state, guidance, ctx.rng)) for k in kinds]
        improved, scanned = False, 0
        while generators and attempted < budget and scanned < max(64, 24*budget):
            live = []
            for gen in generators:
                try:
                    candidate, move = next(gen)
                except StopIteration:
                    continue
                live.append(gen)
                scanned += 1
                key = semantic_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                if _outside(candidate, selected) != exterior:
                    continue
                if protect_assignment and placement_of(candidate) != placement:
                    continue
                attempted += 1
                trial = ctx.evaluate(candidate, label, dict(move, protected=protect_assignment))
                if trial is not None and score_key(trial) < score_key(result):
                    state, result = candidate, trial
                    improved, accepted = True, accepted+1
                    break
                if attempted >= budget:
                    break
            if improved:
                break
            generators = live
        if not improved:
            break
    return state, result, {'attempted': attempted, 'accepted': accepted,
                           'assignment_protected': protect_assignment}


def repair(ctx, state, result, selected, budget, *, protect_assignment, label):
    if not hasattr(ctx, 'policy'):
        return _repair(ctx,state,result,selected,budget,protect_assignment=protect_assignment,label=label)
    previous, before = ctx.deadline, ctx.evaluations
    ctx.deadline = min(previous, perf_counter()+ctx.policy.repair_seconds)
    try:
        with ctx.operation():
            return _repair(ctx,state,result,selected,budget,protect_assignment=protect_assignment,label=label)
    except TimeoutError:
        return state,result,dict(attempted=ctx.evaluations-before, accepted=0,
            assignment_protected=protect_assignment,status='time_budget_exhausted')
    finally:
        ctx.deadline = previous


def run_mgr(ctx, starts, hardware, *, budget, rounds, repair_budget, max_nodes):
    '至多两个不同起点共用严格的评价次数限制。编译使用软时间预算，各起点预留剩余时间的一部分；超时不会替换最后一个可行最优方案。'
    unique, known = [], set()
    for name, state, result in starts:
        key = semantic_key(state)
        if key not in known:
            known.add(key)
            unique.append((name, state, result))
        if len(unique) == 2:
            break
    output = {'starts': [], 'budget': budget, 'attempted': 0, 'round_limit': rounds,
              'returned_global_best': True, 'diagnostics_replays': 0}
    if budget <= 0 or rounds <= 0:
        output['disabled'] = True
        return output
    global_deadline = ctx.deadline
    begin = ctx.evaluations
    try:
        for index, (name, start, start_result) in enumerate(unique):
            if hasattr(ctx, 'check_work'):
                ctx.check_work()
            remaining_starts = len(unique)-index
            remaining_budget = budget-(ctx.evaluations-begin)
            allowance = remaining_budget//remaining_starts
            if allowance <= 0:
                break
            if global_deadline is not None:
                now = perf_counter()
                if now >= global_deadline:
                    break
                share = .75 if hasattr(ctx, 'policy') and remaining_starts == 2 else 1/remaining_starts
                ctx.deadline = now+(global_deadline-now)*share
            start_count = ctx.evaluations
            local, local_result = start, start_result
            record = {'name': name, 'initial_T': start_result['makespan'],
                      'initial_fingerprint': start.fingerprint(), 'allowance': allowance,
                      'rounds': [], 'status': 'completed'}
            ctx.emit({'event': 'mgr_start', **{k: v for k, v in record.items() if k != 'rounds'}})
            def left():
                return allowance-(ctx.evaluations-start_count)
            try:
                # 原生 m 核初解不一定已经执行前置 HC。
                if index and left() > 2:
                    local, local_result, record['initial_repair'] = repair(
                        ctx, local, local_result, ctx.model.nodes, min(repair_budget, left()-1),
                        protect_assignment=True, label='mgr_initial_repair')
                for round_index in range(rounds):
                    if hasattr(ctx, 'check_work'):
                        ctx.check_work()
                    if left() < 2:
                        break
                    observed = ctx.evaluate(local, 'mgr_diagnose',
                                            {'start': name, 'round': round_index}, diagnostics=True)
                    output['diagnostics_replays'] += 1
                    if observed is None:
                        if hasattr(ctx, 'policy'):
                            record['status'] = 'diagnostic_budget_exhausted'
                            break
                        raise RuntimeError('previously valid MGR incumbent became invalid')
                    with ctx.operation() if hasattr(ctx, 'operation') else nullcontext():
                        proposals, region = macro_candidates(ctx.model, local, observed['diagnostics'],
                            hardware, round_index+index, max_nodes=max_nodes)
                    r = {'round': round_index, 'before': local_result['makespan'],
                         'region': region, 'candidates': []}
                    round_base = local_result
                    for candidate, move in proposals:
                        if hasattr(ctx, 'check_work'):
                            ctx.check_work()
                        if left() <= 0:
                            break
                        # 同一轮的所有兄弟候选都从相同的基础方案重构。
                        trial = ctx.evaluate(candidate, 'mgr_macro', dict(move, start=name, round=round_index))
                        entry = {'kind': move['kind'], 'valid': trial is not None,
                                 'affected_cores': move['affected_cores'],
                                 'moved_nodes': move['moved_nodes']}
                        r['candidates'].append(entry)
                        if trial is None:
                            continue
                        entry.update(raw_T=trial['makespan'], worse_than_round_base=
                                     score_key(trial) > score_key(round_base))
                        allowance_repair = min(repair_budget, left())
                        protected = (allowance_repair+1)//2
                        repaired, repaired_result, first = repair(ctx, candidate, trial,
                            move['regional_nodes'], protected, protect_assignment=True,
                            label='mgr_protected_repair')
                        second_budget = min(allowance_repair-first['attempted'], left())
                        repaired, repaired_result, second = repair(ctx, repaired, repaired_result,
                            move['regional_nodes'], second_budget, protect_assignment=False,
                            label='mgr_free_repair')
                        entry.update(repaired_T=repaired_result['makespan'],
                                     protected_repair=first, free_repair=second)
                        if score_key(repaired_result) < score_key(local_result):
                            local, local_result = repaired, repaired_result
                    r.update(after=local_result['makespan'])
                    record['rounds'].append(r)
                    ctx.emit({'event': 'mgr_round_complete', 'start': name, **r,
                              'global_best_T': ctx.best_result['makespan']})
            except TimeoutError:
                record['status'] = 'time_budget_exhausted'
            record.update(final_T=local_result['makespan'], evaluations=ctx.evaluations-start_count,
                          final_fingerprint=local.fingerprint())
            output['starts'].append(record)
            ctx.deadline = global_deadline
    finally:
        ctx.deadline = global_deadline
    output['attempted'] = ctx.evaluations-begin
    output['final_T'] = ctx.best_result['makespan']
    return output
