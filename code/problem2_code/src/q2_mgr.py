"""带预算的多起点宏观重构、受保护局部修复与回退。"""
from __future__ import annotations

from time import perf_counter
import math
from q2_budget import timed_call
from q2_config import (SECOND_START_SHARE, PROTECTED_REPAIR_SHARE,
                       MGR_REPAIR_SHARE, MGR_REPAIR_MAX_SECONDS)

from q2_model import score_key
from q2_macro import macro_candidates, placement_of, semantic_key


def _outside(state, selected):
    return tuple(tuple(v for g in order for v in sorted(state.groups[g]) if v not in selected)
                 for order in state.cores)


def repair(ctx, state, result, selected, budget, *, protect_assignment, label):
    """局部下降从当前宏观候选开始，即使它劣于全局最优方案。
    
    先修复区域内顺序，仅在第二阶段允许改变核心分配。
    两个阶段均保护区域外操作的核心分配及顺序。"""
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


def run_mgr(ctx, starts, hardware, *, budget, rounds, repair_budget, max_nodes, round_offset=0):
    """共用评价次数硬限，最多分配给两个不同起点。
    
    每个起点预留一部分剩余时间；准备和评价均服从该起点截止时间。
    到期后保留已有可行最优方案。"""
    unique, known = [], set()
    for name, state, result in starts:
        key = semantic_key(state)
        if key not in known:
            known.add(key)
            unique.append((name, state, result))
        if len(unique) == 2:
            break
    output = {'starts': [], 'budget': budget, 'attempted': 0, 'round_limit': rounds,
              'returned_global_best': True, 'diagnostics_replays': 0,
              'completed_macro_candidates':0, 'no_regions':False}
    if budget <= 0 or rounds <= 0:
        output['disabled'] = True
        return output
    global_deadline = ctx.deadline
    available_seconds = None if global_deadline is None else max(0.0,global_deadline-perf_counter())
    second_seconds = None if available_seconds is None else SECOND_START_SHARE*available_seconds
    begin = ctx.evaluations
    try:
        for index, (name, start, start_result) in enumerate(unique):
            remaining_starts = len(unique)-index
            remaining_budget = budget-(ctx.evaluations-begin)
            allowance = (max(1,int(remaining_budget*(1-SECOND_START_SHARE)))
                         if remaining_starts>1 else remaining_budget)
            if allowance <= 0:
                break
            if global_deadline is not None:
                now = perf_counter()
                if now >= global_deadline:
                    break
                ctx.deadline = (global_deadline-second_seconds if index==0 and remaining_starts>1
                                else min(global_deadline,now+second_seconds) if index else global_deadline)
            start_count = ctx.evaluations
            local, local_result = start, start_result
            record = {'name': name, 'initial_T': start_result['makespan'],
                      'initial_fingerprint': start.fingerprint(), 'allowance': allowance,
                      'rounds': [], 'status': 'completed'}
            ctx.emit({'event': 'mgr_start', **{k: v for k, v in record.items() if k != 'rounds'}})
            def left():
                return allowance-(ctx.evaluations-start_count)
            try:
                # 当前核数原生初解不一定经历过前面的 HC。
                if index and left() > 2:
                    local, local_result, record['initial_repair'] = repair(
                        ctx, local, local_result, ctx.model.nodes, min(repair_budget, left()-1),
                        protect_assignment=True, label='mgr_initial_repair')
                for round_index in range(rounds):
                    if left() < 2:
                        break
                    observed = ctx.evaluate(local, 'mgr_diagnose',
                                            {'start': name, 'round': round_index}, diagnostics=True)
                    output['diagnostics_replays'] += 1
                    if observed is None:
                        raise RuntimeError('previously valid MGR incumbent became invalid')
                    proposals, region = timed_call(ctx.deadline,macro_candidates,ctx.model,local,
                        observed['diagnostics'],hardware,round_offset+round_index+index,max_nodes=max_nodes)
                    if not proposals:
                        output['no_regions'] = True
                    r = {'round': round_index, 'before': local_result['makespan'],
                         'region': region, 'candidates': []}
                    round_base = local_result
                    for candidate, move in proposals:
                        if left() <= 0:
                            break
                        # 同一轮的各兄弟候选均从相同基准方案重构。
                        start_deadline = ctx.deadline
                        if start_deadline is not None:
                            now = perf_counter()
                            ctx.deadline = now+.65*max(0.0,start_deadline-now)
                        try:
                            trial = ctx.evaluate(candidate,'mgr_macro',dict(move,start=name,round=round_index))
                        except TimeoutError:
                            record['status'] = 'time_budget_exhausted'
                            r['candidates'].append({'kind':move['kind'],'status':'timed_out'})
                            continue
                        finally:
                            ctx.deadline = start_deadline
                        entry = {'kind': move['kind'], 'valid': trial is not None,
                                 'affected_cores': move['affected_cores'],
                                 'moved_nodes': move['moved_nodes']}
                        r['candidates'].append(entry)
                        if trial is None:
                            continue
                        output['completed_macro_candidates'] += 1
                        entry.update(raw_T=trial['makespan'], worse_than_round_base=
                                     score_key(trial) > score_key(round_base))
                        allowance_repair = min(repair_budget, left())
                        protected = math.ceil(allowance_repair*PROTECTED_REPAIR_SHARE)
                        repaired,repaired_result = candidate,trial
                        first,second = {'attempted':0},{'attempted':0}
                        repair_begin_count = ctx.evaluations
                        repair_end = ctx.deadline
                        if repair_end is not None:
                            now = perf_counter()
                            estimate = getattr(ctx,'evaluation_estimate',lambda:.001)()
                            seconds = min(MGR_REPAIR_MAX_SECONDS,
                                max(estimate,MGR_REPAIR_SHARE*available_seconds),
                                .4*max(0.0,repair_end-now))
                            repair_end = min(repair_end,now+seconds)
                            ctx.deadline = now+PROTECTED_REPAIR_SHARE*seconds
                        try:
                            repaired,repaired_result,first = repair(ctx,candidate,trial,
                                move['regional_nodes'],protected,protect_assignment=True,
                                label='mgr_protected_repair')
                        except TimeoutError:
                            record['status'] = 'time_budget_exhausted'
                            first = {'attempted':ctx.evaluations-repair_begin_count,'status':'timed_out'}
                        finally:
                            ctx.deadline = repair_end
                        second_budget = min(allowance_repair-first['attempted'], left())
                        try:
                            repaired,repaired_result,second = repair(ctx,repaired,repaired_result,
                                move['regional_nodes'],second_budget,protect_assignment=False,
                                label='mgr_free_repair')
                        except TimeoutError:
                            record['status'] = 'time_budget_exhausted'
                            second = {'attempted':ctx.evaluations-repair_begin_count-first['attempted'],
                                      'status':'timed_out'}
                        finally:
                            ctx.deadline = start_deadline
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
