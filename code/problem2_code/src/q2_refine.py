"""短搜索窗口、按实测代价分配时间片，以及由改进收益决定的延时。"""
import math
from time import perf_counter

from q2_budget import initial_seconds
from q2_config import (TIME_POLICY, SEARCH_SHARES, SEARCH_EXTENSION_SECONDS,
    NORMAL_SEARCH_SECONDS, MIN_RELATIVE_GAIN, STALE_WINDOWS)
from q2_hc import run_hc
from q2_limits import remaining_seconds
from q2_mgr import run_mgr


def refine(ctx, cores, t1, native, native_result, hardware, budgets, mgr_config,
           case_budget, remaining_counts):
    from q2_search import multi_search, hill_climb
    settings = ctx.settings
    # 根据初解效率分配首次搜索时间，并服从当前核数的剩余预算。
    first, factor = initial_seconds(settings.search_seconds, cores, t1,
                                     ctx.best_result['makespan'])
    estimate = ctx.evaluation_estimate()
    hard_remaining = remaining_seconds()
    reserve = max(1.0, 2*estimate)
    # 先为最终导出及回放保留时间，搜索可用额度取各级上限的最小值。
    allowance = max(0.0, min(settings.search_max_seconds,
        case_budget.search_allowance(remaining_counts, estimate),
        hard_remaining-reserve))
    mgr_limit = ((mgr_config['budget'] if settings.mgr_budget is None else settings.mgr_budget)
                 if settings.mgr else 0)
    left = {k:budgets[k] for k in ('order','migrate','swap','multi')}
    left['mgr'] = mgr_limit
    initial = ctx.best_result['makespan']
    started = perf_counter()
    outer_end = started+allowance
    committed = min(first, allowance)
    windows, hc_calls, mgr_calls = [], [], []
    aggregates = {k:dict(stage=k, status='completed', before=initial, after=initial,
                        evaluations=0, seconds=0.0, details={'slices':[]})
                  for k in ('hc','multi','mgr')}
    hc_covered, mgr_covered = set(), not bool(mgr_limit)
    stale = 0
    pre_mgr = initial
    # 标准搜索始终启用；是否提前结束由收敛与预算规则决定。
    stop = None
    window_seconds = committed
    index = 0
    # 每个搜索窗口后检查改善收益和停滞条件；只有满足续时条件才追加预算。
    while stop is None:
        now = perf_counter()
        if window_seconds <= 0 or now >= outer_end:
            stop = 'time_budget_exhausted'; break
        if not any(left.values()):
            stop = 'evaluation_caps_exhausted'; break
        begin = now
        end = min(outer_end, begin+window_seconds)
        before = ctx.best_result['makespan']
        completed_before = ctx.completed_evaluations
        slices = []
        exhausted = set()
        shares = dict(SEARCH_SHARES)
        if not mgr_limit:
            shares['hc'] += shares.pop('mgr')
        boundary = begin
        for name, share in shares.items():
            boundary += share*(end-begin)
            kind = 'hc' if name == 'final_hc' else name
            now = perf_counter()
            estimate = ctx.evaluation_estimate()
            if name=='multi' and mgr_limit and share*(end-begin)<1.25*estimate:
                slices.append({'phase':name,'status':'reallocated_to_macro_probe'})
                continue
            # 避免分配不足以完成一次评价的碎片时间；未用时间向后结转。
            stage_end = min(end, max(boundary, now+1.25*estimate))
            if name != 'final_hc':
                stage_end = min(stage_end, end-min(.15*(end-begin), estimate))
            if now >= stage_end or stage_end-now < .8*estimate:
                slices.append({'phase':name, 'status':'insufficient_probe_time'})
                continue
            if kind == 'hc' and not any(left[k] for k in ('order','migrate','swap')):
                continue
            if kind != 'hc' and left[kind] <= 0:
                continue
            ctx.deadline = stage_end
            ctx.current_phase = name
            count, phase_before = ctx.evaluations, ctx.best_result['makespan']
            tick = perf_counter()
            status, info = 'completed', {}
            try:
                if kind == 'hc':
                    if settings.hc_mode == 'targeted':
                        _, _, info = run_hc(ctx, ctx.best, ctx.best_result,
                            {k:left[k] for k in ('order','migrate','swap')})
                        status = info['status']
                        for family, record in info['families'].items():
                            left[family] = max(0,left[family]-record['attempted'])
                            if record['valid'] or (record['attempted']==0 and status=='completed'):
                                hc_covered.add(family)
                    else:
                        info = {'families':{}}
                        for family in ('order','migrate','swap'):
                            n = ctx.evaluations
                            try:
                                info['families'][family] = hill_climb(ctx,family,left[family])
                                hc_covered.add(family)
                            finally:
                                left[family] = max(0,left[family]-(ctx.evaluations-n))
                    hc_calls.append(info)
                    if info.get('neighborhoods_exhausted'):
                        exhausted.add('hc')
                elif kind == 'multi':
                    info = multi_search(ctx,left['multi'],budgets['beam'],budgets['depth'])
                    if not info['attempted']:
                        exhausted.add('multi')
                else:
                    if not mgr_calls:
                        pre_mgr = phase_before
                    starts = [('incumbent',ctx.best,ctx.best_result)]
                    if native is not None:
                        starts.append(('native_m_core',native,native_result))
                    info = run_mgr(ctx,starts,hardware,budget=left['mgr'],
                        rounds=mgr_config['rounds'],repair_budget=mgr_config['repair'],
                        max_nodes=settings.mgr_max_nodes,round_offset=index*mgr_config['rounds'])
                    mgr_calls.append(info)
                    mgr_covered |= bool(info.get('completed_macro_candidates') or info.get('no_regions'))
                    if any(s['status']=='time_budget_exhausted' for s in info['starts']):
                        status = 'time_budget_exhausted'
                    if info.get('no_regions') and not info['completed_macro_candidates'] and status=='completed':
                        exhausted.add('mgr')
            except TimeoutError:
                status = 'time_budget_exhausted'
            finally:
                used = ctx.evaluations-count
                if kind in ('multi','mgr'):
                    left[kind] = max(0,left[kind]-used)
                ctx.deadline = None
            record = dict(phase=name,status=status,before=phase_before,
                after=ctx.best_result['makespan'],evaluations=used,seconds=perf_counter()-tick,
                details=info)
            slices.append(record)
            aggregate = aggregates[kind]
            if not aggregate['details']['slices']:
                aggregate['before'] = phase_before
            aggregate['after'] = record['after']
            aggregate['seconds'] += record['seconds']
            aggregate['evaluations'] += used
            aggregate['status'] = status
            aggregate['details']['slices'].append(record)
            ctx.emit(dict(event='search_slice_complete',window=index,**record))
        gain = max(0.0,(before-ctx.best_result['makespan'])/max(1.0,before))
        completed = ctx.completed_evaluations-completed_before
        eligible_hc = {k for k in ('order','migrate','swap') if budgets[k]}
        covered = eligible_hc <= hc_covered and mgr_covered
        # 仅发生超时不能作为搜索已经停滞的证据。
        if completed and covered:
            stale = 0 if gain >= MIN_RELATIVE_GAIN else stale+1
        window = {'window':index,'seconds':perf_counter()-begin,'granted_seconds':end-begin,
            'before':before,'after':ctx.best_result['makespan'],'relative_gain':gain,
            'completed_evaluations':completed,'basic_coverage':covered,
            'stale_windows':stale,'slices':slices}
        windows.append(window)
        ctx.emit(dict(event='search_window_complete',**{k:v for k,v in window.items() if k!='slices'}))
        enabled = {'hc','multi'} | ({'mgr'} if mgr_limit else set())
        if gain==0 and enabled <= exhausted:
            stop = 'neighborhoods_exhausted'; break
        if stale >= STALE_WINDOWS:
            stop = 'stagnation'; break
        if not any(left.values()):
            stop = 'evaluation_caps_exhausted'; break
        if committed >= allowance or perf_counter() >= outer_end:
            stop = 'time_budget_exhausted'; break
        if committed >= NORMAL_SEARCH_SECONDS and gain < MIN_RELATIVE_GAIN:
            stop = 'normal_limit_without_recent_gain'; break
        # 高代价评价可能需要合并多个续时增量，才能容纳一次诊断回放和宏观候选评价；
        # 合并后仍须服从普通搜索与最大搜索时间上限。
        wanted = SEARCH_EXTENSION_SECONDS*max(1,math.ceil(
            3*ctx.evaluation_estimate()/SEARCH_EXTENSION_SECONDS))
        limit = allowance if gain >= MIN_RELATIVE_GAIN else min(allowance,NORMAL_SEARCH_SECONDS)
        window_seconds = min(wanted,limit-committed)
        if window_seconds <= 0:
            stop = 'normal_limit_without_recent_gain'; break
        committed += window_seconds
        index += 1
    ctx.deadline = None
    ctx.current_phase = None
    for stage in aggregates.values():
        if not stage['details']['slices']:
            stage.update(before=ctx.best_result['makespan'],after=ctx.best_result['makespan'],
                         status=stop or 'skipped')
        ctx.stages.append(stage)
        ctx.emit(dict(event='stage_complete',**stage))
    mgr_info = {'starts':[s for call in mgr_calls for s in call['starts']],
        'windows':mgr_calls,'budget':mgr_limit,
        'attempted':sum(call['attempted'] for call in mgr_calls),
        'disabled':not bool(mgr_limit) or stop=='search_disabled',
        'returned_global_best':True,'final_T':ctx.best_result['makespan']}
    timing = {'policy':TIME_POLICY,'base_search_seconds':settings.search_seconds,
        'dynamic_factor':factor,'factor_frozen':True,'initial_search_seconds':first,
        'allowed_search_seconds':allowance,'committed_search_seconds':committed,
        'search_seconds':perf_counter()-started,'normal_limit_seconds':NORMAL_SEARCH_SECONDS,
        'maximum_search_seconds':settings.search_max_seconds,'stop_reason':stop,
        'case_target_seconds':case_budget.target,'case_elapsed_seconds':case_budget.elapsed(),
        'stage_shares':SEARCH_SHARES,'windows':windows}
    hc_info = {'slices':hc_calls,'attempted':sum(x.get('attempted',0) for x in hc_calls),
        'accepted_moves':sum(x.get('accepted_moves',0) for x in hc_calls),
        'first_probe_families':list(dict.fromkeys(k for x in hc_calls for k in x.get('first_probe_families',[]))),
        'evaluation_order':[k for x in hc_calls for k in x.get('evaluation_order',[])],
        'allocation_cost':'observed_seconds','families':{}}
    for call in hc_calls:
        for name, stats in call.get('families',{}).items():
            total = hc_info['families'].setdefault(name,{})
            for key,value in stats.items():
                if isinstance(value,(int,float)):
                    total[key] = total.get(key,0)+value
    return hc_info,mgr_info,pre_mgr,timing
