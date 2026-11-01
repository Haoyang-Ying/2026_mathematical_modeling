'完整竞争 BSPg 与 MGR/HC 求解流程；全部计算进程统一使用随机种子 1234。'
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from reproducibility import initialize_seed, child_environment, seed_metadata, PACKAGE_REVISION
initialize_seed(reexec_script=__name__ == '__main__')

from dagp_model import Model,read_hardware
from dagp_execution import TemplateEvaluator
from guided_initial import generate_guided_plan
from stage_guidance import generate_stage_plan,pad_plan
from hybrid_search import score_summary
from search_worker import atomic_json
from time_budget import base_seconds as budget_seconds, refinement_budget, POLICY, SCALE
from core_limits import supervise, mark_core, remaining_seconds, worker_guard
import bspg_original
import bspg_shared
from bspg_aggregate import aggregate_bspg_plan, ALGORITHM_REVISION


def plan_hash(plan):
    return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _refine(case_path,config_path,initial_path,m,mode,seconds,destination,initial,metrics,
            alternate_paths=(),time_budget=None):
    # 为当前核数的最终提交和进程回收保留时间；所有小项跟随实际搜索额度同比例收缩。
    requested = seconds
    cleanup = budget_seconds(2.0)
    seconds = min(seconds, max(0.0, remaining_seconds()-cleanup))
    if seconds <= 0:
        return initial, {'best_metrics':score_summary(metrics), 'final_makespan':metrics['makespan'],
                         'stop_reason':'core_budget_reserved_for_export', 'wall_seconds':0.0,
                         'search_budget_seconds':0.0, 'time_budget':time_budget}
    if time_budget is not None:
        time_budget['dynamic_factor'] *= seconds/requested
        time_budget['effective_search_seconds'] = seconds
        time_budget['fixed_slice_multiplier'] = SCALE*time_budget['dynamic_factor']
    request = destination.with_suffix('.request.json')
    checkpoint = destination.with_suffix('.checkpoint.json')
    log_path = destination.with_suffix('.log')
    def portable(path):
        try:return str(Path(path).relative_to(ROOT))
        except ValueError:return str(path)
    atomic_json(request,{'case_path':portable(case_path),'config_path':portable(config_path),
                         'initial_path':portable(initial_path),'m':m,'macro':mode=='mgr',
                         'seconds':seconds,'alternate_paths':[portable(p) for p in alternate_paths],
                         'time_budget':time_budget, **worker_guard(), **seed_metadata()})
    # 子进程启动前，主进程已保存完成评价的备选方案，以便超时后回退。
    atomic_json(checkpoint,{'plan':initial,'statistics':{
        'best_metrics':score_summary(metrics),'final_makespan':metrics['makespan'],
        'initial_makespan':metrics['makespan'],'stop_reason':'initial_scored_checkpoint',
        'search_budget_seconds':seconds,'time_budget':time_budget, **seed_metadata()}})
    started = perf_counter()
    killed = False
    with log_path.open('w',encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable,str(ROOT/'src/search_worker.py'),
                                    str(request),str(checkpoint)],stdout=log,stderr=log,cwd=ROOT,
                                    env=child_environment())
        try:
            process.wait(timeout=max(.001,min(seconds+cleanup/2,remaining_seconds()-cleanup/4)))
        except subprocess.TimeoutExpired:
            killed = True
            process.kill()
            process.wait(timeout=cleanup/2)
        except BaseException:
            process.kill();process.wait(timeout=cleanup/2)
            raise
    if process.returncode not in (0,3) and not killed:
        raise RuntimeError(f'{mode} worker failed: '+log_path.read_text(encoding='utf-8')[-4000:])
    payload = json.loads(checkpoint.read_text(encoding='utf-8'))
    payload['statistics'].update(wall_seconds=perf_counter()-started,
                                 hard_stopped=killed,worker_exit_code=process.returncode,
                                 time_budget=time_budget)
    return payload['plan'],payload['statistics']


def solve(case_path,cores,out,*,mode='mgr',guide_seconds=budget_seconds(2.0),search_seconds=budget_seconds(30.0),
          makespan_only=False,coarse_group_policy='validated',aggregate_seconds=budget_seconds(15.0)):
    """各核数独立累计构造、搜索及评价时间，每项最多 600 秒。"""
    cores=tuple(sorted(set(cores)))
    if not cores or any(type(m) is not int or not 1<=m<=5 for m in cores):
        raise ValueError('核数必须为 1 至 5 的整数。')
    return supervise(_solve,(case_path,cores,out),dict(mode=mode,guide_seconds=guide_seconds,
        search_seconds=search_seconds,makespan_only=makespan_only,
        coarse_group_policy=coarse_group_policy,aggregate_seconds=aggregate_seconds),first_core=min(cores))


def _single_plan(case_path):
    """单核采用完整图方案，仍须在相同硬限内完成合法性检查和内部评价。"""
    case_path=Path(case_path)
    model=Model(json.loads(case_path.read_text(encoding='utf-8')),read_hardware(case_path.parent/'config.txt'))
    plan={'node_to_subgraph':{str(node):0 for node in model.ids},
          'core_schedules':[[0]] if model.ids else [[]]}
    model.validate_plan(plan,1)
    with TemplateEvaluator(model) as evaluator:
        evaluator.evaluate(plan)
    return plan


def single_plan(case_path):
    """单核入口与多核入口使用一致的进程监督。"""
    return supervise(_single_plan,(case_path,),first_core=1)


def _solve(case_path,cores,out,*,mode,guide_seconds,search_seconds,
           makespan_only,coarse_group_policy,aggregate_seconds):
    '先保存全部初解，再统一评价；独立核验由外部流程执行。'
    initialize_seed()
    out = Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    case_path = Path(case_path).resolve()
    config_path = case_path.parent/'config.txt'
    case = json.loads(case_path.read_text(encoding='utf-8'))
    hardware = read_hardware(config_path)
    model = Model(case,hardware)
    name = case_path.stem
    needed = ([1] if 1 in cores else [])+list(range(2,max(cores)+1))
    # 原 BSPg、共享输入与结构指导方案共同参与初解竞争。
    seeds,records = {},[]
    for m in needed:
        mark_core(m)
        for label,generator in [('original',bspg_original.generate_bspg_plan),
                                ('shared',bspg_shared.generate_bspg_plan),('guided_raw',None)]:
            before = perf_counter();stats = {}
            if generator:
                plan = generator(case,m,bandwidth=hardware.bandwidth,same_wait=hardware.same_wait,
                                 cross_wait=hardware.cross_wait,statistics=stats)
            else:
                plan = generate_guided_plan(case,m,hardware,model=model,
                                            guide_seconds=guide_seconds,statistics=stats,
                                            group_policy=coarse_group_policy)
            # 搜索结束后检查算子覆盖和调度无环性。
            model.validate_plan(plan,m)
            stats['generation_seconds'] = perf_counter()-before
            path = out/'plans'/label/f'm{m}'/(name+'.json')
            atomic_json(path,plan)
            atomic_json(path.with_suffix('.stats.json'),stats)
            seeds[label,m] = (plan,path,stats)
            print(json.dumps({'generated':name,'m':m,'algorithm':label,
                              'seconds':round(stats['generation_seconds'],3)}),flush=True)
    # 容量指导的低核方案可借用空闲核处理低成本前缀；完整评价前先生成并保存这些候选。
    for m in needed:
        mark_core(m)
        for r in needed:
            if r>=m:continue
            base,_,source_stats=seeds['guided_raw',r]
            guidance=source_stats.get('guidance',{})
            if not any(d.get('kind')=='capacity_split' for d in guidance.get('decisions',[])):
                continue
            stats={}
            plan=generate_stage_plan(case,m,hardware,model,guidance['coarse_plan'],
                                     seconds=guide_seconds,statistics=stats)
            if plan is None:continue
            label=f'stage_from_{r}'
            path=out/'plans'/label/f'm{m}'/(name+'.json')
            atomic_json(path,plan);atomic_json(path.with_suffix('.stats.json'),stats)
            seeds[label,m]=(plan,path,stats)
            print(json.dumps({'generated':name,'m':m,'algorithm':label,
                              'seconds':round(stats['generation_seconds'],3)}),flush=True)
    with TemplateEvaluator(model) as evaluator:
        # 共享准备与单核基准评价计入本次首先处理的核数。
        mark_core(min(cores))
        single = {'node_to_subgraph':{str(u):0 for u in model.ids},
                  'core_schedules':[[0]] if model.ids else [[]]}
        # 即使不输出加速比，动态时间分配仍需要单核基准 T1。
        needs_reference = not makespan_only or mode in ('hc','mgr','all')
        single_metrics = evaluator.evaluate(single) if needs_reference else {'makespan':None}
        if needs_reference:
            single_path = out/'plans/single'/(name+'.json');atomic_json(single_path,single)
        def ratios(metrics,m):
            t1=single_metrics['makespan'] if not makespan_only else None
            return {'single_makespan':t1,'speedup':t1/max(1,metrics['makespan']) if t1 is not None else None,
                    'k_over_m':t1/max(1,m*metrics['makespan']) if t1 is not None else None}
        rows = []
        for (label,m),(plan,path,stats) in seeds.items():
            mark_core(m)
            metrics = evaluator.evaluate(plan)
            stats.update(evaluation_kind='internal_full_shared_ddr',metrics=score_summary(metrics))
            atomic_json(path.with_suffix('.stats.json'),stats)
            row = {'case':name,'m':m,'algorithm':label,'path':str(path.relative_to(out)),
                   'plan_sha256':plan_hash(plan),'metrics':score_summary(metrics),
                   'generation_seconds':stats['generation_seconds'],
                   **ratios(metrics,m)}
            rows.append(row)
            # 每完成一次评价即保存，避免后续用例或候选失败时丢失已完成数据。
            atomic_json(out/f'{name}.records.json',rows)
        records.extend(rows)
        initial_by_m = {}
        raw_bspg_by_m = {}
        aggregation_changed_by_m = {}
        for m in sorted(cores):
            mark_core(m)
            candidates = []
            for (label,r),(plan,path,stats) in seeds.items():
                if r>m or (r==1 and m>1):
                    continue
                padded = pad_plan(plan,m)
                # 对补齐空核后的实际方案进行评价，并保留全部已有 任务 编号。
                observed = evaluator.evaluate(padded)
                candidates.append((observed['makespan'],0 if label=='original' and r==m else 1,
                                   r,label,padded,observed))
            # 按完整内部耗时择优，耗时相同时按既定规则稳定选择。
            best = min(candidates,key=lambda x:x[:4])
            baseline = min((x for x in candidates if x[3]=='original'),key=lambda x:x[:4])
            for label,entry in [('original_best',baseline),('bspg',best)]:
                _,_,r,source,plan,metrics = entry
                path = out/'plans'/label/f'm{m}'/(name+'.json')
                atomic_json(path,plan)
                row = {'case':name,'m':m,'algorithm':label,'path':str(path.relative_to(out)),
                       'plan_sha256':plan_hash(plan),'metrics':score_summary(metrics),
                       'seed_algorithm':source,'seed_source_cores':r,
                       **ratios(metrics,m)}
                records.append(row);atomic_json(path.with_suffix('.stats.json'),row)
            initial_by_m[m] = (best[4],best[5],out/'plans/bspg'/f'm{m}'/(name+'.json'))
            raw_bspg_by_m[m] = (plan_hash(best[4]),out/'plans/bspg'/f'm{m}'/(name+'.json'))
            aggregation_changed_by_m[m] = False
            # 对完整 任务 尝试通信聚合，并用内部回放控制接受与回退。
            if aggregate_seconds > 0 and m > 1:
                destination = out/'plans/bspg_aggregate'/f'm{m}'/(name+'.json')
                atomic_json(destination,best[4])
                def save_aggregate(candidate,observed):
                    atomic_json(destination,candidate)
                    atomic_json(destination.with_suffix('.checkpoint.json'),
                                {'plan':candidate,'metrics':score_summary(observed)})
                candidate,observed,stats = aggregate_bspg_plan(
                    model,evaluator,best[4],best[5],seconds=aggregate_seconds,
                    on_improvement=save_aggregate)
                stats.update(seed_metadata())
                stats['competition'] = {
                    'original_makespan': best[5]['makespan'],
                    'selected_makespan': observed['makespan'],
                    'original_copy_bytes': best[5]['copy_bytes'],
                    'selected_copy_bytes': observed['copy_bytes'],
                    'winner': 'bspg' if plan_hash(candidate) == plan_hash(best[4]) else 'bspg_aggregate',
                }
                if (observed['makespan'] > best[5]['makespan']
                        or observed['copy_bytes'] > best[5]['copy_bytes']):
                    raise AssertionError('aggregation discarded a better original seed')
                atomic_json(destination,candidate)
                atomic_json(destination.with_suffix('.stats.json'),stats)
                row = {'case':name,'m':m,'algorithm':'bspg_aggregate',
                       'path':str(destination.relative_to(out)),'plan_sha256':plan_hash(candidate),
                       'metrics':score_summary(observed),'aggregate_seconds':aggregate_seconds,
                       **ratios(observed,m)}
                records.append(row)
                initial_by_m[m] = (candidate,observed,destination)
                aggregation_changed_by_m[m] = plan_hash(candidate) != plan_hash(best[4])
                atomic_json(out/f'{name}.records.json',records)
                print(json.dumps({'aggregated':name,'m':m,'before':best[5]['makespan'],
                                  'after':observed['makespan']}),flush=True)
        atomic_json(out/f'{name}.records.json',records)
    # 低核最优方案仅在同一搜索流程中继承；保留原生 m 核候选作为第二起点，各起点共用总预算。
    refined={}
    with TemplateEvaluator(model) as inherited_evaluator:
      for m in sorted(cores):
        mark_core(m)
        for search_mode in (['hc','mgr'] if mode=='all' else [mode] if mode in ['hc','mgr'] else []):
            initial,metrics,path=initial_by_m[m]
            source={'algorithm':path.parent.parent.name,'cores':m}
            lower=refined.get(search_mode)
            # 只在低核结果严格更优时继承，并补齐空核列表。
            if lower:
                lower_m,lower_plan,lower_metrics=lower
                padded=pad_plan(lower_plan,m)
                observed=inherited_evaluator.evaluate(padded)
                if observed['makespan']<metrics['makespan']:
                    initial,metrics=padded,observed
                    path=out/'plans'/f'{search_mode}_inherited'/f'm{m}'/(name+'.json')
                    atomic_json(path,initial)
                    source={'algorithm':search_mode,'cores':lower_m}
            # 初解可能来自低核方案，因此保留使用更多核的不同起点。通信聚合得到的方案作为额外起点，不替代聚合前的 BSPg 起点；即使两者占用核数相同，也保留不同方案。
            alternate=[]
            raw_hash,raw_path=raw_bspg_by_m[m]
            if aggregation_changed_by_m[m] and raw_hash!=plan_hash(initial):
                alternate.append(raw_path)
            already_selected={plan_hash(initial)}
            if alternate:already_selected.add(raw_hash)
            native=[r for r in records if r['m']==m and r['algorithm'] in
                    ['original','shared','guided_raw','bspg_aggregate']
                    and r['plan_sha256'] not in already_selected
                    and r['metrics']['active_cores']>metrics['active_cores']]
            if native and search_mode=='mgr':
                alternate.append(out/min(native,key=lambda r:r['metrics']['makespan'])['path'])
            destination=out/'plans'/search_mode/f'm{m}'/(name+'.json')
            destination.parent.mkdir(parents=True,exist_ok=True)
            # 初解择优和低核继承完成后冻结时间系数；较差备选或试探候选不会获得新截止时间或额外倍率。
            timing=refinement_budget(search_seconds,m,single_metrics['makespan'],metrics['makespan'])
            plan,stats=_refine(case_path,config_path,path,m,search_mode,timing['effective_search_seconds'],
                              destination,initial,metrics,alternate,time_budget=timing)
            model.validate_plan(plan,m)
            observed=stats['best_metrics']
            # 最终结果不得劣于进入搜索前的可行方案。
            if observed['makespan']>metrics['makespan']:
                raise AssertionError('refinement discarded a better incumbent')
            stats['initial_source']=source
            atomic_json(destination,plan);atomic_json(destination.with_suffix('.stats.json'),stats)
            refined[search_mode]=(m,plan,observed)
            records.append({'case':name,'m':m,'algorithm':search_mode,
                            'path':str(destination.relative_to(out)),'plan_sha256':plan_hash(plan),
                            'metrics':observed,'search_seconds':timing['effective_search_seconds'],
                            'time_budget':timing,
                            'wall_seconds':stats['wall_seconds'],'initial_source':source,
                            **ratios(observed,m)})
            atomic_json(out/f'{name}.records.json',records)
            print(json.dumps({'completed':name,'m':m,'algorithm':search_mode,
                              'makespan':observed['makespan']}),flush=True)
    return records
