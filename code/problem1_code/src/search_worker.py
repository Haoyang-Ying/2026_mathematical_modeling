'由 run.py 控制的独立 HC/MGR 搜索进程，支持检查点保存。'
from reproducibility import initialize_seed, seed_metadata, RANDOM_SEED
if __name__ == '__main__':
    initialize_seed(reexec_script=True)

import json
import os
from pathlib import Path
import sys
import threading
import time
from time import perf_counter

from dagp_model import Model,read_hardware
from dagp_execution import TemplateEvaluator
from hybrid_search import HybridSearch,score_summary,plan_key
from time_budget import seconds as budget_seconds, dynamic_budget


def atomic_json(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    # 原子替换避免中断时留下半写入的正式文件。
    os.replace(temporary,path)


def main(request_path,result_path):
    initialize_seed()
    started = perf_counter()
    request = json.loads(Path(request_path).read_text(encoding='utf-8'))
    if request.get('random_seed', RANDOM_SEED) != RANDOM_SEED:
        raise ValueError('This package requires random_seed=1234')
    timing = request.get('time_budget') or {}
    stop = threading.Event()
    # 内层搜索进程共享外层核数硬限；监督者退出时不继续占用计算资源。
    def watch():
        while not stop.wait(.05):
            now = time.monotonic()
            if now >= request.get('core_deadline_monotonic',float('inf')):
                os._exit(124)
            owner = request.get('supervisor_heartbeat_path')
            if owner:
                try:
                    heartbeat=json.loads(Path(owner).read_text())['heartbeat']
                except (OSError,ValueError,KeyError):
                    os._exit(125)
                if now-heartbeat > 8:
                    os._exit(125)
    threading.Thread(target=watch,daemon=True).start()
    try:
        with dynamic_budget(timing.get('dynamic_factor',1.0)):
            _run(request,result_path,started)
    finally:
        stop.set()


def _run(request,result_path,started):
    try:
        import resource
        limit=1536*1024*1024
        soft,hard=resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS,(min(soft,limit) if soft>0 else limit,hard))
    except (ImportError,ValueError,OSError):
        pass
    # 所有搜索起点共享此绝对截止时刻，不按起点重新计时。
    deadline = started+request['seconds']
    case = json.loads(Path(request['case_path']).read_text(encoding='utf-8'))
    hardware = read_hardware(request['config_path'])
    model = Model(case,hardware)
    initial = json.loads(Path(request['initial_path']).read_text(encoding='utf-8'))
    with TemplateEvaluator(model,deadline=deadline) as evaluator:
        before = perf_counter()
        metrics = evaluator.evaluate(initial)
        preparation = perf_counter()-before
        best_plan,best_metrics,best_stats=initial,metrics,{}
        initial_makespan=metrics['makespan'];runs=[];start_index=0
        starts=[(initial,request['initial_path'])]
        for path in request.get('alternate_paths',[])[:2]:
            alternate=json.loads(Path(path).read_text(encoding='utf-8'))
            if all(plan_key(alternate)!=plan_key(p) for p,_ in starts):
                starts.append((alternate,path))
        # 只有完整评价后的改进才能更新检查点，超时沿用已保存的可行解。
        def checkpoint(plan,stats):
            nonlocal best_plan,best_metrics,best_stats
            if stats['best_metrics']['makespan']<=best_metrics['makespan']:
                best_plan,best_metrics,best_stats=plan,stats['best_metrics'],dict(stats)
                best_stats['best_start_index']=start_index
            aggregate=dict(best_stats,initial_makespan=initial_makespan,
                           best_metrics=score_summary(best_metrics),final_makespan=best_metrics['makespan'],
                           worker_seconds=perf_counter()-started,search_budget_seconds=request['seconds'],
                           time_budget=request.get('time_budget'),
                           fixed_slice_multiplier=budget_seconds(1.0),
                           starts_planned=len(starts),completed_starts=runs,
                           non_regression_verified=best_metrics['makespan']<=initial_makespan,
                           **seed_metadata())
            try:aggregate['peak_rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            except (NameError,AttributeError):pass
            atomic_json(result_path,{'plan':best_plan,'statistics':aggregate})
        for start_index,(plan,source_path) in enumerate(starts):
            if perf_counter()+budget_seconds(.05)>=deadline:break
            before=perf_counter()
            current=metrics if start_index==0 else evaluator.evaluate(plan,deadline=deadline)
            preparation=preparation if start_index==0 else perf_counter()-before
            remaining=deadline-perf_counter()
            # 为备选起点预留 25% 时间；原 BSPg 与额外原生核数起点同时存在时，分别预留 15% 和 10%。未用完的时间直接交给下一起点。
            fraction=(.75 if start_index==0 and len(starts)>1 else
                      .6 if start_index==1 and len(starts)>2 else 1.0)
            end=perf_counter()+fraction*remaining
            search=HybridSearch(model,evaluator,request['m'],end,started,
                                on_checkpoint=checkpoint,enable_macro=request['macro'])
            result,stats=search.run(plan,current,preparation)
            runs.append({'start_index':start_index,'source_path':source_path,
                         'initial_makespan':current['makespan'],
                         'final_makespan':stats['final_makespan'],
                         'seconds':perf_counter()-before,'stop_reason':stats.get('stop_reason')})
            checkpoint(result,stats)


if __name__=='__main__':
    try:
        main(*sys.argv[1:])
    except (TimeoutError,MemoryError):
        # 保留此前完整检查点（若存在），供主进程读取。
        raise SystemExit(3)
