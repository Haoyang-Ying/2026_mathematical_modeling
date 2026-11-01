'通过子进程限制执行时间的第三问接口，仅调用内部评价器。'
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from q3_runtime_io import read,atomic
from q3_time import TimePolicy


def solve_many(graph,cores=(2,3,4,5),*,config=None,seed_plans=(),inherit=True,
               official=False,on_event=None,on_checkpoint=None,on_pool_update=None,
               checkpoint=None,out=None,resume=False,time_policy=None,**options):
    '返回经过严格内部复核的 {m: (plan, report)}。每次调用均使用受监督的子进程，也支持 Windows 中的模块调用。指定 out 并启用 resume 可累计时间预算；回调接收主进程的进度和快照。未完成的批次抛出 TimeoutError，不返回未经核验的分数。'
    if official:raise ValueError('official evaluation is disabled; internal only')
    if checkpoint is not None:
        raise ValueError('resume via out=... and resume=True; in-memory old checkpoints are not accepted')
    if type(inherit) is not bool:raise ValueError('inherit must be boolean')
    from q2_search import Settings
    from q3_io import hardware3
    from q3_timed_runner import run_cases
    p=asdict(time_policy) if isinstance(time_policy,TimePolicy) else dict(time_policy or {})
    for key in asdict(TimePolicy()):
        if key in options:p[key]=options.pop(key)
    options.setdefault('search_seconds',p.get('base_seconds',30.0))
    p['base_seconds']=options['search_seconds']
    policy=TimePolicy(**p).check()
    options.setdefault('core_sharing',inherit)
    if not inherit and options['core_sharing']:raise ValueError('inherit=False conflicts with core_sharing=True')
    settings=Settings(**options);settings.check()
    graph=graph if isinstance(graph,dict) else read(graph)
    seeds=[s if isinstance(s,dict) else read(s) for s in seed_plans]
    with tempfile.TemporaryDirectory(prefix='q3_api_input_') as temporary:
        directory=Path(out).resolve() if out else Path(temporary)/'result'
        directory.mkdir(parents=True,exist_ok=True)
        # 在接口输出目录中保存稳定的输入路径，
        # 使恢复运行不依赖上一次临时目录的名称。
        inputs=directory/'.inputs';inputs.mkdir(exist_ok=True)
        path=inputs/'graph.json'
        if path.exists():
            if read(path)!=graph:raise ValueError('resume refused: API input graph changed')
        else:atomic(path,graph)
        job=dict(case=str(path),name='api_case',out=str(directory),hardware=hardware3(config),seed_plans=seeds)
        def event(e):
            if on_event:on_event(e,None,None)
        def saved(snapshot):
            if on_checkpoint:on_checkpoint(snapshot)
            if on_pool_update:on_pool_update({'event':'pool_snapshot'},snapshot)
        summary=run_cases([job],cores,asdict(settings),policy,out=directory,resume=resume,
                          on_event=event,on_checkpoint=saved)
        if summary['status']!='complete':
            raise TimeoutError('Q3 budget ended with incomplete internal rechecks; checkpoints: '+str(directory))
        return {m:(read(directory/f'm{m}'/'schedule.json'),read(directory/f'm{m}'/'report.json')) for m in sorted(set(cores))}


def solve(graph,cores=4,**options):
    return solve_many(graph,(cores,),**options)[cores]


def validate(graph,plan,*,config=None,seconds=60.0):
    '在受时间限制的子进程中对固定调度方案进行内部回放。'
    from q3_io import hardware3
    from q3_validation_runner import validate_bounded
    graph=graph if isinstance(graph,dict) else read(graph)
    plan=plan if isinstance(plan,dict) else read(plan)
    return validate_bounded(graph,plan,hardware3(config),seconds)


def evaluate(graph,plan,*,config=None,seconds=60.0,**options):
    if options:raise ValueError('bounded evaluate supports config and seconds only')
    return validate(graph,plan,config=config,seconds=seconds)['result']


def optimize_order(graph,plan,*,config=None,search_seconds=30.0,**options):
    from api_q3 import optimize_order as bounded_order
    return bounded_order(graph,plan,config=config,search_seconds=search_seconds,**options)
