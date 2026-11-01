'在有限时间内进行内部评价和固定核分配下的顺序优化。'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from q3_io import hardware3
from q3_runtime_io import read
from q3_validation_runner import validate_bounded,order_bounded


def evaluate(graph,plan,*,config=None,compile_cache_mb=128,diagnostics=False,seconds=60.0):
    graph=graph if isinstance(graph,dict) else read(graph)
    plan=plan if isinstance(plan,dict) else read(plan)
    return validate_bounded(graph,plan,hardware3(config),seconds,diagnostics=diagnostics,
        compile_cache_mb=compile_cache_mb,include_reference=False)['result']


def optimize_order(graph,plan,*,config=None,official=False,on_checkpoint=None,on_event=None,
                   checkpoint=None,search_seconds=30.0,**options):
    if official:raise ValueError('official evaluation is disabled; internal only')
    graph=graph if isinstance(graph,dict) else read(graph)
    plan=plan if isinstance(plan,dict) else read(plan)
    return order_bounded(graph,plan,hardware3(config),search_seconds,options,
        checkpoint=checkpoint,on_checkpoint=on_checkpoint,on_event=on_event)


def validate(graph,plan,*,config=None,seconds=60.0):
    graph=graph if isinstance(graph,dict) else read(graph)
    plan=plan if isinstance(plan,dict) else read(plan)
    return validate_bounded(graph,plan,hardware3(config),seconds)
