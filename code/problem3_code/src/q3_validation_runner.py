'为固定方案评价和固定核分配下的顺序搜索提供子进程边界。'
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

from q3_runtime_io import atomic, read

ROOT=Path(__file__).resolve().parents[1]


def bounded(request, seconds, *, on_checkpoint=None, wall_seconds=None):
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not 0<seconds<=600:
        raise ValueError('seconds must be finite and in (0,600]')
    used=request.get('checkpoint',{}).get('elapsed_seconds',0) if request.get('checkpoint') else 0.0
    used=max(used,request.get('used_seconds',0))
    if not isinstance(used,(int,float)) or not math.isfinite(used) or used<0:
        raise ValueError('invalid cumulative elapsed time')
    remaining=seconds-used
    if wall_seconds is not None:
        if not math.isfinite(wall_seconds) or wall_seconds<=0:raise ValueError('wall_seconds must be positive and finite')
        remaining=min(remaining,wall_seconds)
    if remaining<=0:
        raise TimeoutError('cumulative fixed-order time budget exhausted')
    started=perf_counter()
    with tempfile.TemporaryDirectory(prefix='q3_fixed_worker_') as temp:
        root=Path(temp);req=root/'request.json';result=root/'result.json';cp=root/'checkpoint.json'
        cleanup=min(.5,max(.005,.05*remaining))
        request=dict(request,deadline=started+remaining-cleanup,checkpoint_path=str(cp),result_path=str(result))
        atomic(req,request)
        with (root/'worker.log').open('w',encoding='utf-8') as log:
            worker=subprocess.Popen([sys.executable,str(ROOT/'src/q3_validation_worker.py'),str(req)],
                                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            aborted=None;last_heartbeat=0.0
            try:
                while worker.poll() is None:
                    left=started+remaining-cleanup/2-perf_counter()
                    if left<=0:
                        worker.kill();worker.wait();break
                    try:worker.wait(timeout=min(.1,left))
                    except subprocess.TimeoutExpired:pass
                    if on_checkpoint and cp.exists() and perf_counter()-last_heartbeat>=.5:
                        snapshot=read(cp)
                        snapshot['elapsed_seconds']=min(seconds,used+perf_counter()-started+1.0)
                        on_checkpoint(snapshot);last_heartbeat=perf_counter()
            except BaseException as exc:
                worker.kill();worker.wait();aborted=exc
        saved=read(cp) if cp.exists() else None
        if saved is not None:
            saved['elapsed_seconds']=used+perf_counter()-started
            if on_checkpoint:on_checkpoint(saved)
        if aborted is not None:raise aborted
        if result.exists() and worker.returncode==0:
            payload=read(result)
            payload['elapsed_seconds']=used+perf_counter()-started
            if saved is not None:payload['checkpoint']=saved
            return payload
        if request['mode']=='order' and saved is not None:
            return dict(plan=saved['best_plan'],checkpoint=saved,report=dict(
                problem=3,optimization='fixed_core_read_order',status='time_budget_exhausted',
                baseline=saved['baseline'],final=saved['best'],
                attempted=saved['attempted'],accepted=saved['accepted'],
                fixed_core_assignment_verified=True,internally_validated=True,
                internal_recheck_passed=False,officially_validated=False,
                elapsed_seconds=used+perf_counter()-started))
        if worker.returncode not in (0,124,137,-9):
            raise RuntimeError((root/'worker.log').read_text()[-4000:])
        raise TimeoutError('internal replay did not finish before its deadline')


def validate_bounded(graph,plan,hardware,seconds=60.0,*,diagnostics=False,compile_cache_mb=128,include_reference=True):
    return bounded(dict(mode='validate',graph=graph,plan=plan,hardware=hardware,
        diagnostics=diagnostics,compile_cache_mb=compile_cache_mb,include_reference=include_reference),seconds)


def order_bounded(graph,plan,hardware,seconds=30.0,options=None,*,checkpoint=None,on_checkpoint=None,on_event=None,
                  wall_seconds=None,used_seconds=0.0,memory_mb=1536):
    settings=dict(options or {},search_seconds=seconds)
    result=bounded(dict(mode='order',graph=graph,plan=plan,hardware=hardware,
                        settings=settings,checkpoint=checkpoint,used_seconds=used_seconds,memory_mb=memory_mb),
                   seconds,on_checkpoint=on_checkpoint,wall_seconds=wall_seconds)
    result['report']['elapsed_seconds']=result['elapsed_seconds'] if 'elapsed_seconds' in result else result['report']['elapsed_seconds']
    if on_event:
        for event in result['report'].get('events',[]):on_event(event)
    return result['plan'],result['report']
