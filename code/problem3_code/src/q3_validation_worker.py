'固定方案的内部评价工作进程，包含自身的截止时间监督。'
import os
from pathlib import Path
import sys
import threading
from time import perf_counter

from q3_runtime_io import atomic,read


def main(path):
    r=read(path);done=threading.Event()
    def watch():
        if not done.wait(max(.001,r['deadline']-perf_counter())):os._exit(124)
    threading.Thread(target=watch,daemon=True).start()
    try:
        try:
            import resource
            amount=r.get('memory_mb',1536)*1024**2
            resource.setrlimit(resource.RLIMIT_AS,(amount,amount))
        except ImportError:pass
        from q3_time import deadline,commit_section
        from q3_evaluator import Problem3Evaluator
        from q3_io import summary
        with deadline(r['deadline']):
            if r['mode']=='order':
                from q3_order import OrderSettings,optimize_fixed
                def save(snapshot):
                    with commit_section():atomic(r['checkpoint_path'],snapshot)
                reserve=min(2.0,max(.05,.15*(r['deadline']-perf_counter())))
                try:
                    with deadline(r['deadline']-reserve):
                        plan,report,snapshot=optimize_fixed(r['graph'],r['plan'],r['hardware'],
                            OrderSettings(**r['settings']),checkpoint=r.get('checkpoint'),on_checkpoint=save)
                except TimeoutError:
                    # 外层时间检查可能在内部搜索已返回最优快照后触发，
                    # 此时对该快照执行回放。
                    if not Path(r['checkpoint_path']).exists():raise
                    snapshot=read(r['checkpoint_path'])
                    plan=snapshot['best_plan']
                    report=dict(problem=3,status='time_budget_exhausted',baseline=snapshot['baseline'],
                        final=snapshot['best'],attempted=snapshot['attempted'],accepted=snapshot['accepted'],
                        fixed_core_assignment_verified=True)
                fresh=Problem3Evaluator(r['graph'],**r['hardware']).evaluate(plan,deadline=r['deadline'])
                if summary(fresh)!=report['final']:
                    # 读取旧快照后，将 JSON 字符串键规范化为整数键。
                    from q3_timed_search import canonical
                    if canonical(summary(fresh))!=canonical(report['final']):
                        raise AssertionError('fixed-order internal replay mismatch')
                report.update(internal_recheck_passed=True,internally_validated=True,
                              officially_validated=False,evaluation_kind='internal_only')
                result=dict(plan=plan,report=report,checkpoint=snapshot)
            else:
                evaluator=Problem3Evaluator(r['graph'],**r['hardware'],
                    cache_bytes=r.get('compile_cache_mb',128)*1024**2)
                inner=evaluator.evaluate(r['plan'],deadline=r['deadline'],
                    capture=r.get('diagnostics',False),diagnostics=r.get('diagnostics',False))
                result=dict(valid=True,internally_validated=True,officially_validated=False,
                            evaluation_kind='internal_only',result=inner)
                if r.get('include_reference',True):
                    from q2_model import Model
                    from q3_timed_search import make_evaluator
                    model=Model(r['graph'])
                    plain=make_evaluator(r['graph'],r['hardware'],r.get('compile_cache_mb',128),l2=False)
                    t1=plain.evaluate(model.fallback(1).plan(),deadline=r['deadline'])['makespan']
                    result.update(reference_T1=t1,k=t1/inner['makespan'] if inner['makespan'] else None)
        atomic(r['result_path'],result)
        return 0
    except TimeoutError:
        return 124
    finally:done.set()


if __name__=='__main__':
    raise SystemExit(main(sys.argv[1]))
