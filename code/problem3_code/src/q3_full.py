'连接受子进程时间限制的第三问内部求解器的兼容接口。'
from dataclasses import asdict


def hardware_without_l2(hardware):
    return {key:hardware[key] for key in ('bandwidth','capacity','cross_core_copy_delay')}


def solve_full(graph,counts,hardware,settings=None,*,seed_plans=(),on_event=None,
               on_pool_update=None,on_checkpoint=None,checkpoint=None,official=False,
               out=None,resume=False,time_policy=None):
    from api_q3_full import solve_many
    options=asdict(settings) if settings is not None else {}
    return solve_many(graph,counts,config=hardware,seed_plans=seed_plans,
        on_event=on_event,on_pool_update=on_pool_update,on_checkpoint=on_checkpoint,
        checkpoint=checkpoint,official=official,out=out,resume=resume,time_policy=time_policy,**options)


def validate_full(graph,plan,hardware,*,expected_t1=None,expected_t3=None,official_t1=None,seconds=60.0):
    from q3_validation_runner import validate_bounded
    result=validate_bounded(graph,plan,hardware,seconds)
    if expected_t1 is not None and result['reference_T1']!=expected_t1:
        raise AssertionError('internal original-single-core reference differs')
    if expected_t3 is not None and result['result']['makespan']!=expected_t3:
        raise AssertionError('internal Q3 replay differs')
    result['exact_match']=True
    return result


def audit_results(graph,results,hardware):
    for plan,report in results.values():
        report['validation']=validate_full(graph,plan,hardware,
            expected_t1=report.get('reference_T1'),expected_t3=report.get('final_T'))
        report.update(internal_recheck_passed=True,officially_validated=False)
    return results
