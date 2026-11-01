'读取硬件配置，并在有限时间内对第三问进行内部核验。'
from time import perf_counter

from q2_io import ROOT, hardware, load_json, atomic_json, digest, compact
from q3_evaluator import Problem3Evaluator
from multicore_cut_evaluate_problem_3 import read_cache_config


def hardware3(path=None):
    if isinstance(path,dict):
        return dict(path)
    path = path or ROOT.parent/'100case_data/config.txt'
    return dict(hardware(path), **read_cache_config(path))


def summary(result):
    keys = ('problem', 'makespan', 'num_cores', 'active_cores', 'data_movement_bytes',
            'cache_stats', 'cache_miss_reasons', 'physical_transfer_bytes',
            'memory_peak_by_core', 'cache_capacity_bytes', 'cache_bandwidth_bytes_per_cycle')
    return {k: result[k] for k in keys if k in result}


def validate_problem3(graph, plan, settings=None, *, evaluator=None, seconds=60.0):
    from q3_validation_runner import validate_bounded
    # 由子进程持有评价器，以便对编译工作施加外部时间限制。
    if evaluator is not None:
        raise ValueError('external evaluator overrides are not supported by bounded validation')
    return validate_bounded(graph,plan,hardware3() if settings is None else settings,seconds)
