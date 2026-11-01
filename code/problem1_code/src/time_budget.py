'每个核数累计运算上限为 600 秒；各阶段按固定比例获得额度。细化搜索的动态系数在进入该阶段时确定，并受当前核数剩余时间约束。'
from contextlib import contextmanager
from contextvars import ContextVar
import math

CORE_TIMEOUT_SECONDS = 600.0
SCALE = CORE_TIMEOUT_SECONDS / 300.0
POLICY = 'adaptive_core600'
_DYNAMIC_FACTOR = ContextVar('refinement_time_factor', default=1.0)


def base_seconds(reference_seconds):
    '公开默认预算不受导入顺序或当前搜索上下文影响。'
    return SCALE * reference_seconds


def seconds(reference_seconds):
    '按当前核数额度同比例确定内部阶段时间，并应用冻结的动态系数。'
    return base_seconds(reference_seconds) * _DYNAMIC_FACTOR.get()


@contextmanager
def dynamic_budget(factor):
    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        raise ValueError('dynamic time factor must be a finite positive number')
    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('dynamic time factor must be finite and positive')
    token = _DYNAMIC_FACTOR.set(factor)
    try:
        yield
    finally:
        _DYNAMIC_FACTOR.reset(token)


# 搜索入口确定后固定动态系数，后续候选不能扩展总预算。
def refinement_budget(base_search_seconds, m, single_makespan, initial_makespan):
    '依据已择优且完成内部评价的当前方案，冻结 max(1,m/k)。'
    if isinstance(m, bool) or not isinstance(m, int) or m < 1:
        raise ValueError('m must be a positive integer')
    values = (base_search_seconds, single_makespan, initial_makespan)
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in values):
        raise ValueError('budgets and makespans must be finite numbers')
    if base_search_seconds <= 0 or single_makespan < 0 or initial_makespan < 0:
        raise ValueError('budget must be positive and makespans nonnegative')
    if single_makespan == initial_makespan == 0:
        factor, speedup, efficiency = 1.0, None, None
    elif single_makespan == 0 or initial_makespan == 0:
        raise ValueError('only an empty zero-cost graph may have zero makespans')
    else:
        speedup = single_makespan / initial_makespan
        efficiency = speedup / m
        factor = max(1.0, m * (initial_makespan / single_makespan))
    effective = base_search_seconds * factor
    if not math.isfinite(effective) or not math.isfinite(SCALE * factor):
        raise ValueError('scaled time budget must be finite')
    return {'policy': POLICY, 'core_limit_seconds': CORE_TIMEOUT_SECONDS,
            'base_search_seconds': base_search_seconds,
            'dynamic_factor': factor, 'effective_search_seconds': effective,
            'fixed_slice_multiplier': SCALE * factor,
            'single_makespan': single_makespan, 'initial_makespan': initial_makespan,
            'initial_speedup': speedup, 'initial_k_over_m': efficiency,
            'factor_source': 'selected_incumbent_after_lower_core_inheritance',
            'factor_frozen_for_all_starts': True}
