'仅用于第三问的有限墙钟时间策略及嵌套截止时间。'
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import math
import signal
import threading
from time import perf_counter

REVISION = 'q3_core600_internal'
CORE_TIMEOUT_SECONDS = 600.0
# 阶段上限均由单个核数额度的比例给出，维持各阶段之间的分配关系。
TIME_RATIOS = dict(base_seconds=1/20, soft_seconds=1/2,
                  extension_seconds=1/20, improvement_window=1/20,
                  stagnation_seconds=1/50, constructor_seconds=1/120,
                  coarse_seconds=1/150, candidate_seconds=1/75,
                  macro_candidate_seconds=1/40, repair_seconds=1/200,
                  cleanup_seconds=1/300)
STAGE_SHARES = (('hc', .15), ('multi', .10), ('mgr', .60), ('final_hc', .15))
_END = ContextVar('q3_absolute_deadline', default=None)


@dataclass(frozen=True)
class TimePolicy:
    base_seconds: float = CORE_TIMEOUT_SECONDS * TIME_RATIOS['base_seconds']
    factor_cap: float = 2.0
    soft_seconds: float = CORE_TIMEOUT_SECONDS * TIME_RATIOS['soft_seconds']
    hard_seconds: float = CORE_TIMEOUT_SECONDS
    extension_seconds: float = CORE_TIMEOUT_SECONDS * TIME_RATIOS['extension_seconds']
    extension_gain: float = .001
    improvement_window: float = CORE_TIMEOUT_SECONDS * TIME_RATIOS['improvement_window']
    stagnation_seconds: float = CORE_TIMEOUT_SECONDS * TIME_RATIOS['stagnation_seconds']
    constructor_seconds: float = CORE_TIMEOUT_SECONDS * TIME_RATIOS['constructor_seconds']
    coarse_seconds: float = CORE_TIMEOUT_SECONDS * TIME_RATIOS['coarse_seconds']
    candidate_seconds: float = CORE_TIMEOUT_SECONDS * TIME_RATIOS['candidate_seconds']
    macro_candidate_seconds: float = CORE_TIMEOUT_SECONDS * TIME_RATIOS['macro_candidate_seconds']
    repair_seconds: float = CORE_TIMEOUT_SECONDS * TIME_RATIOS['repair_seconds']
    conversion_fraction: float = .10
    cleanup_seconds: float = CORE_TIMEOUT_SECONDS * TIME_RATIOS['cleanup_seconds']
    workers: int = 4
    memory_mb: int = 1536
    heartbeat_seconds: float = .5
    max_retries: int = 3

    def check(self):
        for key, value in asdict(self).items():
            if key in ('workers', 'memory_mb', 'max_retries'):
                if type(value) is not int or value < (0 if key == 'max_retries' else 1):
                    raise ValueError(key + ' must be a valid positive integer')
            elif isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(key + ' must be finite and positive; zero never means unlimited')
        if not 0 < self.base_seconds <= self.soft_seconds <= self.hard_seconds <= CORE_TIMEOUT_SECONDS:
            raise ValueError('require 0 < base_seconds <= soft_seconds <= hard_seconds <= 600')
        if self.soft_seconds > CORE_TIMEOUT_SECONDS/2 or self.factor_cap < 1 or self.factor_cap > 2:
            raise ValueError('soft_seconds <= 300 and 1 <= factor_cap <= 2 are required')
        if self.workers > 8 or self.conversion_fraction > .25 or self.extension_gain >= 1:
            raise ValueError('workers <= 8, conversion_fraction <= .25 and extension_gain < 1 required')
        return self

    @classmethod
    def for_core_limit(cls, hard_seconds=CORE_TIMEOUT_SECONDS, **options):
        """缩短单项预算时同步缩放各阶段；没有整例或批次总时限。"""
        values = {name: hard_seconds*ratio for name, ratio in TIME_RATIOS.items()}
        values.update(options)
        return cls(hard_seconds=hard_seconds, **values).check()

    def initial_grant(self):
        return min(self.base_seconds * self.factor_cap, self.soft_seconds, self.hard_seconds)

    def nominal_seconds(self, cores, t1, entry_time):
        if t1 == entry_time == 0:
            factor = 1.0
        elif t1 <= 0 or entry_time <= 0:
            raise ValueError('positive reference/entry makespans required for a nonempty graph')
        else:
            factor = min(self.factor_cap, max(1.0, cores * entry_time / t1))
        return min(self.base_seconds * factor, self.soft_seconds), factor


def check(deadline=None):
    ends = [x for x in (deadline, _END.get()) if x is not None]
    if ends and perf_counter() >= min(ends):
        raise TimeoutError('Problem 3 wall-clock deadline reached')


def alarm_supported():
    return hasattr(signal, 'setitimer') and threading.current_thread() is threading.main_thread()


@contextmanager
def deadline(end):
    '子任务截止时间受父任务约束，包括准备阶段。Unix 主线程通过定时信号中断纯 Python 构造和编译；其他平台使用协作检查及外层 Popen 监督。不依据记录中的外来进程号发送信号。'
    parent = _END.get()
    end = min(end, parent) if parent is not None else end
    check(end)
    token = _END.set(end)
    enabled = alarm_supported()
    started = perf_counter()
    if enabled:
        handler = signal.getsignal(signal.SIGALRM)
        timer = signal.getitimer(signal.ITIMER_REAL)
        def expired(*_):
            raise TimeoutError('Problem 3 operation exceeded its deadline')
        signal.signal(signal.SIGALRM, expired)
        duration = end - started
        if timer[0] > 0:
            duration = min(duration, timer[0])
        signal.setitimer(signal.ITIMER_REAL, max(.000001, duration))
    try:
        yield
        check(end)
    finally:
        if enabled:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)
            if timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, max(.000001, timer[0] - (perf_counter()-started)), timer[1])
        _END.reset(token)


@contextmanager
def commit_section():
    '保护短暂的原子快照提交，提交耗时仍计入预算。磁盘读写阻塞时，外部监督仍可终止工作进程；原子替换保证文件保持为完整的旧快照或新快照。'
    enabled = alarm_supported()
    if enabled:
        timer = signal.getitimer(signal.ITIMER_REAL)
        started = perf_counter()
        signal.setitimer(signal.ITIMER_REAL, 0)
    try:
        yield
    finally:
        if enabled and timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, max(.000001, timer[0] - (perf_counter()-started)), timer[1])


class Stagnation(TimeoutError):
    '阶段已完成初始探测，且后续不再取得进展。'
