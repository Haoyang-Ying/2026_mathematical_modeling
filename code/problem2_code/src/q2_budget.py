"""实际经过时间的预算策略；软截止时间不得使未完成评价的分数被发布。"""
from contextlib import contextmanager
import math
import signal
import sys
import threading
from time import perf_counter

from q2_config import TIME_POLICY, DYNAMIC_FACTOR_MAX


def check_time(deadline):
    if deadline is not None and perf_counter() >= deadline:
        raise TimeoutError('operation wall budget exhausted')


@contextmanager
def time_slice(deadline):
    """中断纯 Python 计算；底层原生调用另由进程看门狗约束。
    
    缓存插入、最优方案提交和断点写入必须在此作用域之外执行。
    独立进程看门狗提供可移植的硬超时兜底。
    嵌套作用域保留最早的闹钟，到期退出后恢复外层计时器。"""
    check_time(deadline)
    if deadline is None:
        yield
        return
    if (not hasattr(signal,'setitimer')
            or threading.current_thread() is not threading.main_thread()):
        # Windows 不支持 setitimer，因此只跟踪这些纯准备计算范围，
        # 不跟踪缓存、最优方案和断点提交；退出时恢复原调试器跟踪函数。
        previous_trace = sys.gettrace()
        ticks = 0
        def trace(frame,event,arg):
            nonlocal ticks
            ticks += 1
            if ticks % 128 == 0:
                check_time(deadline)
            return trace
        sys.settrace(trace)
        try:
            yield
            check_time(deadline)
        finally:
            sys.settrace(previous_trace)
        return
    began = perf_counter()
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    delay = deadline-began
    if old_timer[0] > 0:
        delay = min(delay, old_timer[0])
    def expire(signum, frame):
        raise TimeoutError('operation wall budget exhausted')
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, max(1e-6, delay))
    try:
        yield
        check_time(deadline)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > perf_counter()-began:
            signal.setitimer(signal.ITIMER_REAL,
                max(1e-6, old_timer[0]-(perf_counter()-began)), old_timer[1])


def timed_call(deadline, function, *args, **kwargs):
    with time_slice(deadline):
        return function(*args, **kwargs)


def initial_seconds(base, cores, single, incumbent):
    factor = 1.0 if single <= 0 or incumbent <= 0 else min(
        DYNAMIC_FACTOR_MAX, max(1.0, cores*incumbent/single))
    return base*factor, factor


class CaseBudget:
    """仅记录整例累计耗时，不设立整例硬限或软搜索额度。"""
    target = None

    def __init__(self, *, spent=0.0):
        self.spent, self.started = spent, perf_counter()

    def elapsed(self):
        return self.spent+perf_counter()-self.started

    def remaining(self):
        return math.inf

    def search_allowance(self, remaining_counts, evaluation_seconds=0.0):
        # 搜索只受各阶段额度和当前核数 600 秒累计硬限约束。
        return math.inf


class BatchBudget:
    """仅统计控制器累计运行时间，不设置批次时间目标或截止时刻。"""
    def __init__(self, spent=0.0):
        if not math.isfinite(spent) or spent < 0:
            raise ValueError('spent seconds must be finite and nonnegative')
        self.spent, self.started = spent, perf_counter()

    def snapshot(self):
        spent = self.spent+perf_counter()-self.started
        return {'policy': TIME_POLICY, 'spent_seconds': spent,
                'scope': 'cumulative_active_controller_wall', 'hard_stop': False}
