"""按核数累计实际运行时间；包含构造、评价和搜索，不设置整例总时限。"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import tempfile
import threading
import time

from time_budget import CORE_TIMEOUT_SECONDS

_clock = None
_directory = None
POLL_SECONDS = 0.05


class CoreTimeLimitExceeded(TimeoutError):
    """当前核数的累计运行时间已到上限。"""
    def __init__(self, details):
        self.details = details
        super().__init__(f"核数 {details['core']} 的累计运算时间已达到 {details['limit_seconds']:g} 秒。")


def _atomic(path, value, *, binary=False):
    temporary = path.with_suffix(path.suffix + f'.{threading.get_ident()}.tmp')
    if binary:
        temporary.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    else:
        temporary.write_text(json.dumps(value), encoding='utf-8')
    os.replace(temporary, path)


def _elapsed(clock, now):
    return clock['spent'].get(str(clock['core']), 0.0) + now - clock['segment_started']


def _expired(clock, now):
    elapsed = _elapsed(clock, now)
    if elapsed >= clock['limit_seconds']:
        return dict(core=clock['core'], elapsed_seconds=elapsed,
                    limit_seconds=clock['limit_seconds'])


def remaining_seconds():
    """切回已处理的核数时继续计时，不因进入新阶段而重置额度。"""
    return math.inf if _clock is None else max(0.0, _clock['limit_seconds'] - _elapsed(_clock, time.monotonic()))


def mark_core(core):
    """将后续准备、搜索及复核时间记入指定核数。"""
    global _clock
    if _clock is None or _clock['core'] == core:
        return
    now = time.monotonic()
    expired = _expired(_clock, now)
    if expired:
        _atomic(_directory / 'timeout.json', expired)
        raise CoreTimeLimitExceeded(expired)
    spent = dict(_clock['spent'])
    spent[str(_clock['core'])] = _elapsed(_clock, now)
    _clock = dict(_clock, core=core, spent=spent, segment_started=now)
    _atomic(_directory / 'clock.json', _clock)


def worker_guard():
    """向内层搜索进程传递相同的核数截止时刻和监督者心跳位置。"""
    if _clock is None:
        return {}
    return dict(core_deadline_monotonic=time.monotonic() + remaining_seconds(),
                supervisor_heartbeat_path=str(_directory / 'owner.json'))


def _watchdog(stop):
    while not stop.wait(POLL_SECONDS):
        parent = mp.parent_process()
        if parent is not None and not parent.is_alive():
            os._exit(125)
        expired = _expired(_clock, time.monotonic())
        if expired:
            _atomic(_directory / 'timeout.json', expired)
            os._exit(124)


def _worker(target, args, kwargs, clock, directory):
    global _clock, _directory
    _clock, _directory = clock, Path(directory)
    stop = threading.Event()
    threading.Thread(target=_watchdog, args=(stop,), daemon=True).start()
    try:
        result = target(*args, **kwargs)
        expired = _expired(_clock, time.monotonic())
        if expired:
            _atomic(_directory / 'timeout.json', expired)
        else:
            _atomic(_directory / 'result.pkl', ('ok', result), binary=True)
    except BaseException as error:
        # 原始异常由父进程报告；不将不完整方案当作成功输出。
        if isinstance(error, CoreTimeLimitExceeded):
            _atomic(_directory / 'timeout.json', error.details)
        else:
            try:
                _atomic(_directory / 'result.pkl', ('error', error), binary=True)
            except Exception:
                _atomic(_directory / 'result.pkl', ('error', RuntimeError(repr(error))), binary=True)
    finally:
        stop.set()


def supervise(target, args=(), kwargs=None, *, first_core, core_seconds=CORE_TIMEOUT_SECONDS):
    """监督自己创建的计算进程；各核数分别获得额度，批次总耗时不限。"""
    if isinstance(core_seconds, bool) or not isinstance(core_seconds, (int, float)) or not math.isfinite(core_seconds) or not 0 < core_seconds <= CORE_TIMEOUT_SECONDS:
        raise ValueError('单个核数的运算上限必须为正数且不超过 600 秒。')
    if _clock is not None:
        return target(*args, **(kwargs or {}))
    # 使用 spawn，使 Windows 与其他平台采用相同的进程入口规则。
    context = mp.get_context('spawn')
    with tempfile.TemporaryDirectory(prefix='problem1_core_limit_') as temporary:
        directory = Path(temporary)
        now = time.monotonic()
        clock = dict(core=first_core, segment_started=now, spent={}, limit_seconds=core_seconds)
        _atomic(directory / 'clock.json', clock)
        _atomic(directory / 'owner.json', {'heartbeat': now})
        process = context.Process(target=_worker, args=(target, args, kwargs or {}, clock, temporary))
        process.start()
        expired = None
        try:
            while process.is_alive():
                try:
                    clock = json.loads((directory / 'clock.json').read_text())
                except (OSError, ValueError):
                    pass
                now = time.monotonic()
                _atomic(directory / 'owner.json', {'heartbeat': now})
                expired = _expired(clock, now)
                if expired:
                    process.kill()
                    break
                process.join(timeout=POLL_SECONDS)
            process.join(timeout=2)
            timeout = directory / 'timeout.json'
            if timeout.exists():
                expired = json.loads(timeout.read_text())
            if expired:
                raise CoreTimeLimitExceeded(expired)
            result_path = directory / 'result.pkl'
            if not result_path.exists():
                raise RuntimeError(f'计算进程退出但没有完整结果，退出码为 {process.exitcode}。')
            status, result = pickle.loads(result_path.read_bytes())
            if status == 'error':
                raise result
            return result
        finally:
            if process.is_alive():
                process.kill()
            process.join(timeout=2)
            if process.is_alive():
                raise RuntimeError('计算进程尚未退出，不能启动下一任务。')
