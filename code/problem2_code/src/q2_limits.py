"""进程级实际时间硬限制，同时覆盖初始化和精确回放。

调用方仅管理自己创建的子进程；子进程另有看门狗，在到期或父进程失联时退出。
不向记录中的外部进程编号发送信号。中断后仅保留已经提交的断点。"""
from __future__ import annotations

import functools
import inspect
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import tempfile
import threading
import time
import traceback

from q2_config import CORE_TIMEOUT_SECONDS

POLL_SECONDS = 0.05
TIMEOUT_EXIT_CODE = 124
_clock = None
_directory = None
_sender = None


class TimeLimitExceeded(TimeoutError):
    def __init__(self, details, checkpoint=None):
        self.details, self.checkpoint = details, checkpoint
        super().__init__(f"{details['reason']}: core={details.get('cores')}, "
                         f"case_seconds={details['case_elapsed_seconds']:.3f}")


def validate_limits(core_seconds):
    if (isinstance(core_seconds, bool) or not isinstance(core_seconds, (int, float))
            or not math.isfinite(core_seconds) or not 0 < core_seconds <= CORE_TIMEOUT_SECONDS):
        raise ValueError(f'每个核数累计硬限必须为正数且不超过 {CORE_TIMEOUT_SECONDS:g} 秒')


def _atomic(path, data, *, binary=False):
    temp = path.with_suffix(path.suffix+f'.{threading.get_ident()}.tmp')
    if binary:
        temp.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
    else:
        temp.write_text(json.dumps(data), encoding='utf-8')
    os.replace(temp, path)


def _expired(clock, now):
    case_elapsed = now-clock['started']
    core_elapsed = clock['spent'].get(str(clock['core']), 0)+now-clock['segment_started']
    # 不根据整例累计耗时终止；不同核数分别计时，回放阶段继续累计。
    reason = 'core_timeout' if core_elapsed >= clock['core_seconds'] else None
    if reason:
        return {'status': 'timed_out', 'reason': reason, 'cores': clock['core'],
                'core_elapsed_seconds': core_elapsed, 'case_elapsed_seconds': case_elapsed,
                'core_timeout_seconds': clock['core_seconds'],
                'case_timeout_seconds': None}


def in_worker():
    return _clock is not None


def elapsed_seconds():
    return 0.0 if _clock is None else time.monotonic()-_clock['started']


def remaining_seconds():
    if _clock is None:
        return math.inf
    now = time.monotonic()
    core_spent = _clock['spent'].get(str(_clock['core']),0)+now-_clock['segment_started']
    return max(0.0, _clock['core_seconds']-core_spent)


def mark_core(cores):
    """累计核数耗时；最终验证重新切回该核数时继续计入原额度。"""
    global _clock
    if _clock is None or _clock['core'] == cores:
        return
    now = time.monotonic()
    expired = _expired(_clock, now)
    if expired:
        _atomic(_directory/'timeout.json', expired)
        os._exit(TIMEOUT_EXIT_CODE)
    spent = dict(_clock['spent'])
    key = str(_clock['core'])
    spent[key] = spent.get(key, 0)+now-_clock['segment_started']
    _clock = dict(_clock, core=cores, segment_started=now, spent=spent)
    _atomic(_directory/'clock.json', _clock)


def send_callback(name, *args):
    _sender.send(('callback', name, args))


def send_checkpoint(snapshot):
    _sender.send(('checkpoint', snapshot))


def _watchdog(stop):
    while not stop.wait(POLL_SECONDS):
        parent = mp.parent_process()
        if parent is not None and not parent.is_alive():
            os._exit(125)
        expired = _expired(_clock, time.monotonic())
        if expired:
            _atomic(_directory/'timeout.json', expired)
            os._exit(TIMEOUT_EXIT_CODE)


def _worker(target, args, kwargs, clock, directory, sender):
    global _clock, _directory, _sender
    _clock, _directory, _sender = clock, Path(directory), sender
    stop = threading.Event()
    threading.Thread(target=_watchdog, args=(stop,), daemon=True).start()
    try:
        result = target(*args, **kwargs)
        expired = _expired(_clock, time.monotonic())
        if expired:
            _atomic(_directory/'timeout.json', expired)
            return
        _atomic(_directory/'result.pkl', ('ok', result), binary=True)
    except BaseException as error:
        try:
            _atomic(_directory/'result.pkl', ('error', error, traceback.format_exc()), binary=True)
        except Exception:
            _atomic(_directory/'result.pkl', ('error', RuntimeError(repr(error)),
                                             traceback.format_exc()), binary=True)
    finally:
        stop.set()
        sender.close()


def _read_clock(path, fallback):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return fallback


# 仅清理当前调用方实际持有的子进程对象，不使用历史记录中的外部进程编号。
def _kill_owned(process):
    if process.is_alive():
        process.kill()
    process.join(timeout=2)
    if process.is_alive():
        raise RuntimeError('Timed worker did not exit; refusing to start another case')


# 由父进程监督自己创建的计算子进程，仅在当前核数累计额度耗尽时结束该计算。
def supervise(target, args=(), kwargs=None, *, first_core,
              core_seconds=CORE_TIMEOUT_SECONDS,
              callbacks=None):
    validate_limits(core_seconds)
    if in_worker():
        return target(*args, **(kwargs or {}))
    callbacks = callbacks or {}
    # 在 POSIX 系统上使用 fork 保留调用方的研究性补丁；Windows 使用 spawn。
    context = mp.get_context('fork' if os.name == 'posix' else 'spawn')
    with tempfile.TemporaryDirectory(prefix='q2_limits_') as temporary:
        directory = Path(temporary)
        now = time.monotonic()
        clock = {'started': now, 'segment_started': now, 'core': first_core, 'spent': {},
                 'core_seconds': core_seconds}
        _atomic(directory/'clock.json', clock)
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_worker,
            args=(target, args, kwargs or {}, clock, temporary, sender))
        last_checkpoint = None
        expired = None
        process.start()
        sender.close()
        connected = True
        try:
            while True:
                clock = _read_clock(directory/'clock.json', clock)
                expired = _expired(clock, time.monotonic())
                if expired:
                    _kill_owned(process)
                    break
                if connected and receiver.poll(POLL_SECONDS):
                    try:
                        message = receiver.recv()
                    except (EOFError, OSError):
                        connected = False
                    else:
                        if message[0] == 'checkpoint':
                            last_checkpoint = message[1]
                            callback = callbacks.get('on_checkpoint')
                            if callback:
                                callback(last_checkpoint)
                        else:
                            _, name, values = message
                            if callbacks.get(name):
                                callbacks[name](*values)
                        continue
                elif not connected and process.is_alive():
                    process.join(timeout=POLL_SECONDS)
                if not process.is_alive():
                    if connected and receiver.poll():
                        continue
                    break
            process.join(timeout=2)
            timeout_path = directory/'timeout.json'
            if timeout_path.exists():
                expired = json.loads(timeout_path.read_text())
            if expired:
                raise TimeLimitExceeded(expired, last_checkpoint)
            result_path = directory/'result.pkl'
            if not result_path.exists():
                raise RuntimeError(f'Calculation process exited {process.exitcode} without a result; checkpoints retained')
            result = pickle.loads(result_path.read_bytes())
            if result[0] == 'error':
                raise result[1]
            return result[1]
        finally:
            receiver.close()
            _kill_owned(process)


def bounded_api(function):
    """对公共接口调用实施进程监督；回调观察复制后的工作进程状态。"""
    signature = inspect.signature(function)
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        if in_worker():
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        options = values.get('options', {})
        core_seconds = options.get('core_timeout', values.get('core_timeout', CORE_TIMEOUT_SECONDS))
        cores = values['cores']
        first_core = cores if isinstance(cores, int) else min(cores)
        clean = dict(kwargs)
        callbacks = {name: clean.pop(name, None) for name in
                     ('on_event', 'on_pool_update', 'on_checkpoint')}
        from api import _api_worker
        return supervise(_api_worker,
            (function.__name__, args, clean, tuple(k for k,v in callbacks.items() if v)),
            first_core=first_core, core_seconds=core_seconds,
            callbacks=callbacks)
    return wrapped
