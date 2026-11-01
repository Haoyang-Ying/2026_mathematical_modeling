"""并行运行第二问的独立 case，并保存每次发现的更优方案。"""
from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from q2_config import (DEFAULT_SEED, DEFAULT_ESCAPE_WEIGHT, DEFAULT_SEARCH_SECONDS,
                       CORE_TIMEOUT_SECONDS, MAX_SEARCH_SECONDS)
from q2_budget import BatchBudget
from q2_limits import validate_limits, TIMEOUT_EXIT_CODE
from q2_core_pool import checked_counts
from q2_io import atomic_json, digest
from q2_search import PROFILES
from run import code_hash, expand_cases


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


def case_row(out, number):
    folder = out / f'case_{number:03d}'
    checkpoint = read(folder / 'checkpoint.json')
    report = read(folder / 'report.json', {})
    row = {'case': f'case_{number:03d}', 'status': 'pending',
           'completed_phases': [], 'saved_core_counts': [], 'best': {}}
    if checkpoint:
        solver = checkpoint['solver']
        entries = solver['pool']['entries']
        row.update(status='incomplete', completed_phases=solver['completed_phases'],
                   saved_core_counts=sorted(map(int, entries)),
                   current_phase_at_checkpoint=solver.get('current_phase'))
        t1 = solver['pool']['reference_T1']
        for key, entry in entries.items():
            t = entry['result']['makespan']
            row['best'][key] = {'T': t, 'T1': t1, 'k': t1 / t if t else None,
                               'k_per_core': t1 / t / int(key) if t else None,
                               'active_cores': entry['result']['active_cores']}
    if report.get('status') == 'complete':
        row['status'] = 'complete'
    execution = read(folder / 'execution.json', {})
    if execution.get('status') in ('failed', 'timeout', 'interrupted'):
        row['status'] = execution['status']
    return row


def summary(out, config, active=None):
    active = active or {}
    rows = []
    for number in config['cases']:
        row = case_row(out, number)
        if number in active:
            row.update(status='running', running_seconds=round(time.monotonic() - active[number]['start'], 1))
        rows.append(row)
    return {'updated_at': utc(), 'seed': config['seed'], 'escape_weight': config['escape_weight'],
            'time_budget':read(out/'budget.json',{}),
            'total_cases': len(rows), 'total_plans': len(rows) * len(config['cores']),
            'completed_cases': sum(r['status'] == 'complete' for r in rows),
            'completed_phases': sum(len(r['completed_phases']) for r in rows),
            'saved_candidates': sum(len(r['saved_core_counts']) for r in rows),
            'active_cases': [f'case_{n:03d}' for n in active],
            'failed_cases': [r['case'] for r in rows if r['status'] in ('failed', 'timeout', 'interrupted')],
            'official_validation_started': False, 'rows': rows}


# 完成状态检查必须覆盖所请求的全部核数；候选文件或阶段完成标志不能替代最终验证。
def verify_complete(folder, counts):
    report = read(folder / 'report.json', {})
    checkpoint = read(folder / 'checkpoint.json', {})
    solver = checkpoint.get('solver', {})
    if (report.get('status') != 'complete' or solver.get('completed_phases') != counts
            or report.get('job_signature') != checkpoint.get('job_signature')):
        raise ValueError('incomplete or inconsistent final report/checkpoint')
    for m in counts:
        plan = read(folder / f'm{m}' / 'schedule.json')
        item = read(folder / f'm{m}' / 'report.json')
        expected = report['plans'][str(m)]
        entry = solver['pool']['entries'][str(m)]
        if digest(plan) != expected or item['plan_sha256'] != expected or entry['plan_sha256'] != expected:
            raise ValueError(f'm{m}: final schedule checksum mismatch')
        if len(plan['core_schedules']) != m or item['final_T'] != entry['score_key'][0]:
            raise ValueError(f'm{m}: final score/count mismatch')


def memory_anon():
    try:
        fields = dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
        return int(fields['anon'])
    except (OSError, KeyError):
        return 0


# 根据可用内存与预留余量限制并行准入，避免仅按用户设定的工作进程数盲目派发。
def memory_slots(memory_mb, worker_cap):
    """按完整工作进程预留内存，不将可回收的文件缓存计入不可回收占用。"""
    slots = worker_cap
    try:
        limit = int(Path('/sys/fs/cgroup/memory.max').read_text())
        fields = dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
        occupied = sum(int(fields.get(k, 0)) for k in ('anon', 'kernel', 'shmem'))
        reserve = max(1024**3, int(.15 * limit))
        per_worker = (memory_mb + 256) * 1024**2
        slots = min(slots, max(0, (limit - occupied - reserve) // per_worker))
    except (OSError, ValueError):
        slots = min(slots, 1)
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            slots = min(slots, max(1, int(quota) // int(period)))
        slots = min(slots, len(os.sched_getaffinity(0)))
    except (OSError, ValueError, AttributeError):
        pass
    return int(slots)


def worker_command(args, number, folder):
    # 向每个工作进程显式传递选定的随机种子和惩罚系数。
    command = [sys.executable, str(ROOT / 'run.py'), 'solve-many', '--case', str(number),
            '--cores', *map(str, args.cores), '--profile', args.profile,
            '--seed', str(args.seed), '--escape-weight', str(args.escape_weight),
            '--search-seconds', str(args.search_seconds), '--cache-mb', str(args.cache_mb),
            '--search-max-seconds',str(args.search_max_seconds),
            '--core-timeout', str(args.core_timeout),
            '--memory-mb', str(args.memory_mb), '--out', str(folder), '--resume']
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', help='comma-separated IDs/ranges; default 1-100, or saved campaign cases')
    parser.add_argument('--cores', type=int, nargs='+', default=[2, 3, 4, 5])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-new-cases', type=int, default=0,
                        help='pause after this many case starts; 0 runs the whole queue')
    parser.add_argument('--skip-terminal', action='store_true',
                        help='retain hard timeouts and deterministic failures without retrying')
    parser.add_argument('--memory-aware', action='store_true',
                        help='admit workers only with CPU and memory reserves')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--escape-weight', type=float, default=DEFAULT_ESCAPE_WEIGHT)
    parser.add_argument('--profile', choices=PROFILES, default='standard')
    parser.add_argument('--search-seconds', type=float, default=DEFAULT_SEARCH_SECONDS, choices=(DEFAULT_SEARCH_SECONDS,),
                        help='首次搜索基数为 30 秒，各核数独立使用 600 秒累计总额度')
    parser.add_argument('--search-max-seconds',type=float,default=MAX_SEARCH_SECONDS, choices=(MAX_SEARCH_SECONDS,))
    parser.add_argument('--cache-mb', type=int, default=512)
    parser.add_argument('--memory-mb', type=int, default=2048)
    parser.add_argument('--core-timeout', type=float, default=CORE_TIMEOUT_SECONDS)
    parser.add_argument('--out', default=str(ROOT / 'results' / 'parallel'))
    parser.add_argument('--resume', action='store_true', default=True,
                        help='resume matching checkpoints (always enabled for safety)')
    parser.add_argument('--status', action='store_true', help='read existing progress without starting workers')
    args = parser.parse_args(argv)
    args.cores = list(checked_counts(args.cores))
    out = Path(args.out).resolve()
    saved = read(out / 'campaign.json')
    selected = expand_cases(args.cases) if args.cases else saved['cases'] if saved else list(range(1, 101))
    config = {'cases': selected, 'cores': args.cores, 'profile': args.profile,
              'seed': args.seed, 'escape_weight': args.escape_weight,
              'search_seconds': args.search_seconds, 'cache_mb': args.cache_mb,
              'search_max_seconds':args.search_max_seconds,
              'core_timeout': args.core_timeout, 'case_timeout': None}
    if args.status:
        value = summary(out, saved or config)
        observed = read(out / 'progress.json', {})
        value.pop('active_cases', None)
        value.update(last_observed_active_cases=observed.get('active_cases', []),
                     last_runtime_update=observed.get('updated_at'), activity_status='last_observed_only')
        print(json.dumps(value, ensure_ascii=False))
        return 0
    if (args.workers < 1 or args.max_new_cases < 0 or args.memory_mb < 1 or args.cache_mb < 0
            or args.search_seconds != DEFAULT_SEARCH_SECONDS
            or args.search_max_seconds != MAX_SEARCH_SECONDS
            or not math.isfinite(args.escape_weight) or args.escape_weight < 0):
        raise ValueError('invalid worker, memory, time or penalty settings')
    validate_limits(args.core_timeout)
    budget_snapshot = read(out/'budget.json',{})
    ledger = BatchBudget(budget_snapshot.get('spent_seconds',0.0))
    config['code_sha256'] = code_hash()
    paths = [ROOT.parent / '100case_data' / f'case_{number:03d}.json' for number in selected]
    paths.append(ROOT.parent / '100case_data' / 'config.txt')
    config['input_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.controller.lock').open('a+b')
    try:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:
        import msvcrt
        lock.seek(0)
        lock.write(b'1')
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    # 取得文件锁后重新读取配置，防止两个控制器相互覆盖同一计算任务。
    saved = read(out / 'campaign.json')
    if saved is not None and saved != config:
        raise ValueError('resume refused: cases, counts, seed, settings, code or inputs changed; use a new --out')
    if saved is None:
        if any((out / f'case_{n:03d}').exists() for n in selected):
            raise ValueError('case directories exist without campaign metadata; use a new --out')
        atomic_json(out / 'campaign.json', config)
    queue = collections.deque()
    previous_terminal = []
    for number in selected:
        folder = out / f'case_{number:03d}'
        if read(folder / 'report.json', {}).get('status') == 'complete':
            verify_complete(folder, args.cores)
        elif args.skip_terminal and read(folder / 'execution.json', {}).get('status') in ('timeout', 'failed'):
            previous_terminal.append(number)
        else:
            queue.append(number)
    active, failed = {}, list(previous_terminal)
    launched = 0
    session = {'started_at': utc(), 'pid': os.getpid(), 'workers': args.workers,
               'memory_mb': args.memory_mb, 'core_timeout': args.core_timeout, 'case_timeout': None, 'selected_cases': selected,
               'seed': args.seed, 'escape_weight': args.escape_weight}
    atomic_json(out / 'controller.json', dict(session, status='running'))
    def handle_termination(signum, frame):
        raise KeyboardInterrupt('controller received SIGTERM')
    signal.signal(signal.SIGTERM, handle_termination)
    last_progress = 0
    try:
        while queue or active:
            if not active and args.max_new_cases and launched >= args.max_new_cases:
                break
            available_slots = memory_slots(args.memory_mb, args.workers) if args.memory_aware else args.workers
            if queue and not active and available_slots == 0:
                break
            while (queue and len(active) < available_slots and not (out / 'PAUSE').exists()
                   and (not args.max_new_cases or launched < args.max_new_cases)):
                if active and memory_anon() > 6 * 1024**3:
                    break
                number = queue.popleft()
                folder = out / f'case_{number:03d}'
                folder.mkdir(parents=True, exist_ok=True)
                # 各核数独立计时，批量排队不会削减单项搜索额度。
                command = worker_command(args, number, folder)
                history = read(folder / 'execution.json', {}).get('attempts', [])
                attempt = {'started_at': utc(), 'memory_mb': args.memory_mb,
                           'core_timeout': args.core_timeout, 'case_timeout': None, 'command': command}
                log = (folder / 'worker.log').open('a', encoding='utf-8')
                try:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                history.append(attempt)
                launched += 1
                active[number] = {'process': process, 'log': log, 'start': time.monotonic(),
                                  'folder': folder, 'history': history, 'attempt': attempt}
                atomic_json(folder / 'execution.json', {'status': 'running', 'attempts': history})
                print(json.dumps({'event': 'case_started', 'case': number, 'pid': process.pid, 'at': utc()}), flush=True)
            for number, job in list(active.items()):
                elapsed = time.monotonic() - job['start']
                code = job['process'].poll()
                if code is None:
                    continue
                job['log'].close()
                status = ('timeout' if code == TIMEOUT_EXIT_CODE else
                          'complete' if code == 0 else 'failed')
                if status == 'failed':
                    with (job['folder'] / 'worker.log').open('rb') as stream:
                        stream.seek(max(0, stream.seek(0, 2)-65536))
                        memory_failure = b'MemoryError' in stream.read()
                    if code in (-9, -15) or memory_failure:
                        status = 'interrupted'
                error = None
                if status == 'complete':
                    try:
                        verify_complete(job['folder'], args.cores)
                    except Exception as exc:
                        status, error = 'failed', repr(exc)
                job['attempt'].update(ended_at=utc(), seconds=elapsed, returncode=code, status=status, error=error)
                atomic_json(job['folder'] / 'execution.json', {'status': status, 'attempts': job['history']})
                if status != 'complete':
                    failed.append(number)
                del active[number]
                print(json.dumps({'event': 'case_finished', 'case': number, 'status': status,
                                  'seconds': round(elapsed, 2), 'error': error}), flush=True)
                last_progress = 0
            now = time.monotonic()
            atomic_json(out/'budget.json',ledger.snapshot())
            if now - last_progress >= 15:
                value = summary(out, config, active)
                atomic_json(out / 'progress.json', value)
                print(json.dumps({k: value[k] for k in ('updated_at', 'completed_cases', 'completed_phases',
                                                       'saved_candidates', 'active_cases', 'failed_cases')}), flush=True)
                last_progress = now
            if (out / 'PAUSE').exists() and not active:
                break
            if queue or active:
                time.sleep(.2)
    except BaseException:
        for job in active.values():
            if job['process'].poll() is None:
                job['process'].terminate()
        for job in active.values():
            try:
                job['process'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                job['process'].kill()
                job['process'].wait()
            job['log'].close()
            job['attempt'].update(ended_at=utc(), seconds=time.monotonic() - job['start'], status='interrupted')
            atomic_json(job['folder'] / 'execution.json', {'status': 'interrupted', 'attempts': job['history']})
        atomic_json(out / 'controller.json', dict(session, status='interrupted', ended_at=utc()))
        atomic_json(out / 'progress.json', summary(out, config))
        atomic_json(out/'budget.json',ledger.snapshot())
        raise
    atomic_json(out/'budget.json',ledger.snapshot())
    atomic_json(out / 'progress.json', summary(out, config))
    atomic_json(out / 'controller.json', dict(session, status='paused' if queue else 'failed' if failed else 'finished',
                                            ended_at=utc(), failed_cases=failed))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
