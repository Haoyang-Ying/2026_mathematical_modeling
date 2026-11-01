"""第二问独立运行入口；需要 Python 3.10 或以上版本，仅使用标准库。"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from q2_config import (DEFAULT_SEED, DEFAULT_ESCAPE_WEIGHT, DEFAULT_SEARCH_SECONDS,
                       CORE_TIMEOUT_SECONDS, MAX_SEARCH_SECONDS)
from q2_budget import BatchBudget
from q2_limits import supervise, TimeLimitExceeded, TIMEOUT_EXIT_CODE, mark_core
from q2_model import Model, State, score_key
from q2_search import Settings, PROFILES, optimize, optimize_many
from q2_core_pool import CoreSolutionPool, checked_counts
from q2_io import atomic_json, load_json, hardware, digest, validate_plan, write_submission


def case_path(value):
    if value.isdecimal():
        path = ROOT.parent/'100case_data'/f'case_{int(value):03d}.json'
    else:
        path = Path(value)
        if not path.is_absolute() and not path.exists():
            path = ROOT/path
    if not path.is_file():
        raise FileNotFoundError(f'graph not found: {path}')
    return path.resolve()


def code_hash():
    files = [ROOT/'run.py', ROOT/'api.py'] + sorted((ROOT/'src').glob('*.py'))
    files += sorted((ROOT/'official_runtime').glob('*.py'))
    return digest({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def job_signature(graph_path, hw, settings, cores, inherited_plans=()):
    return digest({'graph': hashlib.sha256(graph_path.read_bytes()).hexdigest(),
                   'hardware': hw, 'settings': asdict(settings), 'cores': cores, 'code': code_hash(),
                   'inherited_plans': [digest(plan) for plan in inherited_plans]})


def settings_from(args):
    return Settings(args.profile, args.seed, args.group_size, args.ready_window,
                    args.cache_mb, args.search_seconds, args.constructor, args.escape_weight,
                    args.coarse_candidates, args.mgr, args.mgr_max_nodes, args.mgr_budget,
                    args.hc_mode, args.core_sharing, args.core_timeout,
                    args.search_max_seconds)


def apply_memory_limit(megabytes):
    if not megabytes:
        return 'disabled'
    try:
        import resource
    except ImportError:
        return 'unavailable_on_this_platform'
    resource.setrlimit(resource.RLIMIT_AS, (megabytes*1024**2,)*2)
    return 'enforced'


def solve_command(args):
    if args.cores < 1 or args.memory_mb < 0:
        raise ValueError('cores must be positive and memory_mb nonnegative')
    memory_status = apply_memory_limit(args.memory_mb)
    path = case_path(args.case)
    out = Path(args.out) if args.out else ROOT/'results'/f'{path.stem}_m{args.cores}'
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    hw, settings = hardware(args.config or path.parent/'config.txt'), settings_from(args)
    settings.check()
    inherited_plans = [(Path(p), load_json(p)) for p in args.inherit_plan]
    signature = job_signature(path, hw, settings, args.cores, [p for _, p in inherited_plans])
    if args.resume and (out/'report.json').exists():
        previous = load_json(out/'report.json')
        if previous.get('job_signature') != signature:
            raise ValueError('resume refused: graph, hardware, settings, core count or code changed; choose a new --out')
        if previous.get('status') == 'complete' and (not args.official or previous.get('officially_validated')):
            if digest(load_json(out/'schedule.json')) != previous['plan_sha256']:
                raise ValueError('saved schedule changed since completion')
            write_submission(out/f'{path.stem}_multicore_res.json', load_json(out/'schedule.json'), args.cores)
            print(json.dumps({'status': 'already_complete', 'out': str(out)}, ensure_ascii=False))
            return
    model = Model(load_json(path))
    inherited = [(p.parent.name + '/' + p.name, State.from_plan(model, plan))
                 for p, plan in inherited_plans]
    warm, saved_key = None, None
    if args.resume and (out/'checkpoint.json').exists():
        checkpoint = load_json(out/'checkpoint.json')
        if checkpoint['job_signature'] != signature:
            raise ValueError('checkpoint belongs to different inputs/settings/code; choose a new --out')
        if digest(checkpoint['plan']) != checkpoint.get('plan_sha256'):
            raise ValueError('checkpoint plan checksum differs')
        warm = State.from_plan(model, checkpoint['plan'])
        recheck = validate_plan(model.graph, checkpoint['plan'], hw, official=False)
        saved_key = score_key(recheck['internal_audit'])
    counts = tuple(range(2, args.cores+1)) if settings.core_sharing and args.cores > 1 else (args.cores,)
    def save_pool(event, pool):
        snapshot = pool.snapshot()
        atomic_json(out/'pool_checkpoint.json', {'job_signature': signature, 'pool': snapshot})
        target = event['target_cores']
        plan, result = pool.final_result(target)
        atomic_json(out/'shared'/f'm{target}'/'schedule.json', plan)
        atomic_json(out/'shared'/f'm{target}'/'report.json', result)
    pool = CoreSolutionPool(model, hw, counts, cache_mb=settings.cache_mb,
                            sharing=settings.core_sharing, on_update=save_pool)
    if args.resume and (out/'pool_checkpoint.json').exists():
        saved = load_json(out/'pool_checkpoint.json')
        if saved.get('job_signature') != signature:
            raise ValueError('resume refused: core pool inputs/settings/code changed')
        pool.restore(saved['pool'])
    # 新一次尝试开始时，不得残留旧的已完成报告，
    # 避免后续恢复将其误认为本次中断任务已经完成。
    atomic_json(out/'report.json', {'status': 'running', 'job_signature': signature})
    with (out/'events.jsonl').open('a' if args.resume else 'w', encoding='utf-8') as stream:
        def callback(event, best, result):
            nonlocal saved_key
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
            if event['event'] == 'greedy_complete':
                atomic_json(out/'greedy_plan.json', best.plan())
            if event.get('new_best') and (saved_key is None or score_key(result) < saved_key):
                plan = best.plan()
                saved_key = score_key(result)
                # 该原子写入文件是中断恢复时的权威状态。
                atomic_json(out/'checkpoint.json', {'job_signature': signature, 'plan': plan,
                            'plan_sha256': digest(plan), 'score_key': list(saved_key), 'event': event})
                atomic_json(out/'best_plan.json', plan)
                stream.flush()
            if event['event'] in ('bspg_complete', 'greedy_complete', 'stage_complete', 'complete',
                                   'mgr_start', 'mgr_round_complete'):
                stream.flush()
                print(json.dumps({'case': path.stem, 'cores': args.cores, **event},
                                 ensure_ascii=False), flush=True)
        plan, report = optimize(model, args.cores, hw, settings, callback, warm_start=warm,
                                inherited=inherited, core_pool=pool)
    report['core_solutions'] = pool.snapshot()
    # 恢复断点时，同时重新生成由断点派生的导出文件。
    for target in pool.entries:
        save_pool({'target_cores': target}, pool)
    atomic_json(out/'schedule.json', plan)
    report.update(status='generated', job_signature=signature, plan_sha256=digest(plan),
                  input_graph=path.name, hardware=hw, python=sys.version,
                  platform=platform.platform(), memory_limit_mb=args.memory_mb,
                  memory_limit_status=memory_status)
    atomic_json(out/'report.json', report)
    audit = validate_plan(model.graph, plan, hw, official=args.official,
                          expected_t1=report['reference_T1'])
    if audit['internal_audit']['makespan'] != report['final_T']:
        raise AssertionError('independent internal replay differs from final score')
    atomic_json(out/'validation.json', audit)
    report['independent_internal_replay_validated'] = True
    if args.official:
        report.update(officially_validated=True, validation_exact_match=audit['exact_match'])
    report['status'] = 'complete'
    atomic_json(out/'report.json', report)
    # 仅将已经通过最终内部回放的方案导出为附件 B 标准文件。
    write_submission(out/f'{path.stem}_multicore_res.json', plan, args.cores)
    print(json.dumps({'status': 'complete', 'case': path.stem, 'cores': args.cores,
                      'T': report['final_T'], 'k': report['k'], 'k_per_core': report['k_per_core'],
                      'official': report['officially_validated'], 'out': str(out)}, ensure_ascii=False), flush=True)


def validate_command(args):
    path, plan = case_path(args.case), load_json(args.plan)
    audit = validate_plan(load_json(path), plan, hardware(args.config or path.parent/'config.txt'), official=not args.internal_only)
    destination = Path(args.out) if args.out else Path(args.plan).with_name('validation.json')
    atomic_json(destination, audit)
    print(json.dumps({'valid': True, 'official': audit['officially_validated'],
                      'exact_match': audit.get('exact_match'), 'T': audit['internal_audit']['makespan'],
                      'out': str(destination.resolve())}, ensure_ascii=False))


def solve_many_command(args):
    """仅负责保存与恢复；跨核数共享和候选接受规则全部由 optimize_many 实现。"""
    counts = checked_counts(args.cores)
    if args.memory_mb < 0:
        raise ValueError('memory_mb must be nonnegative')
    memory_status = apply_memory_limit(args.memory_mb)
    path = case_path(args.case)
    out = (Path(args.out) if args.out else ROOT/'results'/path.stem).resolve()
    out.mkdir(parents=True, exist_ok=True)
    hw, settings, graph = hardware(args.config or path.parent/'config.txt'), settings_from(args), load_json(path)
    settings.check()
    seed_plans = [load_json(p) for p in args.seed_plan]
    signature = job_signature(path, hw, settings, counts, seed_plans)
    checkpoint = None
    if args.resume and (out/'report.json').exists():
        previous = load_json(out/'report.json')
        if previous.get('job_signature') != signature:
            raise ValueError('resume refused: graph/hardware/settings/counts/code/seeds changed')
        if previous.get('status') == 'complete':
            for m in counts:
                if digest(load_json(out/f'm{m}'/'schedule.json')) != previous['plans'][str(m)]:
                    raise ValueError('saved schedule changed since completion')
            if not args.official or previous.get('officially_validated'):
                for m in counts:
                    write_submission(out/f'm{m}'/f'{path.stem}_multicore_res.json',
                                     load_json(out/f'm{m}'/'schedule.json'), m)
                print(json.dumps({'status': 'already_complete', 'out': str(out)}))
                return
    if args.resume and (out/'checkpoint.json').exists():
        saved = load_json(out/'checkpoint.json')
        if saved.get('job_signature') != signature:
            raise ValueError('resume refused: checkpoint inputs/settings/code changed')
        checkpoint = saved['solver']
    model = Model(graph)
    seeds = [(f'seed_{i}', State.from_plan(model, plan)) for i, plan in enumerate(seed_plans)]
    atomic_json(out/'report.json', {'status': 'running', 'job_signature': signature})
    def persist(snapshot):
        # 完整解池和已完成阶段前缀一起提交到同一个原子写入文件。
        atomic_json(out/'checkpoint.json', {'job_signature': signature, 'solver': snapshot})
        t1 = snapshot['pool']['reference_T1']
        for key, entry in snapshot['pool']['entries'].items():
            target, t = int(key), entry['result']['makespan']
            directory = out/f'm{target}'
            atomic_json(directory/'schedule.json', entry['plan'])
            progress = {
                'status': 'phase_complete' if target in snapshot['completed_phases'] else 'candidate_saved',
                'cores': target, 'final_T': t, 'reference_T1': t1,
                'k': t1/t if t else None, 'k_per_core': t1/t/target if t else None,
                'plan_sha256': entry['plan_sha256'], 'best_origin': entry['origin'],
                'job_signature': signature,
                'internally_validated': True, 'officially_validated': False}
            atomic_json(directory/'incumbent.json', progress)
            # 不得将上一次运行的已验证报告与本次新方案混放。
            atomic_json(directory/'report.json', progress)
    with (out/'events.jsonl').open('a' if args.resume else 'w', encoding='utf-8') as stream:
        def event_callback(event, best, result):
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
            if event['event'] in ('core_pool_update', 'greedy_complete', 'stage_complete', 'complete'):
                stream.flush()
                print(json.dumps({'case': path.stem, **event}, ensure_ascii=False), flush=True)
        results = optimize_many(model, counts, hw, settings, on_event=event_callback,
                                on_checkpoint=persist, checkpoint=checkpoint, seeds=seeds)
    # 先生成并保存全部调度方案，再执行可选的官方复核。
    for m, (plan, report) in results.items():
        report.update(status='generated', job_signature=signature, plan_sha256=digest(plan),
                      input_graph=path.name, hardware=hw, memory_limit_mb=args.memory_mb,
                      memory_limit_status=memory_status)
        atomic_json(out/f'm{m}'/'schedule.json', plan)
        atomic_json(out/f'm{m}'/'report.json', report)
    for m, (plan, report) in results.items():
        mark_core(m)
        audit = validate_plan(graph, plan, hw, official=args.official,
                              expected_t1=report['reference_T1'])
        if audit['internal_audit']['makespan'] != report['final_T']:
            raise AssertionError('independent internal replay differs from final score')
        atomic_json(out/f'm{m}'/'validation.json', audit)
        report['independent_internal_replay_validated'] = True
        if args.official:
            report.update(officially_validated=True, validation_exact_match=audit['exact_match'])
        report['status'] = 'complete'
        atomic_json(out/f'm{m}'/'report.json', report)
        write_submission(out/f'm{m}'/f'{path.stem}_multicore_res.json', plan, m)
    summary = {'status': 'complete', 'job_signature': signature, 'cores': list(counts),
               'independent_internal_replay_validated': True,
               'officially_validated': bool(args.official),
               'plans': {str(m): digest(plan) for m, (plan, _) in results.items()},
               'rows': [{k: report[k] for k in ('cores', 'reference_T1', 'final_T', 'k', 'k_per_core',
                         'active_cores', 'best_origin', 'replaced_by_core_pool', 'officially_validated')}
                        for _, report in results.values()]}
    atomic_json(out/'report.json', summary)
    print(json.dumps({'status': 'complete', 'case': path.stem, 'rows': summary['rows'],
                      'out': str(out)}, ensure_ascii=False), flush=True)


def expand_cases(value):
    answer = []
    for part in value.split(','):
        if '-' in part:
            a, b = map(int, part.split('-', 1))
            if a > b:
                raise ValueError('case range is reversed')
            answer.extend(range(a, b+1))
        else:
            answer.append(int(part))
    if not answer or any(x < 1 or x > 100 for x in answer):
        raise ValueError('bundled case IDs must be in 1..100')
    return list(dict.fromkeys(answer))


def batch_command(args):
    """每个 case 启动一个联合求解会话；本函数不实现跨核数共享逻辑。"""
    settings_from(args).check()
    if args.memory_mb < 0:
        raise ValueError('memory_mb must be nonnegative')
    counts = checked_counts(args.cores)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    numbers = expand_cases(args.cases)
    previous_budget = load_json(out/'budget.json') if args.resume and (out/'budget.json').exists() else {}
    ledger = BatchBudget(previous_budget.get('spent_seconds',0.0))
    def save():
        atomic_json(out/'budget.json',ledger.snapshot())
        atomic_json(out/'summary.json', {'profile': args.profile, 'rows': rows,
                    'time_budget':ledger.snapshot(),
                    'completed': sum(r['status'] == 'ok' for r in rows),
                    'failed_or_interrupted': sum(r['status'] != 'ok' for r in rows)})
    for index,number in enumerate(numbers):
        job = out/f'case_{number:03d}'
        job.mkdir(exist_ok=True)
        # 批量记录不再折算为单例搜索额度，各核数使用独立硬限。
        command = [sys.executable, str(ROOT/'run.py'), 'solve-many', '--case', str(number),
                   '--cores', *map(str, counts), '--out', str(job), '--memory-mb', str(args.memory_mb)]
        for name in ('profile', 'seed', 'group_size', 'ready_window', 'cache_mb', 'search_seconds',
                     'constructor', 'escape_weight', 'coarse_candidates', 'hc_mode', 'mgr_max_nodes',
                     'core_timeout', 'search_max_seconds'):
            command += ['--'+name.replace('_', '-'), str(getattr(args, name))]
        command += ['--mgr' if args.mgr else '--no-mgr']
        command += ['--core-sharing' if args.core_sharing and args.inherit else '--no-core-sharing']
        if args.mgr_budget is not None:
            command += ['--mgr-budget', str(args.mgr_budget)]
        if args.config:
            command += ['--config', str(Path(args.config).resolve())]
        if args.resume:
            command.append('--resume')
        if args.official:
            command.append('--official')
        started = time.perf_counter()
        process = None
        try:
            with (job/'worker.log').open('a' if args.resume else 'w', encoding='utf-8') as stream:
                process = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
                while process.poll() is None:
                    save()
                    time.sleep(.2)
            status = ('ok' if process.returncode==0 else 'timeout' if
                process.returncode==TIMEOUT_EXIT_CODE else 'failed')
        finally:
            if process is not None and process.poll() is None:
                process.kill(); process.wait(timeout=5)
            save()
        for m in counts:
            row = {'case': f'{number:03d}', 'cores': m, 'directory': str((job/f'm{m}').relative_to(out)),
                   'status': status, 'case_wall_seconds': time.perf_counter()-started,
                   'checkpoint_available': (job/'checkpoint.json').is_file()}
            if status == 'ok':
                report = load_json(job/f'm{m}'/'report.json')
                row.update({key: report[key] for key in ('reference_T1', 'final_T', 'k', 'k_per_core',
                           'active_cores', 'subgraphs', 'evaluations', 'officially_validated', 'best_origin')})
            rows.append(row)
            save()
            print(json.dumps(row, ensure_ascii=False), flush=True)
    # 独立内部回放及可选官方复核均在各 case 的 solve-many 子进程中执行，
    # 分别继续计入对应核数的累计 600 秒预算，不设置整例时限。
    if any(r['status'] != 'ok' for r in rows):
        raise RuntimeError('some jobs failed or timed out; checkpoints and logs were preserved')


def add_search_options(parser):
    parser.add_argument('--core-sharing', action=argparse.BooleanOptionalAction, default=True,
                        help='algorithm-owned sharing of all valid trials across eligible counts')
    parser.add_argument('--hc-mode', choices=('targeted','legacy'), default='targeted',
                        help='use bottleneck/Tensor HC, or retain the previous HC for ablation')
    parser.add_argument('--mgr', action=argparse.BooleanOptionalAction, default=True,
                        help='bottleneck-guided macro reconstruction and repair (default: enabled)')
    parser.add_argument('--mgr-max-nodes', type=int, default=512)
    parser.add_argument('--mgr-budget', type=int, help='override total MGR evaluation-count budget')
    parser.add_argument('--coarse-candidates', type=int, default=3,
                        help='full-score coarse proposals; 0 reproduces the previous seed pipeline')
    parser.add_argument('--constructor', choices=('bspg', 'legacy'), default='bspg')
    parser.add_argument('--escape-weight', type=float, default=DEFAULT_ESCAPE_WEIGHT,
                        help='extra continuity penalty (selected default: 2)')
    parser.add_argument('--profile', choices=PROFILES, default='standard')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='random seed shared by all search stages (default: 1234)')
    parser.add_argument('--group-size', type=int, default=8, help='legacy constructor only')
    parser.add_argument('--ready-window', type=int, default=8, help='legacy constructor only')
    parser.add_argument('--cache-mb', type=int, default=128)
    parser.add_argument('--search-seconds', type=float, default=DEFAULT_SEARCH_SECONDS, choices=(DEFAULT_SEARCH_SECONDS,),
                        help='首次搜索基数为 30 秒，自适应上限为 90 秒')
    parser.add_argument('--search-max-seconds',type=float,default=MAX_SEARCH_SECONDS, choices=(MAX_SEARCH_SECONDS,),
                        help='普通搜索 180 秒，持续改善最多 540 秒，均计入当前核数 600 秒总额度')
    parser.add_argument('--core-timeout', type=float, default=CORE_TIMEOUT_SECONDS,
                        help='hard cumulative seconds per core count; maximum 600')
    parser.add_argument('--config', help='配置路径；默认读取输入图同目录的 config.txt')
    parser.add_argument('--official', action='store_true', help='also run unchanged official evaluator')
    parser.add_argument('--resume', action='store_true')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    solve = commands.add_parser('solve', help='generate one internally validated plan')
    solve.add_argument('--case', default='044', help='bundled case ID or graph JSON path')
    solve.add_argument('--cores', type=int, default=4)
    solve.add_argument('--out')
    solve.add_argument('--memory-mb', type=int, default=0)
    solve.add_argument('--inherit-plan', action='append', default=[],
                       help='schedule JSON with at most the target number of occupied cores')
    add_search_options(solve)
    solve.set_defaults(handler=solve_command)
    joint = commands.add_parser('solve-many', help='one case, one algorithm session, shared core-count incumbents')
    joint.add_argument('--case', default='044')
    joint.add_argument('--cores', type=int, nargs='+', default=[2, 3, 4, 5])
    joint.add_argument('--out')
    joint.add_argument('--memory-mb', type=int, default=0)
    joint.add_argument('--seed-plan', action='append', default=[], help='optional prior schedule JSON')
    add_search_options(joint)
    joint.set_defaults(handler=solve_many_command)
    validate = commands.add_parser('validate', help='independent legality and official/internal timing audit')
    validate.add_argument('--case', required=True)
    validate.add_argument('--plan', required=True)
    validate.add_argument('--config', help='配置路径；默认读取输入图同目录的 config.txt')
    validate.add_argument('--out')
    validate.add_argument('--internal-only', action='store_true')
    validate.set_defaults(handler=validate_command)
    diagnosis = commands.add_parser('diagnose', help='optional global causal diagnostics for a saved plan')
    diagnosis.add_argument('--case', required=True)
    diagnosis.add_argument('--plan', required=True)
    diagnosis.add_argument('--config', help='配置路径；默认读取输入图同目录的 config.txt')
    diagnosis.add_argument('--out', required=True)
    def diagnose_command(args):
        from scene_b_internal import SceneBEvaluator
        path = case_path(args.case)
        result = SceneBEvaluator(load_json(path), **hardware(args.config or path.parent/'config.txt')).evaluate(
            load_json(args.plan), diagnostics=True)
        atomic_json(args.out, result)
        print(json.dumps({'valid': True, 'T': result['makespan'], 'out': str(args.out)}))
    diagnosis.set_defaults(handler=diagnose_command)
    batch = commands.add_parser('batch', help='call the joint algorithm for each case, saving improvements immediately')
    batch.add_argument('--cases', default='001-100')
    batch.add_argument('--cores', type=int, nargs='+', default=[2, 3, 4, 5])
    batch.add_argument('--out', default=str(ROOT/'results/batch'))
    batch.add_argument('--memory-mb', type=int, default=1536)
    batch.add_argument('--inherit', action=argparse.BooleanOptionalAction, default=True,
                       help='compatibility ablation switch; disabling it disables core sharing')
    add_search_options(batch)
    batch.set_defaults(handler=batch_command)
    args = parser.parse_args(argv)
    try:
        if args.command in ('solve', 'solve-many'):
            settings_from(args).check()
            first_core = args.cores if isinstance(args.cores, int) else min(args.cores)
            supervise(_invoke_handler, (args,), first_core=first_core,
                      core_seconds=args.core_timeout)
        elif args.command == 'validate':
            supervise(_invoke_handler, (args,), first_core=None)
        else:
            args.handler(args)
    except TimeLimitExceeded as error:
        path = case_path(args.case)
        default_out = ROOT/'results'/(f'{path.stem}_m{args.cores}' if args.command == 'solve' else path.stem)
        if args.command == 'validate':
            out = Path(args.out).parent if args.out else Path(args.plan).parent
        else:
            out = Path(args.out) if args.out else default_out
        atomic_json(out/'time_limit.json', error.details)
        report_path = out/'report.json'
        if args.command != 'validate' and report_path.exists():
            report = load_json(report_path)
            report.update(status='timed_out', time_limit=error.details)
            atomic_json(report_path, report)
        print(json.dumps(error.details), flush=True)
        return TIMEOUT_EXIT_CODE
    except KeyboardInterrupt:
        print('Interrupted; the last feasible checkpoint remains available.', file=sys.stderr)
        return 130
    except Exception:
        traceback.print_exc()
        return 1
    return 0


def _invoke_handler(args):
    return args.handler(args)


if __name__ == '__main__':
    raise SystemExit(main())
