#!/usr/bin/env python3
'有时间限制的内部评价及固定核分配下的顺序优化。'
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from q3_runtime_io import atomic,read,digest,code_hash,lock
from q3_io import hardware3,summary
from q3_order import OrderSettings
from q3_validation_runner import validate_bounded,order_bounded


def resolve_case(value):
    raw=str(value)
    path=ROOT.parent/'100case_data'/f'case_{int(raw):03d}.json' if raw.isdecimal() else Path(raw)
    if not path.is_file():raise FileNotFoundError(path)
    return path.resolve()


def main(argv=None, *, available_seconds=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('evaluate','validate','optimize'):
        p=commands.add_parser(name)
        p.add_argument('--case',required=True)
        p.add_argument('--plan',required=True,type=Path)
        p.add_argument('--config',type=Path)
        p.add_argument('--out',required=True,type=Path)
        p.add_argument('--seconds',type=float,default=60.0)
        if name=='evaluate':p.add_argument('--diagnostics',action='store_true')
        if name=='optimize':
            p.add_argument('--evaluations',type=int,default=96)
            p.add_argument('--seed',type=int,default=1234)
            p.add_argument('--compile-cache-mb',type=int,default=128)
            p.add_argument('--search-seconds',type=float,default=30.0)
            p.add_argument('--max-targets',type=int,default=24)
            p.add_argument('--memory-mb',type=int,default=1536)
            p.add_argument('--no-cache-guidance',action='store_true')
            p.add_argument('--official',action='store_true',help=argparse.SUPPRESS)
            p.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    started=perf_counter()
    if getattr(args,'official',False):raise ValueError('official evaluation is disabled; internal only')
    graph,plan,hw=read(resolve_case(args.case)),read(args.plan),hardware3(args.config)
    if args.command!='optimize':
        result=validate_bounded(graph,plan,hw,args.seconds,diagnostics=getattr(args,'diagnostics',False),
                                include_reference=args.command=='validate')
        atomic(args.out,result if args.command=='validate' else result['result'] if args.diagnostics else summary(result['result']))
        print(json.dumps(dict(status='complete',out=str(args.out),internal_only=True)));return 0
    settings=OrderSettings(seed=args.seed,max_evaluations=args.evaluations,
        compile_cache_mb=args.compile_cache_mb,search_seconds=args.search_seconds,
        max_targets=args.max_targets,cache_guided=not args.no_cache_guidance)
    settings.check()
    if args.memory_mb<1:raise ValueError('memory-mb must be positive')
    signature=digest(dict(code=code_hash(ROOT),graph=graph,plan=plan,hardware=hw,
                          settings=asdict(settings),memory_mb=args.memory_mb))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    with lock(out/'.order.lock'):
        path=out/'checkpoint.json'
        previous=None;used=0.0
        if args.resume and path.exists():
            saved=read(path)
            if saved.get('identity')!=signature or saved.get('format_version')!=2:
                raise ValueError('resume refused: code/input/settings changed or old checkpoint')
            if saved.get('content_sha256')!=digest({k:v for k,v in saved.items() if k!='content_sha256'}):
                raise ValueError('resume refused: checkpoint content checksum changed')
            previous=saved['search'];used=saved['used_seconds']
            if saved.get('complete'):
                atomic(out/'schedule.json',previous['best_plan'])
                atomic(out/'report.json',saved['report'])
                print(json.dumps(dict(status='already_complete',out=str(out))));return 0
        elif any(not p.name.startswith('.') for p in out.iterdir()):
            raise ValueError('output is not empty; use --resume or a new directory')
        latest=[previous]
        def persist(amount,complete=False,report=None):
            value=dict(format_version=2,identity=signature,search=latest[0],used_seconds=amount,
                       complete=complete,report=report,evaluation_kind='internal_only')
            value['content_sha256']=digest(value);atomic(path,value)
            if latest[0]:atomic(out/'schedule.json',latest[0]['best_plan'])
        remaining=max(0.0,args.search_seconds-used-(perf_counter()-started))
        if available_seconds is not None:remaining=min(remaining,available_seconds-(perf_counter()-started))
        if remaining<=.02:
            print(json.dumps(dict(status='time_budget_exhausted',out=str(out))));return 2
        reserved=used+(perf_counter()-started)+remaining
        # 主进程崩溃后，遗留子进程已消耗的时间额度仍计入预算。
        persist(reserved)
        atomic(out/'input_schedule.json',plan)
        def save(snapshot):
            latest[0]=snapshot;persist(reserved)
        try:
            best,report=order_bounded(graph,plan,hw,args.search_seconds,
                {k:v for k,v in asdict(settings).items() if k!='search_seconds'},
                checkpoint=previous,on_checkpoint=save,used_seconds=used+perf_counter()-started,
                wall_seconds=max(.001,remaining-.02),memory_mb=args.memory_mb)
        except TimeoutError:
            report=dict(problem=3,status='time_budget_exhausted',internal_recheck_passed=False,
                        internally_validated=latest[0] is not None,officially_validated=False)
        except BaseException:
            persist(min(reserved,used+perf_counter()-started));raise
        elapsed=used+perf_counter()-started
        report.update(elapsed_seconds=elapsed,evaluation_kind='internal_only',identity=signature)
        complete=bool(report.get('internal_recheck_passed'))
        persist(elapsed,complete,report)
        atomic(out/'report.json',report)
        print(json.dumps(dict(status='complete' if complete else report['status'],
                              out=str(out),seconds=elapsed,internal_recheck_passed=complete)))
        return 0 if complete else 2


if __name__=='__main__':
    raise SystemExit(main())
