'第三问运行入口：有限时间预算、四个 case 的并行调度及内部评价。'
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from q3_runtime_io import atomic, read
from q3_time import TimePolicy


def case_path(value):
    value=str(value)
    path=ROOT.parent/'100case_data'/f'case_{int(value):03d}.json' if value.isdecimal() else Path(value)
    if not path.is_absolute():path=path.resolve()
    if not path.is_file():raise FileNotFoundError(path)
    return path


def expand_cases(value):
    answer=[]
    for item in value.split(','):
        if '-' in item:
            a,b=item.split('-',1);answer.extend(range(int(a),int(b)+1))
        else:answer.append(int(item))
    if not answer or any(n<1 or n>100 for n in answer):raise ValueError('case IDs must be in 001..100')
    return list(dict.fromkeys(answer))


def options(parser):
    parser.add_argument('--config',default=str(ROOT.parent/'100case_data/config.txt'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--seed-plan',action='append',default=[])
    parser.add_argument('--profile',choices=('initial','smoke','quick','standard','thorough'),default='standard')
    parser.add_argument('--seed',type=int,default=1234)
    parser.add_argument('--constructor',choices=('bspg','legacy'),default='bspg')
    parser.add_argument('--group-size',type=int,default=8)
    parser.add_argument('--ready-window',type=int,default=8)
    parser.add_argument('--cache-mb','--compile-cache-mb',type=int,default=128)
    parser.add_argument('--escape-weight',type=float,default=2.0)
    parser.add_argument('--coarse-candidates',type=int,default=3)
    parser.add_argument('--mgr',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--mgr-max-nodes',type=int,default=512)
    parser.add_argument('--mgr-budget',type=int)
    parser.add_argument('--hc-mode',choices=('targeted','legacy'),default='targeted')
    parser.add_argument('--core-sharing',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--search-seconds',type=float,default=30.0,
        help='positive BASE wall seconds, multiplied once by min(2,max(1,m/k)); zero is rejected')
    parser.add_argument('--core-soft-seconds',type=float,default=300.0)
    parser.add_argument('--core-hard-seconds',type=float,default=600.0)
    defaults=asdict(TimePolicy())
    for name in ('extension_seconds','extension_gain','factor_cap',
                 'improvement_window','stagnation_seconds','constructor_seconds','coarse_seconds',
                 'candidate_seconds','macro_candidate_seconds','repair_seconds',
                 'conversion_fraction','cleanup_seconds'):
        parser.add_argument('--'+name.replace('_','-'),type=float,default=defaults[name])
    parser.add_argument('--workers',type=int,default=4,help='independent cases; core counts within a case are sequential')
    parser.add_argument('--memory-mb',type=int,default=1536,help='memory cap for each worker process')
    parser.add_argument('--max-retries',type=int,default=3)
    parser.add_argument('--official',action='store_true',help=argparse.SUPPRESS)


def parsed_settings(args):
    from q2_search import Settings
    names=('profile','seed','group_size','ready_window','cache_mb','search_seconds','constructor',
           'escape_weight','coarse_candidates','mgr','mgr_max_nodes','mgr_budget','hc_mode','core_sharing')
    settings=Settings(**{key:getattr(args,key) for key in names});settings.check()
    return asdict(settings)


def parsed_policy(args):
    values=asdict(TimePolicy())
    for key in values:
        if hasattr(args,key):values[key]=getattr(args,key)
    values.update(base_seconds=args.search_seconds,soft_seconds=args.core_soft_seconds,
                  hard_seconds=args.core_hard_seconds)
    return TimePolicy(**values).check()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('solve','solve-many','batch'):
        p=commands.add_parser(name)
        if name=='batch':
            source=p.add_mutually_exclusive_group()
            source.add_argument('--cases',default=None)
            source.add_argument('--jobs',help='JSON list of {case,name?,seed_plans?}; paths relative to this file')
        else:p.add_argument('--case',default='001')
        p.add_argument('--cores',type=int,nargs='+' if name!='solve' else None,
                       default=[2,3,4,5] if name!='solve' else 4)
        p.add_argument('--out')
        options(p)
    p=commands.add_parser('validate',help='fresh internal replay; never invokes an official evaluator')
    p.add_argument('--case',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--config',default=str(ROOT.parent/'100case_data/config.txt'));p.add_argument('--out',required=True)
    p.add_argument('--seconds',type=float,default=60.0)
    p=commands.add_parser('stop',help='request the coordinator and its own workers to stop')
    p.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='stop':
            p=Path(args.out).resolve();p.mkdir(parents=True,exist_ok=True)
            (p/'STOP').write_text('requested stop\n')
            print(json.dumps(dict(status='stop_requested',out=str(p))));return 0
        if args.command=='validate':
            from api_q3_full import validate
            result=validate(str(case_path(args.case)),args.plan,config=args.config,seconds=args.seconds)
            atomic(args.out,result);print(json.dumps(dict(status='complete',out=args.out,internal_only=True)));return 0
        if args.official:
            raise ValueError('This Q3 revision uses internal evaluation only; --official is disabled')
        policy=parsed_policy(args);settings=parsed_settings(args)
        from q3_io import hardware3
        from q3_timed_runner import run_cases
        hw=hardware3(args.config)
        counts=(args.cores,) if args.command=='solve' else tuple(args.cores)
        seeds=[read(p) for p in args.seed_plan]
        if args.command=='batch':
            out=Path(args.out or ROOT/'results_timed/batch').resolve()
            if args.jobs:
                jobfile=Path(args.jobs).resolve();raw=read(jobfile)
                if not isinstance(raw,list):raise ValueError('jobs must be a JSON list')
                def resolve(value):
                    if str(value).isdecimal():return case_path(str(value))
                    p=Path(value);return p if p.is_absolute() else jobfile.parent/p
                jobs=[]
                for item in raw:
                    path=resolve(item['case']).resolve();name=item.get('name',path.stem)
                    target=(out/name).resolve()
                    if target==out or not target.is_relative_to(out):raise ValueError('job name must be inside --out')
                    jobs.append(dict(case=str(path),name=name,out=str(target),hardware=hw,
                        seed_plans=seeds+[read(resolve(p)) for p in item.get('seed_plans',[])]))
            else:
                jobs=[dict(case=str(case_path(n)),name=f'case_{n:03d}',out=str(out/f'case_{n:03d}'),
                    hardware=hw,seed_plans=seeds) for n in expand_cases(args.cases or '001-100')]
        else:
            path=case_path(args.case)
            out=Path(args.out or ROOT/'results_timed'/path.stem).resolve()
            jobs=[dict(case=str(path),name=path.stem,out=str(out),hardware=hw,seed_plans=seeds)]
        def event(e):print(json.dumps(e,ensure_ascii=False,allow_nan=False),flush=True)
        summary=run_cases(jobs,counts,settings,policy,out=out,resume=args.resume,on_event=event)
        print(json.dumps(summary,ensure_ascii=False,allow_nan=False),flush=True)
        return 0 if summary['status']=='complete' else 2
    except KeyboardInterrupt:
        print('Interrupted. Complete scored checkpoints and charged time were retained.',file=sys.stderr)
        return 130
    except Exception:
        traceback.print_exc();return 1


if __name__=='__main__':
    raise SystemExit(main())
