#!/usr/bin/env python3
'按顺序执行有时间限制的固定分配任务；完整第三问批量入口为 run_q3_full.py。'
import argparse
import math
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from q3_runtime_io import atomic,read,digest,code_hash,lock
from run_q3 import main as run_order,resolve_case


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs',required=True,type=Path)
    p.add_argument('--evaluations',type=int,default=96)
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--compile-cache-mb',type=int,default=128)
    p.add_argument('--search-seconds','--timeout',type=float,default=30.0)
    p.add_argument('--memory-mb',type=int,default=1536)
    p.add_argument('--official',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--summary',required=True,type=Path)
    args=p.parse_args(argv);started=perf_counter()
    if args.official:raise ValueError('official evaluation is disabled; internal only')
    if not 0<args.search_seconds<=600:
        p.error('require 0 < search-seconds <= 600')
    jobs=read(args.jobs)
    if not isinstance(jobs,list) or not jobs:p.error('jobs must be a nonempty JSON list')
    base=args.jobs.resolve().parent
    def resolve(value):
        path=Path(value);return path if path.is_absolute() else base/path
    normalized=[]
    for job in jobs:
        case=resolve_case(job['case'] if str(job['case']).isdecimal() else resolve(job['case']))
        plan=resolve(job['plan']);out=resolve(job['out'])
        config=resolve(job['config']) if job.get('config') else ROOT.parent/'100case_data/config.txt'
        normalized.append(dict(case=str(case),plan=str(plan),out=str(out),config=str(config),
            inputs=digest([read(case),read(plan),config.read_text()])))
    identity=digest(dict(code=code_hash(ROOT),jobs=normalized,options={k:v for k,v in vars(args).items()
        if k not in ('resume','jobs','summary')}))
    records=[];prior=0.0
    with lock(args.summary.with_suffix('.lock')):
        if args.summary.exists():
            if not args.resume:raise ValueError('summary exists; use --resume')
            old=read(args.summary)
            if old.get('identity')!=identity:raise ValueError('resume refused: batch identity changed')
            prior=max(old['spent_seconds'],old.get('reserved_seconds',0));records=old['results']
        def save(status,reserved=None):
            spent=prior+perf_counter()-started
            atomic(args.summary,dict(identity=identity,status=status,jobs=len(jobs),results=records,
                spent_seconds=spent,reserved_seconds=spent if reserved is None else reserved,
                evaluation_kind='internal_only'))
        for index,job in enumerate(normalized):
            if index<len(records) and records[index]['status']=='complete':continue
            # 每项独立分配时间，前面的任务不会耗用后续任务的额度。
            grant=args.search_seconds
            save('running',prior+perf_counter()-started+grant)
            command=['optimize','--case',job['case'],'--plan',job['plan'],'--out',job['out'],
                '--config',job['config'],'--evaluations',str(args.evaluations),'--seed',str(args.seed),
                '--compile-cache-mb',str(args.compile_cache_mb),'--search-seconds',str(args.search_seconds),
                '--memory-mb',str(args.memory_mb)]
            if args.resume and (Path(job['out'])/'checkpoint.json').exists():command.append('--resume')
            try:code=run_order(command,available_seconds=grant)
            except BaseException:
                save('interrupted');raise
            row=dict(index=index,case=job['case'],out=job['out'],returncode=code,
                     status='complete' if code==0 else 'partial')
            if index<len(records):records[index]=row
            else:records.append(row)
            save('running')
        complete=len(records)==len(jobs) and all(r['status']=='complete' for r in records)
        save('complete' if complete else 'partial')
        return 0 if complete else 2


if __name__=='__main__':
    raise SystemExit(main())
