'支持四个 case 的协调器，使用有限租约、累计计时及 Popen 进程监督。每个 case 顺序推进核数，完整评分解池跨进程传递；进入下一阶段前回收已结束子进程。主进程不保留模拟器或编译后的图。'
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep, time
import uuid

from q3_runtime_io import atomic, read, digest, code_hash, lock
from q3_time import TimePolicy, REVISION

ROOT = Path(__file__).resolve().parents[1]

def verify_entry(entry):
    if digest(entry['plan']) != entry.get('plan_sha256') or digest(dict(
            plan=entry['plan'],result=entry['result'])) != entry.get('content_sha256'):
        raise ValueError('checkpoint plan/metrics checksum changed; resume refused')
    if entry['result'].get('problem') != 3:
        raise ValueError('checkpoint objective is not Problem 3')


def ledger_row():
    return dict(spent_seconds=0.0,attempts=0,base_done=False,retries=0,
                rechecks=0,extension_eligible=False,last_grant_seconds=0.0)


def is_verified(case, core):
    key = str(core)
    e, r = case['entries'].get(key), case['reports'].get(key,{})
    return bool(e and r.get('internal_recheck_passed') and r.get('plan_sha256') == e['plan_sha256'])


def merge_attempt(case, folder, attempt, core, *, result_allowed=True):
    '只恢复完整、身份匹配且分数未退步的工作进程快照。'
    def identified(path):
        value=read(path)
        if value.get('identity')!=case['identity'] or value.get('attempt')!=attempt:
            raise ValueError('worker checkpoint/result identity mismatch')
        return value
    checkpoint=folder/'checkpoint.json'
    if checkpoint.exists():
        saved=identified(checkpoint)
        for key,entry in saved['entries'].items():
            verify_entry(entry)
            if key not in case['ledger'] or len(entry['plan']['core_schedules'])!=int(key):
                raise ValueError('worker checkpoint core count mismatch')
            def rank(e):
                r=e['result'];b=r['data_movement_bytes']
                return r['makespan'],b['spill_added_copy_bytes'],b['scheduled_copy_bytes']
            previous=case['entries'].get(key)
            if previous is None or rank(entry)<=rank(previous):case['entries'][key]=entry
        case['references'].update(saved['references'])
        for key, progress in case['references'].get('search_progress', {}).items():
            if key in case['ledger']:
                case['ledger'][key]['base_done'] = bool(progress.get('search_finished'))
    path=folder/'result.json'
    if result_allowed and path.exists():
        value=identified(path)
        report=value.get('report')
        entry=case['entries'].get(str(core))
        if report and entry and report.get('plan_sha256')==entry['plan_sha256']:
            if report['final_T']!=entry['result']['makespan']:
                raise ValueError('worker report score mismatch')
            case['reports'][str(core)]=report
        return value
    return None


def export_case(case, directory):
    '先提交权威快照，恢复运行时再生成派生文件。'
    case['content_sha256']=digest({k:v for k,v in case.items() if k!='content_sha256'})
    atomic(directory/'checkpoint.json',case)
    rows = []
    for core in case['counts']:
        key = str(core);entry = case['entries'].get(key)
        if entry is None:
            continue
        report = deepcopy(case['reports'].get(key,{}))
        verified = is_verified(case,core)
        if not verified:
            report = dict(problem=3,backend='problem3',algorithm=REVISION,cores=core,
                status='internal_recheck_pending',final_T=entry['result']['makespan'],
                plan_sha256=entry['plan_sha256'],internally_validated=True,
                internal_recheck_passed=False,officially_validated=False,
                evaluation_kind='internal_only', **{k:entry['result'][k] for k in (
                    'cache_stats','cache_miss_reasons','data_movement_bytes','memory_peak_by_core',
                    'physical_transfer_bytes','active_cores')})
            if 'reference_T1' in case['references']:
                t1,t3 = case['references']['reference_T1'],report['final_T']
                report.update(reference_T1=t1,k=t1/t3 if t3 else None,
                              k_per_core=t1/t3/core if t3 else None)
        report.update(search_finished=case['ledger'][key]['base_done'],
                      status='complete' if verified and case['ledger'][key]['base_done'] else 'search_pending' if not case['ledger'][key]['base_done'] else 'internal_recheck_pending',
                      cumulative_seconds=case['ledger'][key]['spent_seconds'],
                      attempts=case['ledger'][key]['attempts'],job_signature=case['identity'])
        atomic(directory/f'm{core}'/'schedule.json',entry['plan'])
        atomic(directory/f'm{core}'/'report.json',report)
        rows.append(report)
    complete = all(is_verified(case,c) and case['ledger'][str(c)]['base_done'] for c in case['counts'])
    summary = dict(format_version=2,algorithm=REVISION,problem=3,
        status='complete' if complete else 'partial',job_signature=case['identity'],
        cores=case['counts'],rows=rows,internally_validated=complete,
        official_requested=False,officially_validated=False,
        evaluation_kind='internal_only',ledger=case['ledger'])
    atomic(directory/'session.json',summary)
    atomic(directory/'report.json',summary)
    return summary


def prepare_case(job, counts, settings, policy, version, resume):
    directory = Path(job['out']).resolve()
    graph_path = Path(job['case']).resolve()
    signature_payload = dict(code=version,graph_sha256=hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        hardware=job['hardware'],counts=counts,settings=settings,policy=asdict(policy),
        seed_plans=job.get('seed_plans',[]),evaluation_kind='internal_only')
    signature=digest(signature_payload)
    path = directory/'checkpoint.json'
    if resume and path.exists():
        case = read(path)
        if case.get('format_version') != 2 or case.get('identity') != signature:
            raise ValueError('resume refused: old format or graph/settings/code/time policy changed')
        if case.get('content_sha256') != digest({k:v for k,v in case.items() if k!='content_sha256'}):
            raise ValueError('resume refused: checkpoint content checksum changed')
        for entry in case['entries'].values():
            verify_entry(entry)
        # 崩溃前已分配的时间仍计费，恢复时不重置累计时钟。
        for key,row in case['ledger'].items():
            reservation=row.pop('active_reservation',None)
            if reservation:
                attempt=reservation['attempt']
                merge_attempt(case,directory/'attempts'/f'm{key}_{attempt:04d}',attempt,int(key))
                row['extension_eligible']=False
                case['history'].append(dict(core=int(key),attempt=attempt,
                    status='recovered_after_interruption',charged_seconds=reservation['seconds']))
    else:
        visible = list(directory.iterdir()) if directory.exists() else []
        if any(not p.name.startswith('.') for p in visible):
            raise ValueError('output is not empty; choose a new --out or use --resume')
        case = dict(format_version=2,identity=signature,algorithm=REVISION,
                    counts=list(counts),entries={},references={},reports={},
                    ledger={str(c):ledger_row() for c in counts},history=[])
    directory.mkdir(parents=True,exist_ok=True)
    return case


def next_task(case, policy, *, extras=False, allow_extensions=True):
    # 先让每个核数获得首次搜索机会，再分配修复或额外改进时间。
    for core in case['counts']:
        row=case['ledger'][str(core)]
        if row['attempts']==0 and not row.get('budget_exhausted'):
            return core,'search',False,policy.initial_grant()
    for core in case['counts']:
        key=str(core);row=case['ledger'][key]
        if row.get('budget_exhausted'):continue
        if row['base_done'] and key in case['entries'] and not is_verified(case,core) and row['rechecks']<=policy.max_retries:
            cost=case['references'].get('search_progress',{}).get(key,{}).get('evaluation_seconds',0)
            desired=max(policy.initial_grant(),4*cost,2*row.get('last_grant_seconds',0))
            return core,'recheck',False,desired
    if extras:
        # 已保存备用解不代表基础搜索已经完成。
        for core in case['counts']:
            key=str(core);row=case['ledger'][key]
            if not row['base_done'] and not row.get('budget_exhausted') and row['retries']<policy.max_retries:
                cost=case['references'].get('search_progress',{}).get(key,{}).get('evaluation_seconds',0)
                desired=max(2*policy.initial_grant(),2*row.get('last_grant_seconds',0),8*cost)
                return core,'search',False,desired
        if not allow_extensions:return None
        choices=[c for c in case['counts'] if is_verified(case,c)
                 and case['ledger'][str(c)]['base_done']
                 and case['ledger'][str(c)]['extension_eligible']
                 and not case['ledger'][str(c)].get('budget_exhausted')
                 and case['ledger'][str(c)]['spent_seconds']<policy.soft_seconds-policy.cleanup_seconds]
        if choices:
            core=max(choices,key=lambda c:(case['reports'][str(c)].get('recent_gain_fraction',0),-c))
            return core,'search',True,policy.extension_seconds
    return None


def run_cases(jobs, counts, settings, policy=None, *, out, resume=False,
              on_event=None, on_checkpoint=None):
    policy=(policy or TimePolicy()).check()
    counts=tuple(sorted(set(counts)))
    if not counts or any(type(c) is not int or c<1 or c>5 for c in counts):
        raise ValueError('core counts must be integers in 1..5')
    if settings.get('search_seconds',0) <= 0:
        raise ValueError('search_seconds must be positive; zero is not supported in timed Q3')
    if not jobs:
        raise ValueError('at least one case is required')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    destinations=[Path(j['out']).resolve() for j in jobs]
    if len(set(destinations)) != len(destinations):
        raise ValueError('duplicate case output directories')
    version=code_hash(ROOT)
    batch_payload=dict(jobs=[dict(case=str(Path(j['case']).resolve()),out=str(p),
        graph=hashlib.sha256(Path(j['case']).read_bytes()).hexdigest(),hardware=j['hardware'],
        seeds=j.get('seed_plans',[])) for j,p in zip(jobs,destinations)],
        counts=counts,settings=settings,policy=asdict(policy),code=version)
    batch_identity=digest(batch_payload)
    path=out/'batch_state.json';owner=out/'.coordinator_owner.json';stop=out/'STOP'
    active={};cases=[];states=[]
    started=perf_counter();spent_before=0.0
    token=uuid.uuid4().hex
    cancelled=False
    def emit(event):
        if on_event:
            on_event(event)
    with ExitStack() as stack:
        stack.enter_context(lock(out/'.coordinator.lock'))
        for destination in destinations:
            stack.enter_context(lock(destination/'.case.lock'))
        if path.exists():
            old=read(path)
            if not resume:
                raise ValueError('batch output already exists; use --resume or a new --out')
            if old.get('identity') != batch_identity:
                raise ValueError('resume refused: batch identity or time policy changed')
            spent_before=max(old.get('spent_seconds',0),old.get('reserved_through_seconds',0))
        # 读取遗留进程最后一次原子快照前，先使其所属令牌失效。
        atomic(owner,dict(run_token=token,heartbeat_epoch=time(),status='recovering'))
        if resume:sleep(.15)
        for job in jobs:
            cases.append(prepare_case(job,counts,settings,policy,version,resume))
        def elapsed():return spent_before+perf_counter()-started
        last_save=[0.0]
        def save_batch(status='running',force=False):
            if not force and status=='running' and perf_counter()-last_save[0]<policy.heartbeat_seconds:
                return
            last_save[0]=perf_counter()
            amount=elapsed()
            # 主进程意外退出后，子进程最长可能继续运行 8 秒。
            # 该宽限时间也需预留，正常结束时退还未用时间。
            lease=amount+8.2+2*policy.heartbeat_seconds if status=='running' else amount
            atomic(path,dict(format_version=2,identity=batch_identity,algorithm=REVISION,
                status=status,spent_seconds=amount,reserved_through_seconds=lease,
                policy=asdict(policy),counts=list(counts),
                official_requested=False,results=states))
            atomic(owner,dict(run_token=token,heartbeat_epoch=time(),status=status))
        def export(index):
            session=export_case(cases[index],destinations[index])
            row=dict(case=jobs[index].get('name',Path(jobs[index]['case']).stem),
                     directory=str(destinations[index]),status=session['status'],
                     verified_cores=[c for c in counts if is_verified(cases[index],c)],
                     rows=[{k:r[k] for k in ('cores','final_T','k','internal_recheck_passed','cumulative_seconds') if k in r}
                           for r in session['rows']])
            if index<len(states):states[index]=row
            else:states.append(row)
            if on_checkpoint:on_checkpoint(deepcopy(cases[index]))
        for i in range(len(cases)):export(i)
        save_batch(force=True)
        def finish(index, forced=False):
            item=active.pop(index);process=item['process'];case=cases[index]
            if forced and process.poll() is None:
                process.kill()
            process.wait(timeout=max(.2,policy.cleanup_seconds))
            item['log'].close()
            used=perf_counter()-item['start']
            core=item['core'];row=case['ledger'][str(core)]
            row['spent_seconds'] += used-item['grant']
            row.pop('active_reservation',None)
            row.update(last_elapsed_seconds=used,last_returncode=process.returncode)
            result=merge_attempt(case,item['checkpoint'].parent,item['attempt'],core,
                                 result_allowed=process.returncode==0)
            if result is not None:
                if item['mode']=='search':
                    row['extension_eligible']=bool(result['details']['extension_eligible'])
                    if result['details'].get('dynamic_factor') is not None:
                        row.setdefault('frozen_budget',{k:result['details'][k]
                            for k in ('dynamic_factor','nominal_seconds')})
            else:
                row['extension_eligible']=False
            case['history'].append(dict(core=core,mode=item['mode'],extension=item['extension'],
                attempt=item['attempt'],seconds=used,returncode=process.returncode,forced=forced,
                granted_seconds=item['grant']))
            export(index)
            emit(dict(event='core_finish',case=states[index]['case'],cores=core,
                      seconds=used,returncode=process.returncode,
                      verified=is_verified(case,core),forced=forced))
        coverage_finished=False
        try:
            while True:
                now=perf_counter()
                cancelled=stop.exists()
                global_stop=cancelled
                for index,item in list(active.items()):
                    if item['process'].poll() is not None:
                        finish(index)
                    elif global_stop or now>=item['end']:
                        finish(index,forced=True)
                if global_stop:
                    break
                # 只有全部普通 case 工作分配完成后，
                # 再依次安排仍有单核额度的重试和时间扩展。
                base_tasks=[next_task(c,policy,extras=False) for c in cases]
                coverage_finished = coverage_finished or (not any(base_tasks) and not active)
                all_base_done=coverage_finished and not any(base_tasks)
                basic_pending=any(next_task(c,policy,extras=True,allow_extensions=False) for c in cases)
                started_any=False
                for index,case in enumerate(cases):
                    if len(active)>=min(policy.workers,len(jobs)):break
                    if index in active:continue
                    task=next_task(case,policy,extras=all_base_done,allow_extensions=not basic_pending)
                    if task is None:continue
                    core,mode,extension,desired=task
                    row=case['ledger'][str(core)]
                    remaining=policy.hard_seconds-row['spent_seconds']
                    if mode=='search' and extension:
                        remaining=min(remaining,policy.soft_seconds-row['spent_seconds'])
                    # 某个配置耗尽预算时，
                    # 仍应继续推进其他核数及其他 case。
                    if remaining<=.5:
                        row.update(budget_exhausted=True,extension_eligible=False,
                                   retries=policy.max_retries,rechecks=policy.max_retries+1)
                        export(index);continue
                    # 只看当前核数的累计余额，不使用任何整例或全局截止时间。
                    grant=min(desired,remaining)
                    if grant<=.5:continue
                    before=perf_counter()
                    attempt=row['attempts']+1
                    folder=destinations[index]/'attempts'/f'm{core}_{attempt:04d}'
                    folder.mkdir(parents=True,exist_ok=True)
                    cp=folder/'checkpoint.json';result=folder/'result.json'
                    # 启动子进程前先预留完整时间额度，
                    # 防止协调进程崩溃并恢复后重复分配同一份时间。
                    row['spent_seconds']+=grant
                    row.update(attempts=attempt,last_grant_seconds=grant,
                               active_reservation=dict(seconds=grant,attempt=attempt))
                    if mode=='recheck':row['rechecks']+=1
                    elif attempt>1 and not extension:row['retries']+=1
                    cleanup=min(policy.cleanup_seconds,max(.005,.05*grant))
                    req=dict(identity=case['identity'],graph_path=str(Path(jobs[index]['case']).resolve()),
                        hardware=jobs[index]['hardware'],settings=settings,policy=asdict(policy),
                        counts=list(counts),cores=core,entries=case['entries'],references=case['references'],
                        seed_plans=jobs[index].get('seed_plans',[]),mode=mode,extension=extension,
                        frozen_budget=row.get('frozen_budget'),
                        started=before,deadline=before+grant-cleanup,attempt=attempt,
                        checkpoint_path=str(cp),result_path=str(result),error_path=str(folder/'error.json'),
                        events_path=str(destinations[index]/'events.jsonl'),owner_path=str(owner),
                        run_token=token,stop_path=str(stop))
                    atomic(folder/'request.json',req)
                    export(index);save_batch(force=True)
                    log=(folder/'worker.log').open('w',encoding='utf-8')
                    environment=dict(os.environ,PYTHONHASHSEED=str(settings['seed']))
                    try:
                        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'src/q3_timed_worker.py'),
                            str(folder/'request.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=environment)
                    except BaseException:
                        log.close();raise
                    active[index]=dict(process=process,log=log,start=before,
                        end=before+grant-cleanup/2,grant=grant,core=core,mode=mode,
                        extension=extension,attempt=attempt,checkpoint=cp,result=result)
                    emit(dict(event='core_start',case=states[index]['case'],cores=core,
                              mode=mode,extension=extension,granted_seconds=grant))
                    started_any=True
                save_batch()
                if not active and not started_any:
                    # 显示预算耗尽的任务行可能刚刚完成状态推进。
                    possible=[next_task(c,policy,extras=True) for c in cases]
                    if not any(possible):break
                    if all(t is None or cases[i]['ledger'][str(t[0])]['spent_seconds']>=policy.hard_seconds-.5
                           for i,t in enumerate(possible)):break
                if active:
                    nearest=min(item['end'] for item in active.values())-perf_counter()
                    sleep(max(.001,min(.1,policy.heartbeat_seconds,nearest)))
                elif not started_any:break
        except BaseException:
            for index in list(active):finish(index,forced=True)
            save_batch('interrupted')
            raise
        finally:
            for index in list(active):finish(index,forced=True)
        complete=all(s['status']=='complete' for s in states)
        exhausted=any(row.get('budget_exhausted') for case in cases for row in case['ledger'].values())
        status='cancelled' if cancelled else 'complete' if complete else 'budget_exhausted' if exhausted else 'partial'
        save_batch(status)
        summary=dict(status=status,algorithm=REVISION,cores=list(counts),results=states,
            jobs=len(jobs),completed=sum(s['status']=='complete' for s in states),
            verified_configurations=sum(len(s['verified_cores']) for s in states),
            requested_configurations=len(jobs)*len(counts),spent_seconds=elapsed(),
            core_limit_seconds=policy.hard_seconds,evaluation_kind='internal_only',official_requested=False)
        atomic(out/'summary.json',summary)
        return summary
