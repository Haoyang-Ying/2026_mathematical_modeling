'生成低成本的分阶段并行候选，不进行详细评价。'
from collections import defaultdict
from time import perf_counter

from coarse_guidance import Guidance,MemoryProbe,_components
from dagp_execution import _check_deadline
from bspg_branch_split import generate_bspg_plan
from time_budget import base_seconds as budget_seconds


def pad_plan(plan,m):
    '保留 任务 编号、成员映射和所有已占用核的编号。'
    if len(plan['core_schedules'])>m:
        raise ValueError('cannot pad to fewer cores')
    return {'node_to_subgraph':dict(plan['node_to_subgraph']),
            'core_schedules':[list(x) for x in plan['core_schedules']]
                             +[[] for _ in range(m-len(plan['core_schedules']))]}


def _estimate(model,plan):
    blocks=defaultdict(set)
    for u,t in plan['node_to_subgraph'].items():blocks[t].add(int(u))
    pred,succ,_=model.validate_partition(blocks)
    pred={t:set(ps) for t,ps in pred.items()}
    succ={t:set(cs) for t,cs in succ.items()}
    owner={t:p for p,order in enumerate(plan['core_schedules']) for t in order}
    for order in plan['core_schedules']:
        for a,b in zip(order,order[1:]):pred[b].add(a);succ[a].add(b)
    degree={t:len(ps) for t,ps in pred.items()};queue=sorted(t for t,d in degree.items() if not d)
    ends={};copy=0
    for t in queue:
        c=model.cost(frozenset(blocks[t]));copy+=c.copy_cycles
        # 使用保守的局部流水线计算量与边界搬运量，并加入全局 DDR 下界；该代理只用于候选排序。
        duration=sum(c.pipe_work.values())+c.copy_cycles
        ends[t]=duration+max((ends[p]+(model.hardware.same_wait if owner[p]==owner[t]
                    else model.hardware.cross_wait) for p in pred[t]),default=0)
        for child in sorted(succ[t]):
            degree[child]-=1
            if not degree[child]:queue.append(child)
    if len(ends)!=len(blocks):raise ValueError('phase proposal has a cycle')
    return max(max(ends.values(),default=0),copy),copy


def generate_stage_plan(case,m,hardware,model,base_plan,*,seconds=budget_seconds(2.0),statistics=None):
    '为现有容量切分方案的前缀借用空闲核。\n\n输入为粗粒度方案，不是已有优化结果。每个前缀都是独立分量内的拓扑前缀；比例描述工作量，不依赖层编号或特定用例位置。仅对选中候选执行存活内存探测，最终方案随后由调用方完整评价。'
    started=perf_counter();deadline=started+max(0,seconds)
    stats={'strategy':'capacity_phase_prefix_parallelism','full_evaluations':0}
    probe=None
    try:
        _check_deadline(deadline)
        base=pad_plan(base_plan,m)
        idle=[p for p,o in enumerate(base['core_schedules']) if not o]
        if not idle:return None
        _,comp_of=_components(model,deadline)
        blocks=defaultdict(set)
        for u,t in base['node_to_subgraph'].items():blocks[t].add(int(u))
        pred,_,_=model.validate_partition(blocks)
        candidates=[]
        roots=[o[0] for o in base['core_schedules'] if o and not pred[o[0]]]
        roots=sorted(roots,key=lambda t:(-sum(model.cycles[u] for u in blocks[t]),t))[:m]
        for task in roots:
            seqs=defaultdict(list)
            for u in sorted(blocks[task],key=model.rank.get):seqs[comp_of[u]].append(u)
            if len(seqs)<2 or any(len(s)<2 for s in seqs.values()):continue
            for fraction in (.5,2/3,.75):
                _check_deadline(deadline)
                prefixes={};cuts={}
                for branch,seq in seqs.items():
                    target=fraction*sum(model.cycles[u] for u in seq);work=0;cut=1
                    for i,u in enumerate(seq[:-1]):
                        work+=model.cycles[u];cut=i+1
                        if work>=target:break
                    prefixes[branch]=seq[:cut];cuts[branch]=cut
                loads={p:0 for p in idle};groups={p:set() for p in idle}
                for branch,seq in sorted(prefixes.items(),key=lambda x:(-sum(model.cycles[u] for u in x[1]),x[0])):
                    p=min(idle,key=lambda p:(loads[p],p))
                    groups[p].update(seq);loads[p]+=sum(model.cycles[u] for u in seq)
                trial=pad_plan(base,m);next_task=max(blocks)+1
                for p,members in groups.items():
                    if not members:continue
                    for u in members:trial['node_to_subgraph'][str(u)]=next_task
                    trial['core_schedules'][p].append(next_task);next_task+=1
                try:
                    model.validate_plan(trial,m)
                    proxy=_estimate(model,trial)
                except ValueError:continue
                candidates.append((proxy,task,fraction,trial,cuts))
        if not candidates:return None
        probe=MemoryProbe(model,deadline)
        # 最多尝试两个结构代理最优候选，仅接受其中一个满足约束者；构造过程中不计算完整完工时间。
        for proxy,task,fraction,trial,cuts in sorted(candidates,key=lambda x:x[:3])[:2]:
            _check_deadline(deadline)
            new_blocks=defaultdict(set)
            for u,t in trial['node_to_subgraph'].items():new_blocks[t].add(int(u))
            if any(not probe.profile(g)['fits'] for g in new_blocks.values()):continue
            group_of={int(u):t for u,t in trial['node_to_subgraph'].items()}
            owner={t:p for p,o in enumerate(trial['core_schedules']) for t in o}
            priority={t:i for o in trial['core_schedules'] for i,t in enumerate(o)}
            stats.update(proxy_makespan=proxy[0],source_task=task,work_fraction=fraction,
                         cut_lengths=cuts,borrowed_cores=idle,coarse_plan=trial,
                         candidate_count=len(candidates),fallback=False)
            guidance=Guidance(group_of,owner,priority,stats)
            fine={}
            plan=generate_bspg_plan(case,m,bandwidth=hardware.bandwidth,
                       same_wait=hardware.same_wait,cross_wait=hardware.cross_wait,
                       guidance=guidance,statistics=fine)
            model.validate_plan(plan,m);stats['bspg']=fine
            return plan
        stats.update(fallback=True,reason='candidate memory exceeds capacity')
    except TimeoutError:
        stats.update(fallback=True,reason='phase proposal deadline reached')
    finally:
        if probe:probe.builder.close()
        stats['generation_seconds']=perf_counter()-started
        if statistics is not None:statistics.update(stats)
    return None
