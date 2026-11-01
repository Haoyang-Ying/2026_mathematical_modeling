'在有限时间内为 BSPg 提供结构指导，此处不计算完整执行耗时。\n\n全部决策仅依据图结构、张量标识、官方容量和计算量，不使用用例名称、根节点编号或固定层位置制定规则。'
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
import heapq
from statistics import median
from time import perf_counter

from dagp_execution import TemplateEvaluator, _check_deadline, _preparation_deadline
from hybrid_macro import Region, _rebuild_blocks, _schedule_region
from time_budget import base_seconds as budget_seconds


@dataclass
class Guidance:
    group_of: dict[int, int]
    owner: dict[int, int]
    priority: dict[int, int]
    statistics: dict
    hard_boundaries: bool = True


class MemoryProbe:
    '采用官方边界构造和 步骤 1 顺序，再计算张量存活区间。\n\n不执行 步骤 2、步骤 3 或共享 DDR 回放。支持绝对边界序号以便与完整方案比较，并限制缓存规模。'
    def __init__(self, model, deadline=None):
        self.model, self.deadline = model, deadline
        self.builder = TemplateEvaluator(model, deadline=deadline, max_cached_nodes=0)
        self.cache = OrderedDict()
        self.calls = 0

    def profile(self, members, ordinal=0):
        _check_deadline(self.deadline)
        members = frozenset(members)
        key = (members, ordinal)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        b = self.builder
        count = b._boundary_count(members, self.deadline)
        b._ensure_boundary_ids(ordinal + count, self.deadline)
        with _preparation_deadline(self.deadline):
            graph, boundary = b._task_graph(members, self.deadline, ordinal)
            seq = b.step1.step1_schedule(graph)
            step = {op:i for i,op in enumerate(seq)}
            first, last = {}, {}
            for i, edge in enumerate(graph['edges']):
                if i % 1024 == 0:
                    _check_deadline(self.deadline)
                a, c = edge['source'], edge['target']
                if a in step and c not in step:
                    tid, s = c, step[a]
                elif c in step and a not in step:
                    tid, s = a, step[c]
                else:
                    continue
                first[tid] = min(first.get(tid,s),s)
                last[tid] = max(last.get(tid,s),s)
            delta = {p:[0]*(len(seq)+1) for p in self.model.hardware.capacity}
            for t in graph['tensors']:
                tid, p, size = t['id'], t['pos'], t['size']
                if p in delta and tid in first:
                    delta[p][first[tid]] += size
                    delta[p][last[tid]+1] -= size
            peaks = {}
            for p, changes in delta.items():
                live = peak = 0
                for d in changes:
                    live += d
                    peak = max(peak,live)
                peaks[p] = peak
            result = {'peak_bytes':peaks, 'boundary_copy_bytes':boundary,
                      'fits':all(peaks[p]<=cap for p,cap in self.model.hardware.capacity.items())}
        self.calls += 1
        self.cache[key] = result
        while len(self.cache)>64:
            self.cache.popitem(last=False)
        # 本次探测无需在整个对象生命周期内保留成员集合缓存。
        if len(b._boundary_counts)>128:
            b._boundary_counts.clear()
        return result


def _components(model, deadline):
    unseen = set(model.ids)
    components, component_of = [], {}
    for seed in model.topo:
        if seed not in unseen:
            continue
        stack, members = [seed], set()
        unseen.remove(seed)
        while stack:
            u = stack.pop()
            members.add(u)
            component_of[u] = len(components)
            for v in model.pred[u] + model.succ[u]:
                if v in unseen:
                    unseen.remove(v)
                    stack.append(v)
            if len(members)%256 == 0:
                _check_deadline(deadline)
        components.append(members)
    return components, component_of


def _contract_dependencies(model,m,deadline):
    '在计算量上限约束下贪心收缩边，并保持无环。\n\n仅当不存在其他从 u 到 v 的路径时，才允许收缩 u→v，因此商图仍为 DAG，无需构造传递闭包矩阵。'
    blocks = {u:{u} for u in model.ids}
    pred = {u:set(model.pred[u]) for u in model.ids}
    succ = {u:set(model.succ[u]) for u in model.ids}
    work = {u:max(1,model.cycles[u]) for u in model.ids}
    target = max(2*model.hardware.cross_wait,sum(work.values())/max(1,2*m))
    original_span = {}
    for u in model.topo:
        original_span[u] = work[u]+max((original_span[p] for p in pred[u]),default=0)
    span_limit = 1.15*max(max(original_span.values(),default=0),sum(work.values())/max(1,m))
    version = dict.fromkeys(model.ids,0)
    heap = []
    def push(u,v):
        if u==v or u not in blocks or v not in blocks or v not in succ[u]:
            return
        if work[u]+work[v]>target:
            return
        # 短依赖算子的成本主要来自同步；相同优先级时以边界字节量优先选择直接数据复用。所有数值均来自原图。
        tids = set(t for x in blocks[u] for t in model.touched[x])
        shared_bytes = sum(model.tensors[t].size for t in tids
                           if model.tensors[t].producers & blocks[u]
                           and model.tensors[t].consumers & blocks[v])
        benefit = model.hardware.cross_wait + shared_bytes/model.hardware.bandwidth
        heapq.heappush(heap,(-benefit/(work[u]+work[v]),u,v,version[u],version[v]))
    for u in model.ids:
        for v in succ[u]:
            push(u,v)
    attempts = accepted = 0
    while heap:
        if attempts%64==0:
            _check_deadline(deadline)
        _,u,v,vu,vv = heapq.heappop(heap)
        if u not in blocks or v not in blocks or version[u]!=vu or version[v]!=vv or v not in succ[u]:
            continue
        attempts += 1
        todo = list(succ[u]-{v})
        seen = set(todo)
        cyclic = False
        while todo:
            x = todo.pop()
            if v in succ[x]:
                cyclic = True
                break
            for y in succ[x]:
                if y not in seen:
                    seen.add(y)
                    todo.append(y)
        if cyclic:
            continue
        old_pred = {x:set(pred[x]) for x in (succ[u]|succ[v]|{u,v})}
        old_succ = {x:set(succ[x]) for x in (pred[u]|pred[v]|{u,v})}
        parents = (pred[u]|pred[v])-{u,v}
        children = (succ[u]|succ[v])-{u,v}
        for p in parents:
            succ[p].discard(u);succ[p].discard(v);succ[p].add(u)
        for c in children:
            pred[c].discard(u);pred[c].discard(v);pred[c].add(u)
        pred[u],succ[u] = parents,children
        # 拒绝使全图过度串行化的收缩；仅保证无环，仍可能把几乎整个 DAG 压成一条链。
        degree = {x:len(ps) for x,ps in pred.items() if x!=v}
        queue = [x for x,d in degree.items() if not d]
        end = {}
        for x in queue:
            end[x] = (work[u]+work[v] if x==u else work[x])+max(
                (end[p] for p in pred[x]),default=0)
            for y in succ[x]:
                degree[y] -= 1
                if not degree[y]:
                    queue.append(y)
        if max(end.values(),default=0)>span_limit:
            pred.update(old_pred);succ.update(old_succ)
            continue
        blocks[u].update(blocks.pop(v))
        work[u] += work.pop(v)
        pred[u],succ[u] = parents,children
        del pred[v],succ[v]
        version[u] += 1
        accepted += 1
        for p in parents:
            push(p,u)
        for c in children:
            push(u,c)
    ordered = sorted(blocks,key=lambda u:min(model.rank[x] for x in blocks[u]))
    result = {i:frozenset(blocks[u]) for i,u in enumerate(ordered)}
    model.validate_partition(result)
    return result, {'contraction_attempts':attempts,'contractions':accepted,
                    'target_work':target,'contracted_span_limit':span_limit}


def _family_plan(model, components, external_inputs, external_sizes, m, probe, deadline):
    '先均衡独立连通分量，再切分共享输入的阶段组。\n\n各分量在计算 DAG 中仍相互独立；每次切分取各分量内部的拓扑前缀，因此对齐后的阶段分组仍然无环。'
    comp_of = {u:i for i,c in enumerate(components) for u in c}
    consumers = defaultdict(set)
    for u in model.ids:
        for tid in external_inputs.get(u,()):
            consumers[tid].add(comp_of[u])
    shared = {t for t,cs in consumers.items() if len(cs)>1}
    families = defaultdict(list)
    for i, comp in enumerate(components):
        signature = frozenset(t for u in comp for t in external_inputs.get(u,()) if t in shared)
        families[signature].append(i)
    blocks, orders = {}, [[] for _ in range(m)]
    loads = [0.0]*m
    logs = []
    next_group = 0
    for signature, indices in sorted(families.items(),key=lambda item:min(item[1])):
        _check_deadline(deadline)
        component_owner = {}
        # 按整条分支的计算量执行 LPT 分配，避免阶段之间频繁迁移。
        for i in sorted(indices,key=lambda i:(-sum(model.cycles[u] for u in components[i]),i)):
            p = min(range(m),key=lambda p:(loads[p],p))
            component_owner[i] = p
            loads[p] += model.cost(frozenset(components[i])).compute
        seqs = {i:sorted(components[i],key=model.rank.__getitem__) for i in indices}
        pending = [seqs]
        phases = []
        while pending:
            _check_deadline(deadline)
            phase = pending.pop(0)

            def grouped(parts):
                out = [set() for _ in range(m)]
                for i, ops in parts.items():
                    out[component_owner[i]].update(ops)
                return [frozenset(x) for x in out]

            groups = grouped(phase)
            before = [probe.profile(g) for g in groups if g]
            needs_split = any(not r['fits'] for r in before)
            if not needs_split or len(phases)+len(pending)>=7:
                phases.append(phase)
                continue
            # 利用输入权重的中点及其附近少量算子，寻找权重较重的转折位置，无需预设网络层数。
            pivots = {}
            for i, ops in phase.items():
                seen, cumulative = set(), []
                total = 0
                for u in ops:
                    fresh = set(external_inputs.get(u,()))-seen
                    total += sum(external_sizes[t] for t in fresh)
                    seen.update(fresh)
                    cumulative.append(total)
                if total:
                    pivot = next(k+1 for k,c in enumerate(cumulative) if c>=total/2)
                else:
                    work = sum(model.cycles[u] for u in ops)
                    partial = 0
                    pivot = len(ops)//2
                    for k,u in enumerate(ops):
                        partial += model.cycles[u]
                        if partial>=work/2:
                            pivot = k+1
                            break
                pivots[i] = pivot
            best = None
            for offset in (0,-1,1,-2,2,-3,3):
                _check_deadline(deadline)
                left, right, cuts = {}, {}, {}
                for i, ops in phase.items():
                    if len(ops)<2:
                        left[i],right[i],cuts[i] = ops,[],len(ops)
                    else:
                        k = min(len(ops)-1,max(1,pivots[i]+offset))
                        left[i],right[i],cuts[i] = ops[:k],ops[k:],k
                if not any(right.values()):
                    continue
                trial = [probe.profile(g) for part in (left,right) for g in grouped(part) if g]
                over = sum(max(0,r['peak_bytes'][p]-cap)
                           for r in trial for p,cap in model.hardware.capacity.items())
                peak_ratio = max(r['peak_bytes'][p]/max(1,cap)
                                 for r in trial for p,cap in model.hardware.capacity.items())
                boundary = sum(r['boundary_copy_bytes'] for r in trial)
                key = (over, boundary, peak_ratio, abs(offset), offset)
                if best is None or key<best[0]:
                    best = (key,left,right,cuts,trial)
            original_over = sum(max(0,r['peak_bytes'][p]-cap)
                                for r in before for p,cap in model.hardware.capacity.items())
            if best is None or best[0][0]>=original_over:
                phases.append(phase)
                logs.append({'kind':'capacity_unresolved','components':indices,
                             'before':before})
                continue
            logs.append({'kind':'capacity_split','components':indices,
                         'cut_lengths':best[3], 'before':before, 'after':best[4],
                         'boundary_delta':best[0][1]-sum(r['boundary_copy_bytes'] for r in before)})
            pending[0:0] = [best[1],best[2]]
        for phase in phases:
            grouped_phase = [set() for _ in range(m)]
            for i,ops in phase.items():
                grouped_phase[component_owner[i]].update(ops)
            for p,g in enumerate(grouped_phase):
                if g:
                    blocks[next_group] = frozenset(g)
                    orders[p].append(next_group)
                    next_group += 1
    plan = {'node_to_subgraph':{str(u):b for b,g in blocks.items() for u in g},
            'core_schedules':orders}
    model.validate_plan(plan,m)
    return plan, logs


def build_guidance(model, m, op_graph, *, time_limit=budget_seconds(2.0), group_policy='validated'):
    if group_policy not in ('legacy','validated'):
        raise ValueError('group_policy must be legacy or validated')
    start = perf_counter()
    deadline = start+max(0.0,time_limit)
    stats = {'version':2,'full_evaluations':0,'time_limit_seconds':time_limit,
             'fallback':False,'memory_probe_calls':0,'group_policy':group_policy,
             'legacy_count_limit_exceeded':False,'count_limit_bypassed':False}
    probe = None
    hard_boundaries = True
    try:
        _check_deadline(deadline)
        components, comp_of = _components(model,deadline)
        span = {}
        for u in model.topo:
            span[u] = max(1,model.cycles[u])+max((span[p] for p in model.pred[u]),default=0)
        stats.update(compute_ops=len(model.ids),components=len(components),
                     work=sum(model.cycles.values()),span=max(span.values(),default=0),
                     median_op_cycles=median(model.cycles.values()) if model.ids else 0,
                     pipe_cycles=dict(Counter({p:sum(model.cycles[u] for u in model.ids if model.pipes[u]==p)
                                               for p in set(model.pipes.values())})))
        if not model.ids:
            plan = {'node_to_subgraph':{},'core_schedules':[[] for _ in range(m)]}
            logs = []
        elif len(components)>1:
            probe = MemoryProbe(model,deadline)
            plan, logs = _family_plan(model,components,op_graph.external_inputs,
                                     op_graph.external_sizes,m,probe,deadline)
            stats['strategy'] = 'independent_components_shared_input_capacity'
        else:
            # 通用粗粒度分组扩展已保持拓扑输出顺序，并将较晚汇合点留在组入口。
            original = {0:frozenset(model.ids)}
            region = Region(frozenset({0}),frozenset(model.ids),tuple(() for _ in range(m)),
                            tuple(() for _ in range(m)),(0,),1.0,0,1)
            blocks, contraction_stats = _contract_dependencies(model,m,deadline)
            selected = frozenset(blocks)
            stats['proposed_contraction_groups'] = len(blocks)
            stats['contraction'] = contraction_stats
            limit=max(256,64*m)
            stats['legacy_count_limit']=limit
            if len(blocks)>limit:
                stats['legacy_count_limit_exceeded']=True
                if group_policy=='legacy':
                    raise TimeoutError('structural complexity: too many coarse blocks')
                # 调度仍检查当前截止时间；返回指导信息前，完整粗粒度方案必须通过合法性验证。
                stats['count_limit_bypassed']=True
            old = {'node_to_subgraph':{str(u):0 for u in model.ids},
                   'core_schedules':[[0]]+[[] for _ in range(m-1)]}
            plan = _schedule_region(model,old,{},region,blocks,selected,
                                    'branch_affinity',deadline,False)
            # 这些块仅建议归属核；BSPg 可跨块边界合并，并在建议核忙碌时转移工作。它们尚不是最终 任务，逐块性能分析会产生误导。
            hard_boundaries = False
            logs = [{'kind':'strong_dependency_grouping','groups':len(blocks),
                     'hard_boundaries':False,**contraction_stats}]
            stats['strategy'] = 'acyclic_dependency_contraction_soft_owners'
        model.validate_plan(plan,m)
        group_of = {int(u):g for u,g in plan['node_to_subgraph'].items()}
        owner = {g:p for p,order in enumerate(plan['core_schedules']) for g in order}
        stats['proposed_groups'] = len(owner)
        if hard_boundaries:
            limit=max(32,16*m)
            stats['legacy_count_limit']=limit
            if len(owner)>limit:
                stats['legacy_count_limit_exceeded']=True
                if group_policy=='legacy':
                    raise TimeoutError('structural complexity: too many coarse groups across families')
                # 该完整方案已通过 validate_plan；仅凭分组数量不能断定调度不合适。
                stats['count_limit_bypassed']=True
        priority = {g:i for order in plan['core_schedules'] for i,g in enumerate(order)}
        stats.update(groups=len(owner),decisions=logs,coarse_plan=plan,
                     hard_boundaries=hard_boundaries)
    except TimeoutError as exc:
        group_of,owner,priority = {},{},{}
        reason = str(exc)
        if 'deadline reached' in reason:
            reason = 'coarse construction deadline reached'
        stats.update(fallback=True,reason=reason,groups=0)
    finally:
        stats['seconds'] = perf_counter()-start
        if probe:
            stats['memory_probe_calls'] = probe.calls
            probe.builder.close()
    return Guidance(group_of,owner,priority,stats,hard_boundaries)
