"""场景 B 的 HC：结合回放热点、张量片段及公平分配的精确评价。

近似代价仅用于候选排序；静态过滤只检查必要条件。
先进先出、内存和跨核搬运的可行性及最终分数仍由完整评价器判定。"""
from __future__ import annotations

import hashlib
import heapq
import pickle
from collections import Counter, deque
from time import perf_counter
from q2_budget import timed_call

from q2_model import PIPES, State, View, score_key
from q2_macro import placement_of, semantic_key

KINDS = ('order', 'migrate', 'swap')


def semantic_digest(state):
    return hashlib.sha256(pickle.dumps(semantic_key(state), protocol=5)).digest()


def static_feasible(model, state):
    """不要求物理核心通信图无环，也不依据近似内存量拒绝候选。"""
    nodes = [v for values in state.groups.values() for v in values]
    scheduled = [g for order in state.cores for g in order]
    if (len(nodes) != len(model.nodes) or set(nodes) != set(model.nodes)
            or len(scheduled) != len(state.groups) or set(scheduled) != set(state.groups)
            or any(not vs for vs in state.groups.values())):
        return False
    view = View(model, state)
    positions = {g:i for order in state.cores for i,g in enumerate(order)}
    degree = {g:len(view.pred[g]) for g in state.groups}
    for g, succs in view.succ.items():
        for h in succs:
            if view.core[g] == view.core[h] and positions[g] >= positions[h]:
                return False
    ready = deque(g for g,d in degree.items() if not d)
    count = 0
    while ready:
        g = ready.popleft(); count += 1
        for h in view.succ[g]:
            degree[h] -= 1
            if not degree[h]:
                ready.append(h)
    return count == len(state.groups)


def outside_signature(state, selected):
    """保留区域外的核心分配、优先级桶及其相对顺序。"""
    return tuple(tuple(part for g in order
                       if (part := tuple(sorted(v for v in state.groups[g] if v not in selected))))
                 for order in state.cores)


class Neighborhood:
    def __init__(self, model, state, result, rng, hardware, selected=None):
        self.model, self.state, self.result, self.hw = model, state, result, hardware
        self.view, self.placement = View(model, state), placement_of(state)
        self.selected = set(model.nodes if selected is None else selected)
        guide = result.get('hc_guidance', {})
        self.guide = guide if guide.get('source_fingerprint') == state.fingerprint() else {}
        self.times = self.guide.get('compute_times', {})
        self.priority = {}
        self.core_nodes = [[] for _ in state.cores]
        self.loads = [Counter() for _ in state.cores]
        self.direct_by_node = {v:[] for v in model.nodes}
        for v in model.nodes:
            c = self.placement[v]
            self.core_nodes[c].append(v)
            self.loads[c][model.ops[v]['pipe']] += model.work[v]
            for j,(u,size) in enumerate(model.direct_in[v]):
                edge = (u,v,j,size)
                self.direct_by_node[u].append(edge)
                self.direct_by_node[v].append(edge)
        for r in self.guide.get('pipes', ()):
            self.loads[r['core']][r['pipe']] = r['standalone_work']
        self.base_load = max((max(load.values(), default=0) for load in self.loads), default=0)
        self.core_cost = [sum(result.get('step3_by_core', {}).get(c, {}).get('pipe_op_counts', {}).values())
                          or len(self.core_nodes[c]) for c in range(len(state.cores))]
        def add(v, importance):
            if v in self.selected:
                self.priority[v] = max(self.priority.get(v, 0), importance)
        for r in self.guide.get('operations', ()):
            v = r['op']
            if v not in model.ops or self.placement[v] != r['core']:
                members = state.groups.get(r['group'], ())
                v = min(members, key=lambda u: abs(self.times.get(u,(0,0))[0]-r['start']), default=None)
            add(v, r['priority'])
            blocker = r.get('blocker')
            if blocker and blocker[1] in model.ops:
                add(blocker[1], r['priority'])
        for r in self.guide.get('spills', ()):
            importance = r['size']*(1+int(r['spill_out_copies_data']))/hardware['bandwidth']
            for v in (r.get('prev_use_op'), r.get('next_use_op'), *model.producers.get(r['logical_tid'], ())):
                add(v, importance)
        for r in self.guide.get('transfers', ()):
            importance = hardware['cross_core_copy_delay']+2*r['size']/hardware['bandwidth']
            for v in (*model.producers.get(r['tensor_id'], ()), *model.consumers.get(r['tensor_id'], ())):
                add(v, importance)
        if not self.priority:
            for v in self.selected:
                add(v, model.rank[v])
        hot = heapq.nlargest(12, self.priority, key=lambda v:(self.priority[v],-v))
        others = sorted(self.selected-set(hot))
        self.anchors = hot + rng.sample(others, min(3,len(others)))
        eligible = [g for g,vs in state.groups.items() if set(vs)<=self.selected]
        spans,ends = result.get('durations',{}),result.get('ends',{})
        # 保留少量组级移动；若将所有原有移动都替换为不超过4个节点的小片段，
        # 会不必要地丢失有价值的粗粒度优先级和分配调整。
        broad = heapq.nlargest(6,eligible,key=lambda g:(spans.get(g,0)+.2*ends.get(g,0),g))
        self.group_anchors = list(dict.fromkeys([*broad,*(self.view.mapping[v] for v in self.anchors
                                                      if self.view.mapping[v] in eligible)]))
        self.partner_pool = {}
        for c, nodes in enumerate(self.core_nodes):
            pool = []
            for pipe in PIPES:
                pool += heapq.nlargest(8, (v for v in nodes if v in self.selected and model.ops[v]['pipe']==pipe),
                                       key=lambda v:(model.work[v],-v))
            self.partner_pool[c] = pool

    def traffic_delta(self, assignment):
        """插入换入换出操作之前，精确计算边界搬运字节数的变化。"""
        model, old = self.model, self.placement
        affected = {t for v in assignment for t in model.inputs[v]+model.outputs[v]}
        def owner(v, changed):
            return assignment.get(v,old[v]) if changed else old[v]
        def contribution(t, changed):
            sources = {owner(v,changed) for v in model.producers[t]}
            targets = {owner(v,changed) for v in model.consumers[t]}
            pairs = sum(a != b for a in sources for b in targets)
            copies = 2*pairs + (len(targets) if not sources else 0)
            if sources and (not model.consumers[t] or t in model.output_tensors):
                copies += len(sources)
            return model.tensors[t]['size']*copies, pairs
        byte_delta, pair_delta = 0, 0
        for t in affected:
            before, after = contribution(t,False), contribution(t,True)
            byte_delta += after[0]-before[0]; pair_delta += after[1]-before[1]
        direct = {e for v in assignment for e in self.direct_by_node[v]}
        for u,v,_,size in direct:
            change = int(owner(u,True)!=owner(v,True))-int(old[u]!=old[v])
            byte_delta += 2*size*change; pair_delta += change
        return byte_delta, pair_delta

    def load_delta(self, assignment):
        loads = [Counter(x) for x in self.loads]
        for v,c in assignment.items():
            old = self.placement[v]; pipe = self.model.ops[v]['pipe']
            loads[old][pipe] -= self.model.work[v]
            loads[c][pipe] += self.model.work[v]
        return max((max(x.values(),default=0) for x in loads),default=0)-self.base_load

    def fragments(self, v):
        """生成连通的生产者与消费者片段、共享输入批次以及单节点候选。"""
        model, source = self.model, self.placement[v]
        chosen = {v}
        for _ in range(3):
            choices = {u for w in chosen for u in (*model.pred[w],*model.succ[w])
                       if u not in chosen and u in self.selected and self.placement[u]==source}
            if not choices:
                break
            touched = {t for w in chosen for t in model.inputs[w]+model.outputs[w]}
            u = max(choices, key=lambda x:(sum(model.tensors[t]['size'] for t in
                set(model.inputs[x]+model.outputs[x]) & touched), model.work[x],-x))
            chosen.add(u)
        linked = tuple(sorted(chosen,key=model.position.__getitem__))
        shared = ()
        tids = [t for t in model.inputs[v] if not model.producers[t] and len(model.consumers[t])>1]
        if tids:
            t = max(tids,key=lambda t:model.tensors[t]['size'])
            users = [u for u in model.consumers[t] if u in self.selected and self.placement[u]==source and u!=v]
            users.sort(key=lambda u:(abs(self.times.get(u,(model.position[u],0))[0]-
                                        self.times.get(v,(model.position[v],0))[0]),u))
            shared = tuple(sorted((v,*users[:3]),key=model.position.__getitem__))
        return list(dict.fromkeys(x for x in (linked,shared,(v,)) if x))

    def splice(self, specs):
        """提取原始计算片段并插入单节点优先级桶。
        
        原组必须在被移除节点处切开：若从 a→b→c 中抽走 b 后仍将 a、c 留在同一桶，
        会在商图中产生环。"""
        model, state = self.model, self.state
        removed = {v for nodes,_,_ in specs for v in nodes}
        if sum(len(nodes) for nodes,_,_ in specs) != len(removed):
            return None
        groups, orders, next_id = {}, [[] for _ in state.cores], max(state.groups,default=-1)+1
        for c, order in enumerate(state.cores):
            for g in order:
                values = state.groups[g]
                if not removed.intersection(values):
                    groups[g] = values; orders[c].append(g)
                    continue
                run, used_old = [], False
                for v in (*sorted(values,key=model.position.__getitem__),None):
                    if v is None or v in removed:
                        if run:
                            h = g if not used_old else next_id
                            if used_old: next_id += 1
                            groups[h] = tuple(run); orders[c].append(h)
                            run, used_old = [], True
                    else:
                        run.append(v)
        bundles = []
        for nodes,c,mode in specs:
            bundle = []
            for v in sorted(nodes,key=model.position.__getitem__):
                groups[next_id] = (v,); bundle.append(next_id); next_id += 1
            bundles.append((tuple(bundle),c,mode,nodes))
        tmp = [list(x) for x in orders]
        for bundle,c,_,_ in bundles: tmp[c].extend(bundle)
        view = View(model,State(groups,tuple(tuple(x) for x in tmp)))
        for bundle,c,mode,nodes in bundles:
            order = orders[c]; positions = {g:i for i,g in enumerate(order)}
            lo = max((positions[p]+1 for g in bundle for p in view.pred[g] if p in positions),default=0)
            hi = min((positions[s] for g in bundle for s in view.succ[g] if s in positions),default=len(order))
            if lo > hi:
                return None
            center = sum(self.times.get(v,(model.position[v],0))[0] for v in nodes)/len(nodes)
            natural = next((i for i,g in enumerate(order)
                if sum(self.times.get(v,(model.position[v],0))[0] for v in groups[g])/len(groups[g])>center),len(order))
            pos = lo if mode=='early' else hi if mode=='late' else max(lo,min(hi,natural))
            orders[c] = order[:pos]+list(bundle)+order[pos:]
        return State(groups,tuple(tuple(x) for x in orders))

    def proposal(self, specs, kind, variant):
        candidate = self.splice(specs)
        if candidate is None:
            return None
        assignment = {v:c for nodes,c,_ in specs for v in nodes}
        return candidate,self.move_record(assignment,kind,variant)

    def move_record(self, assignment, kind, variant, *, basic=False):
        byte_delta, pair_delta = self.traffic_delta(assignment)
        load_delta = self.load_delta(assignment)
        affected = set(assignment.values()) | {self.placement[v] for v in assignment}
        if any(c != self.placement[v] for v,c in assignment.items()):
            for v in assignment:
                for t in self.model.inputs[v]+self.model.outputs[v]:
                    affected.update(self.placement[u] for u in (*self.model.producers[t],*self.model.consumers[t]))
        cost = sum(self.core_cost[c] for c in affected)/max(1,sum(self.core_cost))
        potential = (max((self.priority.get(v,0) for v in assignment),default=1)
                     + max(0,-load_delta) + max(0,-byte_delta)/self.hw['bandwidth']
                     + max(0,-pair_delta)*self.hw['cross_core_copy_delay'])
        move = {'kind':kind, 'variant':variant, 'nodes':sorted(assignment),
                'basic_neighborhood':basic,
                'target_cores':sorted(set(assignment.values())), 'affected_cores_estimate':sorted(affected),
                'boundary_copy_delta':byte_delta, 'cross_connection_delta':pair_delta,
                'pipe_load_delta_proxy':load_delta, 'compile_fraction_proxy':cost,
                'priority_proxy':potential/(.2+cost)}
        return move

    def targeted_order(self):
        for v in self.anchors:
            source = self.placement[v]
            parts = [(v,)] + [f for f in self.fragments(v) if len(f)>1]
            for nodes in parts:
                for mode in ('early','late','natural'):
                    p = self.proposal([(nodes,source,mode)],'order','consume_or_delay' if len(nodes)==1 else 'tensor_batch')
                    if p: yield p
            # 仅合并较小且完全位于当前区域内的相邻组，不预先假定搬运量一定减少。
            g = self.view.mapping[v]; order = self.state.cores[source]; i = order.index(g)
            if i+1 < len(order):
                h = order[i+1]; union = (*self.state.groups[g],*self.state.groups[h])
                if len(union)<=16 and set(union)<=self.selected:
                    groups = dict(self.state.groups); groups[g] = tuple(union); del groups[h]
                    cores = list(self.state.cores); cores[source] = order[:i+1]+order[i+2:]
                    yield State(groups,tuple(cores)), {'kind':'order','variant':'merge_priority',
                        'nodes':list(union),'affected_cores_estimate':[source],
                        'priority_proxy':self.priority.get(v,1),'boundary_copy_delta':0}

    def targeted_migrate(self):
        for v in self.anchors:
            for nodes in self.fragments(v):
                source = self.placement[v]
                targets = [c for c in range(len(self.state.cores)) if c!=source]
                targets.sort(key=lambda c:(self.load_delta({u:c for u in nodes})+
                    self.traffic_delta({u:c for u in nodes})[0]/self.hw['bandwidth'],c))
                for c in targets[:2]:
                    for mode in ('natural','early','late'):
                        p = self.proposal([(nodes,c,mode)],'migrate','tensor_fragment')
                        if p: yield p

    def targeted_swap(self):
        for v in self.anchors:
            a = self.placement[v]
            fragment = self.fragments(v)[0]
            partners = [u for c,pool in self.partner_pool.items() if c!=a for u in pool]
            def key(u):
                b = self.placement[u]; assignment = {w:b for w in fragment}; assignment[u] = a
                return (self.load_delta(assignment)+self.traffic_delta(assignment)[0]/self.hw['bandwidth'],u)
            for u in sorted(partners,key=key)[:3]:
                b = self.placement[u]
                for mode in ('natural','early','late'):
                    p = self.proposal([(fragment,b,mode),((u,),a,mode)],'swap','pipe_complement')
                    if p: yield p

    def basic_order(self):
        for g in self.group_anchors:
            c = self.view.core[g]; old = self.state.cores[c]; i = old.index(g)
            rest = old[:i]+old[i+1:]; lo,hi = self.view.insertion_bounds(g,rest)
            for j in dict.fromkeys((i-1,i+1,i-3,i+3,lo,hi)):
                if not lo<=j<=hi or j==i: continue
                cores = list(self.state.cores); cores[c] = rest[:j]+(g,)+rest[j:]
                move = self.move_record({v:c for v in self.state.groups[g]},'order','group_priority',basic=True)
                yield State(self.state.groups,tuple(cores)),move

    def basic_migrate(self):
        for g in self.group_anchors:
            nodes = self.state.groups[g]
            if len(nodes)>64: continue
            a = self.view.core[g]
            targets = [c for c in range(len(self.state.cores)) if c!=a]
            targets.sort(key=lambda c:(self.load_delta({v:c for v in nodes})+
                self.traffic_delta({v:c for v in nodes})[0]/self.hw['bandwidth'],c))
            for b in targets[:2]:
                target = self.state.cores[b]; lo,hi = self.view.insertion_bounds(g,target)
                if lo>hi: continue
                center = sum(self.model.position[v] for v in nodes)/len(nodes)
                natural = next((j for j,h in enumerate(target) if sum(self.model.position[v]
                    for v in self.state.groups[h])/len(self.state.groups[h])>center),len(target))
                natural = max(lo,min(hi,natural))
                for j in dict.fromkeys((natural,max(lo,natural-1),min(hi,natural+1),lo,hi)):
                    cores = list(self.state.cores); cores[a] = tuple(h for h in cores[a] if h!=g)
                    cores[b] = target[:j]+(g,)+target[j:]
                    move = self.move_record({v:b for v in nodes},'migrate','group_migrate',basic=True)
                    yield State(self.state.groups,tuple(cores)),move

    def basic_swap(self):
        eligible = [g for g,vs in self.state.groups.items() if len(vs)<=64 and set(vs)<=self.selected]
        for g in self.group_anchors:
            if len(self.state.groups[g])>64: continue
            a = self.view.core[g]
            partners = [h for h in eligible if self.view.core[h]!=a]
            def assignment(h):
                return {**{v:self.view.core[h] for v in self.state.groups[g]},
                        **{v:a for v in self.state.groups[h]}}
            partners.sort(key=lambda h:(self.load_delta(assignment(h))+
                self.traffic_delta(assignment(h))[0]/self.hw['bandwidth'],h))
            for h in partners[:3]:
                b = self.view.core[h]; cores = [list(x) for x in self.state.cores]
                i,j = cores[a].index(g),cores[b].index(h); cores[a][i],cores[b][j] = h,g
                move = self.move_record(assignment(h),'swap','group_swap',basic=True)
                yield State(self.state.groups,tuple(tuple(x) for x in cores)),move

    def blended(self,kind,*,basic_first=False):
        new,base = iter(getattr(self,'targeted_'+kind)()),iter(getattr(self,'basic_'+kind)())
        active = {'new':new,'base':base}
        order = ('base','new','new') if basic_first else ('new','new','base')
        while active:
            for name in order:
                if name not in active: continue
                try: yield next(active[name])
                except StopIteration: del active[name]

    def order(self):
        return self.blended('order')

    def migrate(self):
        return self.blended('migrate')

    def swap(self):
        return self.blended('swap')


def mixed_neighborhoods(model,state,result,rng,hardware):
    """为既有的最优前缀多步搜索提供经过过滤的轮转候选。"""
    space = Neighborhood(model,state,result,rng,hardware)
    kinds = list(KINDS); rng.shuffle(kinds)
    generators = [iter(space.blended(k,basic_first=rng.randrange(3)==0)) for k in kinds]
    seen, scans = {semantic_digest(state)}, 0
    while generators and scans < 384:
        live = []
        for gen in generators:
            for _ in range(32):
                try: candidate,move = next(gen)
                except StopIteration: break
                scans += 1
                key = semantic_digest(candidate)
                if key in seen: continue
                seen.add(key)
                if not static_feasible(model,candidate): continue
                live.append(gen); yield candidate,move; break
            else:
                live.append(gen)
        generators = live


def run_hc(ctx,state,result,budgets,*,total_budget=None,selected=None,protect_assignment=False,
           label='hc',family_order=KINDS):
    """供 MGR 前的 HC 以及 MGR 两个修复阶段共用的引擎。"""
    kinds = [k for k in family_order if budgets.get(k,0)>0 and (not protect_assignment or k=='order')]
    total_budget = sum(budgets.get(k,0) for k in kinds) if total_budget is None else total_budget
    selected = set(ctx.model.nodes if selected is None else selected)
    exterior = outside_signature(state,selected)
    original_placement = placement_of(state)
    stats = {k:dict(attempted=0,valid=0,accepted=0,filtered=0,duplicates=0,seconds=0.0,
                   compile_cost_proxy=0.0,gain=0,timeouts=0)
             for k in kinds}
    history = getattr(ctx,'hc_seen',{})
    history_key = (semantic_digest(state),tuple(sorted(selected)),protect_assignment,label)
    seen = history.setdefault(history_key,{semantic_digest(state)})
    attempts_history = getattr(ctx,'hc_attempts',{})
    prior = attempts_history.setdefault(history_key,{k:0 for k in kinds})
    while len(attempts_history)>24:
        attempts_history.pop(next(iter(attempts_history)))
    while len(history)>24:
        history.pop(next(iter(history)))
    schedule, count, accepted, status = [], 0, 0, 'completed'
    global_deadline = ctx.deadline
    generations, pools, exhausted, proposed = {}, {}, set(), set()
    def reset():
        nonlocal generations,pools,exhausted,proposed
        space = timed_call(global_deadline,Neighborhood,ctx.model,state,result,ctx.rng,ctx.hardware,selected)
        generations = {k:iter(getattr(space,k)()) for k in kinds}
        pools = {k:[] for k in kinds}; exhausted = set(); proposed = set()
    try:
        if kinds and total_budget>0 and selected:
            reset()
        while count<total_budget and kinds and selected:
            if global_deadline is not None and perf_counter()>=global_deadline:
                status = 'time_budget_exhausted'; break
            available = [k for k in kinds if stats[k]['attempted']<budgets[k]
                         and (k not in exhausted or pools[k])]
            if not available: break
            untried = sorted((k for k in available if stats[k]['attempted']==0),
                             key=lambda k:(prior.get(k,0),kinds.index(k)))
            if untried:
                kind = untried[0]
            else:
                def utility(k):
                    s = stats[k]
                    # 仅按评价次数限额运行时保持可复现性，不让实际耗时抖动改变候选路径。
                    # 限时运行使用观测代价；
                    # 非限时运行使用确定性的核心工作量。
                    cost = s['seconds'] if global_deadline is not None else s['compile_cost_proxy']
                    mean = cost/max(1,s['attempted'])
                    return ((s['gain']+.001*max(1,result['makespan'])/(1+s['attempted']))/
                            max(.001,cost+mean), -s['attempted'], -kinds.index(k))
                kind = max(available,key=utility)
            s = stats[kind]
            for _ in range(32):
                if len(pools[kind])>=3 or kind in exhausted: break
                if global_deadline is not None and perf_counter()>=global_deadline: break
                try: candidate,move = timed_call(global_deadline,next,generations[kind])
                except StopIteration:
                    exhausted.add(kind); break
                key = semantic_digest(candidate)
                if key in seen or key in proposed:
                    s['duplicates'] += 1; continue
                proposed.add(key)
                if (outside_signature(candidate,selected)!=exterior
                        or (protect_assignment and placement_of(candidate)!=original_placement)
                        or not static_feasible(ctx.model,candidate)):
                    s['filtered'] += 1; continue
                pools[kind].append((candidate,move,key))
            if not pools[kind]:
                # 限制扫描长度，避免在首次精确评价之前，
                # 过长的无效邻域扫描耗尽其他候选族的机会。
                exhausted.add(kind); continue
            # 即使热点近似反复将小片段排在前面，
            # 仍为基准组级移动保留一定覆盖机会。
            prefer_basic = s['attempted']%3==2
            choices = [i for i,(_,m,_) in enumerate(pools[kind])
                       if bool(m.get('basic_neighborhood'))==prefer_basic]
            index = max(choices or range(len(pools[kind])),
                        key=lambda i:pools[kind][i][1].get('priority_proxy',0))
            candidate,move,key = pools[kind].pop(index)
            seen.add(key); count += 1; s['attempted'] += 1; schedule.append(kind)
            prior[kind] = prior.get(kind,0)+1
            s['compile_cost_proxy'] += .2+move.get('compile_fraction_proxy',1.0)
            # 预计能够完成一次完整评价时，为尚未尝试的候选族保留首次探测时间。
            # 仅完成的探测才算作已覆盖。
            if global_deadline is not None and len(untried)>1:
                now = perf_counter()
                share = max(0,global_deadline-now)/len(untried)
                estimate = getattr(ctx,'evaluation_estimate',lambda:0.0)()
                if share >= 1.25*estimate:
                    ctx.deadline = now+share
            start = perf_counter(); trial = None
            try:
                trial = ctx.evaluate(candidate,label,move,guide_against=result)
            except TimeoutError:
                s['timeouts'] += 1
                seen.discard(key)  # 未完成的评价允许在后续重新尝试。
            finally:
                ctx.deadline = global_deadline
                s['seconds'] += perf_counter()-start
            if trial is not None:
                s['valid'] += 1
                if score_key(trial)<score_key(result):
                    s['gain'] += max(0,result['makespan']-trial['makespan'])
                    state,result = candidate,trial
                    accepted += 1; s['accepted'] += 1
                    reset()
    except TimeoutError:
        status = 'time_budget_exhausted'
    finally:
        ctx.deadline = global_deadline
    return state,result,{'attempted':count,'accepted_moves':accepted,'families':stats,
        'neighborhoods_exhausted':status=='completed' and count<total_budget,
        'evaluation_order':schedule,'status':status,'assignment_protected':protect_assignment,
        'first_probe_families':[k for k in kinds if stats[k]['attempted']],
        'allocation_cost':'observed_seconds' if global_deadline is not None else 'compile_work_proxy',
        'diagnostic_source':result.get('hc_guidance',{}).get('source','structural_fallback')}
