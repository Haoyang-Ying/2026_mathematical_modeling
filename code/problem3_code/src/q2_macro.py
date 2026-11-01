'场景 B 的宏区域及四类重构方法。区域为保守联合拓扑序的区间，这只是构造约束，并非额外的官方可行性条件。允许拆分分组，保持区域外的计算核分配和相对顺序；精确评价时不固定区域外的执行时间。'
from __future__ import annotations

import heapq
import math
from bisect import bisect_left
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass

from q2_model import PIPES, State, View
from q2_coarse import proxy


def placement_of(state):
    return {v: c for c, order in enumerate(state.cores) for g in order for v in state.groups[g]}


def semantic_key(state):
    '分组重命名只是形式变化，成员、顺序和核分配则影响实际方案。'
    return tuple(tuple(tuple(sorted(state.groups[g])) for g in order) for order in state.cores)


def boundary_traffic(model, placement):
    '精确计算换出前的边界 COPY 字节数，包括外部输入复制。张量连接按实际源核和目标核去重；直接算子边分别保留，与官方构造器一致。'
    cross, inputs, outputs, pairs = 0, 0, 0, 0
    for tid, tensor in model.tensors.items():
        sources = {placement[v] for v in model.producers[tid]}
        targets = {placement[v] for v in model.consumers[tid]}
        connections = sum(a != b for a in sources for b in targets)
        cross += tensor['size'] * connections
        pairs += connections
        if not sources:
            inputs += tensor['size'] * len(targets)
        if sources and (not model.consumers[tid] or tid in model.output_tensors):
            outputs += tensor['size'] * len(sources)
    for v in model.nodes:
        for u, size in model.direct_in[v]:
            if placement[u] != placement[v]:
                cross += size
                pairs += 1
    return {'cross_tensor_bytes': cross, 'cross_pairs': pairs,
            'external_read_bytes': inputs, 'output_write_bytes': outputs,
            'boundary_copy_bytes': inputs + outputs + 2*cross}


@dataclass
class Region:
    nodes: tuple
    base_order: tuple
    first: int
    last: int
    reason: str
    anchor: tuple
    source_cores: tuple


def joint_order(model, state, diagnostics):
    view = View(model, state)
    successors = {g: set(s) for g, s in view.succ.items()}
    for order in state.cores:
        for a, b in zip(order, order[1:]):
            successors[a].add(b)
    degree = {g: 0 for g in state.groups}
    for children in successors.values():
        for g in children:
            degree[g] += 1
    times = {}
    for r in diagnostics['operations']:
        if r['group'] in state.groups:
            times[r['group']] = min(times.get(r['group'], r['start']), r['start'])
    ready = [(times.get(g, 0), g) for g in state.groups if not degree[g]]
    heapq.heapify(ready)
    order = []
    while ready:
        _, g = heapq.heappop(ready)
        order.extend(sorted(state.groups[g], key=model.position.__getitem__))
        for h in sorted(successors[g]):
            degree[h] -= 1
            if not degree[h]:
                heapq.heappush(ready, (times.get(h, 0), h))
    return tuple(order) if len(order) == len(model.nodes) else None


def select_region(model, state, diagnostics, round_index, max_nodes=512, bandwidth=60):
    order = joint_order(model, state, diagnostics)
    if not order:
        return None
    position = {v: i for i, v in enumerate(order)}
    records = {(r['core'], r['op']): r for r in diagnostics['operations']}
    placements = placement_of(state)
    group_times = {g: sorted((records[placements[v], v]['start'], v) for v in vs)
                   for g, vs in state.groups.items()}
    def original(c, op):
        if op in model.ops:
            return (op,)
        r = records.get((c, op))
        if r and r['group'] in state.groups:
            times = group_times[r['group']]
            # 将 COPY 映射回邻近的原始决策，
            # 不把生成的 COPY 作为提交方案中可直接移动的算子。
            index = bisect_left(times, (r['start'], -1))
            choices = times[max(0, index-1):index+1]
            return (min(choices, key=lambda pair: (abs(pair[0]-r['start']), pair[1]))[1],)
        return ()
    anchors = []
    span = max(1, diagnostics['makespan'])
    for r in diagnostics['operations']:
        impact = max(r['pipe_head_idle'], r['ready_but_queued'], r['ddr_slowdown'],
                     r['end']-r['start']) / (1 + r['slack']/max(1, span*.05))
        related = list(original(r['core'], r['op']))
        for c, op, _ in r['blockers']:
            related.extend(original(c, op))
        if related:
            reason = 'pipe_wait' if r['pipe_head_idle'] else 'critical_tail'
            anchors.append((impact, reason, tuple(dict.fromkeys(related))))
    for r in diagnostics['spills']:
        nodes = list(original(r['core'], r['prev_use_op']))
        nodes.extend(original(r['core'], r['next_use_op']))
        nodes.extend(model.producers.get(r['logical_tid'], ()))
        if nodes:
            anchors.append((r['size']*(1+int(r['spill_out_copies_data']))/bandwidth,
                            'spill_interval', tuple(dict.fromkeys(nodes))))
    for r in diagnostics['transfers']:
        nodes = list(model.producers.get(r['tensor_id'], ()))
        nodes.extend(model.consumers.get(r['tensor_id'], ()))
        if not nodes:
            nodes = list(original(r['source_core'], r['source_copy_out_id']))
            nodes += list(original(r['target_core'], r['target_copy_in_id']))
        if nodes:
            anchors.append((r['release']-r['source_end']+2*r['size']/bandwidth,
                            'cross_release', tuple(dict.fromkeys(nodes))))
    # 轮换不同停顿原因，不只选择最大停顿的重复副本。
    anchors.sort(key=lambda x: (-x[0], x[1], x[2]))
    diverse, reasons = [], set()
    for a in anchors:
        if a[1] not in reasons:
            diverse.append(a)
            reasons.add(a[1])
    for a in anchors:
        if a not in diverse and len(diverse) < 16:
            diverse.append(a)
    if not diverse:
        return None
    impact, reason, anchor = diverse[round_index % len(diverse)]
    n = len(order)
    fraction = (.08, .18, .35)[round_index % 3]
    target = min(n, max_nodes, max(16, math.ceil(n*fraction)))
    center = position[anchor[0]]
    lo = max(0, center-target//2)
    hi = min(n, lo+target)
    lo = max(0, hi-target)
    # 在容量允许时，向实际生产者、消费者和共享输入使用者扩展。
    related = list(anchor)
    for v in anchor:
        related.extend(model.pred[v])
        related.extend(model.succ[v])
        for t in model.inputs[v]:
            related.extend(model.consumers[t][:32])
    cap = min(n, max_nodes, max(target, int(target*1.5)))
    for v in related:
        a, b = min(lo, position[v]), max(hi, position[v]+1)
        if b-a <= cap:
            lo, hi = a, b
    selected = order[lo:hi]
    return Region(selected, order, lo, hi, reason, anchor,
                  tuple(sorted({placements[v] for v in selected})))


class PipePlanner:
    '按流水线估计区域代价，跟踪张量释放及已有存储状态。已观测的前缀释放时间只作构造提示；共享 DDR、官方 先进先出 和 Belady 换出行为由构造后的整核编译和完整全局回放决定。'
    def __init__(self, model, state, region, diagnostics, hardware):
        self.model, self.hw = model, hardware
        self.placement = placement_of(state)
        self.free = [dict.fromkeys(PIPES, 0) for _ in state.cores]
        self.end = {}
        self.transfers = {}
        self.resident = [OrderedDict() for _ in state.cores]
        prefix = set(region.base_order[:region.first])
        touched_groups = {g for g, vs in state.groups.items() if set(vs) <= prefix}
        for r in diagnostics['operations']:
            if r['op'] in prefix or r['group'] in touched_groups:
                c = r['core']
                self.free[c][r['pipe']] = max(self.free[c][r['pipe']], r['end'])
                if r['op'] in prefix:
                    self.end[r['op']] = r['end']
        frontier = [max(f.values()) for f in self.free]
        # 区域入口并非空存储，使用当前最优方案在各核估计切分点的驻留状态，
        # 包括后缀仍需要的张量。
        for r in diagnostics['tensor_lifetimes']:
            t, c = r['logical_tid'], r['core']
            if t in model.tensors and r['start'] <= frontier[c] < r['end']:
                self.resident[c][t] = None
        for r in diagnostics['transfers']:
            t = r['tensor_id']
            if t in model.tensors and all(v in prefix for v in model.producers[t]):
                self.transfers[t, r['source_core'], r['target_core']] = r['target_end']

    def estimate(self, v, c, commit=False):
        model, bw = self.model, self.hw['bandwidth']
        free = [dict(row) for row in self.free]
        new_transfers = {}
        ready = max((self.end.get(u, 0) for u in model.pred[v]), default=0)
        bytes_added = 0
        def transfer(key, source, size, release):
            nonlocal bytes_added
            if key in self.transfers:
                return self.transfers[key]
            if key in new_transfers:
                return new_transfers[key]
            duration = max(1, math.ceil(size/bw))
            if source is not None:
                out_end = max(free[source]['PIPE_MTE3'], release)+duration
                free[source]['PIPE_MTE3'] = out_end
                release = out_end+self.hw['cross_core_copy_delay']
                bytes_added += size
            incoming = max(free[c]['PIPE_MTE2'], release)+duration
            free[c]['PIPE_MTE2'] = incoming
            bytes_added += size
            new_transfers[key] = incoming
            return incoming
        for t in model.inputs[v]:
            sources = {self.placement[u] for u in model.producers[t]}
            if not sources and t not in self.resident[c]:
                ready = max(ready, transfer(('input_or_reload', t, c, v), None, model.tensors[t]['size'], 0))
            for p in sources-{c}:
                release = max(self.end.get(u, 0) for u in model.producers[t]
                              if self.placement[u] == p)
                ready = max(ready, transfer((t, p, c), p, model.tensors[t]['size'], release))
        for j, (u, size) in enumerate(model.direct_in[v]):
            p = self.placement[u]
            if p != c:
                ready = max(ready, transfer(('direct', u, v, j, p, c), p, size, self.end.get(u, 0)))
        pipe = model.ops[v]['pipe']
        finish = max(ready, free[c][pipe])+model.work[v]
        free[c][pipe] = finish
        required = set(model.inputs[v]) | set(model.outputs[v])
        union = required | self.resident[c].keys()
        pressure = sum(max(0, sum(model.tensors[t]['size'] for t in union
                                 if model.memory_pos(t) == p)-cap)
                       for p, cap in self.hw['capacity'].items())
        if commit:
            self.free, self.end[v], self.placement[v] = free, finish, c
            self.transfers.update(new_transfers)
            for t in sorted(required):
                self.resident[c][t] = None
                self.resident[c].move_to_end(t)
            for p, cap in self.hw['capacity'].items():
                used = sum(model.tensors[t]['size'] for t in self.resident[c] if model.memory_pos(t) == p)
                for t in list(self.resident[c]):
                    if used <= cap:
                        break
                    if t not in required and model.memory_pos(t) == p:
                        del self.resident[c][t]
                        used -= model.tensors[t]['size']
        return finish, bytes_added, pressure


def _focus(model, region, kind):
    selected = set(region.nodes)
    if kind == 'shared_input':
        choices = [(model.tensors[t]['size'] * len(set(users) & selected), t)
                   for t, users in model.consumers.items()
                   if not model.producers[t] and len(set(users) & selected) > 1]
        if choices:
            _, t = max(choices)
            return tuple(v for v in model.consumers[t] if v in selected)[:64]
        return ()
    # 相连的依赖片段包含执行较短但产生长生命周期张量的生产者，
    # 不设置最小计算工作量变化的筛选条件。
    root = next((v for v in region.anchor if v in selected), region.nodes[0])
    chosen, queue = {root}, [root]
    for v in queue:
        neighbors = [u for u in (*model.pred[v], *model.succ[v]) if u in selected and u not in chosen]
        neighbors.sort(key=lambda u: (-sum(model.tensors[t]['size'] for t in
            set(model.inputs[u]+model.outputs[u]) & set(model.inputs[v]+model.outputs[v])), u))
        for u in neighbors:
            if len(chosen) >= min(64, max(2, len(region.nodes)//3)):
                break
            chosen.add(u)
            queue.append(u)
    return tuple(v for v in region.nodes if v in chosen)


def _fragment_target(model, placement, focus, cores, hardware):
    chosen = set(focus)
    base_load, added = [Counter() for _ in range(cores)], Counter()
    for v in model.nodes:
        if v in chosen:
            added[model.ops[v]['pipe']] += model.work[v]
        else:
            base_load[placement[v]][model.ops[v]['pipe']] += model.work[v]
    options = []
    for c in range(cores):
        altered = dict(placement)
        altered.update((v, c) for v in chosen)
        traffic = boundary_traffic(model, altered)
        work = max((base_load[p][pipe]+(added[pipe] if p == c else 0)
                    for p in range(cores) for pipe in PIPES), default=0)
        options.append((work + traffic['boundary_copy_bytes']/hardware['bandwidth'],
                        traffic['boundary_copy_bytes'], c))
    return min(options)[-1]


def reconstruct(model, state, region, diagnostics, hardware, kind, variant=0):
    planner = PipePlanner(model, state, region, diagnostics, hardware)
    selected = set(region.nodes)
    degree = {v: sum(u in selected for u in model.pred[v]) for v in region.nodes}
    ready = [(-model.rank[v], v) for v in region.nodes if not degree[v]]
    heapq.heapify(ready)
    placement = placement_of(state)
    locks = dict(placement) if kind != 'fork_join' else {}
    focus = _focus(model, region, kind) if kind in ('dependency_fragment', 'shared_input') else ()
    if focus:
        target = _fragment_target(model, placement, focus, len(state.cores), hardware)
        locks.update((v, target) for v in focus)
    sequence = []
    remaining = Counter(t for v in region.base_order[region.first:] for t in model.inputs[v])
    while ready:
        window = [heapq.heappop(ready) for _ in range(min(32, len(ready)))]
        options = []
        for _, v in window:
            targets = (locks[v],) if v in locks else range(len(state.cores))
            for c in targets:
                finish, traffic, pressure = planner.estimate(v, c)
                reused = sum(model.tensors[t]['size'] for t in model.inputs[v] if t in planner.resident[c])
                released = sum(model.tensors[t]['size'] for t in model.inputs[v] if remaining[t] == 1)
                memory_gain = (reused + released)/hardware['bandwidth']
                # 该估计只决定候选选择，不能直接接受方案。
                key = (finish + .3*traffic/hardware['bandwidth'] + .5*pressure/hardware['bandwidth']
                       - (.4 if kind == 'shared_input' else .1)*memory_gain,
                       -model.rank[v], int(c != placement[v]), v, c)
                options.append((key, v, c))
        _, v, c = min(options)
        for item in window:
            if item[1] != v:
                heapq.heappush(ready, item)
        planner.estimate(v, c, commit=True)
        placement[v] = c
        sequence.append(v)
        for t in model.inputs[v]:
            remaining[t] -= 1
        for w in model.succ[v]:
            if w in selected:
                degree[w] -= 1
                if not degree[w]:
                    heapq.heappush(ready, (-model.rank[w], w))
    if len(sequence) != len(selected):
        raise ValueError('regional reconstruction did not cover every selected op')
    grain = (1, 4, 16)[variant % 3]
    return rebuild_state(model, state, region, sequence, placement, grain, hardware)


def rebuild_state(model, state, region, sequence, placement, grain, hardware):
    old_view = View(model, state)
    groups, orders = {}, [[] for _ in state.cores]
    next_id = max(state.groups, default=-1)+1
    def put(nodes, c, preferred=None):
        nonlocal next_id
        if not nodes:
            return
        g = preferred
        if g is None or g in groups:
            g = next_id
            next_id += 1
        groups[g] = tuple(nodes)
        orders[c].append(g)
    def outside(nodes):
        run, last = [], None
        for v in nodes:
            g = old_view.mapping[v]
            if run and g != last:
                put(run, old_view.core[last], last)
                run = []
            run.append(v)
            last = g
        if run:
            put(run, old_view.core[last], last)
    outside(region.base_order[:region.first])
    run, touched, last = [], set(), None
    for v in sequence:
        c = placement[v]
        union = touched | set(model.inputs[v]+model.outputs[v])
        pressure = any(sum(model.tensors[t]['size'] for t in union if model.memory_pos(t) == p)
                       > .5*cap for p, cap in hardware['capacity'].items())
        if run and (c != last or len(run) >= grain or pressure):
            put(run, last)
            run, touched = [], set()
        run.append(v)
        touched.update(model.inputs[v]+model.outputs[v])
        last = c
    if run:
        put(run, last)
    outside(region.base_order[region.last:])
    return State(groups, tuple(tuple(order) for order in orders))


def macro_candidates(model, state, diagnostics, hardware, round_index, max_nodes=512):
    region = select_region(model, state, diagnostics, round_index, max_nodes, hardware['bandwidth'])
    if region is None:
        return [], {'skipped': 'no conservative joint topological region'}
    info = {'nodes': len(region.nodes), 'reason': region.reason, 'anchor': list(region.anchor),
            'source_cores': list(region.source_cores), 'first': region.first, 'last': region.last}
    base_key = semantic_key(state)
    seen, result = {base_key}, []
    before = placement_of(state)
    families = ['order_regroup', 'dependency_fragment', 'shared_input', 'fork_join']
    families = families[round_index % 4:] + families[:round_index % 4]
    for kind in families:
        candidate = reconstruct(model, state, region, diagnostics, hardware, kind, round_index)
        key = semantic_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        after = placement_of(candidate)
        affected = [c for c in range(len(state.cores))
                    if base_key[c] != key[c]]
        result.append((candidate, {'kind': kind, 'region': info,
            'regional_nodes': list(region.nodes), 'affected_cores': affected,
            'moved_nodes': sum(before[v] != after[v] for v in region.nodes),
            'traffic': boundary_traffic(model, after), 'proxy': proxy(model, candidate, hardware)}))
    return result, info
