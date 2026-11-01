"""位于 BSPg 初始构造和局部改进之间的场景 B 粗粒度调度。

粗粒度组仅表示优先级桶，不清空内存、不产生同核搬运，也不设置任务完成屏障。
近似代价只用于筛选候选；每个被接受的候选都必须经过完整场景 B 评价器评分。"""
from __future__ import annotations

import heapq
from collections import Counter, OrderedDict, defaultdict

from q2_model import PIPES, State


def components(model):
    seen, answer = set(), []
    for root in model.topo:
        if root in seen:
            continue
        seen.add(root)
        pending = [root]
        for v in pending:
            for w in (*model.pred[v], *model.succ[v]):
                if w not in seen:
                    seen.add(w)
                    pending.append(w)
        answer.append(tuple(sorted(pending, key=model.position.__getitem__)))
    return answer


def memory_order(model, members, capacity):
    """优先消费仍存活中间结果的拓扑遍历。
    
    这里的生命周期仅针对当前连通分量，不是物理缓存模拟。
    限制就绪窗口大小以控制宽有向无环图的处理成本，不删除任何依赖边。"""
    eligible = set(members)
    degree = {v: sum(p in eligible for p in model.pred[v]) for v in members}
    remaining = Counter(t for v in members for t in model.inputs[v])
    ready = [v for v in members if degree[v] == 0]
    heapq.heapify(ready)
    live, used, order = set(), Counter(), []
    touched = {v: set(model.inputs[v]) | set(model.outputs[v]) for v in members}
    while ready:
        window = [heapq.heappop(ready) for _ in range(min(32, len(ready)))]

        def key(v):
            added, freed = Counter(), Counter()
            for t in touched[v]:
                pos, size = model.memory_pos(t), model.tensors[t]['size']
                if t not in live:
                    added[pos] += size
                if remaining[t] - int(t in model.inputs[v]) == 0:
                    freed[pos] += size
            overflow = sum(max(0, used[p] + added[p] - capacity[p]) for p in capacity)
            after = sum(used[p] + added[p] - freed[p] for p in capacity)
            return overflow, after, -sum(freed.values()), -model.rank[v], v

        v = min(window, key=key)
        for w in window:
            if w != v:
                heapq.heappush(ready, w)
        for t in touched[v] - live:
            used[model.memory_pos(t)] += model.tensors[t]['size']
        live.update(touched[v])
        for t in model.inputs[v]:
            remaining[t] -= 1
        for t in touched[v]:
            if remaining[t] == 0:
                live.discard(t)
                used[model.memory_pos(t)] -= model.tensors[t]['size']
        order.append(v)
        for w in model.succ[v]:
            if w in eligible:
                degree[w] -= 1
                if degree[w] == 0:
                    heapq.heappush(ready, w)
    if len(order) != len(members):
        raise ValueError('coarse traversal did not cover its component')
    return tuple(order)


def _external(model, members):
    return {t for v in members for t in model.inputs[v] if not model.producers[t]}


def reuse_placement(model, units, cores, hardware):
    """按流水线负载与去重后的 DDR 输入代价分配完整连通分量。
    
    此方法不切断计算依赖；在连通图上可能限制并行度，因此仅作为附加候选。"""
    loads = [Counter() for _ in range(cores)]
    inputs = [set() for _ in range(cores)]
    work = [Counter() for _ in units]
    external = [_external(model, seq) for seq in units]
    for i, seq in enumerate(units):
        for v in seq:
            work[i][model.ops[v]['pipe']] += model.work[v]
    placement = {}
    for i in sorted(range(len(units)), key=lambda i: (-max(work[i].values(), default=0), i)):
        def key(p):
            added = sum(model.tensors[t]['size'] for t in external[i] - inputs[p])
            times = {q: loads[p][q] + work[i][q] for q in PIPES}
            times['PIPE_MTE2'] += added / hardware['bandwidth']
            return max(times.values()), added, sum(loads[p].values()), p
        p = min(range(cores), key=key)
        added = sum(model.tensors[t]['size'] for t in external[i] - inputs[p])
        loads[p].update(work[i])
        loads[p]['PIPE_MTE2'] += added / hardware['bandwidth']
        inputs[p].update(external[i])
        placement.update((v, p) for v in units[i])
    return placement


def _chunks(model, seq, grain, capacity):
    parts, part, touched = [], [], set()
    for v in seq:
        now = set(model.inputs[v]) | set(model.outputs[v])
        union = touched | now
        # 工作集压力只控制优先级桶，不会清空物理内存。
        pressure = any(sum(model.tensors[t]['size'] for t in union
                           if model.memory_pos(t) == p) > .5 * capacity[p] for p in capacity)
        if part and (len(part) >= grain or pressure):
            parts.append(tuple(part))
            part, touched = [], set()
        part.append(v)
        touched.update(now)
    if part:
        parts.append(tuple(part))
    return parts


def tiled_state(model, units, placement, cores, hardware, width, grain):
    """对独立连通分量分批，并交错安排其拓扑片段。
    
    每个完整分量采用一个全局拓扑序，将其切为连续区间后再交错独立分量，
    可保持组商图及核内顺序图无环。物理核心间允许双向通信，不要求核心通信图无环。"""
    external = [_external(model, seq) for seq in units]
    pending = set(range(len(units)))
    groups, schedules = {}, [[] for _ in range(cores)]
    while pending:
        first = min(pending)
        pending.remove(first)
        batch, shared = [first], set(external[first])
        # 仅组合参与核心集合相同的连通分量。
        # 这只是局部性偏好，不是方案可行性的必要条件。
        owners = {placement[v] for v in units[first]}
        choices = [i for i in sorted(pending)[:128]
                   if {placement[v] for v in units[i]} == owners]
        while choices and len(batch) < width:
            i = min(choices, key=lambda i: (
                -sum(model.tensors[t]['size'] for t in shared & external[i]), i))
            pending.remove(i)
            choices.remove(i)
            batch.append(i)
            shared.update(external[i])
        parts = {i: _chunks(model, units[i], grain, hardware['capacity']) for i in batch}
        for phase in range(max(map(len, parts.values()), default=0)):
            by_core = defaultdict(list)
            for i in batch:
                if phase < len(parts[i]):
                    # 每次核心切换都切开片段；
                    # 将同核但不连续的片段合并，可能引入商图中的环。
                    run, previous = [], None
                    for v in parts[i][phase]:
                        p = placement[v]
                        if run and p != previous:
                            by_core[previous].append(tuple(run))
                            run = []
                        run.append(v)
                        previous = p
                    if run:
                        by_core[previous].append(tuple(run))
            for p, runs in sorted(by_core.items()):
                # 不同独立分量中整分量的单核连续段可以合并；
                # 涉及多个核心的连续段仍分别保留。
                if len(owners) == 1:
                    runs = [tuple(v for run in runs for v in run)]
                for run in runs:
                    g = len(groups)
                    groups[g] = run
                    schedules[p].append(g)
    return State(groups, tuple(tuple(order) for order in schedules))


def proxy(model, state, hardware):
    """整核最近最少使用缓存策略的搬运量近似，生命周期跨越所有组。
    
    该近似忽略官方组内顺序、流水线重叠和 DDR 争用，
    只用来排序，不是精确换出量，也不是下界。"""
    bw, capacity = hardware['bandwidth'], hardware['capacity']
    total_read, total_write, reload, overloaded = 0, 0, 0, 0
    max_work, peaks = 0, []
    placement = {v: p for p, order in enumerate(state.cores)
                 for g in order for v in state.groups[g]}
    cross = 0
    for t in model.tensors:
        sources = {placement[v] for v in model.producers[t]}
        targets = {placement[v] for v in model.consumers[t]}
        cross += model.tensors[t]['size'] * sum(a != b for a in sources for b in targets)
    for v in model.nodes:
        cross += sum(size for u, size in model.direct_in[v] if placement[u] != placement[v])
    for order in state.cores:
        seq = [v for g in order for v in state.groups[g]]
        remaining = Counter(t for v in seq for t in model.inputs[v])
        cache = {p: OrderedDict() for p in capacity}
        used, peak, work = Counter(), Counter(), Counter()
        seen, dirty = set(), set()
        for v in seq:
            work[model.ops[v]['pipe']] += model.work[v]
            required = set(model.inputs[v]) | set(model.outputs[v])
            for p in capacity:
                amount = sum(model.tensors[t]['size'] for t in required if model.memory_pos(t) == p)
                overloaded = max(overloaded, amount - capacity[p])
            for t in sorted(required):
                p, size = model.memory_pos(t), model.tensors[t]['size']
                if t in cache[p]:
                    cache[p].move_to_end(t)
                else:
                    while used[p] + size > capacity[p]:
                        victim = next((x for x in cache[p] if x not in required), None)
                        if victim is None:
                            break
                        cache[p].pop(victim)
                        used[p] -= model.tensors[victim]['size']
                        if victim in dirty and remaining[victim]:
                            total_write += model.tensors[victim]['size']
                        dirty.discard(victim)
                    if t in model.inputs[v]:
                        total_read += size
                        if t in seen:
                            reload += size
                    cache[p][t] = None
                    used[p] += size
                    seen.add(t)
                if t in model.outputs[v]:
                    dirty.add(t)
                peak[p] = max(peak[p], used[p])
            for t in model.inputs[v]:
                remaining[t] -= 1
            for t in required:
                if remaining[t] == 0:
                    p = model.memory_pos(t)
                    if t in cache[p]:
                        cache[p].pop(t)
                        used[p] -= model.tensors[t]['size']
                    dirty.discard(t)
        max_work = max(max_work, max(work.values(), default=0))
        peaks.append(dict(peak))
    traffic = total_read + total_write + 2 * cross
    return {'proxy_cost': max_work + traffic / bw,
            'proxy_reload_bytes': reload, 'proxy_write_bytes': total_write,
            'cross_core_tensor_bytes': cross, 'proxy_peak_by_core': peaks,
            'single_op_over_capacity_bytes': max(0, overloaded)}


def coarse_candidates(model, states, cores, hardware, limit=3):
    """返回规模受限且具有多样性的候选列表；现有最优方案由外部保留。"""
    if limit <= 0 or not model.nodes:
        return [], {'proposals': 0, 'shortlisted': 0}
    units = [memory_order(model, c, hardware['capacity']) for c in components(model)]
    placements, known = [], set()
    for label, state in states:
        placement = {v: p for p, order in enumerate(state.cores)
                     for g in order for v in state.groups[g]}
        key = tuple(placement[v] for v in model.nodes)
        if key not in known:
            known.add(key)
            placements.append((label, placement))
    extra = reuse_placement(model, units, cores, hardware)
    if tuple(extra[v] for v in model.nodes) not in known:
        placements.append(('reuse_placement', extra))
    proposals, fingerprints = [], {s.fingerprint() for _, s in states}
    # 小优先级桶有利于探索内存访问顺序；按输入规模设置桶大小下限以控制大图的组数。
    # 该下限不约束 BSPg 自身生成的组大小。
    grain = max(4, (len(model.nodes) + 1024 * cores - 1) // (1024 * cores))
    for label, placement in placements:
        for width in (1, 4, 16):
            state = tiled_state(model, units, placement, cores, hardware, width, grain)
            fingerprint = state.fingerprint()
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            info = {'source': label, 'batch_width': width, 'fragment_nodes': grain,
                    'components': len(units), 'groups': len(state.groups),
                    'fingerprint': fingerprint, **proxy(model, state, hardware)}
            proposals.append((state, info))
    proposals.sort(key=lambda item: (item[1]['proxy_cost'], item[1]['proxy_reload_bytes'],
                                      item[1]['groups'], item[1]['fingerprint']))
    # 条件允许时为每种分配方式保留一个名额，其余名额按近似代价排序填充。
    selected, sources = [], set()
    for item in proposals:
        if item[1]['source'] not in sources and len(selected) < limit:
            selected.append(item)
            sources.add(item[1]['source'])
    for item in proposals:
        if len(selected) >= limit:
            break
        if all(item[1]['fingerprint'] != old[1]['fingerprint'] for old in selected):
            selected.append(item)
    return selected, {'proposals': len(proposals), 'shortlisted': len(selected),
                      'component_count': len(units), 'proxy_is_exact': False}
