'从场景 B 的全局执行结果中提取可选的因果诊断。就绪约束可能重叠，应取最大值而不是相加。松弛量只适用于本次回放的耗时和 先进先出 条件，不能视为改变 DDR 竞争后的界。诊断不影响执行及可行性。'
from collections import deque
from heapq import nlargest


def compact_guidance(cores, starts, ends, links, delay, limit=64):
    '从已有回放中提取数量受限的 HC 热点，不额外评价。沿紧前驱使用已观测耗时追踪；就绪时间取约束最大值，不累加等待。无需构造完整生命周期或第二个全局依赖图。'
    fifo = [[-1]*len(t.op_ids) for t in cores]
    cross, transfers = {}, []
    for c, task in enumerate(cores):
        for order in task.pipe_orders:
            for a, b in zip(order, order[1:]):
                fifo[c][b] = a
    for link in links:
        sc, dc = link['source_core'], link['target_core']
        si = cores[sc].index[link['source_copy_out_id']]
        di = cores[dc].index[link['target_copy_in_id']]
        release = ends[sc][si]+delay
        if release > cross.get((dc, di), (0, None))[0]:
            cross[dc, di] = (release, (sc, si, 'cross'))
        transfers.append(dict(link, release=release, source_end=ends[sc][si],
                              target_start=starts[dc][di], target_end=ends[dc][di]))
    trace_pred, rows, times, pipes = {}, [], {}, []
    makespan = max((e for row in ends for e in row), default=0)
    for c, task in enumerate(cores):
        for i, op in enumerate(task.op_ids):
            pred = max(task.predecessors[i], key=lambda p: ends[c][p], default=-1)
            dep_ready = ends[c][pred] if pred >= 0 else 0
            pipe_prev = fifo[c][i]
            pipe_ready = ends[c][pipe_prev] if pipe_prev >= 0 else 0
            cross_ready, cross_pred = cross.get((c, i), (0, None))
            kind = 'data' if pred in task.data_predecessors[i] else 'memory'
            choices = [(dep_ready, (c, pred, kind) if pred >= 0 else None),
                       (pipe_ready, (c, pipe_prev, 'pipe') if pipe_prev >= 0 else None),
                       (cross_ready, cross_pred)]
            required, parent = max(choices, key=lambda x: x[0])
            trace_pred[c, i] = parent
            elapsed = ends[c][i]-starts[c][i]
            head_wait = max(0, starts[c][i]-pipe_ready)
            queued = max(0, pipe_ready-max(dep_ready, cross_ready))
            slowdown = max(0, elapsed-task.duration[i]) if task.ddr[i] else 0
            rows.append({'core': c, 'op': op, 'group': task.op_groups[i],
                         'pipe': task.op_pipes[i], 'type': task.op_types[i],
                         'start': starts[c][i], 'end': ends[c][i],
                         'pipe_head_idle': head_wait, 'ready_but_queued': queued,
                         'ddr_slowdown': slowdown, 'ready_at': required,
                         'blocker': (parent[0], cores[parent[0]].op_ids[parent[1]], parent[2])
                                    if parent is not None else None,
                         'impact': max(head_wait, queued, slowdown, elapsed), 'dense_index': i})
            if task.op_types[i] not in ('COPY_IN', 'COPY_OUT'):
                times[op] = (starts[c][i], ends[c][i])
        for order in task.pipe_orders:
            if order:
                pipes.append({'core': c, 'pipe': task.op_pipes[order[0]],
                              'busy': sum(ends[c][i]-starts[c][i] for i in order),
                              'standalone_work': sum(task.duration[i] for i in order)})
    critical = set()
    for c, task in enumerate(cores):
        if not task.op_ids:
            continue
        i = max(range(len(task.op_ids)), key=lambda j: ends[c][j])
        if ends[c][i] < .95*makespan:
            continue
        cursor = (c, i)
        for _ in range(512):
            if cursor in critical:
                break
            critical.add(cursor)
            parent = trace_pred[cursor]
            if parent is None:
                break
            cursor = parent[:2]
    for r in rows:
        r['critical_trace'] = (r['core'], r.pop('dense_index')) in critical
        r['priority'] = r['impact'] * (2 if r['critical_trace'] else .25+r['end']/max(1, makespan))
    hot = nlargest(limit, rows, key=lambda r: (r['priority'], -r['core'], -r['op']))
    spills = [dict(r, core=c) for c, t in enumerate(cores) for r in t.spill_records]
    spills = nlargest(limit//2, spills,
                      key=lambda r: (r['size']*(1+int(r['spill_out_copies_data'])), -r['core'], -r['spill_in_id']))
    transfers = nlargest(limit//2, transfers,
                         key=lambda r: (r['size'], r['release'], -r['source_copy_out_id']))
    return {'operations': hot, 'spills': spills, 'transfers': transfers, 'pipes': pipes,
            'compute_times': times, 'source': 'existing_global_replay',
            'wait_categories_are_additive': False, 'critical_trace_is_counterfactual_bound': False}


def diagnose(cores, starts, ends, links, delay):
    nodes = [(c, i) for c, t in enumerate(cores) for i in range(len(t.op_ids))]
    incoming, outgoing = {x: [] for x in nodes}, {x: [] for x in nodes}
    fifo = {}
    memory_records, spill_records, lifetimes, transfers = [], [], [], []

    def edge(a, b, kind, lag=0):
        incoming[b].append((a, kind, lag))
        outgoing[a].append((b, kind, lag))

    for c, task in enumerate(cores):
        pairs = {(task.index[d['source']], task.index[d['target']])
                 for d in task.memory_dependencies}
        for j, preds in enumerate(task.data_predecessors):
            for i in preds:
                edge((c, i), (c, j), 'data')
        for i, j in sorted(pairs):
            edge((c, i), (c, j), 'memory')
        for order in task.pipe_orders:
            for i, j in zip(order, order[1:]):
                fifo[c, j] = i
                edge((c, i), (c, j), 'pipe')
        for d in task.memory_dependencies:
            memory_records.append(dict(d, core=c,
                source_end=ends[c][task.index[d['source']]],
                target_start=starts[c][task.index[d['target']]]))
        for record in task.spill_records:
            d = dict(record, core=c)
            for name in ('prev_use_op', 'next_use_op', 'spill_in_id', 'spill_out_id'):
                op = d.get(name)
                if op in task.index:
                    i = task.index[op]
                    d[name + '_start'], d[name + '_end'] = starts[c][i], ends[c][i]
            spill_records.append(d)
        for item in task.tensor_lifetimes:
            # 共享全局 DDR 后，最后完成的消费者可能发生变化。
            birth = min((starts[c][task.index[o]] for o in item['producers']), default=0)
            death = max((ends[c][task.index[o]] for o in
                         item['consumers'] or item['producers']), default=birth)
            lifetimes.append(dict(item, core=c, start=birth, end=death))
    for link in links:
        sc, dc = link['source_core'], link['target_core']
        si = cores[sc].index[link['source_copy_out_id']]
        di = cores[dc].index[link['target_copy_in_id']]
        edge((sc, si), (dc, di), 'cross', delay)
        transfers.append(dict(link, source_end=ends[sc][si],
            release=ends[sc][si]+delay, target_start=starts[dc][di], target_end=ends[dc][di]))

    degree = {x: len(incoming[x]) for x in nodes}
    ready = deque(x for x in nodes if not degree[x])
    topo = []
    while ready:
        x = ready.popleft()
        topo.append(x)
        for y, _, _ in outgoing[x]:
            degree[y] -= 1
            if not degree[y]:
                ready.append(y)
    if len(topo) != len(nodes):
        raise ValueError('diagnostics encountered a cyclic compiled execution')
    makespan = max((e for row in ends for e in row), default=0)
    latest, records, pipes = {}, [], []
    for c, i in reversed(topo):
        finish = min((latest[y]-lag for y, _, lag in outgoing[c, i]), default=makespan)
        latest[c, i] = finish - (ends[c][i]-starts[c][i])
    for c, i in nodes:
        t = cores[c]
        readiness = dict(data=0, memory=0, pipe=0, cross=0)
        for (pc, pi), kind, lag in incoming[c, i]:
            readiness[kind] = max(readiness[kind], ends[pc][pi]+lag)
        required = max(readiness.values())
        limiting = [kind for kind, when in readiness.items() if when == required and when > 0]
        blockers = [(pc, cores[pc].op_ids[pi], kind)
                    for (pc, pi), kind, lag in incoming[c, i]
                    if ends[pc][pi]+lag == required]
        records.append({'core': c, 'op': t.op_ids[i], 'group': t.op_groups[i],
            'pipe': t.op_pipes[i], 'type': t.op_types[i],
            'start': starts[c][i], 'end': ends[c][i],
            'standalone_duration': t.duration[i],
            'ddr_slowdown': max(0, ends[c][i]-starts[c][i]-t.duration[i]) if t.ddr[i] else 0,
            'slack': max(0, latest[c, i]-starts[c][i]),
            'ready': readiness, 'limiting': limiting, 'blockers': blockers,
            'pipe_head_idle': max(0, starts[c][i]-readiness['pipe']),
            'ready_but_queued': max(0, readiness['pipe'] - max(
                readiness['data'], readiness['memory'], readiness['cross'])),
            'unexplained_wait': starts[c][i]-required})
    for c, t in enumerate(cores):
        for order in t.pipe_orders:
            if not order:
                continue
            busy = sum(ends[c][i]-starts[c][i] for i in order)
            pipes.append({'core': c, 'pipe': t.op_pipes[order[0]],
                          'busy': busy, 'idle_before_last_end': ends[c][order[-1]]-busy,
                          'end': ends[c][order[-1]]})
    return {'operations': records, 'transfers': transfers, 'spills': spill_records,
            'memory_dependencies': memory_records, 'tensor_lifetimes': lifetimes,
            'pipes': pipes, 'makespan': makespan,
            'wait_categories_are_additive': False, 'slack_uses_observed_durations': True}
