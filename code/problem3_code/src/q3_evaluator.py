'第三问独立模拟 DDR 与只读 先进先出 L2 的带宽资源池。复用第二问整核编译缓存，不调用官方第三问模拟器。每次评价均从空硬件缓存开始；软件编译缓存不保存运行时 L2 状态或时间戳。'
from __future__ import annotations

import math
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace

from scene_b_internal import CoreTemplate, SceneBEvaluator, EPS
from scene_b_boundary import SceneBEvaluationError
from evaluation_validation import require_integer, require_number
from schedule_step3 import PIPE_SLOTS, _op_duration


@dataclass(slots=True)
class CacheCoreTemplate(CoreTemplate):
    cache_keys: tuple
    cache_sizes: tuple
    l2_duration: tuple
    copy_sizes: tuple


class Problem3Evaluator(SceneBEvaluator):
    template_class = CacheCoreTemplate

    def __init__(self, graph, bandwidth, capacity, cross_core_copy_delay,
                 cache_capacity_bytes, cache_bandwidth_bytes_per_cycle,
                 cache_bytes=128 * 1024**2, max_iter=1000000):
        require_integer(cache_capacity_bytes, 'cache_capacity_bytes')
        require_number(cache_bandwidth_bytes_per_cycle,
                       'cache_bandwidth_bytes_per_cycle', positive=True)
        if PIPE_SLOTS != 1:
            raise ValueError('this replay implements the supplied one-slot Pipe model')
        self.l2_capacity = cache_capacity_bytes
        self.l2_bandwidth = cache_bandwidth_bytes_per_cycle
        super().__init__(graph, bandwidth, capacity, cross_core_copy_delay,
                         cache_bytes=cache_bytes, max_iter=max_iter)

    def _template_extras(self, task, ids):
        keys, sizes, durations, copy_sizes = [], [], [], []
        for op_id in ids:
            op = task['op_by_id'][op_id]
            tids = (task['out_tids'][op_id] if op['op'] == 'COPY_IN'
                    else task['in_tids'][op_id] if op['op'] == 'COPY_OUT' else [])
            local = [t for t in tids if task['tensor_by_id'][t].get('pos') != 'DDR']
            tensor = task['tensor_by_id'][local[0]] if local else None
            keys.append(tensor.get('logical_tid', local[0]) if tensor is not None else None)
            sizes.append(tensor['size'] if tensor is not None else 0)
            copy_sizes.append(sum(task['tensor_by_id'][t]['size'] for t in tids))
            durations.append(_op_duration(op, task['in_tids'], task['out_tids'],
                                          task['tensor_by_id'], self.l2_bandwidth))
        return dict(cache_keys=tuple(keys), cache_sizes=tuple(sizes),
                    l2_duration=tuple(durations), copy_sizes=tuple(copy_sizes))

    def _run_replay(self, cores, external, external_counts, check_deadline, capture):
        return replay_l2(cores, external, external_counts, self.delay, self.max_iter,
                         check_deadline, self.l2_capacity, self.l2_bandwidth,
                         self.bandwidth, capture=capture)

    def _diagnostic_cores(self, cores, replay_extra):
        # DDR 减速诊断不能把缓存命中计为 DDR 工作量。
        return [replace(t, duration=tuple(replay_extra['_durations'][c]),
                        ddr=tuple(p == 'DDR' for p in replay_extra['_paths'][c]))
                for c, t in enumerate(cores)]


def replay_l2(cores, external, external_counts, delay, max_iter, check_deadline,
              capacity, l2_bandwidth, ddr_bandwidth, *, capture=False):
    pred = [list(t.indegree) for t in cores]
    releases = [[0] * len(t.op_ids) for t in cores]
    starts = [[-1] * len(t.op_ids) for t in cores]
    ends = [[-1] * len(t.op_ids) for t in cores]
    paths = [['ON_CHIP'] * len(t.op_ids) for t in cores]
    durations = [list(t.duration) for t in cores]
    slots = [(c, order) for c, t in enumerate(cores) for order in t.pipe_orders]
    cursors, running, running_ends = [0]*len(slots), [-1]*len(slots), [0]*len(slots)
    pools = {name: {'work': {}, 'last': 0} for name in ('DDR', 'CACHE_READ')}
    entries, used = OrderedDict(), 0
    stats = dict(copy_in_hits=0, copy_in_misses=0, hit_bytes=0, miss_bytes=0)
    records, cache_events = [], []
    requested, first_insert, last_eviction = {}, {}, {}
    missed_bytes, missed_count = Counter(), Counter()
    actual_bytes = Counter()
    remaining = sum(len(t.op_ids) for t in cores)

    def event(value):
        if capture:
            cache_events.append(value)

    def advance(name, now):
        pool = pools[name]
        elapsed = now-pool['last']
        work = pool['work']
        while elapsed > EPS:
            active = [s for s, w in work.items() if w > EPS]
            if not active:
                break
            smallest = min(work[s] for s in active)
            finish_delta = smallest*len(active)
            if finish_delta >= elapsed-EPS:
                share = elapsed/len(active)
                for s in active:
                    work[s] = max(0.0, work[s]-share)
                break
            for s in active:
                work[s] = max(0.0, work[s]-smallest)
            elapsed -= finish_delta
        pool['last'] = now

    def reschedule(name, now):
        ordered = sorted((max(0.0, w), s) for s, w in pools[name]['work'].items())
        cursor, previous, n, j = float(now), 0.0, len(ordered), 0
        while j < len(ordered):
            work = ordered[j][0]
            cursor += (work-previous)*n
            k = j
            while k < len(ordered) and abs(ordered[k][0]-work) <= EPS:
                s = ordered[k][1]
                running_ends[s] = int(math.ceil(cursor-EPS))
                ends[slots[s][0]][running[s]] = running_ends[s]
                k += 1
            n -= k-j
            previous, j = work, k

    def insert(c, i, now):
        nonlocal used
        t = cores[c]
        tid, size = t.cache_keys[i], t.cache_sizes[i]
        if tid is None or size > capacity or tid in entries:
            return
        evicted = []
        while entries and used+size > capacity:
            old, amount = entries.popitem(last=False)
            used -= amount
            evicted.append(old)
            last_eviction[old] = dict(time=now, core_id=c, op_id=t.op_ids[i],
                                      subgraph_id=t.op_groups[i], tensor_id=tid)
        entries[tid] = size
        used += size
        first_insert.setdefault(tid, dict(time=now, core_id=c, op_id=t.op_ids[i],
                                         subgraph_id=t.op_groups[i]))
        event(dict(time=now, event='insert', tensor_id=tid, size_bytes=size,
                   used_bytes=used, evicted_tensor_ids=evicted, core_id=c, op_id=t.op_ids[i]))

    now = 0
    for iteration in range(max_iter):
        if iteration % 128 == 0:
            check_deadline()
        for name in pools:
            advance(name, now)
        changed = set()
        # 与官方完成事件顺序一致：先按核编号，再按流水线顺序处理。
        for s, i in enumerate(running):
            if i < 0 or running_ends[s] > now:
                continue
            c = slots[s][0]
            running[s] = -1
            cursors[s] += 1
            remaining -= 1
            for name, pool in pools.items():
                if s in pool['work']:
                    del pool['work'][s]
                    changed.add(name)
            # 官方实现在每个 COPY_IN 完成时都尝试插入缓存，
            # 包括读取过程中已被换出的命中项。
            if cores[c].op_types[i] == 'COPY_IN':
                insert(c, i, now)
            for j in cores[c].successors[i]:
                pred[c][j] -= 1
            for dc, di in external[c][i]:
                external_counts[dc][di] -= 1
                releases[dc][di] = max(releases[dc][di], ends[c][i]+delay)
        for name in pools:
            if name in changed:
                reschedule(name, now)
        future_release = math.inf
        for s, (c, order) in enumerate(slots):
            if running[s] >= 0 or cursors[s] >= len(order):
                continue
            i = order[cursors[s]]
            if pred[c][i] or external_counts[c][i]:
                continue
            if releases[c][i] > now:
                future_release = min(future_release, releases[c][i])
                continue
            t = cores[c]
            tid, size = t.cache_keys[i], t.cache_sizes[i]
            eligible = t.op_types[i] == 'COPY_IN' and tid is not None and size > 0
            hit = eligible and tid in entries
            duration = t.l2_duration[i] if hit else t.duration[i]
            durations[c][i] = duration
            path = 'CACHE_READ' if hit else 'DDR' if t.ddr[i] else 'ON_CHIP'
            paths[c][i] = path
            running[s] = i
            starts[c][i] = now
            running_ends[s] = ends[c][i] = now+duration
            if eligible:
                stats['copy_in_hits' if hit else 'copy_in_misses'] += 1
                stats['hit_bytes' if hit else 'miss_bytes'] += size
                reason = None
                if not hit:
                    reason = ('oversize' if size > capacity else
                              'cold' if tid not in requested else
                              'inflight_first_load' if tid not in first_insert else 'evicted')
                    missed_bytes[reason] += size
                    missed_count[reason] += 1
                row = dict(core_id=c, op_id=t.op_ids[i], subgraph_id=t.op_groups[i],
                           tensor_id=tid, size_bytes=size, start=now, cache_hit=bool(hit),
                           memory_path=path, miss_reason=reason, _index=i)
                if reason == 'inflight_first_load':
                    row['first_request'] = dict(requested[tid])
                if reason == 'evicted':
                    row['last_eviction'] = dict(last_eviction.get(tid, {}))
                records.append(row)
                requested.setdefault(tid, dict(core_id=c, op_id=t.op_ids[i],
                                               subgraph_id=t.op_groups[i], start=now))
                event(dict(time=now, event='hit' if hit else 'miss', tensor_id=tid,
                           size_bytes=size, core_id=c, op_id=t.op_ids[i], op=t.op_types[i]))
            if t.op_types[i] in ('COPY_IN', 'COPY_OUT'):
                actual_bytes[path] += t.copy_sizes[i]
            if path in pools:
                advance(path, now)
                pools[path]['work'][s] = float(duration)
                reschedule(path, now)
        if remaining == 0:
            break
        next_now = min((running_ends[s] for s, i in enumerate(running) if i >= 0),
                       default=math.inf)
        next_now = min(next_now, future_release)
        if not math.isfinite(next_now) or next_now <= now:
            raise SceneBEvaluationError(f'Problem 3 replay deadlock/no progress at t={now}')
        now = next_now
    else:
        raise SceneBEvaluationError('max_iter exceeded')

    for r in records:
        i = r.pop('_index'); c = r['core_id']
        r['end'] = ends[c][i]
        r['standalone_duration'] = durations[c][i]
        r['pool_slowdown'] = max(0, r['end']-r['start']-durations[c][i])
        if r['miss_reason'] == 'inflight_first_load' and r['tensor_id'] in first_insert:
            r['first_insertion'] = dict(first_insert[r['tensor_id']])
    stats['hits'] = stats['copy_in_hits']
    stats['accesses'] = stats['hits']+stats['copy_in_misses']
    total = stats['hit_bytes']+stats['miss_bytes']
    stats['hit_rate'] = stats['hit_bytes']/total if total else 0.0
    extra = dict(problem=3, cache_mode='read_only', bandwidth_bytes_per_cycle=ddr_bandwidth,
                 cache_capacity_bytes=capacity, cache_bandwidth_bytes_per_cycle=l2_bandwidth,
                 cache_stats=stats, cache_read_records=records,
                 cache_miss_reasons={'counts': dict(missed_count), 'bytes': dict(missed_bytes)},
                 cache_final_entries=[dict(tensor_id=t, size_bytes=s) for t, s in entries.items()],
                 cache_used_bytes_final=used,
                 physical_transfer_bytes={'DDR': actual_bytes['DDR'],
                                          'CACHE_READ': actual_bytes['CACHE_READ'],
                                          'ON_CHIP': actual_bytes['ON_CHIP']},
                 _durations=durations, _paths=paths)
    if capture:
        extra['cache_events'] = cache_events
        extra['memory_path_records'] = [
            (c, t.op_ids[i], paths[c][i], t.cache_keys[i],
             bool(paths[c][i] == 'CACHE_READ'))
            for c, t in enumerate(cores) for i in range(len(t.op_ids))
            if t.op_types[i] in ('COPY_IN', 'COPY_OUT')]
    return starts, ends, iteration+1, extra
