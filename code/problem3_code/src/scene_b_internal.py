'使用整核编译缓存和紧凑的精确场景 B 事件回放。官方 步骤 1、步骤 2、步骤 3 作为参考编译器；独立多核模拟器不调用 evaluate_scene_b。无完整方案分数缓存，每次评价均检查并回放方案。每个评价器持有一份图快照和固定硬件配置，对每个 case 单独创建。'
from __future__ import annotations

import copy
import math
import pickle
import sys
from collections import OrderedDict, deque
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter

RUNTIME = Path(__file__).resolve().parents[1] / "official_runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))
from evaluation_validation import validate_parameters
from multicore_cut_evaluate_problem_1 import _copy_traffic_bytes, _original_tensor_views
from schedule_step1 import step1_schedule
from schedule_step2 import step2_spill_insertion, _build_extended_graph
from schedule_step3 import PIPES, _op_duration, _uses_ddr_bandwidth, prepare_step3_execution, _build_graph_views
from scene_b_boundary import build_core_graphs, _prioritize_task_seq, SceneBEvaluationError

EPS = 1e-9


class WeightedLRU:
    '按估计字节数限制缓存，进程限制约束工作进程总内存。'
    def __init__(self, budget):
        self.budget, self.used = budget, 0
        self.entries = OrderedDict()

    def get(self, key):
        item = self.entries.get(key)
        if item is not None:
            self.entries.move_to_end(key)
            return item[0]
        return None

    def put(self, key, value, weight):
        if weight > self.budget:
            return
        old = self.entries.pop(key, None)
        if old is not None:
            self.used -= old[1]
        while self.entries and self.used + weight > self.budget:
            _, (_, amount) = self.entries.popitem(last=False)
            self.used -= amount
        self.entries[key] = value, weight
        self.used += weight

    def clear(self):
        self.entries.clear()
        self.used = 0


@dataclass(slots=True)
class CoreTemplate:
    op_ids: tuple
    index: dict
    duration: tuple
    ddr: tuple
    successors: tuple
    indegree: tuple
    pipe_orders: tuple
    op_pipes: tuple
    op_types: tuple
    op_groups: tuple
    groups: tuple
    copy_bytes: int
    spill_bytes: int
    memory_peak: dict
    local_makespan: int
    memory_dependency_count: int
    predecessors: tuple
    data_predecessors: tuple
    memory_dependencies: tuple
    spill_records: tuple
    tensor_lifetimes: tuple


class SceneBEvaluator:
    template_class = CoreTemplate

    def _template_extras(self, task, ids):
        return {}

    def _run_replay(self, cores, external, external_counts, check_deadline, capture):
        return (*_replay(cores, external, external_counts, self.delay,
                         self.max_iter, check_deadline), {})

    def _diagnostic_cores(self, cores, replay_extra):
        return cores

    def __init__(self, graph, bandwidth, capacity, cross_core_copy_delay,
                 cache_bytes=128 * 1024 * 1024, max_iter=1000000):
        validate_parameters(bandwidth, capacity, max_iter,
                            cross_core_copy_delay=cross_core_copy_delay)
        if cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")
        self.graph = copy.deepcopy(graph)
        self.bandwidth, self.capacity = bandwidth, dict(capacity)
        self.delay, self.max_iter = cross_core_copy_delay, max_iter
        self.op_by_id = {o['id']: o for o in self.graph['ops']}
        self.tensor_by_id = {t['id']: t for t in self.graph['tensors']}
        self.max_node_id = max((*self.op_by_id, *self.tensor_by_id, 0))
        self.tensor_views = _original_tensor_views(self.graph)
        self.original_copy_bytes = _copy_traffic_bytes(self.graph)
        self.raw_cache = WeightedLRU(cache_bytes // 4)
        self.core_cache = WeightedLRU(cache_bytes * 3 // 4)

    def clear_cache(self):
        self.raw_cache.clear()
        self.core_cache.clear()

    def _compile(self, data, graph, raw_seq):
        seq = _prioritize_task_seq(graph, raw_seq, data['op_subgraph'],
                                   data['subgraph_order'])
        r2 = step2_spill_insertion(graph, seq, capacity=self.capacity) if seq else {
            'new_ops': [], 'new_tensors': [], 'new_edges': [],
            'spill_records': [], 'seq_ext': []}
        spill_bytes = sum(r['size'] * (1 + int(r['spill_out_copies_data']))
                          for r in r2.get('spill_records', []))
        op_groups = dict(data['op_subgraph'])
        for spill in r2.get('spill_records', []):
            next_group = op_groups.get(spill.get('next_use_op'))
            prev_group = op_groups.get(spill.get('prev_use_op'))
            if spill['spill_out_id'] is not None:
                op_groups[spill['spill_out_id']] = (
                    prev_group if prev_group is not None else next_group)
            op_groups[spill['spill_in_id']] = next_group
        ext_graph = _build_extended_graph(graph, r2)
        task = prepare_step3_execution(ext_graph, capacity=self.capacity,
                                       bandwidth=self.bandwidth)
        ids = tuple(task['seq'])
        index = {op: i for i, op in enumerate(ids)}
        ops = [task['op_by_id'][op] for op in ids]
        _, _, data_preds, _ = _build_graph_views(ext_graph['ops'], ext_graph['edges'])
        # 保留因果标识，不把 步骤 3 的单核时间戳当作多核执行时间；
        # 只有启用诊断时才构造完整日志。
        lifetime = {}
        for event in task['step3']['memory_events']:
            tid = event['tid']
            item = lifetime.setdefault(tid, {'tensor_id': tid,
                'logical_tid': task['tensor_by_id'][tid].get('logical_tid', tid),
                'pos': task['tensor_by_id'][tid]['pos'],
                'size': task['tensor_by_id'][tid]['size'],
                'producers': [], 'consumers': []})
        for op in ids:
            for tid in task['out_tids'][op]:
                if tid in lifetime:
                    lifetime[tid]['producers'].append(op)
            for tid in task['in_tids'][op]:
                if tid in lifetime:
                    lifetime[tid]['consumers'].append(op)
        return self.template_class(
            op_ids=ids, index=index,
            duration=tuple(_op_duration(op, task['in_tids'], task['out_tids'],
                                        task['tensor_by_id'], self.bandwidth) for op in ops),
            ddr=tuple(_uses_ddr_bandwidth(op, task['in_tids'], task['out_tids'],
                                         task['tensor_by_id']) for op in ops),
            successors=tuple(tuple(index[s] for s in task['op_succs'][op]) for op in ids),
            indegree=tuple(len(task['op_preds'][op]) for op in ids),
            pipe_orders=tuple(tuple(index[o] for o in task['pipe_ops'][p]) for p in PIPES),
            op_pipes=tuple(op['pipe'] for op in ops),
            op_types=tuple(op['op'] for op in ops),
            op_groups=tuple(op_groups.get(op) for op in ids),
            groups=tuple(data['subgraph_order']),
            copy_bytes=_copy_traffic_bytes(graph), spill_bytes=spill_bytes,
            memory_peak=dict(task['step3']['memory_peak']),
            local_makespan=task['step3']['makespan'],
            memory_dependency_count=len(task['step3']['memory_dependencies']),
            predecessors=tuple(tuple(index[p] for p in task['op_preds'][op]) for op in ids),
            data_predecessors=tuple(tuple(index[p] for p in data_preds[op]) for op in ids),
            memory_dependencies=tuple(dict(d, previous_logical_tids=[
                task['tensor_by_id'][tid].get('logical_tid', tid) for tid in d['previous_tensor_ids']])
                for d in task['step3']['memory_dependencies']),
            spill_records=tuple(dict(r) for r in r2.get('spill_records', ())),
            tensor_lifetimes=tuple(lifetime.values()),
            **self._template_extras(task, ids))

    def evaluate(self, plan, *, capture=False, diagnostics=False, deadline=None,
                 hc_guidance=False, guidance_below=None):
        '非法方案、超时或死锁均抛出异常，不返回部分分数。deadline 为 perf_counter 的绝对时间，在各阶段之间检查；编译的硬时间和内存限制由外层子进程提供。capture 保存核验所需的紧凑操作和传输记录。'
        def check_deadline():
            if deadline is not None and perf_counter() >= deadline:
                raise TimeoutError("Scene B evaluation deadline reached")

        check_deadline()
        t0 = perf_counter()
        data_by_core, links, cross_bytes, view = build_core_graphs(
            self.graph, plan, self.bandwidth, self)
        t1 = perf_counter()
        cores = []
        stats = {'core_hits': 0, 'core_misses': 0, 'step1_hits': 0, 'step1_misses': 0}
        for core_id in range(view['num_cores']):
            check_deadline()
            data = data_by_core[core_id]
            graph = {'ops': data['ops'], 'tensors': list(data['tensors'].values()),
                     'edges': data['edges']}
            # 有序序列化内容保留官方编号及平分时的排序依据。
            raw_blob = pickle.dumps(graph, protocol=5)
            raw_key = sha256(raw_blob).digest()
            core_key = (raw_key, sha256(pickle.dumps(
                (data['op_subgraph'], data['subgraph_order']), protocol=5)).digest())
            template = self.core_cache.get(core_key)
            if template is None:
                stats['core_misses'] += 1
                raw_seq = self.raw_cache.get(raw_key)
                if raw_seq is None:
                    stats['step1_misses'] += 1
                    raw_seq = tuple(step1_schedule(graph)) if graph['ops'] else ()
                    self.raw_cache.put(raw_key, raw_seq, 128 + 40 * len(raw_seq))
                else:
                    stats['step1_hits'] += 1
                template = self._compile(data, graph, raw_seq)
                weight = (2048 + 1152 * len(template.op_ids)
                          + 96 * sum(map(len, template.successors))
                          + 1024 * len(template.spill_records)
                          + 768 * len(template.memory_dependencies)
                          + 512 * len(template.tensor_lifetimes))
                self.core_cache.put(core_key, template, weight)
            else:
                stats['core_hits'] += 1
            cores.append(template)
        # 编译完成后释放完整预处理图和 步骤 3 日志。
        del data_by_core
        t2 = perf_counter()
        check_deadline()
        external = [[] for _ in cores]
        external_counts = [[0] * len(t.op_ids) for t in cores]
        for c, task in enumerate(cores):
            external[c] = [[] for _ in task.op_ids]
        dense_links = []
        for link in links:
            sc, dc = link['source_core'], link['target_core']
            si = cores[sc].index[link['source_copy_out_id']]
            di = cores[dc].index[link['target_copy_in_id']]
            external[sc][si].append((dc, di))
            external_counts[dc][di] += 1
            dense_links.append((sc, si, dc, di))
        _validate_global(cores, dense_links)
        t3 = perf_counter()
        starts, ends, events, replay_extra = self._run_replay(
            cores, external, external_counts, check_deadline, capture or diagnostics)
        check_deadline()
        t4 = perf_counter()
        task_bytes = sum(t.copy_bytes for t in cores)
        spill_bytes = sum(t.spill_bytes for t in cores)
        boundary_bytes = task_bytes - self.original_copy_bytes
        core_ends = [max(e, default=0) for e in ends]
        group_starts, group_ends = {}, {}
        for c, task in enumerate(cores):
            for i, g in enumerate(task.op_groups):
                if g is not None:
                    group_starts[g] = min(group_starts.get(g, starts[c][i]), starts[c][i])
                    group_ends[g] = max(group_ends.get(g, ends[c][i]), ends[c][i])
        result = {
            'scene': 'B', 'makespan': max(core_ends, default=0),
            'num_cores': len(cores), 'task_count': len(cores),
            'active_cores': sum(bool(t.op_ids) for t in cores),
            'execution_ops': sum(len(t.op_ids) for t in cores),
            'cross_task_traffic': cross_bytes,
            'data_movement_bytes': {
                'original_graph_copy_bytes': self.original_copy_bytes,
                'scheduled_copy_bytes': task_bytes + spill_bytes,
                'added_copy_bytes': boundary_bytes + spill_bytes,
                'partition_added_copy_bytes': boundary_bytes,
                'spill_added_copy_bytes': spill_bytes},
            'memory_peak_by_core': {c: dict(t.memory_peak) for c, t in enumerate(cores)},
            'step3_by_core': {c: {
                'local_makespan': t.local_makespan,
                'memory_dependency_count': t.memory_dependency_count,
                'pipe_op_counts': {p: len(t.pipe_orders[j]) for j, p in enumerate(PIPES)}}
                for c, t in enumerate(cores)},
            'core_ends': core_ends, 'starts': group_starts, 'ends': group_ends,
            'durations': {g: group_ends[g] - group_starts[g] for g in group_starts},
            'event_count': events, 'cache_stats': stats,
            'stage_seconds': {'boundary_and_validation': t1-t0, 'compile': t2-t1,
                              'global_validation': t3-t2, 'replay': t4-t3},
            'cache_estimated_bytes': self.raw_cache.used + self.core_cache.used}
        result['compile_cache_stats'] = stats
        result.update({k: v for k, v in replay_extra.items() if not k.startswith('_')})
        if capture:
            result['op_records'] = [
                (c, op, starts[c][i], ends[c][i], t.op_groups[i],
                 t.op_pipes[i], t.op_types[i])
                for c, t in enumerate(cores) for i, op in enumerate(t.op_ids)]
            result['transfer_records'] = [
                (sc, cores[sc].op_ids[si], dc, cores[dc].op_ids[di],
                 ends[sc][si], ends[sc][si] + self.delay, starts[dc][di], ends[dc][di])
                for sc, si, dc, di in dense_links]
        if diagnostics:
            from q2_diagnostics import diagnose
            result['diagnostics'] = diagnose(
                self._diagnostic_cores(cores, replay_extra), starts, ends, links, self.delay)
        key = (result['makespan'], spill_bytes, task_bytes+spill_bytes)
        if hc_guidance and (guidance_below is None or key < tuple(guidance_below)):
            from q2_diagnostics import compact_guidance
            hint_start = perf_counter()
            result['hc_guidance'] = compact_guidance(
                self._diagnostic_cores(cores, replay_extra), starts, ends, links, self.delay)
            result['stage_seconds']['hc_guidance'] = perf_counter()-hint_start
        return result


def _validate_global(cores, links):
    if not links:
        return  # 步骤 3 已分别检查各独立核。
    offsets, total = [], 0
    for task in cores:
        offsets.append(total)
        total += len(task.op_ids)
    successors = [[] for _ in range(total)]
    indegree = [0] * total

    def edge(a, b):
        successors[a].append(b)
        indegree[b] += 1

    for c, task in enumerate(cores):
        off = offsets[c]
        for i, succs in enumerate(task.successors):
            for j in succs:
                edge(off+i, off+j)
        for order in task.pipe_orders:
            for i, j in zip(order, order[1:]):
                edge(off+i, off+j)
    for sc, si, dc, di in links:
        edge(offsets[sc]+si, offsets[dc]+di)
    ready = deque(i for i, d in enumerate(indegree) if d == 0)
    visited = 0
    while ready:
        i = ready.popleft()
        visited += 1
        for j in successors[i]:
            indegree[j] -= 1
            if indegree[j] == 0:
                ready.append(j)
    if visited != total:
        raise SceneBEvaluationError('global execution (local dependencies/FIFO/cross-core COPY) contains a cycle')


def _replay(cores, external, external_counts, delay, max_iter, check_deadline):
    '只允许固定流水线队首发射，等待中的 COPY 不占用流水线。'
    pred = [list(t.indegree) for t in cores]
    releases = [[0] * len(t.op_ids) for t in cores]
    starts = [[-1] * len(t.op_ids) for t in cores]
    ends = [[-1] * len(t.op_ids) for t in cores]
    slots = [(c, order) for c, t in enumerate(cores) for order in t.pipe_orders]
    cursors = [0] * len(slots)
    running = [-1] * len(slots)
    running_ends = [0] * len(slots)
    ddr_work = {}  # 将槽位映射到按独占带宽计算的剩余周期。
    ddr_last = 0
    remaining = sum(len(t.op_ids) for t in cores)

    def advance_ddr(now):
        nonlocal ddr_last
        elapsed = now - ddr_last
        while elapsed > EPS:
            active = [s for s, w in ddr_work.items() if w > EPS]
            if not active:
                break
            smallest = min(ddr_work[s] for s in active)
            finish_delta = smallest * len(active)
            if finish_delta >= elapsed - EPS:
                share = elapsed / len(active)
                for s in active:
                    ddr_work[s] = max(0.0, ddr_work[s] - share)
                break
            for s in active:
                ddr_work[s] = max(0.0, ddr_work[s] - smallest)
            elapsed -= finish_delta
        ddr_last = now

    def reschedule(now):
        ordered = sorted((max(0.0, w), s) for s, w in ddr_work.items())
        cursor, previous, active_count, i = float(now), 0.0, len(ordered), 0
        while i < len(ordered):
            work = ordered[i][0]
            cursor += (work - previous) * active_count
            j = i
            while j < len(ordered) and abs(ordered[j][0] - work) <= EPS:
                s = ordered[j][1]
                running_ends[s] = int(math.ceil(cursor - EPS))
                ends[slots[s][0]][running[s]] = running_ends[s]
                j += 1
            active_count -= j - i
            previous, i = work, j

    now = 0
    for iteration in range(max_iter):
        if iteration % 128 == 0:
            check_deadline()
        advance_ddr(now)
        retired_ddr = False
        for s, i in enumerate(running):
            if i < 0 or running_ends[s] > now:
                continue
            c = slots[s][0]
            running[s] = -1
            cursors[s] += 1
            remaining -= 1
            if s in ddr_work:
                del ddr_work[s]
                retired_ddr = True
            for j in cores[c].successors[i]:
                pred[c][j] -= 1
            for dc, di in external[c][i]:
                external_counts[dc][di] -= 1
                releases[dc][di] = max(releases[dc][di], ends[c][i] + delay)
        if retired_ddr:
            reschedule(now)
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
            duration = cores[c].duration[i]
            running[s] = i
            starts[c][i] = now
            running_ends[s] = ends[c][i] = now + duration
            if cores[c].ddr[i]:
                advance_ddr(now)
                ddr_work[s] = float(duration)
                reschedule(now)
        if remaining == 0:
            return starts, ends, iteration + 1
        next_now = min((running_ends[s] for s, i in enumerate(running) if i >= 0),
                       default=math.inf)
        next_now = min(next_now, future_release)
        if not math.isfinite(next_now) or next_now <= now:
            raise SceneBEvaluationError(f'multicore replay deadlock/no progress at t={now}')
        now = next_now
    raise SceneBEvaluationError('max_iter exceeded')
