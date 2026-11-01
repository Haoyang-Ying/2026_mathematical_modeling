'缓存场景 A 的 任务 准备结果，并执行不生成时间线的四流水线模拟。\n\n任务 准备复用官方 步骤 1、步骤 2 和步骤 3，包含边界 COPY、Belady 换出与重载及内存复用依赖。模板按成员集合和官方边界编号分配前缀缓存；前缀一致时，核分配及核内顺序可复用模板。内部全局评价不生成官方时间线或 DDR 诊断日志，也不调用 evaluate_scene_a。\n\nUnix 主线程通过 ITIMER_REAL 在准备阶段截止时中断；其他环境在阶段或事件边界协作检查截止时间，外层工作进程提供硬时间限制。'

from __future__ import annotations

import importlib
import math
import os
import signal
import sys
import threading
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


PIPES = ("PIPE_MTE2", "PIPE_MTE3", "PIPE_M", "PIPE_V")
_EPS = 1.0e-9


def _check_deadline(deadline):
    if deadline is not None and perf_counter() >= deadline:
        raise TimeoutError("Task execution evaluation deadline reached")


@contextmanager
def _preparation_deadline(deadline):
    '在 Unix 上限制未经内部插桩的官方准备流程耗时。'
    _check_deadline(deadline)
    supported = (deadline is not None and hasattr(signal, "setitimer")
                 and threading.current_thread() is threading.main_thread())
    if not supported:
        yield
        _check_deadline(deadline)
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = perf_counter()

    def expired(signum, frame):
        raise TimeoutError("Task template preparation deadline reached")

    signal.signal(signal.SIGALRM, expired)
    duration = max(1.0e-6, deadline - started)
    if previous_timer[0] > 0:
        duration = min(duration, previous_timer[0])
    signal.setitimer(signal.ITIMER_REAL, duration)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL,
                            max(1.0e-6, previous_timer[0] - (perf_counter() - started)),
                            previous_timer[1])
    _check_deadline(deadline)


def _official_modules(official_dir=None):
    official_dir = official_dir or os.environ.get("DAGP_OFFICIAL_DIR")
    if official_dir is None:
        here = Path(__file__).resolve().parent
        candidates = [here / "official_runtime", here / "official_support",
                      here.parent / "official" / "code"]
        official_dir = next((p for p in candidates if (p / "schedule_step3.py").is_file()), None)
    if official_dir is None:
        raise FileNotFoundError("Official Step1/2/3 support files are required")
    folder = str(Path(official_dir).resolve())
    if folder not in sys.path:
        sys.path.insert(0, folder)
    return tuple(importlib.import_module(name) for name in
                 ("schedule_step1", "schedule_step2", "schedule_step3"))


@dataclass(frozen=True, slots=True)
class TaskTemplate:
    # 节点按照官方 seq_ext 顺序使用连续索引。
    duration: tuple
    ddr: tuple
    successors: tuple
    indegree: tuple
    pipe_orders: tuple
    copy_bytes: int
    boundary_copy_bytes: int
    spill_bytes: int
    memory_dependencies: int
    local_makespan: int

    @property
    def nodes(self):
        return len(self.duration)


class TemplateEvaluator:
    '准备并缓存 任务 模板，评价完整固定方案。\n\ndeadline 是 time.perf_counter() 的绝对截止时刻。超时后已完成模板仍可复用；每个实例固定原始图、硬件和官方实现，作为缓存上下文。'

    def __init__(self, model, deadline=None, official_dir=None, source_graph=None,
                 max_cached_nodes=750_000):
        self.model = model
        self.deadline = deadline
        self.hardware = model.hardware
        source = source_graph if source_graph is not None else getattr(model, "source_graph", None)
        if source is None:
            raise ValueError("TemplateEvaluator requires the original source_graph")
        self.source = source
        self.ops = {op["id"]: op for op in source["ops"]}
        self.tensors = {t["id"]: t for t in source["tensors"]}
        self.direct = defaultdict(list)
        for index, edge in enumerate(source["edges"]):
            if index % 4096 == 0:
                _check_deadline(deadline)
            if edge["source"] in self.ops and edge["target"] in self.ops:
                self.direct[edge["source"]].append(edge["target"])
        # 严格保留官方绝对编号。步骤 3 遍历 set(TensorID) 时的释放顺序会影响内存额度依赖，因此即使重编号保持相对顺序，也不能假定安全。
        self._used_ids = set(self.ops) | set(self.tensors)
        self._next_op = max((*self.ops, 0)) + 1
        self._next_tensor = max((*self.tensors, 10000)) + 1
        self._boundary_ids = []
        self._boundary_counts = {}
        self.step1, self.step2, self.step3 = _official_modules(official_dir)
        self.cache = OrderedDict()
        self.max_cached_nodes = max_cached_nodes
        self.cached_nodes = 0
        self.cache_hits = self.cache_misses = 0
        self.prepare_seconds = 0.0
        self.evaluations = 0

    # 结束当前评价器时释放模板缓存，避免不同用例累积内存。
    def close(self):
        self.cache.clear()
        self.cached_nodes = 0

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def _deadline(self, deadline):
        if self.deadline is None:
            return deadline
        if deadline is None:
            return self.deadline
        return min(deadline, self.deadline)

    def _boundary_count(self, members, deadline):
        if members in self._boundary_counts:
            return self._boundary_counts[members]
        touched = set()
        for op in members:
            touched.update(self.model.touched[op])
        count = 0
        for index, tid in enumerate(touched):
            if index % 256 == 0:
                _check_deadline(deadline)
            info = self.model.tensors[tid]
            prod, cons = info.producers & members, info.consumers & members
            count += int(bool(cons) and not prod)
            count += int(bool(prod) and (info.required_output or not info.consumers
                                         or not info.consumers <= members))
        self._boundary_counts[members] = count
        return count

    def _ensure_boundary_ids(self, count, deadline):
        while len(self._boundary_ids) < count:
            if len(self._boundary_ids) % 256 == 0:
                _check_deadline(deadline)
            while self._next_tensor in self._used_ids:
                self._next_tensor += 1
            ddr = self._next_tensor
            self._used_ids.add(ddr)
            self._next_tensor += 1
            while self._next_op in self._used_ids:
                self._next_op += 1
            op = self._next_op
            self._used_ids.add(op)
            self._next_op += 1
            self._boundary_ids.append((ddr, op))

    def _task_graph(self, members, deadline, boundary_ordinal):
        ops = [dict(self.ops[op]) for op in sorted(members)]
        touched = set()
        for index, op in enumerate(members):
            if index % 256 == 0:
                _check_deadline(deadline)
            touched.update(self.model.touched[op])
        tensors, edges = [], []
        cursor = boundary_ordinal
        boundary_bytes = 0
        # 该 任务 使用与完整官方方案一致的绝对编号。
        for index, tid in enumerate(sorted(touched)):
            if index % 256 == 0:
                _check_deadline(deadline)
            info = self.model.tensors[tid]
            prod, cons = info.producers & members, info.consumers & members
            tensor = dict(self.tensors[tid])
            if tensor["pos"] == "DDR":
                tensor["pos"] = "UB"
            tensors.append(tensor)
            edges.extend({"source": op, "target": tid} for op in sorted(prod))
            edges.extend({"source": tid, "target": op} for op in sorted(cons))
            read = bool(cons) and not prod
            write = bool(prod) and (info.required_output or not info.consumers
                                    or not info.consumers <= members)
            for incoming, required in ((True, read), (False, write)):
                if not required:
                    continue
                ddr_id, copy_id = self._boundary_ids[cursor]
                cursor += 1
                tensors.append({"id": ddr_id, "pos": "DDR", "size": info.size})
                ops.append({"id": copy_id,
                            "op": "COPY_IN" if incoming else "COPY_OUT",
                            "pipe": "PIPE_MTE2" if incoming else "PIPE_MTE3",
                            "cycles": max(1, math.ceil(info.size / self.hardware.bandwidth))})
                source, target = (ddr_id, tid) if incoming else (tid, ddr_id)
                edges.extend(({"source": source, "target": copy_id},
                              {"source": copy_id, "target": target}))
                boundary_bytes += info.size
        for op in sorted(members):
            edges.extend({"source": op, "target": child}
                         for child in self.direct.get(op, ()) if child in members)
        return {"ops": ops, "tensors": tensors, "edges": edges}, boundary_bytes

    def template(self, members, deadline=None, boundary_ordinal=0):
        deadline = self._deadline(deadline)
        _check_deadline(deadline)
        members = frozenset(members)
        key = (members, boundary_ordinal)
        if key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.cache_misses += 1
        # 可变分配器的更新必须在异步定时中断范围外完成；协作式截止检查只放在完整边界编号对之间。
        self._ensure_boundary_ids(boundary_ordinal + self._boundary_count(members, deadline), deadline)
        started = perf_counter()
        try:
            with _preparation_deadline(deadline):
                graph, boundary_bytes = self._task_graph(members, deadline, boundary_ordinal)
                seq = self.step1.step1_schedule(graph)
                _check_deadline(deadline)
                result2 = self.step2.step2_spill_insertion(graph, seq, self.hardware.capacity)
                spill_bytes = sum(record["size"] * (1 + int(record["spill_out_copies_data"]))
                                  for record in result2["spill_records"])
                _check_deadline(deadline)
                extended = self.step2._build_extended_graph(graph, result2)
                prepared = self.step3.prepare_step3_execution(
                    extended, capacity=self.hardware.capacity, bandwidth=self.hardware.bandwidth)
                _check_deadline(deadline)
                order = prepared["seq"]
                dense = {op: index for index, op in enumerate(order)}
                duration, ddr = [], []
                for op in order:
                    row = prepared["op_by_id"][op]
                    duration.append(self.step3._op_duration(
                        row, prepared["in_tids"], prepared["out_tids"],
                        prepared["tensor_by_id"], self.hardware.bandwidth))
                    ddr.append(self.step3._uses_ddr_bandwidth(
                        row, prepared["in_tids"], prepared["out_tids"], prepared["tensor_by_id"]))
                result = TaskTemplate(
                    tuple(duration), tuple(ddr),
                    tuple(tuple(dense[v] for v in sorted(prepared["op_succs"][op])) for op in order),
                    tuple(len(prepared["op_preds"][op]) for op in order),
                    tuple(tuple(dense[op] for op in prepared["pipe_ops"][pipe]) for pipe in PIPES),
                    boundary_bytes + spill_bytes, boundary_bytes, spill_bytes,
                    len(prepared["step3"]["memory_dependencies"]), prepared["step3"]["makespan"])
        finally:
            self.prepare_seconds += perf_counter() - started
        self.cache[key] = result
        self.cached_nodes += result.nodes
        while self.cached_nodes > self.max_cached_nodes and len(self.cache) > 1:
            _, expired = self.cache.popitem(last=False)
            self.cached_nodes -= expired.nodes
        return result

    def evaluate(self, plan, deadline=None):
        deadline = self._deadline(deadline)
        _check_deadline(deadline)
        orders = plan["core_schedules"]
        self.model.validate_plan(plan, len(orders))
        blocks = defaultdict(set)
        for op, task in plan["node_to_subgraph"].items():
            blocks[task].add(int(op))
        blocks = {task: frozenset(members) for task, members in blocks.items()}
        pred, succ, _ = self.model.validate_partition(blocks)
        ordinals = {}
        ordinal = 0
        for task in sorted(blocks):
            ordinals[task] = ordinal
            ordinal += self._boundary_count(blocks[task], deadline)
        # 优先准备较小的 任务，即使大型候选因截止时间中断，缓存中仍保留可用的完整模板。
        templates = {}
        for task in sorted(blocks, key=lambda t: (len(blocks[t]), t)):
            templates[task] = self.template(blocks[task], deadline, ordinals[task])
        metrics = self._replay(orders, pred, succ, templates, deadline)
        self.evaluations += 1
        metrics.update({
            "proxy": "cached_official_task_four_pipe_shared_ddr",
            "template_cache_hits": self.cache_hits,
            "template_cache_misses": self.cache_misses,
            "template_cache_entries": len(self.cache),
            "template_cached_nodes": self.cached_nodes,
            "template_prepare_seconds": self.prepare_seconds,
            "detailed_evaluations": self.evaluations,
        })
        return metrics

    def _replay(self, orders, pred, succ, templates, deadline):
        m = len(orders)
        owner = {task: core for core, order in enumerate(orders) for task in order}
        task_pending = {task: len(parents) for task, parents in pred.items()}
        release = dict.fromkeys(templates, 0)
        core_available = [0] * m
        core_cursor = [0] * m
        active = [None] * m
        op_pending = [None] * m
        op_left = [0] * m
        pipe_cursor = [[0] * 4 for _ in range(m)]
        # 每个执行槽保存连续算子索引和预计整数完成时刻。
        slots = [[None] * 4 for _ in range(m)]
        remaining = {}  # 以（核编号，流水线）为键，保存剩余 DDR 服务周期。
        starts, ends = {}, {}
        now = last_ddr_update = 0
        ddr_busy = 0
        events = 0

        def advance_ddr(time):
            nonlocal last_ddr_update, ddr_busy
            elapsed = float(time - last_ddr_update)
            if remaining:
                ddr_busy += time - last_ddr_update
            while elapsed > _EPS:
                live = [key for key, work in remaining.items() if work > _EPS]
                if not live:
                    break
                smallest = min(remaining[key] for key in live)
                finish = smallest * len(live)
                if finish >= elapsed - _EPS:
                    share = elapsed / len(live)
                    for key in live:
                        remaining[key] = max(0.0, remaining[key] - share)
                    break
                for key in live:
                    remaining[key] = max(0.0, remaining[key] - smallest)
                elapsed -= finish
            last_ddr_update = time

        def reschedule_ddr():
            ordered = sorted((max(0.0, work), key) for key, work in remaining.items())
            time, previous, count, index = float(now), 0.0, len(ordered), 0
            while index < len(ordered):
                work = ordered[index][0]
                time += (work - previous) * count
                stop = index
                while stop < len(ordered) and abs(ordered[stop][0] - work) <= _EPS:
                    core, pipe = ordered[stop][1]
                    slots[core][pipe][1] = int(math.ceil(time - _EPS))
                    stop += 1
                count -= stop - index
                previous, index = work, stop

        while len(ends) < len(templates):
            if events % 64 == 0:
                _check_deadline(deadline)
            events += 1
            advance_ddr(now)
            retired_ddr = False
            for core in range(m):
                task = active[core]
                if task is None:
                    continue
                template = templates[task]
                for pipe in range(4):
                    slot = slots[core][pipe]
                    if slot is None or slot[1] > now:
                        continue
                    op = slot[0]
                    slots[core][pipe] = None
                    pipe_cursor[core][pipe] += 1
                    op_left[core] -= 1
                    for child in template.successors[op]:
                        op_pending[core][child] -= 1
                    if (core, pipe) in remaining:
                        del remaining[core, pipe]
                        retired_ddr = True
                if op_left[core] == 0:
                    ends[task] = now
                    active[core] = None
                    core_cursor[core] += 1
                    core_available[core] = now + self.hardware.same_wait
                    for child in succ[task]:
                        task_pending[child] -= 1
                        lag = self.hardware.cross_wait if owner[child] != core else 0
                        release[child] = max(release[child], now + lag)
            if retired_ddr:
                reschedule_ddr()
            # 同一时刻先处理全部完成事件，再激活 任务 并发射算子。
            for core in range(m):
                if active[core] is not None or core_cursor[core] >= len(orders[core]):
                    continue
                task = orders[core][core_cursor[core]]
                if task_pending[task] or max(release[task], core_available[core]) > now:
                    continue
                active[core] = task
                starts[task] = now
                template = templates[task]
                op_pending[core] = list(template.indegree)
                op_left[core] = template.nodes
                pipe_cursor[core] = [0] * 4
            issued_ddr = False
            for core in range(m):
                task = active[core]
                if task is None:
                    continue
                template = templates[task]
                for pipe in range(4):
                    if slots[core][pipe] is not None:
                        continue
                    order = template.pipe_orders[pipe]
                    cursor = pipe_cursor[core][pipe]
                    if cursor >= len(order):
                        continue
                    op = order[cursor]
                    if op_pending[core][op]:
                        continue
                    slots[core][pipe] = [op, now + template.duration[op]]
                    if template.ddr[op]:
                        remaining[core, pipe] = float(template.duration[op])
                        issued_ddr = True
            if issued_ddr:
                reschedule_ddr()
            if len(ends) == len(templates):
                break
            next_time = min((slot[1] for row in slots for slot in row if slot is not None),
                            default=math.inf)
            for core in range(m):
                if active[core] is not None or core_cursor[core] >= len(orders[core]):
                    continue
                task = orders[core][core_cursor[core]]
                if task_pending[task] == 0:
                    ready = max(release[task], core_available[core])
                    if ready > now:
                        next_time = min(next_time, ready)
            if not math.isfinite(next_time) or next_time <= now:
                raise RuntimeError("Cached Task event simulation is deadlocked")
            now = next_time
        _check_deadline(deadline)
        durations = {task: ends[task] - starts[task] for task in templates}
        return {
            "score": now, "makespan": now,
            "copy_bytes": sum(t.copy_bytes for t in templates.values()),
            "boundary_copy_bytes": sum(t.boundary_copy_bytes for t in templates.values()),
            "spill_bytes": sum(t.spill_bytes for t in templates.values()),
            "memory_dependency_count": sum(t.memory_dependencies for t in templates.values()),
            "execution_ops": sum(t.nodes for t in templates.values()),
            "task_count": len(templates), "active_cores": sum(bool(order) for order in orders),
            "ddr_busy_cycles": ddr_busy, "event_count": events,
            "starts": starts, "ends": ends, "durations": durations,
            "core_ends": [ends[order[-1]] if order else 0 for order in orders],
        }
