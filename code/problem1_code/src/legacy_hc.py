"""竞争算法使用的 HC 局部代理与邻域公共函数。"""
from __future__ import annotations
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterator

def _members(plan: dict[str, Any]) -> dict[int, frozenset[int]]:
    """将 操作 -> 任务 转换为 任务 -> 操作 集合。"""
    groups: dict[int, set[int]] = defaultdict(set)
    for op, task in plan["node_to_subgraph"].items():
        groups[task].add(int(op))
    return {task: frozenset(ops) for task, ops in groups.items()}


@dataclass(slots=True)
class _ProxyState:
    """某个方案的近似 任务 时长及在 任务 DAG 上的估计完成时间。"""

    score: float
    makespan: float
    io_bytes: int
    durations: dict[int, float]
    ends: dict[int, float]
    dependencies: dict[tuple[int, int], int]
    slack: dict[int, float]


class _LocalProxy:
    """仅对改动的 任务 重算局部代价，再在线性规模 任务 DAG 上传播。"""

    def __init__(self, case: dict[str, Any], graph: Any, bandwidth: int,
                 cross_wait: int, same_wait: int, io_weight: float):
        self.graph = graph
        self.bandwidth = bandwidth
        self.cross_wait = cross_wait
        self.same_wait = same_wait
        self.io_weight = io_weight
        self.op_ids = tuple(sorted(graph.cycles))
        self.pipes = {op["id"]: op["pipe"] for op in case["ops"] if op["id"] in graph.cycles}
        self.op_cycles = {op: max(1, cycles) for op, cycles in graph.cycles.items()}
        self.offset: dict[int, float] = {}

        # 原始 张量 的边界 COPY 判定只依赖于当前 任务 的 操作 集合。
        op_by_id = {op["id"]: op for op in case["ops"]}
        tensor_by_id = {item["id"]: item for item in case["tensors"]}
        producers: dict[int, set[int]] = defaultdict(set)
        consumers: dict[int, set[int]] = defaultdict(set)
        for edge in case["edges"]:
            a, b = edge["source"], edge["target"]
            if a in op_by_id and b in tensor_by_id:
                producers[b].add(a)
            elif a in tensor_by_id and b in op_by_id:
                consumers[a].add(b)
        eligible = set(self.op_ids)
        self.tensors: dict[int, tuple[int, frozenset[int], frozenset[int], bool]] = {}
        self.touched: dict[int, set[int]] = {op: set() for op in eligible}
        self.pair_bytes: dict[tuple[int, int], int] = defaultdict(int)
        for tensor_id, tensor in tensor_by_id.items():
            prod = frozenset(producers[tensor_id] & eligible)
            cons = frozenset(consumers[tensor_id] & eligible)
            if not prod and not cons:
                continue
            has_copy_out = any(
                op_by_id[op]["op"] == "COPY_OUT" for op in consumers[tensor_id]
            )
            size = tensor["size"]
            self.tensors[tensor_id] = (size, prod, cons, has_copy_out)
            for op in prod | cons:
                self.touched[op].add(tensor_id)
            for a in prod:
                for b in cons:
                    if a != b:
                        self.pair_bytes[(a, b)] += size

        # 原始 操作 图只排序一次；评估单个 任务 时沿这个顺序求内部最长链。
        indegree = {op: len(graph.predecessors[op]) for op in self.op_ids}
        ready = deque(op for op in self.op_ids if indegree[op] == 0)
        order: list[int] = []
        while ready:
            op = ready.popleft()
            order.append(op)
            for nxt in graph.successors[op]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)
        if len(order) != len(self.op_ids):
            raise ValueError("计算 Op 图有环")
        self.op_order = tuple(order)
        self.op_rank = {op: index for index, op in enumerate(order)}

    @lru_cache(maxsize=256)
    def task_cost(self, members: frozenset[int]) -> tuple[float, int]:
        """估计单 任务 时长及其所有边界 COPY 字节数。"""
        if not members:
            return 0.0, 0
        workload: dict[str, int] = defaultdict(int)
        touched: set[int] = set()
        for op in members:
            workload[self.pipes[op]] += self.op_cycles[op]
            touched.update(self.touched[op])

        chain_end: dict[int, int] = {}
        for op in sorted(members, key=self.op_rank.__getitem__):
            chain_end[op] = self.op_cycles[op] + max(
                (chain_end[parent] for parent in self.graph.predecessors[op]
                 if parent in members), default=0,
            )
        io_bytes = 0
        for tensor in touched:
            size, producers, consumers, has_copy_out = self.tensors[tensor]
            local_prod = bool(producers & members)
            local_cons = bool(consumers & members)
            if local_cons and not local_prod:
                io_bytes += size
            if local_prod and (has_copy_out or not consumers or not consumers <= members):
                io_bytes += size
        # PIPE_SLOTS=1；内部链和各 流水线 累计工作量取较大者。
        # COPY 耗时在边界额外计入，后续官方时间线会修正这种近似的偏差。
        compute = max(max(workload.values(), default=0), max(chain_end.values(), default=0))
        return compute + io_bytes / self.bandwidth, io_bytes

    def calibrate(self, plan: dict[str, Any], result: dict[str, Any],
                  members: dict[int, frozenset[int]]) -> None:
        """用最近的官方结果校准现有 任务；未评估的候选沿用该偏移。"""
        actual = {
            entry["task_id"]: entry["duration"]
            for core in result["per_core_timeline"] for entry in core["tasks"]
        }
        self.offset = {
            task: actual[task] - self.task_cost(ops)[0]
            for task, ops in members.items()
        }

    def dependency_counts(self, plan: dict[str, Any]) -> dict[tuple[int, int], int]:
        """只在校准或建图时扫描原始 操作 边，记录跨 任务 边的重数。"""
        mapping = plan["node_to_subgraph"]
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for op, successors in self.graph.successors.items():
            source = mapping[str(op)]
            for nxt in successors:
                target = mapping[str(nxt)]
                if source != target:
                    counts[(source, target)] += 1
        return dict(counts)

    def _task_graph_times(self, plan: dict[str, Any], durations: dict[int, float],
                          dependencies: dict[tuple[int, int], int]):
        """任务 DAG 最长路：跨核前驱等待，同核连续 任务 顺序等待。"""
        schedules = plan["core_schedules"]
        task_core = {task: core for core, order in enumerate(schedules) for task in order}
        predecessors: dict[int, dict[int, int]] = {task: {} for task in task_core}
        for source, target in dependencies:
            delay = self.cross_wait if task_core[source] != task_core[target] else 0
            predecessors[target][source] = delay
        for order in schedules:
            for source, target in zip(order, order[1:]):
                predecessors[target][source] = max(
                    self.same_wait, predecessors[target].get(source, 0)
                )

        successors_by_task: dict[int, list[int]] = {task: [] for task in task_core}
        indegree = {task: len(parents) for task, parents in predecessors.items()}
        for target, parents in predecessors.items():
            for source in parents:
                successors_by_task[source].append(target)
        pending = deque(task for task, degree in indegree.items() if degree == 0)
        ends: dict[int, float] = {}
        while pending:
            task = pending.popleft()
            earliest = max((ends[prev] + delay for prev, delay in
                            predecessors[task].items()), default=0.0)
            ends[task] = earliest + durations[task]
            for nxt in successors_by_task[task]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    pending.append(nxt)
        if len(ends) != len(task_core):
            return None
        makespan = max(ends.values(), default=0.0)
        latest = dict.fromkeys(ends, makespan)
        for task in reversed(ends):
            for nxt in successors_by_task[task]:
                latest[task] = min(latest[task], latest[nxt] - durations[nxt]
                                   - predecessors[nxt][task])
        return ends, {task: max(0.0, latest[task] - end) for task, end in ends.items()}

    def state(self, plan: dict[str, Any],
              members: dict[int, frozenset[int]]) -> _ProxyState | None:
        """校准后重算当前方案的 任务 代价与全局关键路径。"""
        durations: dict[int, float] = {}
        io_total = 0
        for task, ops in members.items():
            local_duration, local_io = self.task_cost(ops)
            durations[task] = max(1.0, local_duration + self.offset.get(task, 0.0))
            io_total += local_io
        dependencies = self.dependency_counts(plan)
        times = self._task_graph_times(plan, durations, dependencies)
        if times is None:
            return None
        ends, slack = times
        makespan = max(ends.values(), default=0.0)
        return _ProxyState(makespan + self.io_weight * io_total / self.bandwidth,
                           makespan, io_total, durations, ends, dependencies, slack)

    def state_after_merge(
        self, plan: dict[str, Any], new_members: dict[int, frozenset[int]],
        current: _ProxyState, old_members: dict[int, frozenset[int]],
        source: int, target: int,
    ) -> _ProxyState | None:
        """仅计算源和目标 任务 的局部变化，随后更新全局关键路径。"""
        durations = dict(current.durations)
        io_total = current.io_bytes
        for task in (source, target):
            io_total -= self.task_cost(old_members[task])[1]
            if task not in new_members:
                durations.pop(task)
                continue
            local_duration, new_io = self.task_cost(new_members[task])
            io_total += new_io
            durations[task] = max(1.0, local_duration + self.offset.get(task, 0.0))
        # 只更新被移动 操作 的关联边；其余 操作 边不再逐候选扫描。
        group = old_members[source] - new_members.get(source, frozenset())
        mapping = plan["node_to_subgraph"]
        dependencies = dict(current.dependencies)

        def replace_edge(a: int, b: int) -> None:
            new_a, new_b = mapping[str(a)], mapping[str(b)]
            old_a = source if a in group else new_a
            old_b = source if b in group else new_b
            if old_a != old_b:
                key = (old_a, old_b)
                count = dependencies[key] - 1
                if count:
                    dependencies[key] = count
                else:
                    dependencies.pop(key)
            if new_a != new_b:
                key = (new_a, new_b)
                dependencies[key] = dependencies.get(key, 0) + 1

        for op in group:
            for nxt in self.graph.successors[op]:
                replace_edge(op, nxt)
            for prev in self.graph.predecessors[op]:
                if prev not in group:
                    replace_edge(prev, op)
        times = self._task_graph_times(plan, durations, dependencies)
        if times is None:
            return None
        ends, slack = times
        makespan = max(ends.values(), default=0.0)
        return _ProxyState(makespan + self.io_weight * io_total / self.bandwidth,
                           makespan, io_total, durations, ends, dependencies, slack)


def _pairs(plan: dict[str, Any], graph: Any, state: _ProxyState,
           limit: int = 200, offset: int = 0) -> list[tuple[int, int]]:
    """只允许相邻 任务 或有直接计算依赖的 任务 互相合并。"""
    pairs: set[tuple[int, int]] = set()
    for source, target in state.dependencies:
        pairs.add((source, target))
        pairs.add((target, source))
    for order in plan["core_schedules"]:
        for a, b in zip(order, order[1:]):
            pairs.add((a, b))
            pairs.add((b, a))
    # 七成名额关注关键链，三成关注依赖密集处，避免一直只看尾部 任务。
    critical = sorted(pairs, key=lambda pair: (
        min(state.slack[pair[0]], state.slack[pair[1]]),
        -max(state.durations[pair[0]], state.durations[pair[1]]), pair))
    affinity = sorted(pairs, key=lambda pair: (
        -(state.dependencies.get(pair, 0) + state.dependencies.get(pair[::-1], 0)),
        min(state.slack[pair[0]], state.slack[pair[1]]), pair))
    chosen = {}
    i = j = 0
    while i < len(critical) or j < len(affinity):
        for pair in critical[i:i + 7] + affinity[j:j + 3]:
            chosen[pair] = None
        i += 7
        j += 3
    return list(chosen)[offset:offset + limit]


def _batch_groups(source: int, target: int,
                  members: dict[int, frozenset[int]], proxy: _LocalProxy,
                  max_batch_ops: int) -> Iterator[tuple[frozenset[int], str]]:
    """优先尝试整个 任务，再试最多 max_batch_ops 个边界 操作 的批量合并。"""
    source_ops = members[source]
    target_ops = members[target]
    yield source_ops, "整 Task"
    boundary = [
        op for op in source_ops
        if any(neighbor in target_ops for neighbor in
               (*proxy.graph.predecessors[op], *proxy.graph.successors[op]))
    ]
    if not boundary:
        return

    def affinity(op: int) -> int:
        return sum(
            proxy.pair_bytes.get((op, other), 0)
            + proxy.pair_bytes.get((other, op), 0)
            for other in (*proxy.graph.predecessors[op], *proxy.graph.successors[op])
            if other in target_ops
        )

    boundary.sort(key=lambda op: (-affinity(op), -proxy.op_cycles[op], op))
    length = min(max_batch_ops, len(boundary))
    sizes = sorted({length, max(2, length // 2), max(2, length // 4), 1}, reverse=True)
    for count in sizes:
        if count <= length:
            group = frozenset(boundary[:count])
            if group != source_ops:
                yield group, "边界批量" if count > 1 else "单 Op"


def _merge(
    plan: dict[str, Any], members: dict[int, frozenset[int]],
    source: int, target: int, group: frozenset[int],
) -> tuple[dict[str, Any], dict[int, frozenset[int]]]:
    """复制方案，将 group 从源 任务 一次性并入目标 任务。"""
    mapping = dict(plan["node_to_subgraph"])
    schedules = [list(order) for order in plan["core_schedules"]]
    new_members = dict(members)
    new_members[target] = members[target] | group
    remainder = members[source] - group
    if remainder:
        new_members[source] = remainder
    else:
        new_members.pop(source)
        for order in schedules:
            if source in order:
                order.remove(source)
                break
    for op in group:
        mapping[str(op)] = target
    return {"node_to_subgraph": mapping, "core_schedules": schedules}, new_members

