'场景 A 的 任务 调度与轻量级共享 DDR 代理评价。\n\n列表调度器使用 任务 独立耗时进行低成本的最早完成时间分配，再用共享资源模型评价固定核内顺序：每个 任务 依次读取、计算、写回，所有活跃读写阶段均分一个 DDR 通道；DDR 忙碌时不同核仍可计算。标称读写服务时间保留逐张量取整规则。\n\n该模型将每个 任务 聚合处理，不复现官方算子发射顺序、任务 内搬运与计算重叠、缓存重读、换出插入或内存复用依赖。score 是待最小化的近似完工周期，不是赛题官方评分。'

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any


_EPS = 1.0e-7


def _selected_cores(m: int, active_cores: int | Sequence[int] | None) -> tuple[int, ...]:
    if type(m) is not int or m < 1:
        raise ValueError("m must be a positive integer")
    if active_cores is None:
        return tuple(range(m))
    if type(active_cores) is int:
        if not 1 <= active_cores <= m:
            raise ValueError("active_cores must lie between 1 and m")
        return tuple(range(active_cores))
    selected = tuple(active_cores)
    if (not selected or len(set(selected)) != len(selected)
            or any(type(core) is not int or not 0 <= core < m for core in selected)):
        raise ValueError("active_cores must contain distinct valid core indices")
    return tuple(sorted(selected))


def _phases(cost: Any, bandwidth: float) -> tuple[float, float, float]:
    '计算独占 DDR 通道时的读取、计算和写回耗时。'
    read = float(getattr(cost, "read_cycles", cost.read_bytes / bandwidth))
    write = float(getattr(cost, "write_cycles", cost.write_bytes / bandwidth))
    compute = float(cost.compute)
    if any(not math.isfinite(value) or value < 0 for value in (read, compute, write)):
        raise ValueError("Task proxy costs must be finite and nonnegative")
    return read, compute, write


def _plan(blocks: Mapping[int, frozenset[int]], schedules: list[list[int]]) -> dict[str, Any]:
    return {
        "node_to_subgraph": {
            str(op): task for task, members in blocks.items() for op in sorted(members)
        },
        "core_schedules": schedules,
    }


def _checked_orders(
    blocks: Mapping[int, frozenset[int]],
    plan: Mapping[str, Any] | Sequence[Sequence[int]],
    pred: Mapping[int, set[int]],
    succ: Mapping[int, set[int]],
) -> tuple[list[list[int]], dict[int, int], dict[int, int], tuple[int, ...]]:
    '加入核内顺序边后，检查 任务 恰好覆盖一次及全图无环。'
    if isinstance(plan, Mapping):
        if set(plan) != {"node_to_subgraph", "core_schedules"}:
            raise ValueError("plan must contain node_to_subgraph and core_schedules")
        expected = {str(op): task for task, members in blocks.items() for op in members}
        if plan["node_to_subgraph"] != expected:
            raise ValueError("plan mapping does not agree with the supplied partition")
        raw_orders = plan["core_schedules"]
    else:
        raw_orders = plan
    if (not isinstance(raw_orders, Sequence) or isinstance(raw_orders, (str, bytes))
            or not raw_orders):
        raise ValueError("core_schedules must contain at least one core")
    orders = []
    for order in raw_orders:
        if not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
            raise ValueError("each core order must be a sequence")
        orders.append(list(order))
    tasks = [task for order in orders for task in order]
    if (any(type(task) is not int for task in tasks)
            or len(tasks) != len(set(tasks)) or set(tasks) != set(blocks)):
        raise ValueError("each Task must occur exactly once in the core schedules")
    owner = {task: core for core, order in enumerate(orders) for task in order}
    previous = {right: left for order in orders for left, right in zip(order, order[1:])}
    aug_succ = {task: set(children) for task, children in succ.items()}
    indegree = {task: len(parents) for task, parents in pred.items()}
    for right, left in previous.items():
        if right not in aug_succ[left]:
            aug_succ[left].add(right)
            indegree[right] += 1
    queue = deque(sorted(task for task, degree in indegree.items() if degree == 0))
    topo = []
    while queue:
        task = queue.popleft()
        topo.append(task)
        for child in sorted(aug_succ[task]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(topo) != len(blocks):
        raise ValueError("Task dependencies and core orders together contain a cycle")
    return orders, owner, previous, tuple(topo)


def list_schedule(
    model: Any,
    blocks: Mapping[int, frozenset[int]],
    m: int,
    active_cores: int | Sequence[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    '根据底部层级和预计最早完成时间分配就绪 任务。\n\n按一个全局 任务 拓扑顺序仅向各核队尾追加，因此与真实依赖合并后仍无环。active_cores=k 使用前 k 个核，但仍输出 m 个核列表；也可传入核编号序列。不同 任务 放在同一核不会获得额外亲和收益，因为场景 A 仍计算它们的边界 COPY。'
    selected = _selected_cores(m, active_cores)
    pred, succ, topo = model.validate_partition(blocks)
    bandwidth = float(model.hardware.bandwidth)
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be positive and finite")
    costs = {task: model.cost(frozenset(members)) for task, members in blocks.items()}
    phases = {task: _phases(cost, bandwidth) for task, cost in costs.items()}
    duration = {task: sum(value) for task, value in phases.items()}
    same_wait = float(model.hardware.same_wait)
    cross_wait = float(model.hardware.cross_wait)
    min_wait = same_wait if len(selected) == 1 else min(same_wait, cross_wait)
    bottom = {}
    for task in reversed(topo):
        bottom[task] = duration[task] + max(
            (min_wait + bottom[child] for child in succ[task]), default=0.0
        )
    indegree = {task: len(parents) for task, parents in pred.items()}
    ready = [(-bottom[task], task) for task in topo if indegree[task] == 0]
    heapq.heapify(ready)
    schedules: list[list[int]] = [[] for _ in range(m)]
    free = [0.0] * m
    owner: dict[int, int] = {}
    ends: dict[int, float] = {}
    while ready:
        _, task = heapq.heappop(ready)
        best = None
        for core in selected:
            # 同核相邻 任务 即使独立，也需要计入 same_wait。
            start = free[core] + (same_wait if schedules[core] else 0.0)
            for parent in pred[task]:
                lag = cross_wait if owner[parent] != core else 0.0
                start = max(start, ends[parent] + lag)
            candidate = (start + duration[task], start, core)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        finish, _, core = best
        schedules[core].append(task)
        free[core] = ends[task] = finish
        owner[task] = core
        for child in succ[task]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, (-bottom[child], child))
    if len(owner) != len(blocks):
        raise ValueError("partition has cyclic dependencies")
    plan = _plan(blocks, schedules)
    metrics = _evaluate(model, blocks, plan, pred, succ, costs, phases)
    metrics["construction_makespan"] = max(ends.values(), default=0.0)
    metrics["requested_active_cores"] = len(selected)
    return plan, metrics


def evaluate_plan(
    model: Any,
    blocks: Mapping[int, frozenset[int]],
    plan: Mapping[str, Any] | Sequence[Sequence[int]],
) -> dict[str, Any]:
    '使用共享 DDR 模型评价固定 任务 核分配，不调用官方完整评价器。\n\n输入可为官方格式方案或仅 core_schedules；检查分组、任务 完整覆盖及依赖边与核内顺序边合并后的无环性。'
    pred, succ, _ = model.validate_partition(blocks)
    bandwidth = float(model.hardware.bandwidth)
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be positive and finite")
    costs = {task: model.cost(frozenset(members)) for task, members in blocks.items()}
    phases = {task: _phases(cost, bandwidth) for task, cost in costs.items()}
    return _evaluate(model, blocks, plan, pred, succ, costs, phases)


def _evaluate(
    model: Any,
    blocks: Mapping[int, frozenset[int]],
    plan: Mapping[str, Any] | Sequence[Sequence[int]],
    pred: Mapping[int, set[int]],
    succ: Mapping[int, set[int]],
    costs: Mapping[int, Any],
    phases: Mapping[int, tuple[float, float, float]],
) -> dict[str, Any]:
    orders, owner, previous, topo = _checked_orders(blocks, plan, pred, succ)
    m = len(orders)
    same_wait = float(model.hardware.same_wait)
    cross_wait = float(model.hardware.cross_wait)
    if any(not math.isfinite(value) or value < 0 for value in (same_wait, cross_wait)):
        raise ValueError("Task waits must be finite and nonnegative")
    nominal = {task: sum(value) for task, value in phases.items()}
    unshared_ends = {}
    for task in topo:
        start = unshared_ends[previous[task]] + same_wait if task in previous else 0.0
        for parent in pred[task]:
            lag = cross_wait if owner[parent] != owner[task] else 0.0
            start = max(start, unshared_ends[parent] + lag)
        unshared_ends[task] = start + nominal[task]

    # 每核最多一个活跃 任务，因此每个事件最多扫描 m 项，无需逐周期模拟；复杂度为 O(m × 任务 数 + 边数)。
    cursor = [0] * m
    current: list[int | None] = [None] * m
    stage = [-1] * m                 # 阶段编码：-1 空闲，0 读取，1 计算，2 写回。
    remaining = [0.0] * m            # 以独占 DDR 或单个计算核为基准的服务周期。
    core_available = [0.0] * m
    pending = {task: len(parents) for task, parents in pred.items()}
    dep_release = dict.fromkeys(blocks, 0.0)
    starts: dict[int, float] = {}
    ends: dict[int, float] = {}
    now = 0.0
    ddr_busy = 0.0
    peak_ddr_flows = 0

    while len(ends) != len(blocks):
        # 先完成零时长或已结束阶段，传播释放状态并启动当前时刻所有可执行 任务，再选择严格更晚的下一个事件。
        changed = True
        while changed:
            changed = False
            for core in range(m):
                task = current[core]
                if task is not None and remaining[core] <= _EPS:
                    changed = True
                    stage[core] += 1
                    if stage[core] < 3:
                        remaining[core] = phases[task][stage[core]]
                    else:
                        ends[task] = now
                        current[core] = None
                        stage[core] = -1
                        cursor[core] += 1
                        core_available[core] = now + same_wait
                        for child in succ[task]:
                            pending[child] -= 1
                            lag = cross_wait if owner[child] != core else 0.0
                            dep_release[child] = max(dep_release[child], now + lag)
                if current[core] is None and cursor[core] < len(orders[core]):
                    task = orders[core][cursor[core]]
                    release = max(core_available[core], dep_release[task])
                    if pending[task] == 0 and release <= now + _EPS:
                        starts[task] = now
                        current[core] = task
                        stage[core] = 0
                        remaining[core] = phases[task][0]
                        changed = True
        if len(ends) == len(blocks):
            break

        flows = sum(phase in (0, 2) for phase in stage)
        peak_ddr_flows = max(peak_ddr_flows, flows)
        dt = math.inf
        for core in range(m):
            if stage[core] == 1:
                dt = min(dt, remaining[core])
            elif stage[core] in (0, 2):
                dt = min(dt, remaining[core] * flows)
            elif cursor[core] < len(orders[core]):
                task = orders[core][cursor[core]]
                if pending[task] == 0:
                    dt = min(dt, max(core_available[core], dep_release[task]) - now)
        if not math.isfinite(dt) or dt <= 0:
            raise RuntimeError("proxy event simulation could not advance")
        if flows:
            ddr_busy += dt
        for core in range(m):
            if stage[core] == 1:
                remaining[core] -= dt
            elif stage[core] in (0, 2):
                remaining[core] -= dt / flows
        now += dt

    durations = {task: ends[task] - starts[task] for task in blocks}
    copy_cycles = sum(value[0] + value[2] for value in phases.values())
    unshared_makespan = max(unshared_ends.values(), default=0.0)
    return {
        "proxy": "task_read_compute_write_shared_ddr",
        "score": now,
        "makespan": now,
        "unshared_makespan": unshared_makespan,
        # 此界仅适用于聚合代理模型，不是官方执行时间的界。
        "proxy_lower_bound": max(unshared_makespan, copy_cycles),
        "copy_bytes": sum(cost.copy_bytes for cost in costs.values()),
        "copy_cycles": copy_cycles,
        "compute_cycles": sum(value[1] for value in phases.values()),
        "ddr_busy_cycles": ddr_busy,
        "ddr_busy_fraction": ddr_busy / now if now else 0.0,
        "peak_ddr_flows": peak_ddr_flows,
        "task_count": len(blocks),
        "active_cores": sum(bool(order) for order in orders),
        "starts": starts,
        "ends": ends,
        "durations": durations,
        "nominal_durations": nominal,
        "core_ends": [ends[order[-1]] if order else 0.0 for order in orders],
        "core_busy": [sum(durations[task] for task in order) for order in orders],
    }
