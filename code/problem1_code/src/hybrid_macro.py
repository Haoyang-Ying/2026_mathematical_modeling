'为场景 A 构造保持结构合法的大范围破坏与重建操作。\n\n本模块只生成方案，接受与否由调用方的完整共享 DDR 评价决定，通常在短时 HC 修复后进行。区域取旧 任务 与核内顺序联合图的某个拓扑序中的连续区间；区域外 任务 成员和顺序保留。在每个核的对应切口插入重建结果，包括原区域没有 任务 的核。\n\n此处的小规模调度估计仅用于构造启发，不是最终评分。'

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator


@dataclass(frozen=True)
class Region:
    tasks: frozenset[int]
    ops: frozenset[int]
    prefixes: tuple[tuple[int, ...], ...]
    suffixes: tuple[tuple[int, ...], ...]
    topo: tuple[int, ...]  # 旧联合图的完整拓扑顺序。
    work_fraction: float
    first: int
    last: int  # 右端点不包含在区间内。
    reason: str = "topological_window"


def _check(deadline: float | None) -> None:
    if deadline is not None and perf_counter() >= deadline:
        raise TimeoutError("macro construction deadline reached")


def _blocks(plan):
    blocks = defaultdict(set)
    for op, task in plan["node_to_subgraph"].items():
        blocks[task].add(int(op))
    return {task: frozenset(ops) for task, ops in blocks.items()}


def _numbers(metrics, key):
    return {int(task): float(value) for task, value in metrics.get(key, {}).items()}


def _joint(model, plan, metrics, deadline):
    _check(deadline)
    blocks = _blocks(plan)
    pred, succ, _ = model.validate_partition(blocks)
    joint_succ = {task: set(children) for task, children in succ.items()}
    degree = {task: len(parents) for task, parents in pred.items()}
    for order in plan["core_schedules"]:
        for left, right in zip(order, order[1:]):
            if right not in joint_succ[left]:
                joint_succ[left].add(right)
                degree[right] += 1
    starts = _numbers(metrics, "starts")
    ready = [(starts.get(task, 0.0), task) for task in blocks if not degree[task]]
    heapq.heapify(ready)
    topo = []
    while ready:
        if len(topo) % 256 == 0:
            _check(deadline)
        _, task = heapq.heappop(ready)
        topo.append(task)
        for child in sorted(joint_succ[task]):
            degree[child] -= 1
            if degree[child] == 0:
                heapq.heappush(ready, (starts.get(child, 0.0), child))
    if len(topo) != len(blocks):
        raise ValueError("macro input Task dependencies and core orders contain a cycle")
    return blocks, pred, succ, tuple(topo)


def _region_from_interval(model, plan, topo, first, last, reason="topological_window"):
    '构造区域的内部辅助函数，也便于单独检查语义。'
    if not 0 <= first < last <= len(topo):
        raise ValueError("region must be a nonempty topological interval")
    rank = {task: index for index, task in enumerate(topo)}
    tasks = frozenset(topo[first:last])
    ops = frozenset(int(op) for op, task in plan["node_to_subgraph"].items()
                    if task in tasks)
    # 不按每核首末选中 任务 定位切口；某核可能没有被选 任务，但其前后缀插入切口仍可确定。
    prefixes = tuple(tuple(task for task in order if rank[task] < first)
                     for order in plan["core_schedules"])
    suffixes = tuple(tuple(task for task in order if rank[task] >= last)
                     for order in plan["core_schedules"])
    total = sum(max(1, model.cycles[op]) for op in model.ids)
    work = sum(max(1, model.cycles[op]) for op in ops)
    return Region(tasks, ops, prefixes, suffixes, topo, work / max(1, total),
                  first, last, reason)


def select_region(model, plan, metrics, round_index=0, deadline=None):
    '在等待、尾部或高计算量 任务 附近选择足够大的区间。\n\n目标计算量在每核平均工作量的一倍与两倍之间插值，上限为全图的 65%，使 2–5 核下保持三种不同规模。以完整旧 任务 为单位扩展，实际比例可能超过目标。超过截止时间抛出 TimeoutError；空图返回 None。'
    blocks, pred, succ, topo = _joint(model, plan, metrics, deadline)
    if not topo:
        return None
    m = len(plan["core_schedules"])
    starts, ends = _numbers(metrics, "starts"), _numbers(metrics, "ends")
    work = {task: sum(max(1, model.cycles[op]) for op in ops)
            for task, ops in blocks.items()}
    total = sum(work.values())
    low = min(0.65, 1.0 / max(1, m))
    high = min(0.65, 2.0 / max(1, m))
    target = total * (low + (round_index % 3) * (high - low) / 2)
    rank = {task: index for index, task in enumerate(topo)}
    anchors = []
    for core, order in enumerate(plan["core_schedules"]):
        previous_end = 0.0
        for task in order:
            gap = max(0.0, starts.get(task, previous_end) - previous_end)
            if gap:
                # 延迟前的 任务 可能比等待 任务 本身更适合修复，因此区间同时向两侧扩展。
                anchors.append((gap, rank[task], "idle_gap"))
            previous_end = ends.get(task, starts.get(task, previous_end))
        if order:
            anchors.append((previous_end / max(1, m), rank[order[-1]], "tail"))
    for task in topo:
        anchors.append((float(work[task]), rank[task], "heavy_work"))
    anchors.sort(key=lambda item: (-item[0], item[1], item[2]))
    # 按加权位置保持候选多样性，避免相近空闲间隙锚点让各轮重复重建同一个极小邻域。
    cumulative = [0]
    for task in topo:
        cumulative.append(cumulative[-1] + work[task])
    diverse = []
    for anchor in anchors:
        position = cumulative[anchor[1]] + work[topo[anchor[1]]] / 2
        if all(abs(position - pos) >= 0.2 * target for _, pos in diverse):
            diverse.append((anchor, position))
        if len(diverse) >= 8:
            break
    (_, center, reason), _ = diverse[(round_index // 3) % len(diverse)]
    first, last = center, center + 1
    selected_work = work[topo[center]]
    while selected_work < target and (first > 0 or last < len(topo)):
        _check(deadline)
        left_work = cumulative[center] - cumulative[first]
        right_work = cumulative[last] - cumulative[center + 1]
        # 尾部窗口向前扩展；其他窗口尽量均衡两侧计算量。
        take_left = first > 0 and (last == len(topo) or left_work <= right_work)
        if take_left:
            first -= 1
            selected_work += work[topo[first]]
        else:
            selected_work += work[topo[last]]
            last += 1
    return _region_from_interval(model, plan, topo, first, last, reason)


def _duration(model, members, task, metrics, preserve):
    durations = metrics.get("durations", {})
    if preserve:
        value = durations.get(task, durations.get(str(task)))
        if value is not None:
            return max(1.0, float(value))
    cost = model.cost(members)
    return max(1.0, float(cost.compute + cost.copy_cycles))


def _schedule_region(model, old_plan, metrics, region, blocks, selected,
                     mode, deadline, preserve):
    '区域内采用只向队尾追加的列表调度，保持全局顺序无环。'
    pred, succ, topo = model.validate_partition(blocks)
    selected = frozenset(selected)
    old_owner = {task: core for core, order in enumerate(old_plan["core_schedules"])
                 for task in order}
    ends = _numbers(metrics, "ends")
    same = float(model.hardware.same_wait)
    cross = float(model.hardware.cross_wait)
    m = len(region.prefixes)
    schedules = [[] for _ in range(m)]
    free = [ends.get(prefix[-1], 0.0) if prefix else 0.0
            for prefix in region.prefixes]
    duration = {}
    for index, task in enumerate(selected):
        if index % 64 == 0:
            _check(deadline)
        duration[task] = _duration(model, blocks[task], task, metrics, preserve)
    bottom = {}
    for task in reversed(topo):
        if task in selected:
            bottom[task] = duration[task] + max(
                (bottom[child] + min(same, cross)
                 for child in succ[task] if child in selected), default=0.0)
    degree = {task: len(pred[task] & selected) for task in selected}
    ready = [(-bottom[task], task) for task in selected if not degree[task]]
    heapq.heapify(ready)
    placed_owner = {}
    placed_ends = {}
    while ready:
        _check(deadline)
        _, task = heapq.heappop(ready)
        internal_parents = [p for p in pred[task] if p in selected]
        anchor = max(internal_parents, key=lambda p: (placed_ends[p], bottom[p], -p),
                     default=None)
        affinity = placed_owner.get(anchor)
        candidates = []
        for core in range(m):
            has_previous = bool(region.prefixes[core] or schedules[core])
            start = free[core] + (same if has_previous else 0.0)
            for parent in pred[task]:
                parent_core = placed_owner[parent] if parent in selected else old_owner[parent]
                parent_end = placed_ends[parent] if parent in selected else ends.get(parent, 0.0)
                start = max(start, parent_end + (cross if parent_core != core else 0.0))
            finish = start + duration[task]
            candidates.append((finish, start, core))
        best = min(candidates)
        if mode == "branch_affinity" and affinity is not None:
            anchored = candidates[affinity]
            # 有限的亲和偏好使链尽量连续，但不强制所有分支挤到主链所在核。
            allowance = 0.08 * duration[task] + min(same, cross)
            if anchored[0] <= best[0] + allowance:
                best = anchored
        finish, _, core = best
        schedules[core].append(task)
        free[core] = placed_ends[task] = finish
        placed_owner[task] = core
        for child in succ[task] & selected:
            degree[child] -= 1
            if degree[child] == 0:
                heapq.heappush(ready, (-bottom[child], child))
    if len(placed_owner) != len(selected):
        raise ValueError("regional replacement is cyclic")
    return {
        "node_to_subgraph": {str(op): task for task, ops in blocks.items() for op in sorted(ops)},
        "core_schedules": [list(prefix) + middle + list(suffix)
                           for prefix, middle, suffix in zip(region.prefixes, schedules, region.suffixes)],
    }


def _rebuild_blocks(model, blocks, region, m, mode, deadline):
    '将满足依赖的算子输出顺序切成连续分组。\n\n局部就绪遵循原 Model 边；原张量边界始终保留，因此评价器仍插入外部输入输出 COPY。在就绪分支前沿切分以保持并行性。'
    eligible = region.ops
    op_topo = [op for op in model.topo if op in eligible]
    degree = {op: sum(parent in eligible for parent in model.pred[op]) for op in op_topo}
    bottom = {}
    for op in reversed(op_topo):
        bottom[op] = max(1, model.cycles[op]) + max(
            (bottom[child] for child in model.succ[op] if child in eligible), default=0)
    ready = [(-bottom[op], op) for op in op_topo if not degree[op]]
    heapq.heapify(ready)
    available = {op for op in op_topo if not degree[op]}
    emitted = set()
    groups = []
    total = sum(max(1, model.cycles[op]) for op in eligible)
    target = max(total / max(1, (4 if mode == "parallel" else 2) * m),
                 min(total, 2 * total / max(1, len(eligible))))
    max_ops = 384 if mode == "parallel" else 768
    seal_work = target * (0.15 if mode == "parallel" else 0.4)
    while available:
        _check(deadline)
        while ready and ready[0][1] not in available:
            heapq.heappop(ready)
        _, seed = heapq.heappop(ready)
        local = [(-bottom[seed], seed)]
        group = set()
        work = 0
        while local and len(group) < max_ops:
            if len(group) % 64 == 0:
                _check(deadline)
            _, op = heapq.heappop(local)
            if op not in available:
                continue
            if group and work >= target:
                break
            available.remove(op)
            emitted.add(op)
            group.add(op)
            work += max(1, model.cycles[op])
            children = []
            for child in model.succ[op]:
                if child not in eligible:
                    continue
                degree[child] -= 1
                if degree[child] == 0:
                    available.add(child)
                    heapq.heappush(ready, (-bottom[child], child))
                    children.append(child)
            if len(children) > 1 and work >= seal_work:
                break
            for child in children:
                # 涉及更早分组的汇合点留在下一 任务 的入口，避免推迟当前整组的释放。
                if all(parent not in eligible or parent in group
                       for parent in model.pred[child]):
                    heapq.heappush(local, (-bottom[child], child))
            refill_work = target * (0.6 if mode == "parallel" else 1.0)
            if not local and work < refill_work and available:
                while ready and ready[0][1] not in available:
                    heapq.heappop(ready)
                if ready:
                    next_op = ready[0][1]
                    parents = [parent for parent in model.pred[next_op] if parent in eligible]
                    # 较小的独立链可共用 任务；当前链与更早 任务 的汇合仍单独分开。
                    mixed_join = any(parent in group for parent in parents) and any(
                        parent not in group for parent in parents)
                    if not mixed_join:
                        heapq.heappush(local, ready[0])
        groups.append(frozenset(group))
    if emitted != eligible:
        raise ValueError("regional Op reconstruction failed to cover the region")
    # 优先复用腾出的编号，区域外全部 任务 编号严格不变。
    ids = iter(sorted(region.tasks))
    next_id = max(blocks, default=-1) + 1
    new_blocks = {task: members for task, members in blocks.items() if task not in region.tasks}
    selected = []
    for group in groups:
        task = next(ids, None)
        if task is None:
            task, next_id = next_id, next_id + 1
        new_blocks[task] = group
        selected.append(task)
    return new_blocks, frozenset(selected)


def _changes(model, old_plan, new_plan, region):
    old_blocks, new_blocks = _blocks(old_plan), _blocks(new_plan)
    old_map = {int(op): task for op, task in old_plan["node_to_subgraph"].items()}
    new_map = {int(op): task for op, task in new_plan["node_to_subgraph"].items()}
    old_core = {task: core for core, order in enumerate(old_plan["core_schedules"]) for task in order}
    new_core = {task: core for core, order in enumerate(new_plan["core_schedules"]) for task in order}
    old_core_sequences = Counter(tuple(old_blocks[task] for task in order)
                                 for order in old_plan["core_schedules"])
    new_core_sequences = Counter(tuple(new_blocks[task] for task in order)
                                 for order in new_plan["core_schedules"])
    pure_core_permutation = old_core_sequences == new_core_sequences
    # 成员签名忽略 任务 重命名；前驱分组签名用于区分执行顺序变化，无需比较任意编号。
    old_prev, new_prev = {}, {}
    for plan, blocks, previous in ((old_plan, old_blocks, old_prev), (new_plan, new_blocks, new_prev)):
        for order in plan["core_schedules"]:
            before = None
            for task in order:
                previous[task] = before
                before = blocks[task]
    moved, regrouped, reordered = set(), set(), set()
    for op in region.ops:
        left, right = old_map[op], new_map[op]
        if old_core[left] != new_core[right]:
            moved.add(op)
        if old_blocks[left] != new_blocks[right]:
            regrouped.add(op)
        elif old_core[left] == new_core[right] and old_prev[left] != new_prev[right]:
            reordered.add(op)
    all_changed = moved | regrouped | reordered
    total_work = sum(max(1, model.cycles[op]) for op in model.ids)
    changed_work = sum(max(1, model.cycles[op]) for op in all_changed)
    selected_work = sum(max(1, model.cycles[op]) for op in region.ops)
    return {
        "region_tasks": len(region.tasks), "region_ops": len(region.ops),
        "region_work_fraction": region.work_fraction, "region_reason": region.reason,
        "core_changed_op_count": len(moved),
        "core_changed_work": sum(max(1, model.cycles[op]) for op in moved),
        "partition_changed": bool(regrouped), "partition_changed_ops": len(regrouped),
        "order_changed_ops": len(reordered), "changed_op_count": len(all_changed),
        "changed_work": changed_work, "changed_work_fraction": changed_work / max(1, total_work),
        "pure_core_permutation": pure_core_permutation and bool(moved),
        "macro_effective": (changed_work >= 0.1 * selected_work and bool(all_changed)
                            and not pure_core_permutation),
    }


def macro_candidates(model, plan, metrics, region, m, round_index=0, deadline=None
                     ) -> Iterator[tuple[str, dict[str, Any], dict[str, Any]]]:
    '惰性生成最多四个不同的大范围候选，每个均通过完整结构检查。\n\nA 类保留 任务 成员和编号，B 类改变区域 任务 边界。此处不接受候选，也不调用执行评价器。deadline 为 perf_counter 的绝对截止时刻。'
    if region is None:
        return
    if m != len(plan["core_schedules"]) or m != len(region.prefixes):
        raise ValueError("macro core count does not match plan and region")
    blocks = _blocks(plan)
    # 不同变体收敛到同一语义方案时，避免重复构造和评价；去重键也忽略仅为外观变化的 任务 重命名。
    seen = set()
    variants = [("A_critical_eft", "critical_eft", None),
                ("B_parallel", "critical_eft", "parallel"),
                ("A_branch_affinity", "branch_affinity", None),
                ("B_aggregate", "branch_affinity", "aggregate")]
    if round_index % 2:
        variants = [variants[2], variants[3], variants[0], variants[1]]
    for label, schedule_mode, rebuild in variants:
        _check(deadline)
        if rebuild:
            candidate_blocks, selected = _rebuild_blocks(model, blocks, region, m, rebuild, deadline)
        else:
            candidate_blocks, selected = blocks, region.tasks
        candidate = _schedule_region(model, plan, metrics, region, candidate_blocks,
                                     selected, schedule_mode, deadline, not rebuild)
        _check(deadline)
        model.validate_plan(candidate, m)
        key = tuple(tuple(candidate_blocks[task] for task in order)
                    for order in candidate["core_schedules"])
        if key in seen:
            continue
        seen.add(key)
        info = _changes(model, plan, candidate, region)
        if not info["macro_effective"]:
            continue
        info.update({"operator": label, "region_new_tasks": len(selected),
                     "construction_only": True})
        yield label, candidate, info
