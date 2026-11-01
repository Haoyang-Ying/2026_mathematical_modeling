# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Huawei Technologies Co., Ltd.
# Copyright 2026 Haoyang-Ying
# Upstream author: Pal Andras Papp.
# Modified implementation: Python adaptation for the contest task model,
# with local scheduling extensions. See NOTICE for sources and changes.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""适配场景 B 物理核心边界的 BSPg 基础实现。

输入原始 case JSON 和核数 m，输出严格包含 node_to_subgraph 与
core_schedules 的方案。只调度非 COPY 计算 操作；原始 COPY 与 张量
用于恢复计算依赖。虚拟执行时间仍采用 操作 的 cycles；分叉决策使用
有界局部估计，最终结果应交由场景 B 的完整评价器计算。
局部分支的组内归属优先于普通 ready_p 规则。

算法仍称 BSPg；选点分数增加当前 任务 的共享外部输入复用字节数：
  score(o,p) = sum(c(u)/deg+(u) * I(u,p), u in Pred(o))
               + sum(size(t), t in external_inputs[o] & tensor_p[p])
第二项仅在 o 能合法并入当前 任务 时有效。tensor_p 记录计划使用的输入，
不表示实际驻留；新建 任务 时重置，不随虚拟阶段重置。未模拟释放和溢出。
接口与方案格式沿用母本；本文件只生成初步贪心方案，不运行 HC/MGR。

参考：Papp 等，《逐步贴近实际模型的高效多处理器调度》，算法 1 及作者公开的 GreedyBSP 实现。
"""

from __future__ import annotations

import argparse
import collections
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from q2_config import DEFAULT_ESCAPE_WEIGHT

COPY_TYPES = frozenset({"COPY_IN", "COPY_OUT"})


def exit_penalty(weight, delay, pending_bytes, bandwidth):
    '以周期表示的额外启发式损失，不计入官方总完工时间。pending_bytes 是连续性估计量，不是实际换出量预测。'
    return weight * (delay + pending_bytes / bandwidth)


@dataclass(slots=True)
class _OpGraph:
    """一次建图后供整个贪心过程复用的计算图和静态量。"""

    cycles: dict[int, int]
    predecessors: dict[int, tuple[int, ...]]
    successors: dict[int, tuple[int, ...]]
    c: dict[int, int]
    deg_plus: dict[int, int]
    external_inputs: dict[int, frozenset[int]]
    external_sizes: dict[int, int]
    external_consumers: dict[int, tuple[int, ...]]


def _build_op_graph(case: dict[str, Any]) -> _OpGraph:
    """跳过 张量 和原始 COPY，建立非 COPY 操作 间的直接依赖。"""
    ops = case["ops"]
    tensors = case["tensors"]
    edges = case["edges"]
    op_by_id = {op["id"]: op for op in ops}
    tensor_by_id = {tensor["id"]: tensor for tensor in tensors}
    if len(op_by_id) != len(ops) or len(tensor_by_id) != len(tensors):
        raise ValueError("输入图有重复的 Op ID 或 Tensor ID")
    if op_by_id.keys() & tensor_by_id.keys():
        raise ValueError("Op 与 Tensor 的 ID 不可重复")

    cycles: dict[int, int] = {}
    for op in ops:
        if op["op"] in COPY_TYPES:
            continue
        duration = op["cycles"]
        if type(duration) is not int or duration < 0:
            raise ValueError(f"计算 Op {op['id']} 的 cycles 必须是非负整数")
        cycles[op["id"]] = duration

    # 第一项静态量：c(o) 是 o 直接产生的不同 张量 的总字节数。
    output_tensors: dict[int, set[int]] = {op_id: set() for op_id in cycles}
    all_ids = op_by_id.keys() | tensor_by_id.keys()
    full_successors: dict[int, list[int]] = {node_id: [] for node_id in all_ids}
    full_indegree: dict[int, int] = dict.fromkeys(all_ids, 0)
    for edge in edges:
        source, target = edge["source"], edge["target"]
        if source not in all_ids or target not in all_ids:
            raise ValueError(f"原图的边引用了未知节点：{source} -> {target}")
        full_successors[source].append(target)
        full_indegree[target] += 1
        if source in cycles and target in tensor_by_id:
            output_tensors[source].add(target)
    c = {
        op_id: sum(tensor_by_id[tensor_id]["size"] for tensor_id in tids)
        for op_id, tids in output_tensors.items()
    }

    # 检查原图有向无环，避免穿过 COPY/张量 时遇到隐藏的环。
    from collections import deque

    pending = deque(node for node, degree in full_indegree.items() if degree == 0)
    visited = 0
    # 有计算祖先的 COPY_IN 是中间数据搬运，不能算作外部输入。
    has_compute_ancestor: set[int] = set()
    while pending:
        node = pending.popleft()
        visited += 1
        for next_node in full_successors[node]:
            if node in cycles or node in has_compute_ancestor:
                has_compute_ancestor.add(next_node)
            full_indegree[next_node] -= 1
            if full_indegree[next_node] == 0:
                pending.append(next_node)
    if visited != len(full_successors):
        raise ValueError("原始计算图存在有向环")

    # 仅索引 COPY_IN 直接输出、且直接供计算 操作 使用的外部 张量。
    # 使用 张量 ID 去重；大小相同的两个 张量 仍是不同输入。
    external_tensors = {
        tensor_id
        for op_id, op in op_by_id.items() if op["op"] == "COPY_IN"
        for tensor_id in full_successors[op_id]
        if tensor_id in tensor_by_id and tensor_id not in has_compute_ancestor
    }
    external_inputs: dict[int, set[int]] = {op_id: set() for op_id in cycles}
    external_consumers: dict[int, tuple[int, ...]] = {}
    for tensor_id in sorted(external_tensors):
        consumers = tuple(sorted(set(full_successors[tensor_id]) & cycles.keys()))
        external_consumers[tensor_id] = consumers
        for op_id in consumers:
            external_inputs[op_id].add(tensor_id)

    # 从每个计算 操作 穿过中间节点，碰到下一个计算 操作 就停止。
    successor_sets: dict[int, set[int]] = {op_id: set() for op_id in cycles}
    predecessor_sets: dict[int, set[int]] = {op_id: set() for op_id in cycles}
    for op_id in cycles:
        stack = list(full_successors[op_id])
        visited_middle: set[int] = set()
        while stack:
            node = stack.pop()
            if node in cycles:
                if node != op_id:
                    successor_sets[op_id].add(node)
                    predecessor_sets[node].add(op_id)
                continue
            if node not in visited_middle:
                visited_middle.add(node)
                stack.extend(full_successors[node])

    predecessors = {
        op_id: tuple(sorted(neighbors))
        for op_id, neighbors in predecessor_sets.items()
    }
    successors = {
        op_id: tuple(sorted(neighbors))
        for op_id, neighbors in successor_sets.items()
    }

    # 第二项静态量：deg⁺(o) 是直接后继计算 操作 的数量。
    # 终点 操作 的出度为 0；它不会成为分数求和中的前驱。
    deg_plus = {op_id: len(neighbors) for op_id, neighbors in successors.items()}
    return _OpGraph(
        cycles, predecessors, successors, c, deg_plus,
        {op_id: frozenset(tids) for op_id, tids in external_inputs.items()},
        {tensor_id: tensor_by_id[tensor_id]["size"] for tensor_id in external_tensors},
        external_consumers,
    )


def compute_c_and_deg_plus(
    case: dict[str, Any],
) -> tuple[dict[int, int], dict[int, int]]:
    """返回按 操作 ID 索引的 c(o) 字节数和 deg⁺(o) 直接出度。"""
    graph = _build_op_graph(case)
    return graph.c, graph.deg_plus


def _check_plan(graph: _OpGraph, plan: dict[str, Any], m: int) -> None:
    """校验节点覆盖、子图依赖和同核 任务 顺序组成的图无环。"""
    if set(plan) != {"node_to_subgraph", "core_schedules"}:
        raise ValueError("方案只能包含 node_to_subgraph 与 core_schedules")
    mapping = plan["node_to_subgraph"]
    schedules = plan["core_schedules"]
    if not isinstance(mapping, dict) or not isinstance(schedules, list):
        raise ValueError("方案字段类型错误")
    if len(schedules) != m or any(not isinstance(order, list) for order in schedules):
        raise ValueError("核心数量或 core_schedules 格式错误")
    if set(mapping) != {str(op_id) for op_id in graph.cycles}:
        raise ValueError("node_to_subgraph 没有恰好覆盖所有非 COPY Op")
    if any(type(sgid) is not int or sgid < 0 for sgid in mapping.values()):
        raise ValueError("子图 ID 必须是非负整数")

    tasks = [sgid for order in schedules for sgid in order]
    if (any(type(sgid) is not int for sgid in tasks)
            or len(tasks) != len(set(tasks))
            or set(tasks) != set(mapping.values())):
        raise ValueError("每个子图必须恰好出现于一个核的调度序列")

    # 同核连续 任务 的顺序也作为有向边加入，然后统一进行拓扑检查。
    task_successors = {sgid: set() for sgid in tasks}
    task_indegree = {sgid: 0 for sgid in tasks}

    def add_edge(source: int, target: int) -> None:
        if source == target or target in task_successors[source]:
            return
        task_successors[source].add(target)
        task_indegree[target] += 1

    for source, neighbors in graph.successors.items():
        source_task = mapping[str(source)]
        for target in neighbors:
            add_edge(source_task, mapping[str(target)])
    for order in schedules:
        for source, target in zip(order, order[1:]):
            add_edge(source, target)

    pending = [task for task, degree in task_indegree.items() if degree == 0]
    visited = 0
    while pending:
        task = pending.pop()
        visited += 1
        for target in task_successors[task]:
            task_indegree[target] -= 1
            if task_indegree[target] == 0:
                pending.append(target)
    if visited != len(tasks):
        raise ValueError("子图依赖与同核 Task 串行顺序组成有向环")


def validate_bspg_plan(case: dict[str, Any], plan: dict[str, Any]) -> bool:
    """独立验证现有方案；返回 True，否则抛出 ValueError。"""
    schedules = plan.get("core_schedules")
    if not isinstance(schedules, list) or not 1 <= len(schedules) <= 5:
        raise ValueError("core_schedules 必须包含 1 至 5 个核心")
    _check_plan(_build_op_graph(case), plan, len(schedules))
    return True


class _ForkPlanner:
    """有界搜索共同后继，按 任务 的粗略完工时间比较成组外移。"""

    def __init__(
        self, case: dict[str, Any], graph: _OpGraph, *,
        bandwidth: int, same_wait: int, cross_wait: int,
        max_nodes: int, max_candidates: int, escape_weight: float = DEFAULT_ESCAPE_WEIGHT,
    ) -> None:
        self.graph = graph
        self.bandwidth = bandwidth
        self.same_wait = same_wait
        self.cross_wait = cross_wait
        self.max_nodes = max_nodes
        self.max_candidates = max_candidates
        self.escape_weight = escape_weight
        self.penalty_rejections = 0
        self.pipe = {
            op["id"]: op["pipe"] for op in case["ops"] if op["id"] in graph.cycles
        }
        self.tensors = {t["id"]: t["size"] for t in case["tensors"]}
        self.tensor_source: dict[int, int] = {}
        self.inputs: dict[int, list[int]] = collections.defaultdict(list)
        for edge in case["edges"]:
            a, b = edge["source"], edge["target"]
            if b in self.tensors:
                self.tensor_source[b] = a
            elif a in self.tensors and b in graph.cycles:
                self.inputs[b].append(a)

        degree = {u: len(parents) for u, parents in graph.predecessors.items()}
        queue = collections.deque(u for u, count in degree.items() if count == 0)
        order = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in graph.successors[u]:
                degree[v] -= 1
                if degree[v] == 0:
                    queue.append(v)
        if len(order) != len(graph.cycles):
            raise ValueError("计算 Op 图有环")
        self.rank = {u: i for i, u in enumerate(order)}

    def _region(self, fork: int):
        """拓扑上首个所有直接后继均可到达的节点及其前驱锥。"""
        children = self.graph.successors[fork]
        all_bits = (1 << len(children)) - 1
        mask: dict[int, int] = {}
        pending: list[tuple[int, int]] = []
        queued = set()
        for i, child in enumerate(children):
            mask[child] = mask.get(child, 0) | (1 << i)
            if child not in queued:
                queued.add(child)
                heapq.heappush(pending, (self.rank[child], child))
        join = None
        examined = 0
        while pending and examined < self.max_nodes:
            _, u = heapq.heappop(pending)
            examined += 1
            if mask[u] == all_bits:
                join = u
                break
            for v in self.graph.successors[u]:
                mask[v] = mask.get(v, 0) | mask[u]
                if v not in queued:
                    queued.add(v)
                    heapq.heappush(pending, (self.rank[v], v))
        if join is None:
            return None
        region = {join}
        stack = [join]
        while stack:
            u = stack.pop()
            for parent in self.graph.predecessors[u]:
                if parent != fork and parent in mask and parent not in region:
                    region.add(parent)
                    stack.append(parent)
        if not set(children) <= region:
            return None
        # 外部计算前驱会使候选 任务 的开始时刻未知；保留原贪心处理。
        if any(
            parent != fork and parent not in region
            for u in region for parent in self.graph.predecessors[u]
        ):
            return None
        return children, join, sorted(region - {join}, key=self.rank.get), mask

    @staticmethod
    def _groups(children, before, mask):
        """尽早相遇的两支组成对子；各对子/未配对支路为原子组。"""
        paired = set()
        groups = []
        for u in before:
            bits = mask[u]
            if bits.bit_count() != 2:
                continue
            i = (bits & -bits).bit_length() - 1
            k = (bits ^ (1 << i)).bit_length() - 1
            if i not in paired and k not in paired:
                groups.append((i, k))
                paired.update((i, k))
        groups.extend((i,) for i in range(len(children)) if i not in paired)
        return sorted(groups, key=lambda group: group[0])

    def _work(self, nodes, tags):
        pipe_work: dict[object, dict[str, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )
        for u in nodes:
            pipe_work[tags[u]][self.pipe[u]] += self.graph.cycles[u]
        return {tag: max(v.values()) for tag, v in pipe_work.items()}

    def _boundary_bytes(self, nodes, tags, *, cross_only=False):
        '统计物理核之间的 COPY；同核优先级分组之间的边不增加该成本。'
        inputs = set()
        outputs = set()
        for u in nodes:
            dest = tags[u][1]
            for tensor in self.inputs.get(u, ()):
                producer = self.tensor_source.get(tensor)
                origin = tags[producer][1] if producer in tags else "ddr"
                if origin == dest:
                    continue
                if cross_only and origin == "ddr":
                    continue
                inputs.add((tensor, dest))
                if origin != "ddr":
                    outputs.add((tensor, origin))
        return sum(self.tensors[t] for t, _ in inputs | outputs)

    def propose(self, fork: int, p: int, other_cores: list[int]):
        found = self._region(fork)
        if found is None or not other_cores:
            return None
        children, join, before, mask = found
        groups = self._groups(children, before, mask)
        if len(groups) <= 1:
            return None

        all_nodes = before + [join]
        pipe_totals = collections.defaultdict(int)
        critical = {}
        for u in all_nodes:
            pipe_totals[self.pipe[u]] += self.graph.cycles[u]
            critical[u] = self.graph.cycles[u] + max(
                (critical[parent] for parent in self.graph.predecessors[u] if parent in critical),
                default=0,
            )
        stay = max(max(pipe_totals.values()), critical[join])
        stay_tags = {u: ("stay", p) for u in [fork, *all_nodes]}
        stay += math.ceil(self._boundary_bytes(all_nodes, stay_tags) / self.bandwidth)
        group_for_bit = {bit: i for i, group in enumerate(groups) for bit in group}
        group_mask = [sum(1 << bit for bit in group) for group in groups]
        group_work = [
            sum(self.graph.cycles[u] for u in before
                if mask[u] & bits and mask[u] & ~bits == 0)
            for bits in group_mask
        ]

        best = None
        considered = 0
        for retained_count in range(1, len(groups)):
            for start in range(len(groups) - retained_count + 1):
                if considered >= self.max_candidates:
                    break
                considered += 1
                retained = set(range(start, start + retained_count))
                remaining = sorted(
                    (i for i in range(len(groups)) if i not in retained),
                    key=lambda i: (-group_work[i], i),
                )
                slots = other_cores[:min(len(other_cores), len(remaining))]
                loads = dict.fromkeys(slots, 0)
                group_core = {i: p for i in retained}
                for i in remaining:
                    q = min(loads, key=lambda core: (loads[core], core))
                    group_core[i] = q
                    loads[q] += group_work[i]

                tags = {}
                owner = {}
                for u in before:
                    used = {group_core[group_for_bit[bit]]
                            for bit in range(len(children)) if mask[u] & (1 << bit)}
                    core = next(iter(used)) if len(used) == 1 else p
                    tags[u] = ("branch", core) if len(used) == 1 else ("join", p)
                    owner[u] = core
                tags[join] = ("join", p)
                owner[join] = p
                work = self._work(all_nodes, tags)
                tags[fork] = ("old", p)
                boundary = self._boundary_bytes(all_nodes, tags)
                cross_bytes = self._boundary_bytes(all_nodes, tags, cross_only=True)
                remote = max(
                    2 * self.cross_wait + work.get(("branch", q), 0)
                    for q in slots
                )
                local = self.same_wait + work.get(("branch", p), 0)
                split = (
                    max(local, remote) + self.same_wait
                    + work.get(("join", p), 0)
                    + math.ceil(boundary / self.bandwidth)
                )
                penalty = exit_penalty(self.escape_weight, self.cross_wait,
                                       cross_bytes, self.bandwidth)
                penalized = split + penalty
                if split < stay <= penalized:
                    self.penalty_rejections += 1
                if best is None or penalized < best[0]:
                    best = (penalized, owner, tags, boundary, split, penalty, cross_bytes)
            if considered >= self.max_candidates:
                break
        if best is None or best[0] >= stay:
            return None
        return {
            "fork": fork, "join": join, "cores": set(best[1].values()),
            "owner": best[1],
            "pure": {u for u in before if best[2][u][0] == "branch"},
            "join_ops": {u for u in all_nodes if best[2][u][0] == "join"},
            "stay": stay, "split": best[4], "boundary_bytes": best[3],
            "penalized_split": best[0], "exit_penalty_cycles": best[5],
            "cross_copy_bytes": best[6],
            "groups": len(groups),
        }


def generate_bspg_plan(
    case: dict[str, Any], m: int, *, bandwidth: int = 60,
    same_wait: int = 0, cross_wait: int = 500,
    max_nodes: int = 2000, max_candidates: int = 32,
    statistics: dict[str, Any] | None = None,
    guidance=None, escape_weight: float = DEFAULT_ESCAPE_WEIGHT, enable_splits: bool = True,
) -> dict[str, Any]:
    """仅在真正同时释放分支时，尝试封 任务 并成组分配空闲核。

    分数为原计算前驱亲和分加上可在当前 任务 复用的外部输入字节数。
    复用量仅作启发式估计；任务 合并条件、ready 集合和阶段门槛 m 沿用母本。
    时间/带宽默认值对应本赛题场景 A 的官方 config。
    """
    if type(m) is not int or not 1 <= m <= 5:
        raise ValueError("m 必须是 1 至 5 的整数")
    if min(bandwidth, max_nodes, max_candidates) <= 0 or min(same_wait, cross_wait) < 0:
        raise ValueError("带宽、局部搜索界和候选数应为正数，Task 等待应为非负数")
    if same_wait != 0:
        raise ValueError("Scene B has no same-core Task wait")
    if not isinstance(escape_weight, (int, float)) or not 0 <= escape_weight < math.inf:
        raise ValueError("escape_weight must be finite and nonnegative")
    graph = _build_op_graph(case)
    hint_group = guidance.group_of if guidance is not None else {}
    hint_owner = guidance.owner if guidance is not None else {}
    hint_priority = guidance.priority if guidance is not None else {}
    hard_hints = getattr(guidance,'hard_boundaries',True)
    soft_bonus = cross_wait*bandwidth if hint_group and not hard_hints else 0

    def _owner_bonus(op_id,p):
        return soft_bonus if hint_owner.get(hint_group.get(op_id))==p else 0
    hint_heaps = {group:[] for group in hint_owner}
    hint_groups_by_core = [sorted((g for g,p in hint_owner.items() if p==core),
                                 key=lambda g:(hint_priority[g],g)) for core in range(m)]
    task_hint = {}

    def _hint_ready(op_id):
        if op_id in hint_group:
            heapq.heappush(hint_heaps[hint_group[op_id]],op_id)
    planner = _ForkPlanner(
        case, graph, bandwidth=bandwidth, same_wait=same_wait,
        cross_wait=cross_wait, max_nodes=max_nodes, max_candidates=max_candidates,
        escape_weight=escape_weight,
    ) if m > 1 and enable_splits else None
    if statistics is not None:
        statistics.update(
            triggered=0, accepted=0, decisions=[],
            external_input_reuse_scored_ops=0,
            external_input_reuse_bytes_estimate=0,
            guided_choices=0, guidance_fallback_choices=0,
            ready_p_override_attempts=0, ready_p_overrides=0,
            ready_p_overrides_suppressed=0, escape_weight=escape_weight,
        )

    # 直接调用已存储的静态量；此后不再遍历原 JSON 计算大小或出度。
    c, deg_plus = graph.c, graph.deg_plus
    edge_weight = {
        op_id: c[op_id] / deg_plus[op_id]
        for op_id in graph.cycles if deg_plus[op_id] > 0
    }
    unfinished = {op_id: len(parents) for op_id, parents in graph.predecessors.items()}
    ready = {op_id for op_id, count in unfinished.items() if count == 0}
    if not ready and graph.cycles:
        raise ValueError("计算 Op 图没有初始就绪节点")
    for op_id in ready:
        _hint_ready(op_id)

    # 候选堆保存未检查合并条件的分数上界；取候选时核算实际加分。
    ready_all = set(ready)
    ready_p: list[set[int]] = [set() for _ in range(m)]
    all_heaps: list[list[tuple[float, int]]] = [
        [(-_owner_bonus(op_id,p), op_id) for op_id in sorted(ready_all)] for p in range(m)
    ]
    proc_heaps: list[list[tuple[float, int]]] = [[] for _ in range(m)]
    for heap in all_heaps:
        heapq.heapify(heap)

    # active_mask[u] 的第 p 位表示 u 自身或其某个直接后继已分配给 p。
    active_mask = dict.fromkeys(graph.cycles, 0)
    # scores 保持原来的前驱亲和分；外部输入复用分单独按当前 任务 维护。
    scores: list[dict[int, float]] = [{} for _ in range(m)]
    tensor_p: list[set[int]] = [set() for _ in range(m)]
    reuse_scores: list[dict[int, int]] = [{} for _ in range(m)]

    def _upper_score(op_id: int, p: int) -> float:
        return scores[p].get(op_id, 0.0) + reuse_scores[p].get(op_id, 0) + _owner_bonus(op_id,p)

    def _score(op_id: int, p: int, from_ready_p: bool) -> float:
        bonus = reuse_scores[p].get(op_id, 0)
        if bonus and not _can_merge(op_id, p, from_ready_p):
            bonus = 0
        return scores[p].get(op_id, 0.0) + bonus + _owner_bonus(op_id,p)

    def _update_external_inputs(op_id: int, p: int, new_task: bool) -> None:
        """按 任务 更新输入集合，只刷新受影响的未分配候选。"""
        for q in range(m):
            reuse_scores[q].pop(op_id, None)
        affected: set[int] = set()
        if new_task:
            affected.update(reuse_scores[p])
            reuse_scores[p].clear()
            tensor_p[p].clear()
        added = graph.external_inputs[op_id] - tensor_p[p]
        tensor_p[p].update(added)
        for tensor_id in sorted(added):
            size = graph.external_sizes[tensor_id]
            for consumer in graph.external_consumers[tensor_id]:
                if str(consumer) in mapping:
                    continue
                reuse_scores[p][consumer] = reuse_scores[p].get(consumer, 0) + size
                affected.add(consumer)
        for consumer in affected:
            entry = (-_upper_score(consumer, p), consumer)
            if consumer in ready_all:
                heapq.heappush(all_heaps[p], entry)
            elif consumer in ready_p[p]:
                heapq.heappush(proc_heaps[p], entry)

    def _peek(heap: list[tuple[float, int]], candidates: set[int], p: int,
              from_ready_p: bool):
        """按真实分数选点，以堆中上界剪枝；合并条件每次重新检查。

        暂时不能合并的候选必须放回堆中，其合并条件可能随 任务 图改变。
        不能仅修正原堆顶后直接返回，否则会遗漏真正分数更高的候选。
        """
        deferred = []
        best = None
        while heap:
            negative, op_id = heap[0]
            if (op_id not in candidates or reserved_core.get(op_id, p) != p
                    or -negative != _upper_score(op_id, p)):
                heapq.heappop(heap)
                continue
            if best is not None and (negative, op_id) >= (-best[0], best[1]):
                break
            score = _score(op_id, p, from_ready_p)
            if best is None or (-score, op_id) < (-best[0], best[1]):
                best = (score, op_id)
            if score == -negative:
                break
            deferred.append(heapq.heappop(heap))
        for entry in deferred:
            heapq.heappush(heap, entry)
        return best

    def _choose(free: list[bool]):
        """先执行已承诺分组；其余 操作 保持原 BSPg 的选点顺序。"""
        for p in range(m):
            if not free[p]:
                continue
            candidates = [
                u for u in forced_ready[p]
                if u in ready and (
                    splits[op_scope[u]]["remaining_pure"] == 0
                    or u in splits[op_scope[u]]["pure"]
                )
            ]
            if candidates:
                u = min(candidates, key=lambda op_id: (
                    -_score(op_id, p, op_id in ready_p[p]), op_id
                ))
                return u, p, u in ready_p[p]
        if hint_group and not hard_hints:
            best = None
            for p in range(m):
                if not free[p] or p in core_lock:
                    continue
                local = _peek(proc_heaps[p],ready_p[p],p,True)
                candidate = local or _peek(all_heaps[p],ready_all,p,False)
                if candidate is None:
                    continue
                score,u = candidate
                key = (-score,u,p)
                if best is None or key<best[0]:
                    best = (key,u,p,local is not None)
            if best is not None and statistics is not None:
                field = ('guided_choices' if _owner_bonus(best[1],best[2])
                         else 'guidance_fallback_choices')
                statistics[field] += 1
            return None if best is None else best[1:]
        # 分组偏好优先于局部就绪亲和度；例如，
        # 可以先选择另一分支的前缀，再选择当前分支的后缀。
        # 就绪条件以及最终任务图和核内顺序的无环检查仍然有效。
        for p in range(m):
            if not free[p] or p in core_lock:
                continue
            for group in hint_groups_by_core[p]:
                heap = hint_heaps[group]
                deferred, candidates = [], []
                while heap and len(deferred)<16:
                    u = heapq.heappop(heap)
                    if u not in ready:
                        continue
                    deferred.append(u)
                    # 粗粒度阶段可以显式切换到另一个核。
                    # 此处满足逻辑就绪即可，任务依赖
                    # 以及分组商图和核内顺序的无环检查仍会约束实际执行。
                    # 若额外要求 ready_p[p]，会使当前阶段被隐式切回。
                    if u not in reserved_core and u in ready:
                        candidates.append(u)
                for u in deferred:
                    heapq.heappush(heap,u)
                if candidates:
                    u = min(candidates,key=lambda u:(-_score(u,p,u in ready_p[p]),u))
                    # BSPg 通常严格优先选择 ready_p；粗粒度覆盖该规则时，
                    # 必须计入连续性损失，并将周期数换算成
                    # 亲和度分数使用的字节单位，保证比较一致。
                    local = _peek(proc_heaps[p], ready_p[p], p, True)
                    if (u not in ready_p[p] and local is not None
                            and hint_owner.get(hint_group.get(local[1]), p) == p):
                        if statistics is not None:
                            statistics['ready_p_override_attempts'] += 1
                        v = local[1]
                        pending = sum(c[parent] for parent in graph.predecessors[v]
                                      if op_core.get(parent) == p)
                        loss = exit_penalty(escape_weight, cross_wait, pending, bandwidth)
                        gain = (_score(u, p, False) - local[0]) / bandwidth
                        if escape_weight > 0 and gain <= loss:
                            if statistics is not None:
                                statistics['ready_p_overrides_suppressed'] += 1
                            return v, p, True
                        if statistics is not None:
                            statistics['ready_p_overrides'] += 1
                    if statistics is not None:
                        statistics['guided_choices'] += 1
                    return u,p,u in ready_p[p]
        # 目标核繁忙时，推进到该核的下一个完成事件。
        # 全部核空闲时，普通 BSPg 可以从过期提示中恢复。
        if hint_group and not all(free) and ready and all(u in hint_group for u in ready):
            return None
        for p in range(m):
            if free[p] and p not in core_lock:
                candidate = _peek(proc_heaps[p], ready_p[p], p, True)
                if candidate is not None:
                    if statistics is not None and hint_group:
                        statistics['guidance_fallback_choices'] += 1
                    return candidate[1], p, True
        best = None
        for p in range(m):
            if not free[p] or p in core_lock:
                continue
            candidate = _peek(all_heaps[p], ready_all, p, False)
            if candidate is None:
                continue
            score, op_id = candidate
            # 原 C++ 先按 操作 ID、再按核心 ID 扫描，分数相等时取先者。
            key = (-score, op_id, p)
            if best is None or key < best[0]:
                best = (key, op_id, p)
        if best is not None and statistics is not None and hint_group:
            statistics['guidance_fallback_choices'] += 1
        return None if best is None else (best[1], best[2], False)

    def _activate(u: int, p: int) -> None:
        """更新超边所属核心，以及受影响候选节点的增量分数。"""
        bit = 1 << p
        if active_mask[u] & bit:
            return
        active_mask[u] |= bit
        if deg_plus[u] == 0 or c[u] == 0:
            return
        increment = edge_weight[u]
        core_scores = scores[p]
        for successor in graph.successors[u]:
            score = core_scores.get(successor, 0.0) + increment
            core_scores[successor] = score
            if successor in ready_all:
                heapq.heappush(all_heaps[p], (-_upper_score(successor, p), successor))
            elif successor in ready_p[p]:
                heapq.heappush(proc_heaps[p], (-_upper_score(successor, p), successor))

    free = [True] * m
    # 这些状态描述已采纳的局部 任务 切分及合并约束。
    reserved_core: dict[int, int] = {}
    forced_ready: list[set[int]] = [set() for _ in range(m)]
    op_scope: dict[int, int] = {}
    splits: dict[int, dict[str, Any]] = {}
    core_lock: dict[int, int] = {}
    sealed_tasks: set[int] = set()
    blocked_forks: set[int] = set()
    task_scope: dict[int, int] = {}
    # 与原 BSPg 一样，虚拟时间在新阶段归零；-1 是初始时间事件。
    finish_times: list[tuple[int, int]] = [(0, -1)]
    stage = 0
    end_stage = False
    threshold = m
    assigned_count = 0
    op_core: dict[int, int] = {}
    op_stage: dict[int, int] = {}
    mapping: dict[str, int] = {}
    core_schedules: list[list[int]] = [[] for _ in range(m)]
    # 当前已分配 任务 的依赖图：计算依赖及同核 任务 顺序均逐步加入。
    task_successors: dict[int, set[int]] = {}
    task_predecessors: dict[int, set[int]] = {}
    task_core: dict[int, int] = {}
    next_sgid = 0

    def _can_merge(op_id: int, p: int, from_ready_p: bool) -> bool:
        """检查合并后有无新跨核等待或 任务 环；ready_p 保持原有优先权。"""
        if not core_schedules[p]:
            return False
        current_task = core_schedules[p][-1]
        if current_task in sealed_tasks:
            return False
        group = hint_group.get(op_id)
        if hard_hints and group is not None and task_hint.get(current_task)!=group:
            return False
        other_parents = {
            mapping[str(parent)] for parent in graph.predecessors[op_id]
            if mapping[str(parent)] != current_task
        }
        # 对 ready_all，不能让已有 任务 为新操作的跨核前驱重新等待。
        scope = op_scope.get(op_id)
        join_task = (scope in splits and op_id in splits[scope]["join_ops"]
                     and task_scope.get(current_task) == scope
                     and splits[scope]["remaining_pure"] == 0)
        same_hint = hard_hints and group is not None and task_hint.get(current_task)==group
        if not from_ready_p and not join_task and not same_hint and any(
            task_core[parent] != p and parent not in task_predecessors[current_task]
            for parent in other_parents
        ):
            return False
        if not other_parents:
            return True
        # 若存在 当前 任务 -> ... -> 某前驱 任务，添加前驱 -> 当前 任务 会成环。
        pending = [current_task]
        visited = {current_task}
        while pending:
            task = pending.pop()
            for nxt in task_successors[task]:
                if nxt in other_parents:
                    return False
                if nxt not in visited:
                    visited.add(nxt)
                    pending.append(nxt)
        return True

    while ready or finish_times:
        if end_stage and not finish_times:
            stage += 1
            end_stage = False
            ready_all = set(ready)
            ready_p = [set() for _ in range(m)]
            proc_heaps = [[] for _ in range(m)]
            all_heaps = [
                [(-_upper_score(op_id, p), op_id) for op_id in ready_all]
                for p in range(m)
            ]
            for heap in all_heaps:
                heapq.heapify(heap)
            finish_times.append((0, -1))

        if not finish_times:
            raise ValueError("有未分配的 Op，但当前阶段既无执行事件也无法推进")
        time = finish_times[0][0]

        # 相同时间完成的 操作 必须全部处理，再从 ready 集合中选择新任务。
        new_forks = []
        while finish_times and finish_times[0][0] == time:
            _, finished = heapq.heappop(finish_times)
            if finished == -1:
                continue
            p = op_core[finished]
            free[p] = True
            scope = op_scope.get(finished)
            if scope in splits:
                state = splits[scope]
                if finished in state["pure"]:
                    state["remaining_pure"] -= 1
                    if state["remaining_pure"] == 0:
                        for core in state["cores"]:
                            if core_schedules[core]:
                                sealed_tasks.add(core_schedules[core][-1])
                else:
                    state["remaining_join"] -= 1
                    if state["remaining_join"] == 0:
                        for core in state["cores"]:
                            core_lock.pop(core, None)
                        del splits[scope]
            just_ready = []
            for successor in graph.successors[finished]:
                unfinished[successor] -= 1
                if unfinished[successor] != 0:
                    continue
                ready.add(successor)
                _hint_ready(successor)
                just_ready.append(successor)
                if successor in reserved_core:
                    forced_ready[reserved_core[successor]].add(successor)
                if all(
                    op_stage[parent] < stage or op_core[parent] == p
                    for parent in graph.predecessors[successor]
                ):
                    ready_p[p].add(successor)
                    heapq.heappush(
                        proc_heaps[p], (-_upper_score(successor, p), successor)
                    )
            if len(just_ready) >= 2 and deg_plus[finished] == len(just_ready):
                new_forks.append((finished, p))

        # 阶段一旦关闭，只等待本阶段正在运行的 操作 完成。
        if end_stage:
            continue

        # 仅当多个真正可执行的兄弟分支同时出现，且有空闲、无积压的核时估计。
        for fork, p in new_forks:
            if ((hard_hints and fork in hint_group) or planner is None or fork in blocked_forks or p in core_lock
                    or not core_schedules[p]):
                continue
            available = [q for q in range(m) if q != p and free[q]
                         and q not in core_lock and not ready_p[q]
                         and not forced_ready[q]]
            if not available:
                continue
            if statistics is not None:
                statistics["triggered"] += 1
            decision = planner.propose(fork, p, available)
            if decision is None or any(u in reserved_core for u in decision["owner"]):
                continue
            sealed_tasks.add(core_schedules[p][-1])
            splits[fork] = dict(
                pure=decision["pure"], join_ops=decision["join_ops"],
                remaining_pure=len(decision["pure"]),
                remaining_join=len(decision["join_ops"]),
                cores=decision["cores"],
            )
            for core in decision["cores"]:
                core_lock[core] = fork
            blocked_forks.update(u for u in decision["owner"] if u != decision["join"])
            for u, core in decision["owner"].items():
                reserved_core[u] = core
                op_scope[u] = fork
                if u in ready:
                    forced_ready[core].add(u)
            if statistics is not None:
                statistics["accepted"] += 1
                if len(statistics["decisions"]) < 12:
                    statistics["decisions"].append({
                        key: decision[key] for key in (
                            "fork", "join", "groups", "stay", "split", "boundary_bytes",
                            "penalized_split", "exit_penalty_cycles", "cross_copy_bytes"
                        )
                    })

        while True:
            chosen = _choose(free)
            if chosen is None:
                break
            op_id, p, from_ready_p = chosen
            ready.remove(op_id)
            for q in range(m):
                ready_p[q].discard(op_id)
                forced_ready[q].discard(op_id)
            ready_all.discard(op_id)
            reserved_core.pop(op_id, None)

            parent_tasks = {
                mapping[str(parent)] for parent in graph.predecessors[op_id]
            }
            can_merge = _can_merge(op_id, p, from_ready_p)
            if can_merge:
                sgid = core_schedules[p][-1]
                bonus = reuse_scores[p].get(op_id, 0)
                if statistics is not None and bonus:
                    statistics["external_input_reuse_scored_ops"] += 1
                    statistics["external_input_reuse_bytes_estimate"] += bonus
            else:
                sgid = next_sgid
                next_sgid += 1
                task_successors[sgid] = set()
                task_predecessors[sgid] = set()
                task_core[sgid] = p
                if core_schedules[p]:
                    prior = core_schedules[p][-1]
                    task_successors[prior].add(sgid)
                    task_predecessors[sgid].add(prior)
                core_schedules[p].append(sgid)
                if op_id in hint_group:
                    task_hint[sgid] = hint_group[op_id]

            for parent_task in parent_tasks:
                if parent_task != sgid:
                    task_successors[parent_task].add(sgid)
                    task_predecessors[sgid].add(parent_task)

            mapping[str(op_id)] = sgid
            if op_id in op_scope:
                task_scope.setdefault(sgid, op_scope[op_id])
            op_core[op_id] = p
            op_stage[op_id] = stage
            assigned_count += 1
            free[p] = False
            heapq.heappush(finish_times, (time + graph.cycles[op_id], op_id))

            _update_external_inputs(op_id, p, new_task=not can_merge)
            _activate(op_id, p)
            for parent in graph.predecessors[op_id]:
                _activate(parent, p)

        if not ready_all and sum(free) >= threshold and not splits:
            end_stage = True

    if assigned_count != len(graph.cycles):
        raise ValueError("贪心调度没有覆盖所有计算 Op")
    plan = {"node_to_subgraph": mapping, "core_schedules": core_schedules}
    _check_plan(graph, plan, m)
    if statistics is not None:
        statistics['fork_penalty_rejections'] = planner.penalty_rejections if planner else 0
    return plan
