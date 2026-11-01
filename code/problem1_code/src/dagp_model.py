'场景 A 的可移植图模型与局部代价模型。\n\n换出插入前的边界 COPY 数据量严格遵循官方 任务 构造。计算耗时与内存峰值采用低成本代理估计，不能代替官方评价。仅依赖 Python 标准库。'

from __future__ import annotations

import heapq
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


COPY_TYPES = frozenset({"COPY_IN", "COPY_OUT"})
PIPES = frozenset({"PIPE_MTE2", "PIPE_MTE3", "PIPE_M", "PIPE_V"})
CONFIG_KEYS = {
    "capacity": frozenset({"L1", "UB"}),
    "bandwidth": frozenset({"bandwidth"}),
    "multicore_scene_a": frozenset({
        "task_cross_core_wait_cycles", "task_same_core_wait_cycles"}),
    "multicore_scene_b": frozenset({"cross_core_copy_delay_cycles"}),
    "problem_3": frozenset({
        "cache_capacity_bytes", "cache_bandwidth_bytes_per_cycle"}),
}


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}: {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Hardware:
    bandwidth: float
    capacity: dict[str, int]
    same_wait: int
    cross_wait: int

    def __post_init__(self) -> None:
        if (isinstance(self.bandwidth, bool)
                or not isinstance(self.bandwidth, (int, float))
                or not math.isfinite(self.bandwidth) or self.bandwidth <= 0):
            raise ValueError("bandwidth must be finite and positive")
        if not isinstance(self.capacity, dict) or set(self.capacity) != {"L1", "UB"}:
            raise ValueError("capacity must contain exactly L1 and UB")
        for mem, value in self.capacity.items():
            _integer(value, f"capacity.{mem}")
        _integer(self.same_wait, "same_wait")
        _integer(self.cross_wait, "cross_wait")
        # 复制调用方的可变字典，避免长期持有其原对象。
        object.__setattr__(self, "capacity", dict(self.capacity))


def read_hardware(config_path: str | Path) -> Hardware:
    '读取官方 config.txt 的显式配置，不自行补造默认值。\n\n允许出现已知的场景 B 与第三问配置段；未知配置段、未知或重复键、非法取值及缺少场景 A 必需键均报错。'
    path = Path(config_path)
    settings: dict[str, dict[str, int]] = {}
    section: str | None = None
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            if section not in CONFIG_KEYS:
                raise ValueError(f"{path}:{line_no}: unknown section [{section}]")
            settings.setdefault(section, {})
            continue
        parts = line.split()
        if section is None or len(parts) != 2:
            raise ValueError(f"{path}:{line_no}: expected '<key> <integer>' in a section")
        key, raw_value = parts
        if key not in CONFIG_KEYS[section] or key in settings[section]:
            raise ValueError(f"{path}:{line_no}: unknown/duplicate [{section}] key {key}")
        try:
            value = int(raw_value)
        except ValueError:
            raise ValueError(f"{path}:{line_no}: {key} must be an integer") from None
        _integer(value, f"{path}:{line_no}: {key}")
        if key in {"bandwidth", "cache_bandwidth_bytes_per_cycle"} and value == 0:
            raise ValueError(f"{path}:{line_no}: {key} must be positive")
        settings[section][key] = value
    for required in ("capacity", "bandwidth", "multicore_scene_a"):
        missing = CONFIG_KEYS[required] - settings.get(required, {}).keys()
        if missing:
            raise ValueError(f"{path}: [{required}] must define {', '.join(sorted(missing))}")
    return Hardware(
        bandwidth=settings["bandwidth"]["bandwidth"],
        capacity=settings["capacity"],
        same_wait=settings["multicore_scene_a"]["task_same_core_wait_cycles"],
        cross_wait=settings["multicore_scene_a"]["task_cross_core_wait_cycles"],
    )


@dataclass(frozen=True, slots=True)
class Tensor:
    size: int
    producers: frozenset[int]
    consumers: frozenset[int]
    required_output: bool
    mem: str  # 计算 任务 内有效存储位置：原始 DDR 张量转为 UB。


@dataclass(frozen=True, slots=True)
class TaskCost:
    compute: float
    copy_bytes: int
    copy_cycles: int
    pipe_work: dict[str, int]
    peak_bytes: dict[str, int]
    read_bytes: int = 0
    write_bytes: int = 0
    read_cycles: int = 0
    write_cycles: int = 0


def _topological(
    pred: Mapping[int, Any], succ: Mapping[int, Any], context: str,
) -> tuple[int, ...]:
    '采用确定性的 Kahn 拓扑排序，同优先级时先选较小编号。'
    indegree = {node: len(parents) for node, parents in pred.items()}
    ready = [node for node, count in indegree.items() if count == 0]
    heapq.heapify(ready)
    result = []
    while ready:
        node = heapq.heappop(ready)
        result.append(node)
        for child in succ[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(result) != len(indegree):
        raise ValueError(f"{context} contains a directed cycle")
    return tuple(result)


class Model:
    '经过验证的计算 DAG，以及张量级 任务 边界代价。\n\ncycles 保留输入数值；代理使用 max(1, cycles)，与官方对零周期算子的耗时约定一致。tensors 保留全部原始张量；生产者和消费者集合仅包含非 COPY 计算算子的编号，原始 COPY_OUT 另行记录。'

    def __init__(self, case: dict[str, Any], hardware: Hardware):
        if not isinstance(hardware, Hardware):
            raise TypeError("hardware must be Hardware")
        self.hardware = hardware
        if not isinstance(case, dict):
            raise ValueError("case must be an object")
        # 保留原始属性供详细 任务 模板构造使用。构造器在准备执行前复制局部节点，不得修改调用方的原始图。
        self.source_graph = case
        for name in ("ops", "tensors", "edges"):
            if not isinstance(case.get(name), list):
                raise ValueError(f"case.{name} must be a list")
        op_by_id: dict[int, dict[str, Any]] = {}
        tensor_by_id: dict[int, dict[str, Any]] = {}
        all_ids: set[int] = set()
        for name, rows, by_id in (
            ("op", case["ops"], op_by_id),
            ("tensor", case["tensors"], tensor_by_id),
        ):
            for item in rows:
                if not isinstance(item, dict):
                    raise ValueError(f"{name} must be an object")
                ident = _integer(item.get("id"), f"{name}.id")
                if ident in all_ids:
                    raise ValueError(f"duplicate Op/Tensor ID: {ident}")
                all_ids.add(ident)
                by_id[ident] = item
                if name == "op":
                    _integer(item.get("cycles"), f"op {ident} cycles")
                    if item.get("pipe") not in PIPES:
                        raise ValueError(f"op {ident} has an unsupported pipe")
                    if not isinstance(item.get("op"), str) or not item["op"]:
                        raise ValueError(f"op {ident} must have an operation name")
                else:
                    _integer(item.get("size"), f"tensor {ident} size")
                    if item.get("pos") not in {"DDR", "L1", "UB"}:
                        raise ValueError(f"tensor {ident} has an unsupported memory position")

        self.ids = tuple(sorted(op for op in op_by_id
                                if op_by_id[op]["op"] not in COPY_TYPES))
        self.cycles = {op: op_by_id[op]["cycles"] for op in self.ids}
        self.pipes = {op: op_by_id[op]["pipe"] for op in self.ids}
        full_pred: dict[int, set[int]] = {node: set() for node in all_ids}
        full_succ: dict[int, set[int]] = {node: set() for node in all_ids}
        producers: dict[int, set[int]] = defaultdict(set)
        consumers: dict[int, set[int]] = defaultdict(set)
        for edge in case["edges"]:
            if not isinstance(edge, dict):
                raise ValueError("edge must be an object")
            source = _integer(edge.get("source"), "edge.source")
            target = _integer(edge.get("target"), "edge.target")
            if source not in all_ids or target not in all_ids:
                raise ValueError(f"edge references an unknown node: {source} -> {target}")
            if source in tensor_by_id and target in tensor_by_id:
                raise ValueError(f"Tensor-to-Tensor edge is unsupported: {source} -> {target}")
            if target in full_succ[source]:
                raise ValueError(f"duplicate edge: {source} -> {target}")
            full_succ[source].add(target)
            full_pred[target].add(source)
            if source in op_by_id and target in tensor_by_id:
                producers[target].add(source)
            elif source in tensor_by_id and target in op_by_id:
                consumers[source].add(target)
        _topological(full_pred, full_succ, "original input graph")

        pred: dict[int, set[int]] = {op: set() for op in self.ids}
        succ: dict[int, set[int]] = {op: set() for op in self.ids}
        # 穿过所有中间 COPY 和张量节点，遇到下一个计算算子即停止；直接算子边采用同样规则。
        for source in self.ids:
            pending = list(full_succ[source])
            seen_middle: set[int] = set()
            while pending:
                target = pending.pop()
                if target in self.cycles:
                    succ[source].add(target)
                    pred[target].add(source)
                elif target not in seen_middle:
                    seen_middle.add(target)
                    pending.extend(full_succ[target])
        self.pred = {op: tuple(sorted(pred[op])) for op in self.ids}
        self.succ = {op: tuple(sorted(succ[op])) for op in self.ids}
        self.topo = _topological(self.pred, self.succ, "computational Op graph")
        self.rank = {op: index for index, op in enumerate(self.topo)}

        eligible = set(self.ids)
        self.tensors: dict[int, Tensor] = {}
        touched: dict[int, set[int]] = {op: set() for op in self.ids}
        for tid in sorted(tensor_by_id):
            row = tensor_by_id[tid]
            prod = frozenset(producers[tid] & eligible)
            cons = frozenset(consumers[tid] & eligible)
            self.tensors[tid] = Tensor(
                size=row["size"], producers=prod, consumers=cons,
                required_output=any(op_by_id[op]["op"] == "COPY_OUT"
                                    for op in consumers[tid]),
                mem="UB" if row["pos"] == "DDR" else row["pos"],
            )
            for op in prod | cons:
                touched[op].add(tid)
        self.touched = {op: frozenset(tids) for op, tids in touched.items()}
        # 使用实例私有缓存，避免全局持有旧 Model；同时限制缓存条目数和缓存算子编号总量。
        self._cost_cache: OrderedDict[frozenset[int], TaskCost] = OrderedDict()
        self._cached_members = 0

    def clear_cost_cache(self) -> None:
        self._cost_cache.clear()
        self._cached_members = 0

    def cost(self, members: frozenset[int]) -> TaskCost:
        '返回低成本代理代价，以及换出前准确的 任务 边界流量。\n\ncompute 取内部加权最长路径与各流水线总工作量的最大值，是计算下界，并非可实现的多流水线时刻表。copy_cycles 对每个插入的 COPY 累加 max(1, ceil(size / bandwidth))。\n\npeak_bytes 按原始固定拓扑顺序估计存活张量：输入在首次使用时出现，输出在首次生成时出现，缓冲区在最后一次局部使用或生成后释放；同一算子接触的缓冲区同时存在，并假设边界输出及时写出。它既不是官方 步骤 1/步骤 2 的内存顺序，也不保证构成上界或下界，且不模拟共享 DDR 延迟、换出与内存复用。'
        if not isinstance(members, frozenset):
            members = frozenset(members)
        if any(type(op) is not int or op not in self.cycles for op in members):
            raise ValueError("Task members must be known non-COPY Op IDs")
        cached = self._cost_cache.get(members)
        if cached is not None:
            self._cost_cache.move_to_end(members)
            return cached
        order = sorted(members, key=self.rank.__getitem__)
        pipe_work: dict[str, int] = defaultdict(int)
        chain_end: dict[int, int] = {}
        tids: set[int] = set()
        for op in order:
            duration = max(1, self.cycles[op])
            pipe_work[self.pipes[op]] += duration
            chain_end[op] = duration + max(
                (chain_end[parent] for parent in self.pred[op] if parent in members),
                default=0,
            )
            tids.update(self.touched[op])
        compute = float(max(max(pipe_work.values(), default=0),
                            max(chain_end.values(), default=0)))
        read_bytes = write_bytes = read_cycles = write_cycles = 0
        starts: dict[int, list[tuple[str, int]]] = defaultdict(list)
        ends: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for tid in tids:
            tensor = self.tensors[tid]
            local_prod = tensor.producers & members
            local_cons = tensor.consumers & members
            input_boundary = bool(local_cons) and not local_prod
            output_boundary = bool(local_prod) and (
                tensor.required_output or not tensor.consumers
                or not tensor.consumers <= members)
            if input_boundary:
                read_bytes += tensor.size
                read_cycles += max(1, math.ceil(tensor.size / self.hardware.bandwidth))
            if output_boundary:
                write_bytes += tensor.size
                write_cycles += max(1, math.ceil(tensor.size / self.hardware.bandwidth))
            uses = local_prod | local_cons
            first = min(self.rank[op] for op in (local_prod or local_cons))
            last = max(self.rank[op] for op in uses)
            starts[first].append((tensor.mem, tensor.size))
            ends[last].append((tensor.mem, tensor.size))
        live = {mem: 0 for mem in self.hardware.capacity}
        peak = dict(live)
        for op in order:
            index = self.rank[op]
            for mem, size in starts.get(index, ()):
                live[mem] += size
            for mem in live:
                peak[mem] = max(peak[mem], live[mem])
            for mem, size in ends.get(index, ()):
                live[mem] -= size
        result = TaskCost(
            compute=compute, copy_bytes=read_bytes + write_bytes,
            copy_cycles=read_cycles + write_cycles,
            pipe_work=dict(pipe_work), peak_bytes=peak,
            read_bytes=read_bytes, write_bytes=write_bytes,
            read_cycles=read_cycles, write_cycles=write_cycles,
        )
        self._cost_cache[members] = result
        self._cached_members += len(members)
        while len(self._cost_cache) > 256 or self._cached_members > 250_000:
            old_members, _ = self._cost_cache.popitem(last=False)
            self._cached_members -= len(old_members)
        return result

    def validate_partition(
        self, blocks: dict[int, frozenset[int]],
    ) -> tuple[dict[int, set[int]], dict[int, set[int]], tuple[int, ...]]:
        '检查算子恰好覆盖一次及商图无环，并返回相应 DAG。'
        if not isinstance(blocks, dict):
            raise ValueError("blocks must be a dictionary")
        owner: dict[int, int] = {}
        for task, members in blocks.items():
            _integer(task, "Task ID")
            if not isinstance(members, (set, frozenset)) or not members:
                raise ValueError(f"Task {task} must contain a nonempty Op set")
            for op in members:
                if type(op) is not int or op not in self.cycles:
                    raise ValueError(f"Task {task} contains unknown computational Op {op!r}")
                if op in owner:
                    raise ValueError(f"Op {op} belongs to more than one Task")
                owner[op] = task
        if len(owner) != len(self.ids):
            missing = sorted(set(self.ids) - owner.keys())
            raise ValueError(f"partition does not cover all computational Ops: {missing[:20]}")
        pred: dict[int, set[int]] = {task: set() for task in blocks}
        succ: dict[int, set[int]] = {task: set() for task in blocks}
        for op in self.ids:
            source = owner[op]
            for child in self.succ[op]:
                target = owner[child]
                if target != source:
                    succ[source].add(target)
                    pred[target].add(source)
        return pred, succ, _topological(pred, succ, "Task quotient")

    def validate_plan(self, plan: dict[str, Any], m: int) -> bool:
        '检查官方输出格式，以及依赖边与核内顺序边合并后的 DAG。'
        _integer(m, "number of cores", 1)
        if not isinstance(plan, dict) or set(plan) != {"node_to_subgraph", "core_schedules"}:
            raise ValueError("plan must contain exactly node_to_subgraph and core_schedules")
        raw_mapping = plan["node_to_subgraph"]
        schedules = plan["core_schedules"]
        if not isinstance(raw_mapping, dict):
            raise ValueError("node_to_subgraph must be an object")
        if (not isinstance(schedules, list) or len(schedules) != m
                or any(not isinstance(order, list) for order in schedules)):
            raise ValueError("core_schedules must contain one Task list per core")
        mapping: dict[int, int] = {}
        for raw_op, task in raw_mapping.items():
            if not (type(raw_op) is int or (isinstance(raw_op, str)
                    and raw_op.isascii() and raw_op.isdecimal())):
                raise ValueError("node_to_subgraph keys must be integer Op IDs")
            op = int(raw_op)
            _integer(op, "mapped Op ID")
            _integer(task, "mapped Task ID")
            if op in mapping:
                raise ValueError(f"duplicate mapped integer Op ID: {op}")
            mapping[op] = task
        if set(mapping) != set(self.ids):
            raise ValueError("node_to_subgraph must exactly cover non-COPY Ops")
        blocks: dict[int, set[int]] = defaultdict(set)
        for op, task in mapping.items():
            blocks[task].add(op)
        pred, succ, _ = self.validate_partition(dict(blocks))
        listed = [task for order in schedules for task in order]
        for task in listed:
            _integer(task, "scheduled Task ID")
        if len(listed) != len(set(listed)) or set(listed) != set(blocks):
            raise ValueError("core_schedules must contain every Task exactly once")
        for order in schedules:
            for source, target in zip(order, order[1:]):
                succ[source].add(target)
                pred[target].add(source)
        _topological(pred, succ, "Task dependencies plus core order")
        return True
