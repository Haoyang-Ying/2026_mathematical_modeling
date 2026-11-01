"""第二问的数据表示；组仅表示优先级桶，不构成执行屏障。"""
from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass

from scene_b_internal import SceneBEvaluator  # 同时加载随包附带的官方运行依赖。
from evaluation_validation import validate_graph, require_integer
from stub_multicore_cut_and_schedule import (
    _build_op_adjacency, _contract_excluded_copy_nodes, derive_multicore_plan,
)
from multicore_cut_evaluate_problem_1 import _original_tensor_views

PIPES = ('PIPE_MTE2', 'PIPE_MTE3', 'PIPE_M', 'PIPE_V')


@dataclass(frozen=True)
class State:
    """写时复制的搜索状态；局部修改时保留组编号以便复用缓存。"""
    groups: dict[int, tuple[int, ...]]
    cores: tuple[tuple[int, ...], ...]

    def plan(self):
        mapping = {v: g for g, nodes in self.groups.items() for v in nodes}
        return {'node_to_subgraph': {str(v): mapping[v] for v in sorted(mapping)},
                'core_schedules': [list(order) for order in self.cores]}

    def fingerprint(self):
        payload = (sorted((g, sorted(v)) for g, v in self.groups.items()), self.cores)
        return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()

    def padded(self, cores):
        """保留核心分配、组编号及顺序，仅追加空闲核心。"""
        require_integer(cores, 'cores', 1)
        if cores < len(self.cores):
            raise ValueError('cannot inherit a higher-core plan into fewer cores')
        return State(dict(self.groups), self.cores + ((),) * (cores-len(self.cores)))

    def resized(self, cores):
        """仅通过调整空闲核心，将方案复用于另一目标核数。
        
        保留活跃核心的相对顺序、核内节点分配、组编号及组优先级顺序，不合并活跃核心。
        每个转换后的方案仍必须在目标核数下重新评价。"""
        require_integer(cores, 'cores', 1)
        if cores >= len(self.cores):
            return self.padded(cores)
        active = tuple(order for order in self.cores if order)
        if len(active) > cores:
            raise ValueError('active core count exceeds target; cannot remove an occupied core')
        return State(dict(self.groups), active + ((),) * (cores-len(active)))

    @classmethod
    def from_plan(cls, model, plan):
        view = derive_multicore_plan(model.graph, plan)
        return cls({g: tuple(sorted(vs, key=model.position.__getitem__))
                    for g, vs in view['nodes_by_subgraph'].items()},
                   tuple(tuple(x) for x in plan['core_schedules']))


class Model:
    def __init__(self, graph):
        validate_graph(graph)
        self.graph = graph
        self.ops = {o['id']: o for o in graph['ops']
                    if o['op'] not in ('COPY_IN', 'COPY_OUT')}
        self.nodes = tuple(sorted(self.ops))
        _, all_succ = _build_op_adjacency(graph)
        pred, succ = _contract_excluded_copy_nodes(self.nodes, all_succ)
        self.pred = {v: tuple(sorted(pred[v])) for v in self.nodes}
        self.succ = {v: tuple(sorted(succ[v])) for v in self.nodes}
        degree = {v: len(self.pred[v]) for v in self.nodes}
        ready = [v for v in self.nodes if degree[v] == 0]
        heapq.heapify(ready)
        topo = []
        while ready:
            v = heapq.heappop(ready)
            topo.append(v)
            for w in self.succ[v]:
                degree[w] -= 1
                if degree[w] == 0:
                    heapq.heappush(ready, w)
        if len(topo) != len(self.nodes):
            raise ValueError('eligible operation graph is cyclic')
        self.topo = tuple(topo)
        self.position = {v: i for i, v in enumerate(topo)}
        producers, consumers, direct = _original_tensor_views(graph)
        original_ops = {o['id']: o for o in graph['ops']}
        self.output_tensors = {t['id'] for t in graph['tensors']
            if any(original_ops[v]['op'] == 'COPY_OUT'
                   for v in consumers.get(t['id'], ()) if v in original_ops)}
        self.tensors = {t['id']: t for t in graph['tensors']}
        self.producers = {t: tuple(sorted(v for v in producers.get(t, ()) if v in self.ops))
                          for t in self.tensors}
        self.consumers = {t: tuple(sorted(v for v in consumers.get(t, ()) if v in self.ops))
                          for t in self.tensors}
        self.inputs = {v: [] for v in self.nodes}
        self.outputs = {v: [] for v in self.nodes}
        for t in self.tensors:
            for v in self.producers[t]:
                self.outputs[v].append(t)
            for v in self.consumers[t]:
                self.inputs[v].append(t)
        self.direct_in = {v: [] for v in self.nodes}
        for e in direct:
            if e['source'] in self.ops and e['target'] in self.ops:
                self.direct_in[e['target']].append((e['source'], int(e.get('data_size', 0))))
        self.work = {v: max(1, self.ops[v]['cycles']) for v in self.nodes}
        self.rank = {}
        for v in reversed(topo):
            self.rank[v] = self.work[v] + max((self.rank[w] for w in self.succ[v]), default=0)

    def memory_pos(self, tensor):
        pos = self.tensors[tensor]['pos']
        return 'UB' if pos == 'DDR' else pos

    def fallback(self, cores):
        require_integer(cores, 'cores', 1)
        return State({0: self.topo} if self.nodes else {},
                     ((0,) if self.nodes else (),) + ((),) * (cores-1))

    def group_runs(self, order, placement, cores, group_size, capacity):
        """同一全局拓扑序的连续区间可保持商图无环。
        
        这仅是初始构造方法，搜索可离开该结构族；完整执行验证始终不可省略。"""
        require_integer(group_size, 'group_size', 1)
        groups, schedules, current, touched = {}, [[] for _ in range(cores)], [], set()
        previous_core = None

        def flush():
            if current:
                g = len(groups)
                groups[g] = tuple(current)
                schedules[previous_core].append(g)

        for v in order:
            c = placement[v]
            new_tensors = set(self.inputs[v]) | set(self.outputs[v])
            union = touched | new_tensors
            pressure = any(sum(self.tensors[t]['size'] for t in union
                               if self.memory_pos(t) == pos) > .65 * capacity[pos]
                           for pos in ('L1', 'UB'))
            if current and (c != previous_core or len(current) >= group_size or pressure):
                flush()
                current, touched = [], set()
            previous_core = c
            current.append(v)
            touched.update(new_tensors)
        flush()
        return State(groups, tuple(tuple(x) for x in schedules))


class View:
    """低成本构造组依赖，不要求物理核心通信图无环。"""
    def __init__(self, model, state):
        self.mapping = {v: g for g, nodes in state.groups.items() for v in nodes}
        self.core = {g: c for c, order in enumerate(state.cores) for g in order}
        self.pred = {g: set() for g in state.groups}
        self.succ = {g: set() for g in state.groups}
        for v in model.nodes:
            a = self.mapping[v]
            for w in model.succ[v]:
                b = self.mapping[w]
                if a != b:
                    self.succ[a].add(b)
                    self.pred[b].add(a)

    def insertion_bounds(self, g, order):
        positions = {x: i for i, x in enumerate(order)}
        lower = max((positions[p]+1 for p in self.pred[g] if p in positions), default=0)
        upper = min((positions[s] for s in self.succ[g] if s in positions), default=len(order))
        return lower, upper


def score_key(result):
    """首先比较总完工时间；仅在时间完全相等时用搬运量打破平局。"""
    traffic = result['data_movement_bytes']
    return (result['makespan'], traffic['spill_added_copy_bytes'], traffic['scheduled_copy_bytes'])
