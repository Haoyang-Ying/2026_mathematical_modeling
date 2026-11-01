"""利用已有 BSPg 实现构造场景 B 的初始方案。

三种候选保留 BSPg 的 ready_p、ready_all 和亲和性机制。
连通分量引导只使用图结构，不依赖特定 case；不引入场景 A 的任务等待、容量分阶段切分或限时粗粒度收缩流程。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from q2_config import DEFAULT_ESCAPE_WEIGHT
from q2_bspg_core import generate_bspg_plan
from q2_model import State

BSPG_RULES = ('bspg_local', 'bspg_split', 'bspg_components')


@dataclass
class Guidance:
    group_of: dict
    owner: dict
    priority: dict
    hard_boundaries: bool = True


def component_guidance(model, cores):
    """保持计算连通分量完整，并平衡各流水线的工作量。
    
    不同分量仍可能共享 DDR 和外部输入张量。
    分量到核心的分配仅用于初始构造，后续搜索仍可拆分或迁移分量。"""
    seen, components = set(), []
    for root in model.topo:
        if root in seen:
            continue
        pending, members = [root], []
        seen.add(root)
        while pending:
            v = pending.pop()
            members.append(v)
            for w in (*model.pred[v], *model.succ[v]):
                if w not in seen:
                    seen.add(w)
                    pending.append(w)
        components.append(tuple(sorted(members, key=model.position.__getitem__)))
    if len(components) <= 1:
        return None, {'components': len(components), 'guidance_used': False}
    work = []
    for members in components:
        by_pipe = defaultdict(int)
        for v in members:
            by_pipe[model.ops[v]['pipe']] += model.work[v]
        work.append(dict(by_pipe))
    loads = [defaultdict(int) for _ in range(cores)]
    owner = {}
    for g in sorted(range(len(components)),
                    key=lambda g: (-max(work[g].values()), -sum(work[g].values()), g)):
        p = min(range(cores), key=lambda p: (
            max((loads[p][q] + work[g].get(q, 0)
                 for q in set(loads[p]) | set(work[g])), default=0),
            sum(loads[p].values()), p))
        owner[g] = p
        for q, cycles in work[g].items():
            loads[p][q] += cycles
    groups = {v: g for g, members in enumerate(components) for v in members}
    priority = {}
    for p in range(cores):
        for i, g in enumerate(g for g in range(len(components)) if owner[g] == p):
            priority[g] = i
    return Guidance(groups, owner, priority), {
        'components': len(components), 'guidance_used': True,
        'component_sizes': {str(n): sum(len(c) == n for c in components)
                            for n in sorted({len(c) for c in components})},
        'assigned_pipe_work': [dict(x) for x in loads]}


def bspg_greedy(model, cores, hardware, rule, *, escape_weight=DEFAULT_ESCAPE_WEIGHT):
    if rule not in BSPG_RULES:
        raise ValueError(f'unknown BSPg rule: {rule}')
    guidance, structural = None, {}
    if rule == 'bspg_components':
        guidance, structural = component_guidance(model, cores)
    stats = {}
    plan = generate_bspg_plan(
        model.graph, cores, bandwidth=hardware['bandwidth'], same_wait=0,
        cross_wait=hardware['cross_core_copy_delay'], statistics=stats,
        guidance=guidance, escape_weight=escape_weight,
        enable_splits=rule != 'bspg_local')
    state = State.from_plan(model, plan)
    return state, {'rule': rule, 'constructor': 'bspg_scene_b',
                   'escape_weight': escape_weight, 'structure': structural,
                   'bspg_statistics': stats}
