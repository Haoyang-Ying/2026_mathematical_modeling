'仅负责生成方案的接口；完整评价由调用方另行执行。'
from time import perf_counter

from bspg_branch_split import _build_op_graph, generate_bspg_plan
from coarse_guidance import build_guidance
from dagp_model import Model
from time_budget import base_seconds as budget_seconds


def generate_guided_plan(case, m, hardware, *, model=None, guide_seconds=budget_seconds(2.0), statistics=None,
                         max_nodes=2000,max_candidates=32,group_policy='validated'):
    started = perf_counter()
    model = model or Model(case,hardware)
    graph = _build_op_graph(case)
    guidance = build_guidance(model,m,graph,time_limit=guide_seconds,group_policy=group_policy)
    fine = {}
    plan = generate_bspg_plan(case,m,bandwidth=hardware.bandwidth,
                             same_wait=hardware.same_wait,cross_wait=hardware.cross_wait,
                             guidance=guidance,statistics=fine,
                             max_nodes=max_nodes,max_candidates=max_candidates)
    model.validate_plan(plan,m)
    if statistics is not None:
        statistics.update(guidance=guidance.statistics,bspg=fine,
                          generation_seconds=perf_counter()-started)
    return plan
