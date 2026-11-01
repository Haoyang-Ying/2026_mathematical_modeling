"""公共接口：所有默认参数均从当前解压后的代码包内部读取。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from q2_config import (DEFAULT_SEED, DEFAULT_ESCAPE_WEIGHT, DEFAULT_SEARCH_SECONDS,
                       CORE_TIMEOUT_SECONDS, MAX_SEARCH_SECONDS)
from q2_limits import bounded_api, mark_core, send_callback, send_checkpoint
from q2_model import Model, State
from q2_search import Settings, optimize, optimize_many
from q2_io import hardware, validate_plan, load_json


@bounded_api
def solve(graph, cores=4, *, profile='standard', seed=DEFAULT_SEED, config=None,
          group_size=8, ready_window=8, cache_mb=128, search_seconds=DEFAULT_SEARCH_SECONDS,
          constructor='bspg', escape_weight=DEFAULT_ESCAPE_WEIGHT,
          coarse_candidates=3, inherited_plans=(),
          mgr=True, mgr_max_nodes=512, mgr_budget=None,
          hc_mode='targeted',
          core_sharing=True, official=False, on_event=None, on_pool_update=None,
          core_timeout=CORE_TIMEOUT_SECONDS,
          search_max_seconds=MAX_SEARCH_SECONDS):
    """返回（提交方案，报告）；graph 可以是图数据字典，也可以是 JSON 文件路径。"""
    if not isinstance(graph, dict):
        # 保留输入路径以确定默认配置，再读取图数据。
        config = config if config is not None else Path(graph).resolve().parent/'config.txt'
        graph = load_json(graph)
    settings = Settings(profile, seed, group_size, ready_window, cache_mb, search_seconds,
                        constructor, escape_weight, coarse_candidates, mgr, mgr_max_nodes,
                        mgr_budget, hc_mode, core_sharing, core_timeout,
                        search_max_seconds)
    hw = hardware(config)
    model = Model(graph)
    inherited = []
    for i, value in enumerate(inherited_plans):
        if not isinstance(value, dict):
            value = load_json(value)
        inherited.append((f'inherited_{i}', State.from_plan(model, value)))
    plan, report = optimize(model, cores, hw, settings, on_event=on_event,
                            inherited=inherited, on_pool_update=on_pool_update)
    report['validation'] = validate_plan(graph, plan, hw, official=official,
                                        expected_t1=report['reference_T1'])
    if report['validation']['internal_audit']['makespan'] != report['final_T']:
        raise AssertionError('independent internal replay differs from final score')
    report['independent_internal_replay_validated'] = True
    if official:
        report['officially_validated'] = True
    return plan, report


@bounded_api
def solve_many(graph, cores=(2, 3, 4, 5), *, inherit=True, seed_plans=(),
               config=None, official=False, on_event=None, on_pool_update=None,
               on_checkpoint=None, checkpoint=None, **options):
    """调用算法的多核数联合求解器，返回 {m: (plan, final_pool_report)}。
    
    BSPg、HC 和 MGR 运行期间，所有可行候选均可改善其他核数的结果。
    先完成的核数也会接收后续阶段发现的改进。inherit=False 用于关闭双向共享的消融实验。
    每次改善或阶段完成时立即调用断点保存回调；seed_plans 可包含任意声明核数的方案。"""
    if type(inherit) is not bool:
        raise ValueError('inherit must be a boolean')
    if not isinstance(graph, dict):
        # 保留输入路径以确定默认配置，再读取图数据。
        config = config if config is not None else Path(graph).resolve().parent/'config.txt'
        graph = load_json(graph)
    options.setdefault('core_sharing', inherit)
    if not inherit and options['core_sharing']:
        raise ValueError('inherit=False conflicts with core_sharing=True')
    settings, hw, model = Settings(**options), hardware(config), Model(graph)
    seeds = [(f'seed_{i}', State.from_plan(model, value if isinstance(value, dict) else load_json(value)))
             for i, value in enumerate(seed_plans)]
    results = optimize_many(model, cores, hw, settings, on_event=on_event,
                            on_pool_update=on_pool_update, on_checkpoint=on_checkpoint,
                            checkpoint=checkpoint, seeds=seeds)
    # 所有搜索阶段完成后，再复核解池中最终选定的方案。
    for m, (plan, report) in results.items():
        mark_core(m)
        report['validation'] = validate_plan(graph, plan, hw, official=official,
                                            expected_t1=report['reference_T1'])
        if report['validation']['internal_audit']['makespan'] != report['final_T']:
            raise AssertionError('independent internal replay differs from final score')
        report['independent_internal_replay_validated'] = True
        if official:
            report['officially_validated'] = True
    return results


def _api_worker(name, args, kwargs, requested_callbacks):
    """进程间通信适配器；不改变候选生成、评分或接受规则。"""
    from copy import copy
    kwargs = dict(kwargs)
    if 'on_event' in requested_callbacks:
        kwargs['on_event'] = lambda *values: send_callback('on_event', *values)
    def pool_update(event, pool):
        if name == 'solve':
            send_checkpoint({'kind': 'single_core_pool', 'pool': pool.snapshot()})
        if 'on_pool_update' in requested_callbacks:
            observed = copy(pool)
            # 观察回调接收解池状态，不接收其中可变的编译缓存。
            observed.evaluator, observed.on_update = None, None
            send_callback('on_pool_update', event, observed)
    kwargs['on_pool_update'] = pool_update
    if name == 'solve_many':
        kwargs['on_checkpoint'] = send_checkpoint
    return globals()[name](*args, **kwargs)


def validate(graph, plan, *, config=None, official=True):
    """方案不合法，或官方评价与内部评价不一致时，抛出异常。"""
    if not isinstance(graph, dict):
        # 保留输入路径以确定默认配置，再读取图数据。
        config = config if config is not None else Path(graph).resolve().parent/'config.txt'
        graph = load_json(graph)
    if not isinstance(plan, dict):
        plan = load_json(plan)
    return validate_plan(graph, plan, hardware(config), official=official)


def diagnose(graph, plan, *, config=None):
    """可选的全局诊断：包括操作、搬运、换入换出及内存使用。"""
    from scene_b_internal import SceneBEvaluator
    if not isinstance(graph, dict):
        # 保留输入路径以确定默认配置，再读取图数据。
        config = config if config is not None else Path(graph).resolve().parent/'config.txt'
        graph = load_json(graph)
    if not isinstance(plan, dict):
        plan = load_json(plan)
    return SceneBEvaluator(graph, **hardware(config)).evaluate(plan, diagnostics=True)
