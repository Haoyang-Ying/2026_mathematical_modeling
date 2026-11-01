"""场景 B 的边界构造，改编自随题提供的官方评价器。

搬运操作创建顺序和绝对编号与参考实现一致。
仅复用静态来源索引，每次调用仍验证方案。
本模块不调用官方场景 B 构造器或模拟器。"""
import math
from schedule_step1 import _check_topo
from stub_multicore_cut_and_schedule import derive_multicore_plan

class SceneBEvaluationError(RuntimeError):
    pass

def _append_tensor(task_data, tensor):
    task_data['tensors'].setdefault(tensor['id'], dict(tensor))

def _append_op(task_data, op, subgraph_id):
    task_data['ops'].append(dict(op))
    task_data['op_subgraph'][op['id']] = subgraph_id

def _prioritize_task_seq(graph, raw_seq, op_subgraph, subgraph_order):
    """依照核内子图调度序分桶，桶内保留步骤1 顺序。

    输入raw_seq的完整性/拓扑性由步骤 1保证，稳定排序保持覆盖性。
    子图优先级是选手提供的新顺序，可能破坏拓扑，故只校验这一新约束。
    """
    rank = {subgraph_id: index
            for index, subgraph_id in enumerate(subgraph_order)}
    fallback = len(rank)
    seq = sorted(
        raw_seq,
        key=lambda op_id: rank.get(op_subgraph.get(op_id), fallback),
    )
    if not _check_topo(graph, seq):
        raise SceneBEvaluationError(
            'subgraph priority order violates an intra-core dependency')
    return seq

def build_core_graphs(graph_json, plan, bandwidth, source):
    """每核合并为一个任务，仅对跨核边插入 DDR COPY 对。

    原图/方案由本函数开头的derive_multicore_plan进行入口校验，参数由
    evaluate_scene_b校验。局部图保留本核依赖并插入源/汇COPY，仍为DAG；
    子图重排的拓扑性由_prioritize_task_seq保证，步骤2/3无需重复检查。
    """
    plan_view = derive_multicore_plan(graph_json, plan)
    num_cores = plan_view['num_cores']
    op_by_id, tensor_by_id = source.op_by_id, source.tensor_by_id
    mapping = plan_view['mapping']
    core_by_subgraph = plan_view['core_by_subgraph']
    core_by_op = {op_id: core_by_subgraph[subgraph_id]
                  for op_id, subgraph_id in mapping.items()}
    core_orders = plan_view['core_orders']
    producers, consumers, direct_edges = source.tensor_views
    tasks_data = {
        core_id: {
            'ops': [], 'tensors': {}, 'edges': [], 'op_subgraph': {},
            'subgraph_order': core_orders.get(core_id, []),
        }
        for core_id in range(num_cores)
    }
    for op_id, subgraph_id in mapping.items():
        _append_op(tasks_data[core_by_op[op_id]], op_by_id[op_id], subgraph_id)

    max_node_id = source.max_node_id
    next_id = max_node_id + 1
    cross_links = []
    cross_task_traffic = 0

    def new_id():
        nonlocal next_id
        value = next_id
        next_id += 1
        return value

    def add_copy_in(core_id, local_tid, size, subgraph_id, ddr_tid=None):
        ddr_tid = new_id() if ddr_tid is None else ddr_tid
        copy_id = new_id()
        _append_tensor(tasks_data[core_id], {
            'id': ddr_tid, 'pos': 'DDR', 'size': size})
        _append_op(tasks_data[core_id], {
            'id': copy_id, 'op': 'COPY_IN', 'pipe': 'PIPE_MTE2',
            'cycles': max(1, math.ceil(size / bandwidth)),
        }, subgraph_id)
        tasks_data[core_id]['edges'].extend([
            {'source': ddr_tid, 'target': copy_id},
            {'source': copy_id, 'target': local_tid},
        ])
        return copy_id, ddr_tid

    def add_copy_out(core_id, local_tid, size, subgraph_id, ddr_tid=None):
        ddr_tid = new_id() if ddr_tid is None else ddr_tid
        copy_id = new_id()
        _append_tensor(tasks_data[core_id], {
            'id': ddr_tid, 'pos': 'DDR', 'size': size})
        _append_op(tasks_data[core_id], {
            'id': copy_id, 'op': 'COPY_OUT', 'pipe': 'PIPE_MTE3',
            'cycles': max(1, math.ceil(size / bandwidth)),
        }, subgraph_id)
        tasks_data[core_id]['edges'].extend([
            {'source': local_tid, 'target': copy_id},
            {'source': copy_id, 'target': ddr_tid},
        ])
        return copy_id, ddr_tid

    for tensor_id in sorted(tensor_by_id):
        tensor = tensor_by_id[tensor_id]
        eligible_producers = sorted(op for op in producers.get(tensor_id, ())
                                    if op in mapping)
        eligible_consumers = sorted(op for op in consumers.get(tensor_id, ())
                                    if op in mapping)
        producer_cores = sorted({core_by_op[op] for op in eligible_producers})
        consumer_cores = sorted({core_by_op[op] for op in eligible_consumers})
        touched_cores = sorted(set(producer_cores) | set(consumer_cores))
        if not touched_cores:
            continue
        local_tensor = dict(tensor)
        if local_tensor.get('pos') == 'DDR':
            local_tensor['pos'] = 'UB'
        for core_id in touched_cores:
            _append_tensor(tasks_data[core_id], local_tensor)
            for op_id in eligible_producers:
                if core_by_op[op_id] == core_id:
                    tasks_data[core_id]['edges'].append(
                        {'source': op_id, 'target': tensor_id})
            for op_id in eligible_consumers:
                if core_by_op[op_id] == core_id:
                    tasks_data[core_id]['edges'].append(
                        {'source': tensor_id, 'target': op_id})

        # 图输入：每个消费核各自从 DDR 读入一次。
        if eligible_consumers and not eligible_producers:
            for dst_core in consumer_cores:
                dst_ops = [op for op in eligible_consumers
                           if core_by_op[op] == dst_core]
                dst_subgraph = min(
                    (mapping[op] for op in dst_ops),
                    key=lambda sg: core_orders[dst_core].index(sg))
                add_copy_in(dst_core, tensor_id, tensor['size'], dst_subgraph)

        # 图输出：保留对 DDR 的最终写回。
        has_original_copy_out = any(
            op_by_id[op_id].get('op') == 'COPY_OUT'
            for op_id in consumers.get(tensor_id, ()) if op_id in op_by_id)
        if eligible_producers and (has_original_copy_out or not eligible_consumers):
            for src_core in producer_cores:
                src_ops = [op for op in eligible_producers
                           if core_by_op[op] == src_core]
                src_subgraph = max(
                    (mapping[op] for op in src_ops),
                    key=lambda sg: core_orders[src_core].index(sg))
                add_copy_out(src_core, tensor_id, tensor['size'], src_subgraph)

        # 跨核张量：每个实际来源核心 → 目标核心连接一对 COPY。
        for src_core in producer_cores:
            src_ops = [op for op in eligible_producers
                       if core_by_op[op] == src_core]
            src_subgraph = max(
                (mapping[op] for op in src_ops),
                key=lambda sg: core_orders[src_core].index(sg))
            for dst_core in consumer_cores:
                if src_core == dst_core:
                    continue
                dst_ops = [op for op in eligible_consumers
                           if core_by_op[op] == dst_core]
                dst_subgraph = min(
                    (mapping[op] for op in dst_ops),
                    key=lambda sg: core_orders[dst_core].index(sg))
                ddr_tid = new_id()
                out_id, _ = add_copy_out(
                    src_core, tensor_id, tensor['size'], src_subgraph, ddr_tid)
                in_id, _ = add_copy_in(
                    dst_core, tensor_id, tensor['size'], dst_subgraph, ddr_tid)
                cross_links.append({
                    'tensor_id': tensor_id, 'size': tensor['size'],
                    'source_core': src_core, 'target_core': dst_core,
                    'source_copy_out_id': out_id,
                    'target_copy_in_id': in_id,
                })
                cross_task_traffic += tensor['size']

    # 直接 操作之间的 边：同核保留，跨核转成显式 COPY 对。
    for edge in direct_edges:
        src, dst = edge['source'], edge['target']
        if src not in mapping or dst not in mapping:
            continue
        src_core, dst_core = core_by_op[src], core_by_op[dst]
        if src_core == dst_core:
            tasks_data[src_core]['edges'].append(dict(edge))
            continue
        size = max(0, int(edge.get('data_size', 0)))
        local_tid, ddr_tid = new_id(), new_id()
        local_tensor = {'id': local_tid, 'pos': 'UB', 'size': size}
        _append_tensor(tasks_data[src_core], local_tensor)
        _append_tensor(tasks_data[dst_core], local_tensor)
        tasks_data[src_core]['edges'].append({'source': src, 'target': local_tid})
        tasks_data[dst_core]['edges'].append({'source': local_tid, 'target': dst})
        out_id, _ = add_copy_out(
            src_core, local_tid, size, mapping[src], ddr_tid)
        in_id, _ = add_copy_in(
            dst_core, local_tid, size, mapping[dst], ddr_tid)
        cross_links.append({
            'tensor_id': local_tid, 'size': size,
            'source_core': src_core, 'target_core': dst_core,
            'source_copy_out_id': out_id, 'target_copy_in_id': in_id,
        })
        cross_task_traffic += size

    return tasks_data, cross_links, cross_task_traffic, plan_view
