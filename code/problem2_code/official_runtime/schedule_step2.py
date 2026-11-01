# 本文件源于赛题提供的公共运行依赖，可执行逻辑保持原样。
# 来源说明：以下为官方提供的程序，补充了声明并将注释中文化，执行逻辑未改动。

"""
schedule_step2.py — 步骤 2: 缓存换入换出插入

对应文档: docs/核内调度算法.md § 2

输入接口:
    step2_spill_insertion(graph_json, seq, capacity):
        graph_json: dict 含 'ops' (id ≥ 1) 'tensors' (id ≥ 10001) 'edges' (op-张量二部图)
        seq: List[op_id] 步骤 1 输出的访问顺序
        capacity: dict[Type -> int]，来自 config.txt，只包含 L1 和 UB

输出:
    dict {
        'seq_ext': 扩展后的序列 (含 SPILL 节点 id),
        'spill_records': 列表，元素为 {spill_out_id?, spill_in_id, tid, from_tid,
            to_tid, pos, size, backing_tid, spill_out_copies_data, ...},
        'new_edges': 列表，元素为 (src, dst) 扩展图中新增的依赖边,
        'removed_edges': 被重命名替换的逻辑张量的消费者边,
        'ext_edges': 重命名后的完整边集合,
        'new_ops': 列表，元素为 {id, op, pipe, cycles} 扩展图中新增的 SPILL 节点,
        'new_tensors': 持久 DDR 后备副本与重命名片上张量,
        'overflow_log': 列表，元素为 {step, type, resid, capacity} SPILL 触发事件,
        'capacity': 本次使用的容量配置,
    }

不变量 (由 1.4 命题 + 本步溢出处理保证):
    1. 拓扑正确: 任意时刻每种 Type 的驻留量 ≤ 该 Type 容量
    2. 数据流保持: 每个张量在被 SPILL 期间仍可被所有原消费者访问 (经 DDR 跳板)
    3. seq_ext 是原 seq 的扩展 (含原 op + 插入的 SPILL 节点)
    4. 每个逻辑张量最多发生一次 SPILL_OUT DDR 写回；后续只生成 COPY_IN
"""

from contest_io import emit_error
from collections import defaultdict

class Step2SchedulingError(RuntimeError):
    """步骤2 无法在硬件容量内完成某个 op 的分配内存时抛出。"""

def _build_tensor_uses(graph_json, seq):
    """对每个张量给出按 seq 步骤排序的 (step, op_id) 列表"""
    n = len(seq)
    op_id_set = {o['id'] for o in graph_json['ops']}
    op_step = {op_id: i for i, op_id in enumerate(seq)}

    tensor_uses = defaultdict(list)
    for e in graph_json['edges']:
        if e['source'] in op_id_set and e['target'] not in op_id_set:
            tid, op = e['target'], e['source']
        elif e['source'] not in op_id_set and e['target'] in op_id_set:
            tid, op = e['source'], e['target']
        else:
            continue
        if op in op_step:
            tensor_uses[tid].append((op_step[op], op))
    for tid in tensor_uses:
        tensor_uses[tid].sort()
        # 去重: 同一步骤多次使用 (多个 op 在同步骤用同一张量) 只保留一个.
        # 否则中间使用处理会错位: 例如 uses=[(72),(79),(119),(119),(169)] 时,
        # 步骤 119 第一次中间使用错把 new_nu 设为 uses[3][0]=119 (应为 169)
        new_list = []
        last_step = -1
        for (s, op) in tensor_uses[tid]:
            if s != last_step:
                new_list.append((s, op))
                last_step = s
        tensor_uses[tid] = new_list
    return tensor_uses, op_step, op_id_set

def _find_copy_in_backings(graph_json):
    """找出由 COPY_IN 从唯一 DDR 张量拷入的片上张量。"""
    op_by_id = {op['id']: op for op in graph_json['ops']}
    tensor_by_id = {tensor['id']: tensor for tensor in graph_json['tensors']}
    op_ids = set(op_by_id)
    in_tids = defaultdict(list)
    out_tids = defaultdict(list)
    for edge in graph_json['edges']:
        src, dst = edge['source'], edge['target']
        if src not in op_ids and dst in op_ids:
            in_tids[dst].append(src)
        elif src in op_ids and dst not in op_ids:
            out_tids[src].append(dst)

    candidates = defaultdict(set)
    for op_id, op in op_by_id.items():
        if op.get('op') != 'COPY_IN':
            continue
        ddr_inputs = [
            tid for tid in in_tids[op_id]
            if tid in tensor_by_id and tensor_by_id[tid].get('pos') == 'DDR'
        ]
        if len(ddr_inputs) != 1:
            continue
        for tid in out_tids[op_id]:
            if tid in tensor_by_id and tensor_by_id[tid].get('pos') != 'DDR':
                candidates[tid].add(ddr_inputs[0])
    return {
        tid: next(iter(backings))
        for tid, backings in candidates.items() if len(backings) == 1
    }

def step2_spill_insertion(graph_json, seq, capacity):
    """
    Belady SPILL 插入
    输入约束：graph_json为已验证DAG；seq完整且拓扑正确；capacity合法。
    保证来源：选手入口验证图与容量，Step1按前驱访问生成完整拓扑序；
    题目2/3的子图优先级重排另行检查拓扑，此处不重复校验上述约束。
    capacity 必须由调用方从 config.txt 读出后传入，本模块不再内置默认容量。
    """
    n = len(seq)
    tensor_uses, op_step, op_id_set = _build_tensor_uses(graph_json, seq)
    original_copy_in_backing = _find_copy_in_backings(graph_json)

    # 每个张量的生命周期
    tensor_lifecycle = {}
    for t in graph_json['tensors']:
        tid = t['id']
        pos = t['pos']
        if pos == 'DDR':
            continue
        if tid not in tensor_uses:
            continue
        uses = tensor_uses[tid]
        tensor_lifecycle[tid] = {
            'pos': pos,
            'size': t['size'],
            'uses': uses,
            'first': uses[0][0],
            'last': uses[-1][0],
        }

    # 把“每个步骤扫描所有张量生命周期”改为一次性事件索引。
    # 按 tensor_lifecycle 的插入顺序建表，以保持原算法在同一步骤内的处理顺序。
    uses_at_step = [[] for _ in range(n)]
    for tid, info in tensor_lifecycle.items():
        for use_idx, (step, _) in enumerate(info['uses']):
            uses_at_step[step].append((tid, use_idx))

    # id 分配
    max_id = max(o['id'] for o in graph_json['ops'])
    max_tid = max((t['id'] for t in graph_json['tensors']), default=0)
    next_id = max(max_id, max_tid) + 1

    # 模拟状态
    # 驻留记录 active[T][tid] 保存下一次使用步骤与已使用次数，即 (next_use_step, used_count)。
    # dict 同时提供 O(1) 查找/删除，并保留与原 list 一致的插入顺序。
    type_active = {T: {} for T in capacity}
    type_resid = {T: 0 for T in capacity}
    # pending_spills 保存待插入的换出换入记录：换出位置、换入位置、张量编号、存储类型、字节数以及前后使用操作。
    pending_spills = []
    spill_in_at_step = defaultdict(list)
    overflow_log = []
    debug_over_count = []

    def get_resid(T):
        return type_resid[T]

    def trigger_one_spill(t, T, current_step_tids):
        """在步骤 t 触发一次 Belady SPILL, 返回 (victim_tid, freed_size) 或 None.
        排除当前步骤 t 正在使用的缓冲区 (即换出对象.uses 含步骤 t 的使用) —
        否则换出对象仍在被 op 读, SPILL_OUT 只能排在 op 之后, 造成 op 自身瞬时超 capacity.
        """
        # 只在 next_use 不是 None (即还有后续使用) 的候选中选
        finite = [
            (nu, ui, tid)
            for tid, (nu, ui) in type_active[T].items()
            if nu is not None
        ]
        # 排除当前步骤正在使用的缓冲区 (uses 列表里有使用的步骤 == t)
        finite = [item for item in finite if item[2] not in current_step_tids]
        if not finite:
            return None
        finite.sort(key=lambda x: -x[0])  # 优先选择下一次使用位置最远的驻留项。
        nu, ui, victim_tid = finite[0]
        info = tensor_lifecycle[victim_tid]
        v_size = info['size']

        # prev_use / next_use 的取值依据:
        #   active[T] 里存的是 (next_use, use_index)，use_index 表示换出对象
        #   已经被使用过的次数。溢出检查发生在步骤 t 的使用处理之后，因此
        #   ui 已是包含步骤 t 在内的已处理次数。
        #   于是: 下一次使用 = uses[ui] (存在且非 None，见上面的过滤)，
        #         上一次使用 = uses[ui-1] (ui > 0 时)，否则退回到首次分配内存的步骤。
        prev_use_step = info['uses'][ui - 1][0] if ui > 0 else info['first']
        next_use_step = nu  # 下一次使用位置；前面的筛选已保证其不是 None。
        prev_use_op = info['uses'][ui - 1][1] if ui > 0 else None
        next_use_op = info['uses'][ui][1]  # uses[ui] 表示下一次使用，二元组下标 1 对应操作编号。

        # 从 active 移除换出对象
        type_active[T].pop(victim_tid)
        type_resid[T] -= v_size

        pending_spills.append({
            'out_after_step': prev_use_step,
            'in_before_step': next_use_step,
            'tid': victim_tid,
            'pos': T,
            'size': v_size,
            'prev_use_op': prev_use_op,
            'next_use_op': next_use_op,
        })
        spill_in_at_step[next_use_step].append((victim_tid, ui))
        return victim_tid, v_size

    # 主模拟循环
    for t in range(n):
        op = seq[t]

        current_step_tids = {T: set() for T in capacity}
        for tid, _ in uses_at_step[t]:
            current_step_tids[tensor_lifecycle[tid]['pos']].add(tid)

        # 0. 处理已安排的 COPY_IN (在步骤 t 之前的物理换回)
        # 把换出对象重新分配内存到 L1/UB, 跟步骤 t 的 op 消费对齐.
        for tid, u_idx in spill_in_at_step.get(t, ()):
            info = tensor_lifecycle[tid]
            T = info['pos']
            uses = info['uses']
            new_nu = uses[u_idx + 1][0] if u_idx + 1 < len(uses) else None
            # 只在换出对象不在 type_active 时加入 (避免重复)
            if tid not in type_active[T]:
                type_active[T][tid] = (new_nu, u_idx)
                type_resid[T] += info['size']

        # 1. 处理本步使用的分配内存 / next_use 更新，但延迟末次使用后的释放。
        #    物理语义必须是分配内存 -> 容量检查/SPILL -> 执行 -> 释放内存；否则会漏掉
        #    “输出已经申请、末次输入尚未释放”的瞬态峰值。
        release_after_execute = []
        for tid, idx in uses_at_step[t]:
            info = tensor_lifecycle[tid]
            uses = info['uses']
            T = info['pos']
            if idx == 0:
                # 首次使用（生产者） = 申请内存
                next_nu = uses[1][0] if 1 < len(uses) else None
                type_active[T][tid] = (next_nu, 1)
                type_resid[T] += info['size']
            elif idx < len(uses) - 1:
                # 中间使用：更新 next_use。对已有 key 赋值不改变 dict 插入顺序。
                new_nu = uses[idx + 1][0]
                active_item = type_active[T].get(tid)
                if active_item is not None:
                    _, used_count = active_item
                    type_active[T][tid] = (new_nu, used_count + 1)
            # k=1 的张量同时是首次使用和末次使用：先分配内存参与瞬态容量
            # 检查，执行完成后再释放。普通末次输入同理延迟到容量检查之后释放。
            if idx == len(uses) - 1:
                release_after_execute.append((T, tid))

        # 1.5 分配内存后、执行/释放内存前统一 Belady SPILL。
        #     trigger_one_spill 会排除当前 op 使用的所有张量，因此只会换出当前
        #     op 不需要的驻留项。最终生成的 SPILL_OUT 锚定在换出对象上次使用之后，
        #     在扩展序列中物理发生于当前 op 之前。
        for T, C in capacity.items():
            while get_resid(T) > C:
                resid_now = get_resid(T)
                overflow_log.append({'step': t, 'type': T, 'resid': resid_now, 'capacity': C})
                result = trigger_one_spill(t, T, current_step_tids[T])
                if result is None:
                    current_tids = sorted(current_step_tids[T].intersection(type_active[T]))
                    active_desc = sorted(
                        (tid, tensor_lifecycle[tid]['size'], nu)
                        for tid, (nu, _) in type_active[T].items()
                    )
                    message = (
                        '[STEP2 ERROR] no spill victim: step={step} op={op} type={type_} '
                        'alloc_resid={resid} capacity={capacity} current_tids={current} '
                        'active(tid,size,next_use)={active}'
                    ).format(
                        step=t, op=op, type_=T, resid=resid_now, capacity=C,
                        current=current_tids, active=active_desc,
                    )
                    emit_error(message)
                    raise Step2SchedulingError(message)

            # 这是当前 op 真正的分配内存后峰值；此时输入和输出都仍然驻留。
            resid_after_alloc = get_resid(T)
            if resid_after_alloc > C:
                # while 的后置断言，防止未来修改重新引入静默超限。
                message = (
                    '[STEP2 ERROR] alloc peak still exceeds capacity after spill: '
                    'step={step} op={op} type={type_} resid={resid} capacity={capacity}'
                ).format(step=t, op=op, type_=T, resid=resid_after_alloc, capacity=C)
                emit_error(message)
                raise Step2SchedulingError(message)

        # 2. 当前 op 执行完成后，释放本步末次使用的输入/输出。
        for T, tid in release_after_execute:
            if type_active[T].pop(tid, None) is not None:
                type_resid[T] -= tensor_lifecycle[tid]['size']

    # 3. 构建重命名后的扩展图。
    # 每次 COPY_IN 产生一个新的片上张量物理驻留实例；它本身就是一段物理
    # 驻留生命周期。仅释放而不搬运数据的 COPY_OUT 不再创建，真实 DDR 写回仍保留。
    new_ops_list = []
    new_tensors_list = []
    new_edges = []
    removed_edges = []
    spill_records = []

    # 收集插入点
    insert_after = defaultdict(list)  # 从步骤编号映射到该步骤之后需要执行的真实换出操作列表。
    insert_before = defaultdict(list)

    backing_by_tid = dict(original_copy_in_backing)
    backing_origin = {
        tid: 'original_copy_in' for tid in original_copy_in_backing
    }
    current_incarnation = {
        tid: tid for tid in tensor_lifecycle
    }
    incarnation_version = defaultdict(int)
    spill_records_by_tid = defaultdict(list)

    for sp in pending_spills:
        logical_tid = sp['tid']
        from_tid = current_incarnation[logical_tid]
        spill_out_copies_data = sp['tid'] not in backing_by_tid
        spill_out_id = None
        if spill_out_copies_data:
            spill_out_id = next_id
            next_id += 1
        spill_in_id = next_id
        next_id += 1
        if spill_out_copies_data:
            backing_tid = next_id
            next_id += 1
            backing_by_tid[sp['tid']] = backing_tid
            backing_origin[sp['tid']] = 'spill_out'
            new_tensors_list.append({
                'id': backing_tid,
                'pos': 'DDR',
                'size': sp['size'],
            })
            backing_source = 'new_spill_out'
        else:
            backing_tid = backing_by_tid[sp['tid']]
            backing_source = (
                'original_copy_in'
                if backing_origin[sp['tid']] == 'original_copy_in'
                else 'reused_spill_out'
            )

        incarnation_version[logical_tid] += 1
        to_tid = next_id
        next_id += 1
        current_incarnation[logical_tid] = to_tid
        new_tensors_list.append({
            'id': to_tid,
            'logical_tid': logical_tid,
            'version': incarnation_version[logical_tid],
            'pos': sp['pos'],
            'size': sp['size'],
        })

        if spill_out_copies_data:
            new_ops_list.append({
                'id': spill_out_id,
                'op': 'COPY_OUT',
                'pipe': 'PIPE_MTE3',
                'cycles': max(1, sp['size'] // 64),
                'transfer_bytes': sp['size'],
                'spill_logical_tid': logical_tid,
            })
        new_ops_list.append({
            'id': spill_in_id,
            'op': 'COPY_IN',
            'pipe': 'PIPE_MTE2',
            'cycles': max(1, sp['size'] // 64),
            'transfer_bytes': sp['size'],
            'spill_logical_tid': logical_tid,
        })

        # 数据边直接表达物理物理驻留实例的生产与消费。COPY_OUT 也是 from_tid
        # 的一个消费者；步骤3 的引用计数会等所有消费者结束后再释放 from_tid。
        if spill_out_copies_data:
            new_edges.append((from_tid, spill_out_id))
            new_edges.append((spill_out_id, backing_tid))
        new_edges.append((backing_tid, spill_in_id))
        new_edges.append((spill_in_id, to_tid))

        if spill_out_id is not None:
            insert_after[sp['out_after_step']].append(spill_out_id)
        insert_before[sp['in_before_step']].append(spill_in_id)

        record = {
            'spill_out_id': spill_out_id,
            'spill_in_id': spill_in_id,
            'swap_tid': backing_tid,
            'backing_tid': backing_tid,
            'backing_source': backing_source,
            'spill_out_copies_data': spill_out_copies_data,
            'tid': logical_tid,
            'logical_tid': logical_tid,
            'from_tid': from_tid,
            'to_tid': to_tid,
            'version': incarnation_version[logical_tid],
            'pos': sp['pos'],
            'size': sp['size'],
            'prev_use_op': sp['prev_use_op'],
            'next_use_op': sp['next_use_op'],
            'prev_use_step': sp['out_after_step'],
            'next_use_step': sp['in_before_step'],
        }
        spill_records.append(record)
        spill_records_by_tid[logical_tid].append(record)

    # 把每条逻辑张量 → 消费者边改接到该消费者所属的物理驻留实例。
    # 生产者 → 原始张量边保持不变；COPY_IN -> 重命名张量边已在上面加入。
    op_step = {op_id: step for step, op_id in enumerate(seq)}
    consumer_incarnation = {}
    for logical_tid, records in spill_records_by_tid.items():
        record_index = 0
        physical_tid = logical_tid
        for step, consumer_op in tensor_uses[logical_tid]:
            while (record_index < len(records)
                   and step >= records[record_index]['next_use_step']):
                physical_tid = records[record_index]['to_tid']
                record_index += 1
            consumer_incarnation[(logical_tid, consumer_op)] = physical_tid
    rewired_edges = []
    for edge in graph_json['edges']:
        src, dst = edge['source'], edge['target']
        replacement = None
        if src in spill_records_by_tid and dst in op_step:
            physical_tid = consumer_incarnation.get((src, dst), src)
            if physical_tid != src:
                replacement = (physical_tid, dst)
        if replacement is None:
            rewired_edges.append(dict(edge))
        else:
            removed_edges.append((src, dst))
            rewired_edges.append({
                **edge,
                'source': replacement[0],
                'target': replacement[1],
            })

    ext_edges = rewired_edges + [
        {'source': src, 'target': dst, 'data_size': 0}
        for src, dst in new_edges
    ]

    # 构造插入换入换出操作后的扩展序列 seq_ext。
    seq_ext = []
    for i, op in enumerate(seq):
        # '第 i-1 步之后' 事件
        if i > 0:
            for sid in insert_after[i - 1]:
                seq_ext.append(sid)
        # '第 i 步之前' 事件
        for sid in insert_before[i]:
            seq_ext.append(sid)
        seq_ext.append(op)
    # '第 n-1 步之后' 事件 (循环结束后)
    for sid in insert_after[n - 1]:
        seq_ext.append(sid)

    return {
        'seq_ext': seq_ext,
        'spill_records': spill_records,
        'new_edges': new_edges,
        'removed_edges': removed_edges,
        'ext_edges': ext_edges,
        'new_ops': new_ops_list,
        'new_tensors': new_tensors_list,
        'overflow_log': overflow_log,
        'capacity': capacity,
        '_debug_over_count': debug_over_count,
    }


# ==================== 命令行入口 ====================
def _build_extended_graph(graph_json, result):
    """组装Step2输出，不重复校验。

    输入约束：result来自同一原图的step2_spill_insertion。Step2统一分配
    新ID并在原拓扑访问点插入SPILL，故ext_edges与seq_ext匹配、覆盖完整。
    此函数不改顺序和依赖，只组装这两份关联输出。
    """
    ops = list(graph_json['ops'])
    ops.extend(result['new_ops'])
    tensors = list(graph_json['tensors'])
    tensors.extend(result['new_tensors'])
    edges = list(result.get('ext_edges', graph_json['edges']))
    return {
        'ops': ops,
        'tensors': tensors,
        'edges': edges,
        'seq_ext': result['seq_ext'],
    }

def _simulate_occupancy(ext_graph, result, seq_orig):
    """按重命名张量的生产者/消费者引用计数复演 seq_ext 驻留量。"""
    del result, seq_orig
    ops = ext_graph['ops']
    tensors = ext_graph['tensors']
    edges = ext_graph['edges']
    seq = ext_graph['seq_ext']
    n = len(seq)
    tensor_by_id = {t['id']: t for t in tensors}
    op_id_set = {o['id'] for o in ops}
    in_tids = {o['id']: [] for o in ops}
    out_tids = {o['id']: [] for o in ops}
    for e in edges:
        if e['source'] in op_id_set and e['target'] not in op_id_set:
            out_tids[e['source']].append(e['target'])
        elif e['target'] in op_id_set and e['source'] not in op_id_set:
            in_tids[e['target']].append(e['source'])

    consumer_count = defaultdict(int)
    producers = set()
    for op_id in out_tids:
        producers.update(out_tids[op_id])
    for op_id, tids in in_tids.items():
        for tid in set(tids):
            consumer_count[tid] += 1

    l1_occ_alloc = [0] * n
    ub_occ_alloc = [0] * n
    l1_occ = [0] * n
    ub_occ = [0] * n
    events_per_step = [[] for _ in range(n)]
    active = set()
    used = {'L1': 0, 'UB': 0}

    # 无生产者的片上图输入在步骤 0 前已驻留。
    for tid, count in consumer_count.items():
        tensor = tensor_by_id.get(tid)
        if (count and tid not in producers and tensor is not None
                and tensor.get('pos') in used):
            active.add(tid)
            used[tensor['pos']] += tensor['size']
            if n:
                events_per_step[0].append(
                    (tid, tensor['pos'], tensor['size'], 'initial_alloc'))

    for i, op_id in enumerate(seq):
        for tid in set(out_tids[op_id]):
            tensor = tensor_by_id.get(tid)
            if tensor is not None and tensor.get('pos') in used and tid not in active:
                active.add(tid)
                used[tensor['pos']] += tensor['size']
                events_per_step[i].append(
                    (tid, tensor['pos'], tensor['size'], 'alloc'))

        l1_occ_alloc[i], ub_occ_alloc[i] = used['L1'], used['UB']

        for tid in set(in_tids[op_id]):
            if tid not in consumer_count:
                continue
            consumer_count[tid] -= 1
            if consumer_count[tid] == 0 and tid in active:
                tensor = tensor_by_id[tid]
                active.remove(tid)
                used[tensor['pos']] -= tensor['size']
                events_per_step[i].append(
                    (tid, tensor['pos'], tensor['size'], 'free'))
        for tid in set(out_tids[op_id]):
            if consumer_count.get(tid, 0) == 0 and tid in active:
                tensor = tensor_by_id[tid]
                active.remove(tid)
                used[tensor['pos']] -= tensor['size']
                events_per_step[i].append(
                    (tid, tensor['pos'], tensor['size'], 'free'))

        l1_occ[i], ub_occ[i] = used['L1'], used['UB']

    l1_peak = max(l1_occ_alloc) if l1_occ_alloc else 0
    ub_peak = max(ub_occ_alloc) if ub_occ_alloc else 0
    l1_peak_idx = l1_occ_alloc.index(l1_peak) if l1_peak > 0 else -1
    ub_peak_idx = ub_occ_alloc.index(ub_peak) if ub_peak > 0 else -1
    return l1_occ_alloc, ub_occ_alloc, l1_occ, ub_occ, events_per_step, l1_peak, l1_peak_idx, ub_peak, ub_peak_idx
