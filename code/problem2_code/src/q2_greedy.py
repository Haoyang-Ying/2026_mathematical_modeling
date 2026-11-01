"""三种确定性规则驱动的列表构造器，使用低成本近似代价。

近似状态跟踪每核四条流水线、张量目标去重、暴露的搬运延迟及保守的存活张量字节数。
DDR 总工作量仅给出下界，不精确模拟带宽争用。
完整方案最终由 SceneBEvaluator 评价是否可接受。"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from q2_model import PIPES


RULES = {
    'critical': {'rank': 1.0, 'traffic': .10, 'spill': .50, 'release': .0},
    'locality': {'rank': .35, 'traffic': 2.0, 'spill': .75, 'release': .10},
    'memory': {'rank': .40, 'traffic': .50, 'spill': 3.0, 'release': .60},
}


@dataclass
class Trial:
    finish: float
    pipes: list
    receives: dict
    sends: dict
    traffic: int
    spill_estimate: int
    released: int


class Constructor:
    def __init__(self, model, cores, hardware):
        self.model, self.cores = model, cores
        self.bandwidth = hardware['bandwidth']
        self.delay = hardware['cross_core_copy_delay']
        self.capacity = hardware['capacity']
        self.pipes = [{p: 0.0 for p in PIPES} for _ in range(cores)]
        self.loads = [{p: 0.0 for p in PIPES} for _ in range(cores)]
        self.placement, self.finish = {}, {}
        self.receives, self.sends = {}, {}
        self.resident = [set() for _ in range(cores)]
        self.live = [{p: 0 for p in ('L1', 'UB')} for _ in range(cores)]
        self.remaining = {t: len(model.consumers[t]) for t in model.tensors}
        self.ddr_bytes = 0

    def trial(self, v, c):
        m = self.model
        pipes = [dict(x) for x in self.pipes]
        receives, sends = {}, {}
        ready = max((self.finish[p] for p in m.pred[v]), default=0)
        traffic = 0
        for t in m.inputs[v]:
            size = m.tensors[t]['size']
            duration = max(1, math.ceil(size / self.bandwidth))
            sources = {self.placement[p] for p in m.producers[t]}
            if not sources:
                key = (t, c)
                arrival = self.receives.get(key)
                if arrival is None:
                    arrival = pipes[c]['PIPE_MTE2'] + duration
                    pipes[c]['PIPE_MTE2'] = arrival
                    receives[key] = arrival
                    traffic += size
                ready = max(ready, arrival)
            for sc in sorted(sources - {c}):
                key = (t, sc, c)
                arrival = self.sends.get(key)
                if arrival is None:
                    source_end = max(self.finish[p] for p in m.producers[t]
                                     if self.placement[p] == sc)
                    out_end = max(source_end, pipes[sc]['PIPE_MTE3']) + duration
                    pipes[sc]['PIPE_MTE3'] = out_end
                    arrival = max(out_end + self.delay, pipes[c]['PIPE_MTE2']) + duration
                    pipes[c]['PIPE_MTE2'] = arrival
                    sends[key] = arrival
                    traffic += 2 * size
                ready = max(ready, arrival)
        for p, size in m.direct_in[v]:
            sc = self.placement[p]
            if sc == c:
                continue
            duration = max(1, math.ceil(size / self.bandwidth))
            out_end = max(self.finish[p], pipes[sc]['PIPE_MTE3']) + duration
            pipes[sc]['PIPE_MTE3'] = out_end
            arrival = max(out_end + self.delay, pipes[c]['PIPE_MTE2']) + duration
            pipes[c]['PIPE_MTE2'] = arrival
            ready = max(ready, arrival)
            traffic += 2 * size
        added = (set(m.inputs[v]) | set(m.outputs[v])) - self.resident[c]
        peak = dict(self.live[c])
        for t in added:
            peak[m.memory_pos(t)] += m.tensors[t]['size']
        spill = sum(max(0, peak[pos]-self.capacity[pos]) for pos in peak)
        released = sum(m.tensors[t]['size'] for t in m.inputs[v] if self.remaining[t] == 1)
        pipe = m.ops[v]['pipe']
        finish = max(ready, pipes[c][pipe]) + m.work[v]
        pipes[c][pipe] = finish
        # 终端张量即使没有符合条件的消费者，也需要最终写回。
        for t in m.outputs[v]:
            if not m.consumers[t]:
                size = m.tensors[t]['size']
                pipes[c]['PIPE_MTE3'] = max(finish, pipes[c]['PIPE_MTE3']) + max(
                    1, math.ceil(size / self.bandwidth))
                traffic += size
        return Trial(finish, pipes, receives, sends, traffic, spill, released)

    def commit(self, v, c, trial):
        m = self.model
        self.placement[v], self.finish[v] = c, trial.finish
        self.pipes = trial.pipes
        self.receives.update(trial.receives)
        self.sends.update(trial.sends)
        self.ddr_bytes += trial.traffic
        self.loads[c][m.ops[v]['pipe']] += m.work[v]
        for t in set(m.inputs[v]) | set(m.outputs[v]):
            if t not in self.resident[c]:
                self.resident[c].add(t)
                self.live[c][m.memory_pos(t)] += m.tensors[t]['size']
        for t in m.inputs[v]:
            self.remaining[t] -= 1
        # 保守静态存活性：仅当全部尚未调度的消费者均已消失时释放。
        # 不据此选择或修改官方换出决策。
        for t in set(m.inputs[v]) | set(m.outputs[v]):
            if self.remaining[t] == 0:
                for core in range(self.cores):
                    if t in self.resident[core]:
                        self.resident[core].remove(t)
                        self.live[core][m.memory_pos(t)] -= m.tensors[t]['size']


def greedy(model, cores, hardware, rule, group_size=8, ready_window=8):
    weights = RULES[rule]
    builder = Constructor(model, cores, hardware)
    degree = {v: len(model.pred[v]) for v in model.nodes}
    ready = [(-model.rank[v], v) for v in model.nodes if degree[v] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        pool = [heapq.heappop(ready) for _ in range(min(ready_window, len(ready)))]
        top_rank = -pool[0][0]
        best = None
        for _, v in pool:
            for c in range(cores):
                trial = builder.trial(v, c)
                traffic_time = trial.traffic / builder.bandwidth
                # 总搬运量除以带宽仅作为筛选下界。
                projected_end = max(trial.finish,
                                    (builder.ddr_bytes + trial.traffic) / builder.bandwidth)
                value = (projected_end + weights['rank'] * (top_rank-model.rank[v])
                         + weights['traffic'] * traffic_time
                         + weights['spill'] * 2 * trial.spill_estimate / builder.bandwidth
                         - weights['release'] * trial.released / builder.bandwidth)
                key = (value, builder.loads[c][model.ops[v]['pipe']],
                       -model.rank[v], v, c)
                if best is None or key < best[0]:
                    best = key, v, c, trial
        _, v, c, trial = best
        builder.commit(v, c, trial)
        order.append(v)
        for item in pool:
            if item[1] != v:
                heapq.heappush(ready, item)
        for w in model.succ[v]:
            degree[w] -= 1
            if degree[w] == 0:
                heapq.heappush(ready, (-model.rank[w], w))
    state = model.group_runs(order, builder.placement, cores, group_size, hardware['capacity'])
    return state, {'rule': rule, 'weights': dict(weights), 'group_size': group_size,
                   'ready_window': ready_window, 'proxy_ddr_bytes': builder.ddr_bytes,
                   'proxy_makespan': max(builder.finish.values(), default=0)}
