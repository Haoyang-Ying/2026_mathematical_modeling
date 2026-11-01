'适用于场景 A 贪心与区域大邻域搜索的分时间片 HC。\n\n局部代理仅负责候选排序；每个被接受方案均由共享 TemplateEvaluator 完整回放，并保留实际 任务 编号。本模块不调用官方完整评价器。'

from __future__ import annotations

import bisect
from collections import OrderedDict
from time import perf_counter
from types import SimpleNamespace

from dagp_execution import _preparation_deadline
from legacy_hc import _LocalProxy, _batch_groups, _members, _merge, _pairs
from time_budget import seconds as budget_seconds


def _copy_plan(plan):
    return {"node_to_subgraph": {str(op): task for op, task in
                                 plan["node_to_subgraph"].items()},
            "core_schedules": [list(order) for order in plan["core_schedules"]]}


class FineSearch:
    '可复用的 HC 引擎，deadline 为 perf_counter 的绝对截止时刻。\n\non_improvement(plan, metrics) 仅在详细评价完成后调用。指定 region_ops 时，只修改全部成员位于该区域内的 任务；protect_assignment 进一步保留每个算子的归属核。区域外 任务 的成员、编号、核和相对顺序保持不变。'

    def __init__(self, model, evaluator, m):
        self.model, self.evaluator, self.m = model, evaluator, m
        self.graph = SimpleNamespace(cycles=model.cycles,
                                     predecessors=model.pred,
                                     successors=model.succ)
        self.proxy = None
        # 缓存保留真实编号和全部核内顺序；规范化 任务 编号可能改变官方边界 COPY 编号，因此不能随意使用。
        self._scores = OrderedDict()
        self._cursors = OrderedDict()
        # 迁移轮转游标独立于候选对游标和当前方案编号；即使无合并候选导致短时间片之间重置候选对游标，迁移仍应最终遍历其他各核。
        self._move_batch = 0

    def _key(self, plan):
        mapping = plan["node_to_subgraph"]
        return (tuple(mapping[str(op)] for op in self.model.ids),
                tuple(tuple(order) for order in plan["core_schedules"]))

    def _remember(self, key, metrics):
        self._scores[key] = metrics
        self._scores.move_to_end(key)
        while len(self._scores) > 24:
            self._scores.popitem(last=False)

    def _calibrate(self, plan, members, metrics):
        actual = {int(task): value for task, value in metrics["durations"].items()}
        # 每次接受方案和每轮运行后重新计算；仅 任务 编号一致不足以复用，因为成员和边界分配可能已改变。
        self.proxy.offset = {
            task: actual[task] - self.proxy.task_cost(ops)[0]
            for task, ops in members.items()
        }
        return self.proxy.state(plan, members)

    def _candidates(self, plan, members, state, allowed, protect, cursor,
                    deadline, stats):
        '生成小规模有界候选批次；下次运行从另一批候选对继续。'
        move_batch = self._move_batch
        self._move_batch += 1
        proxy = self.proxy
        core_of = {task: core for core, order in enumerate(plan["core_schedules"])
                   for task in order}
        local_end = min(deadline, perf_counter() + min(budget_seconds(.35), max(budget_seconds(.01),
                                  (deadline - perf_counter()) * .20)))
        candidates = []
        seen = set()

        def add(candidate, candidate_members, kind, new_state=None):
            if perf_counter() >= local_end:
                return
            stats["local_candidates"] += 1
            if new_state is None:
                new_state = proxy.state(candidate, candidate_members)
            if new_state is None:
                stats["cycle_rejections"] += 1
                return
            key = self._key(candidate)
            if key in seen:
                return
            seen.add(key)
            candidates.append((new_state.score, kind, candidate, key))

        # 重排与迁移可复用 任务 模板，因此短时间片中先于昂贵的合并模板进行考虑。
        schedules = plan["core_schedules"]
        starts = {task: state.ends[task] - state.durations[task] for task in members}
        ranked = sorted(allowed, key=lambda task: (state.slack[task],
                         -state.durations[task], task))
        if ranked:
            offset = cursor % len(ranked)
            ranked = (ranked[offset:] + ranked[:offset])[:8]
        if not protect and self.m > 1:
            for task in ranked:
                source = core_of[task]
                destinations = sorted((core for core in range(self.m) if core != source),
                        key=lambda core: state.ends[schedules[core][-1]]
                        if schedules[core] else 0)
                if len(destinations) > 2:
                    # 保留最早完成的目标位置，并轮换一个备选位置，不增加本批移动数量。
                    destinations = [destinations[0],
                                    destinations[1 + move_batch % (len(destinations) - 1)]]
                for target in destinations:
                    order = schedules[target]
                    insertion = bisect.bisect_left([starts[t] for t in order], starts[task])
                    for position in sorted({insertion, max(0, insertion - 1)}):
                        if perf_counter() >= local_end:
                            break
                        candidate = _copy_plan(plan)
                        candidate["core_schedules"][source].remove(task)
                        candidate["core_schedules"][target].insert(position, task)
                        add(candidate, members, "move_task")

        swaps = []
        for core, order in enumerate(schedules):
            for position, (a, b) in enumerate(zip(order, order[1:])):
                if a in allowed and b in allowed:
                    swaps.append((min(state.slack[a], state.slack[b]), core, position))
        for _, core, position in sorted(swaps)[cursor:cursor + 8]:
            if perf_counter() >= local_end:
                break
            candidate = _copy_plan(plan)
            order = candidate["core_schedules"][core]
            order[position], order[position + 1] = order[position + 1], order[position]
            add(candidate, members, "swap_adjacent")

        pairs = _pairs(plan, self.graph, state, limit=12, offset=cursor)
        for source, target in pairs:
            if source not in allowed or target not in allowed:
                continue
            if protect and core_of[source] != core_of[target]:
                continue
            for group, label in _batch_groups(source, target, members, proxy, 64):
                if perf_counter() >= local_end:
                    break
                candidate, new_members = _merge(plan, members, source, target, group)
                # 成员集合发生变化后，即使 任务 编号不变，也不能套用旧成员集合校准出的偏移量。
                old_offset = proxy.offset
                proxy.offset = {task: value for task, value in old_offset.items()
                                if task not in (source, target)}
                try:
                    new_state = proxy.state_after_merge(candidate, new_members, state,
                                                         members, source, target)
                finally:
                    proxy.offset = old_offset
                # 这里的 None 表示商图或核内顺序产生环，并非要求通过 add 的通用路径重新计算。
                if new_state is None:
                    stats["local_candidates"] += 1
                    stats["cycle_rejections"] += 1
                    continue
                kind = "merge_task" if group == members[source] else "merge_boundary"
                add(candidate, new_members, kind, new_state)
        return candidates, bool(pairs), cursor + 12

    @staticmethod
    def _shortlist(candidates):
        '保留多个邻域操作类别，也包括代理评价较差的候选。'
        candidates = sorted(candidates, key=lambda row: row[0])
        chosen = []
        used = set()
        for kind in ("move_task", "merge_task", "merge_boundary", "swap_adjacent"):
            for row in candidates:
                if row[1] == kind and row[3] not in used:
                    chosen.append(row)
                    used.add(row[3])
                    break
        for row in candidates:
            if len(chosen) >= 6:
                break
            if row[3] not in used:
                chosen.append(row)
                used.add(row[3])
        return chosen

    def run(self, plan, metrics, deadline, *, region_ops=None,
            protect_assignment=False, on_improvement=None):
        '返回已完成评价的最优方案、详细指标及时间片统计。\n\n准备、局部排序和回放共用同一时间片截止时刻。Unix 定时中断只包围局部代理运算；评价器自行管理准备阶段中断及缓存、编号分配提交。外层工作进程提供可移植的进程级硬限。'
        started = perf_counter()
        best_plan, best_metrics = plan, metrics
        stats = {"local_candidates": 0, "detailed": 0, "accepted": 0,
                 "timeout": False, "candidate_timeouts": 0,
                 "cycle_rejections": 0, "invalid_candidates": 0,
                 "cache_hits": 0, "moves": [], "stop": "exhausted"}
        if started >= deadline:
            stats.update(timeout=True, stop="deadline", seconds=perf_counter() - started)
            return best_plan, best_metrics, stats
        attempted = set()
        try:
            with _preparation_deadline(deadline):
                current = _copy_plan(plan)
                self.model.validate_plan(current, self.m)
                region = None if region_ops is None else frozenset(region_ops)
                if self.proxy is None:
                    hardware = self.model.hardware
                    self.proxy = _LocalProxy(self.model.source_graph, self.graph,
                        hardware.bandwidth, hardware.cross_wait, hardware.same_wait, .03)
                key = self._key(current)
                self._remember(key, metrics)
                cursor_key = (key, region, protect_assignment)
                cursor = self._cursors.get(cursor_key, 0)
                members = _members(current)
                state = self._calibrate(current, members, best_metrics)
                if state is None:
                    raise ValueError("HC input has cyclic Task/core orders")
            stale_batches = 0
            while perf_counter() < deadline:
                allowed = set(members) if region is None else {
                    task for task, ops in members.items() if ops <= region}
                if not allowed:
                    break
                with _preparation_deadline(deadline):
                    candidates, has_pairs, cursor = self._candidates(current, members,
                        state, allowed, protect_assignment, cursor, deadline, stats)
                self._cursors[cursor_key] = cursor if has_pairs else 0
                self._cursors.move_to_end(cursor_key)
                while len(self._cursors) > 16:
                    self._cursors.popitem(last=False)
                improved = False
                for _, kind, candidate, candidate_key in self._shortlist(candidates):
                    if candidate_key in attempted:
                        continue
                    attempted.add(candidate_key)
                    remaining = deadline - perf_counter()
                    if remaining <= budget_seconds(.001):
                        raise TimeoutError("HC deadline reached")
                    observed = self._scores.get(candidate_key)
                    if observed is None:
                        # 昂贵的新 任务 不应消耗整个时间片；已完成模板在中断后仍保留。
                        candidate_deadline = min(deadline, perf_counter() +
                                                 min(budget_seconds(12.0), max(budget_seconds(.01), remaining * .45)))
                        try:
                            with _preparation_deadline(candidate_deadline):
                                self.model.validate_plan(candidate, self.m)
                            stats["detailed"] += 1
                            observed = self.evaluator.evaluate(candidate,
                                                              deadline=candidate_deadline)
                        except TimeoutError:
                            stats["candidate_timeouts"] += 1
                            continue
                        except (ValueError, RuntimeError) as exc:
                            stats["invalid_candidates"] += 1
                            stats["last_invalid"] = str(exc)[:180]
                            continue
                        self._remember(candidate_key, observed)
                    else:
                        stats["cache_hits"] += 1
                    if observed["makespan"] < best_metrics["makespan"] - 1e-9:
                        best_plan, best_metrics = candidate, observed
                        current = candidate
                        stats["accepted"] += 1
                        stats["moves"].append({"kind": kind,
                                               "makespan": observed["makespan"]})
                        if on_improvement is not None:
                            on_improvement(best_plan, best_metrics)
                        with _preparation_deadline(deadline):
                            members = _members(current)
                            state = self._calibrate(current, members, best_metrics)
                        cursor_key = (candidate_key, region, protect_assignment)
                        cursor = self._cursors.get(cursor_key, 0)
                        improved = True
                        stale_batches = 0
                        break
                if improved:
                    continue
                stale_batches += 1
                if not has_pairs or stale_batches >= 3:
                    break
        except TimeoutError:
            stats["timeout"] = True
            stats["stop"] = "deadline"
        stats["seconds"] = perf_counter() - started
        stats["initial_makespan"] = metrics["makespan"]
        stats["final_makespan"] = best_metrics["makespan"]
        return best_plan, best_metrics, stats
