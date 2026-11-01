'宏观重建与 HC 共用一个当前最优方案及一个绝对截止时刻。\n\n本模块不调用官方完整评价器。每个被接受方案均完成带缓存的四流水线与共享 DDR 内部评价；未完成候选被丢弃，任务 编号变化严格按评价时状态保留。'
from __future__ import annotations

import hashlib
import json
from time import perf_counter

from dagp_execution import _check_deadline
from hybrid_hc import FineSearch
from hybrid_macro import macro_candidates, select_region
from time_budget import seconds as budget_seconds


def copy_plan(plan):
    return {"node_to_subgraph": dict(plan["node_to_subgraph"]),
            "core_schedules": [list(order) for order in plan["core_schedules"]]}


def score_summary(metrics):
    return {key: metrics[key] for key in (
        "makespan", "copy_bytes", "boundary_copy_bytes", "spill_bytes",
        "task_count", "active_cores", "ddr_busy_cycles", "event_count",
        "memory_dependency_count", "execution_ops") if key in metrics}


def plan_key(plan):
    # 保留真实编号：修改编号可能改变官方 COPY 分配。
    content = (sorted((int(op), task) for op, task in plan["node_to_subgraph"].items()),
               plan["core_schedules"])
    return hashlib.sha256(json.dumps(content, separators=(",", ":")).encode()).hexdigest()


class HybridSearch:
    def __init__(self, model, evaluator, m, deadline, started, on_checkpoint=None,
                 max_rounds=4, enable_macro=True):
        self.model, self.evaluator, self.m = model, evaluator, m
        self.deadline, self.started = deadline, started
        self.on_checkpoint = on_checkpoint
        self.max_rounds, self.enable_macro = max_rounds, enable_macro
        self.best_plan = self.best_metrics = None
        self.initial_makespan = None
        self.logs = []
        self.accepted = []
        self.hc = None
        self.phase = "initial"
        self.seen = set()

    def statistics(self):
        metrics = self.best_metrics
        return {
            "algorithm": "best 2..m greedy + macro region LNS + shared-evaluator HC v3",
            "evaluation_kind": "cached_official_task_four_pipe_shared_ddr",
            "initial_makespan": self.initial_makespan,
            "final_makespan": metrics["makespan"],
            "best_metrics": score_summary(metrics),
            "tasks": metrics["task_count"], "active_cores": metrics["active_cores"],
            "official_evaluations_during_search": 0,
            "detailed_evaluations": self.evaluator.evaluations,
            "template_cache_hits": self.evaluator.cache_hits,
            "template_cache_misses": self.evaluator.cache_misses,
            "template_prepare_seconds": self.evaluator.prepare_seconds,
            "accepted_improvements": list(self.accepted),
            "search_log": list(self.logs),
            "macro_enabled": self.enable_macro,
            "initial_plan_preserved": True,
            "non_regression_verified": metrics["makespan"] <= self.initial_makespan,
            "elapsed_search_seconds": perf_counter() - self.started,
        }

    def _consider(self, plan, metrics):
        if metrics["makespan"] >= self.best_metrics["makespan"]:
            return False
        self.model.validate_plan(plan, self.m)
        self.best_plan, self.best_metrics = copy_plan(plan), metrics
        self.accepted.append({"phase": self.phase, "makespan": metrics["makespan"],
                              "elapsed_seconds": perf_counter() - self.started})
        if self.on_checkpoint:
            self.on_checkpoint(self.best_plan, self.statistics())
        return True

    def _hc(self, plan, metrics, deadline, *, region_ops=None, protect=False):
        if perf_counter() + budget_seconds(.015) >= min(deadline, self.deadline):
            return plan, metrics
        before = perf_counter()
        try:
            result, observed, stats = self.hc.run(
                plan, metrics, min(deadline, self.deadline), region_ops=region_ops,
                protect_assignment=protect, on_improvement=self._consider)
            self.logs.append({"kind": "hc", "phase": self.phase,
                              "seconds": perf_counter() - before,
                              "before": metrics["makespan"], "after": observed["makespan"],
                              "protected": protect, "statistics": stats})
            self._consider(result, observed)
            return result, observed
        except TimeoutError:
            # 回调已将每次完成评价的改进保存为检查点。
            self.logs.append({"kind": "hc", "phase": self.phase, "timed_out": True,
                              "seconds": perf_counter() - before})
            return plan, metrics

    def run(self, initial_plan, initial_metrics, initial_evaluation_seconds):
        self.best_plan, self.best_metrics = copy_plan(initial_plan), initial_metrics
        self.initial_makespan = initial_metrics["makespan"]
        self.seen.add(plan_key(initial_plan))
        if self.on_checkpoint:
            self.on_checkpoint(self.best_plan, self.statistics())
        if not self.model.ids or perf_counter() >= self.deadline:
            return self.best_plan, self.statistics()
        self.hc = FineSearch(self.model, self.evaluator, self.m)
        available = max(0.0, self.deadline - perf_counter())
        final_reserve = min(budget_seconds(45.0), max(budget_seconds(.05), .15 * available))
        macro_deadline = self.deadline - final_reserve
        self.phase = "initial_hc"
        self._hc(self.best_plan, self.best_metrics,
                 min(macro_deadline, perf_counter() + min(budget_seconds(20.0), .10 * available)))
        stop_reason = "candidate_rounds_completed"
        for round_index in range(self.max_rounds if self.enable_macro else 0):
            if perf_counter() + budget_seconds(.1) >= macro_deadline:
                stop_reason = "macro_budget_exhausted"
                break
            # 同一轮各候选使用相同的基准方案和边界切口。
            base, metrics = copy_plan(self.best_plan), self.best_metrics
            try:
                region = select_region(self.model, base, metrics, round_index=round_index,
                                       deadline=macro_deadline)
            except TimeoutError:
                stop_reason = "region_budget_exhausted"
                break
            if region is None:
                stop_reason = "no_macro_region"
                break
            round_start = perf_counter()
            evaluated = 0
            self.logs.append({"kind": "region", "round": round_index,
                              "tasks": len(region.tasks), "ops": len(region.ops),
                              "work_fraction": region.work_fraction})
            try:
                candidates = macro_candidates(self.model, base, metrics, region, self.m,
                    round_index=round_index, deadline=macro_deadline)
                for label, candidate, info in candidates:
                    _check_deadline(macro_deadline)
                    if evaluated >= 4:
                        break
                    signature = plan_key(candidate)
                    if signature in self.seen:
                        continue
                    self.seen.add(signature)
                    if len(self.seen) > 128:
                        self.seen = {signature, plan_key(self.best_plan)}
                    self.phase = "macro_" + label
                    record = {"kind": "macro", "round": round_index, "operator": label,
                              "changes": info, "base_makespan": metrics["makespan"]}
                    before = perf_counter()
                    remaining = macro_deadline - before
                    # 重建可能较昂贵，因此只分配有限时间比例，并为其他操作和最终 HC 保留时间。
                    allowance = min(budget_seconds(60.0), budget_seconds(max(8.0, initial_evaluation_seconds * 3.0)),
                                    remaining * .65)
                    if allowance < budget_seconds(.02):
                        break
                    try:
                        observed = self.evaluator.evaluate(candidate,
                            deadline=min(macro_deadline, before + allowance))
                    except TimeoutError:
                        record.update(timed_out=True, seconds=perf_counter() - before)
                        self.logs.append(record)
                        continue
                    except (ValueError, RuntimeError) as exc:
                        record.update(rejected=str(exc)[:300], seconds=perf_counter() - before)
                        self.logs.append(record)
                        continue
                    evaluated += 1
                    record["macro_makespan"] = observed["makespan"]
                    # 宏观候选一旦更优即保存，即使其后续修复被中断也不丢失；较差候选仍可尝试 HC 修复。
                    self._consider(candidate, observed)
                    repair_seconds = min(budget_seconds(5.0), max(budget_seconds(.05), .025 * available),
                                         max(0.0, macro_deadline - perf_counter()) * .4)
                    repair_end = min(macro_deadline, perf_counter() + repair_seconds)
                    self.phase = "macro_repair_" + label
                    candidate, observed = self._hc(candidate, observed,
                        min(repair_end, perf_counter() + repair_seconds * .4),
                        region_ops=region.ops, protect=True)
                    candidate, observed = self._hc(candidate, observed, repair_end,
                        region_ops=region.ops, protect=False)
                    record.update(repaired_makespan=observed["makespan"],
                                  best_makespan=self.best_metrics["makespan"],
                                  seconds=perf_counter() - before)
                    self.logs.append(record)
                    self._consider(candidate, observed)
            except TimeoutError:
                stop_reason = "macro_budget_exhausted"
                break
            self.phase = "between_round_hc"
            self._hc(self.best_plan, self.best_metrics,
                     min(macro_deadline, perf_counter() + min(budget_seconds(8.0), .03 * available)))
            self.logs.append({"kind": "round", "round": round_index,
                              "seconds": perf_counter() - round_start,
                              "evaluated": evaluated,
                              "best_makespan": self.best_metrics["makespan"]})
        self.phase = "final_hc"
        self._hc(self.best_plan, self.best_metrics, self.deadline)
        result_stats = self.statistics()
        result_stats["stop_reason"] = stop_reason
        return self.best_plan, result_stats
