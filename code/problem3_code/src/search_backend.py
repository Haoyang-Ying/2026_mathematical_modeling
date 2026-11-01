'为共享搜索和跨核解池显式选择优化目标。硬件模拟状态始终属于单次评价；该描述符只选择模拟器、报告标签及快照命名空间，不保存 L2 内容。'
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchBackend:
    name: str
    problem: int
    algorithm: str
    revision: str
    reference_key: str

    def evaluator(self, graph, hardware, cache_mb):
        if self.problem == 3:
            from q3_evaluator import Problem3Evaluator
            cls = Problem3Evaluator
        else:
            from scene_b_internal import SceneBEvaluator
            cls = SceneBEvaluator
        return cls(graph, **hardware, cache_bytes=cache_mb * 1024**2)

    def label_report(self, report, reference):
        report.update(problem=self.problem, backend=self.name,
                      algorithm=self.algorithm, revision=self.revision)
        if self.problem == 3:
            # 此参考值在启用 L2 的条件下回放得到，
            # 不等于官方原始单核 T1；第三问接口会单独补充后者。
            report.pop('reference_T1', None)
            report.pop('k', None)
            report.pop('k_per_core', None)
            report[self.reference_key] = reference
            t = report['final_T']
            report['speedup_vs_Q3_fallback'] = reference / t if t else None


BACKENDS = {
    'problem2': SearchBackend('problem2', 2, 'problem2_latest_corepool_v1',
                              'scene_b_core_pool_1', 'reference_T1'),
    'problem3': SearchBackend('problem3', 3, 'problem3_full_search_v1',
                              'q3_full_replay_1', 'reference_singlecore_Q3_T'),
}


def get_backend(value='problem2'):
    if isinstance(value, SearchBackend):
        if BACKENDS.get(value.name) != value:
            raise ValueError('unregistered evaluator backend')
        return value
    try:
        return BACKENDS[value]
    except (KeyError, TypeError):
        raise ValueError(f'unknown evaluator backend: {value!r}') from None
