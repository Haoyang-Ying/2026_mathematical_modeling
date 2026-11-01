'算法内部维护多个允许核数的精确评分最优方案。可行候选即使在当前核数下未被接受，仍可能改善其他核数。只增删空闲核，每次改变声明核数后完整回放；本模块不处理批量任务或文件读写。'
from copy import deepcopy
from time import perf_counter

from q2_model import State, score_key
from q2_io import digest
from search_backend import get_backend


def checked_counts(values):
    values = tuple(values)
    if not values or any(type(m) is not int or m < 1 for m in values):
        raise ValueError('core counts must be a nonempty sequence of positive integers')
    return tuple(sorted(set(values)))


class CoreSolutionPool:
    def __init__(self, model, hardware, counts, *, cache_mb=128, on_update=None,
                 sharing=True, backend='problem2'):
        self.model, self.hardware = model, deepcopy(hardware)
        self.backend = get_backend(backend)
        self.counts = checked_counts(counts)
        self.sharing = sharing
        self.identity = digest({'graph': model.graph, 'hardware': hardware,
                                'counts': self.counts, 'sharing': sharing,
                                'backend': self.backend.name, 'revision': self.backend.revision})
        self.evaluator = self.backend.evaluator(model.graph, hardware, cache_mb)
        self.entries = {}
        self.reference_t1 = None
        self.on_update = on_update
        self.seen = set()
        self.stats = {'observed_candidates': 0, 'conversion_evaluations': 0,
                      'conversion_failures': 0, 'updates': 0, 'downward_updates': 0,
                      'upward_updates': 0, 'import_evaluations': 0,
                      'restore_evaluations': 0, 'conversion_seconds': 0.0}

    def set_reference(self, value):
        if self.reference_t1 is not None and value != self.reference_t1:
            raise ValueError('fixed single-core reference changed across searches')
        self.reference_t1 = value

    @staticmethod
    def _summary(result):
        keys = (
            'makespan', 'num_cores', 'active_cores', 'task_count', 'cross_task_traffic',
            'data_movement_bytes', 'memory_peak_by_core', 'step3_by_core',
            'problem', 'cache_mode', 'cache_stats', 'cache_miss_reasons',
            'physical_transfer_bytes', 'cache_capacity_bytes', 'cache_bandwidth_bytes_per_cycle')
        return deepcopy({key: result[key] for key in keys if key in result})

    def offer(self, state, result, stage, *, emit=None):
        '观察所有合法候选，不受局部接受规则影响。来源分数可筛除无意义回放；补空核或保持顺序的压缩只改变空闲核，但更新目标最优方案前仍须完成目标核数评价。转换工作单独计数，耗时计入调用搜索。'
        if result.get('problem', 2) != self.backend.problem:
            raise ValueError('candidate objective differs from the core pool backend')
        self.stats['observed_candidates'] += 1
        source_count = len(state.cores)
        active = sum(bool(row) for row in state.cores)
        for target in self.counts:
            if not self.sharing and target != source_count:
                continue
            if active > target:
                continue
            previous = self.entries.get(target)
            # 第三问对每个尚未评价且允许转换的核数保守地重新回放，
            # 不依据另一核数在共享缓存模拟中的耗时，
            # 直接丢弃目标核数候选。
            if previous is not None and (self.backend.problem == 2 or target == source_count) \
                    and score_key(result) >= score_key(previous['result']):
                continue
            converted = state.resized(target)
            fingerprint = converted.fingerprint()
            token = (target, fingerprint)
            if token in self.seen:
                continue
            self.seen.add(token)
            if target == source_count:
                evaluated = result
            else:
                started = perf_counter()
                self.stats['conversion_evaluations'] += 1
                try:
                    evaluated = self.evaluator.evaluate(converted.plan())
                except (ValueError, RuntimeError) as error:
                    self.stats['conversion_failures'] += 1
                    if emit:
                        emit({'event': 'core_pool_rejected', 'source_cores': source_count,
                              'target_cores': target, 'source_stage': stage,
                              'error': str(error)[:500]})
                    continue
                finally:
                    self.stats['conversion_seconds'] += perf_counter()-started
            if previous is not None and score_key(evaluated) >= score_key(previous['result']):
                continue
            origin = {'source_cores': source_count, 'source_stage': stage,
                      'source_active_cores': active, 'source_fingerprint': state.fingerprint(),
                      'target_cores': target, 'target_fingerprint': fingerprint,
                      'conversion_replayed': target != source_count,
                      'direction': 'downward' if target < source_count else
                                   'upward' if target > source_count else 'same_count'}
            self.entries[target] = {'state': converted, 'result': self._summary(evaluated),
                                    'origin': origin}
            self.stats['updates'] += 1
            self.stats['downward_updates'] += int(target < source_count)
            self.stats['upward_updates'] += int(target > source_count)
            event = {'event': 'core_pool_update', **origin,
                     'before_T': previous['result']['makespan'] if previous else None,
                     'after_T': evaluated['makespan'], 'score_key': list(score_key(evaluated))}
            if self.on_update:
                self.on_update(event, self)
            if emit:
                emit(event)

    def import_state(self, state, label='external_seed', *, emit=None):
        self.stats['import_evaluations'] += 1
        result = self.evaluator.evaluate(state.plan())
        self.offer(state, result, label, emit=emit)
        return result

    def snapshot(self):
        return {'format_version': 1, 'identity': self.identity, 'counts': list(self.counts),
                'backend': self.backend.name, 'revision': self.backend.revision,
                self.backend.reference_key: self.reference_t1, 'stats': deepcopy(self.stats),
                'entries': {str(m): {'plan': e['state'].plan(),
                    'plan_sha256': digest(e['state'].plan()), 'score_key': list(score_key(e['result'])),
                    'result': deepcopy(e['result']), 'origin': deepcopy(e['origin'])}
                    for m, e in sorted(self.entries.items())}}

    def restore(self, snapshot):
        if snapshot.get('format_version') != 1 or snapshot.get('identity') != self.identity:
            raise ValueError('core pool checkpoint belongs to different graph/hardware/counts')
        if snapshot.get('backend') != self.backend.name or snapshot.get('revision') != self.backend.revision:
            raise ValueError('core pool checkpoint evaluator backend changed')
        if snapshot.get('counts') != list(self.counts):
            raise ValueError('core pool checkpoint counts differ')
        restored = {}
        for key, saved in snapshot['entries'].items():
            m = int(key)
            if m not in self.counts or digest(saved['plan']) != saved['plan_sha256']:
                raise ValueError('core pool checkpoint plan or target count changed')
            state = State.from_plan(self.model, saved['plan'])
            if len(state.cores) != m:
                raise ValueError('core pool plan has a wrong declared core count')
            result = self.evaluator.evaluate(state.plan())
            if list(score_key(result)) != saved['score_key']:
                raise ValueError('core pool checkpoint exact score changed')
            restored[m] = {'state': state, 'result': self._summary(result),
                           'origin': deepcopy(saved['origin'])}
        self.entries = restored
        self.reference_t1 = snapshot[self.backend.reference_key]
        self.stats.update(snapshot['stats'])
        self.stats['restore_evaluations'] += len(restored)
        self.seen = {(m,e['state'].fingerprint()) for m,e in restored.items()}

    def final_result(self, cores, phase_report=None):
        entry = self.entries[cores]
        state, result = entry['state'], entry['result']
        report = deepcopy(phase_report) if phase_report is not None else {}
        phase_T = report.get('final_T')
        phase_fp = report.get('final_fingerprint')
        t = result['makespan']
        k = self.reference_t1/t if t else None
        report.update(algorithm='problem2_latest_corepool_v1', revision='scene_b_core_pool_1',
                      cores=cores, reference_T1=self.reference_t1, final_T=t, k=k,
                      k_per_core=k/cores if k is not None else None,
                      active_cores=result['active_cores'], subgraphs=len(state.groups),
                      data_movement_bytes=deepcopy(result['data_movement_bytes']),
                      memory_peak_by_core=deepcopy(result['memory_peak_by_core']),
                      final_fingerprint=state.fingerprint(), internally_validated=True,
                      officially_validated=False, best_origin=deepcopy(entry['origin']),
                      phase_final_T=phase_T,
                      replaced_by_core_pool=phase_fp is not None and phase_fp != state.fingerprint(),
                      core_pool_statistics=deepcopy(self.stats))
        report.pop('validation', None)
        report.pop('validation_exact_match', None)
        self.backend.label_report(report, self.reference_t1)
        if self.backend.problem == 3:
            for key in ('cache_mode', 'cache_stats', 'cache_miss_reasons',
                        'physical_transfer_bytes', 'cache_capacity_bytes',
                        'cache_bandwidth_bytes_per_cycle'):
                report[key] = deepcopy(result[key])
        if phase_T is not None and report['replaced_by_core_pool']:
            report.setdefault('stages', []).append({
                'stage': 'core_pool', 'status': 'completed', 'before': phase_T, 'after': t,
                'source_cores': entry['origin']['source_cores'],
                'source_stage': entry['origin']['source_stage']})
        return state.plan(), report
