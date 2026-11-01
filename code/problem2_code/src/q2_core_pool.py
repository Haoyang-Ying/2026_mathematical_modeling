"""算法内部维护多个允许核数的精确评分最优方案。

每个可行候选均可能改善其他核数，即使它不优于当前搜索核数的最优方案。
只增删空闲核心；每次改变声明核数后均完整精确回放。
本模块不处理批量任务和文件读写。"""
from copy import deepcopy
from time import perf_counter

from q2_model import State, score_key
from q2_io import digest
from scene_b_internal import SceneBEvaluator


def checked_counts(values):
    values = tuple(values)
    if not values or any(type(m) is not int or m < 1 for m in values):
        raise ValueError('core counts must be a nonempty sequence of positive integers')
    return tuple(sorted(set(values)))


class CoreSolutionPool:
    def __init__(self, model, hardware, counts, *, cache_mb=128, on_update=None, sharing=True):
        self.model, self.hardware = model, deepcopy(hardware)
        self.counts = checked_counts(counts)
        self.sharing = sharing
        self.identity = digest({'graph': model.graph, 'hardware': hardware,
                                'counts': self.counts, 'sharing': sharing})
        self.evaluator = SceneBEvaluator(model.graph, **hardware, cache_bytes=cache_mb*1024**2)
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
        return deepcopy({key: result[key] for key in (
            'makespan', 'num_cores', 'active_cores', 'task_count', 'cross_task_traffic',
            'data_movement_bytes', 'memory_peak_by_core', 'step3_by_core')})

    # 候选先参与来源核数择优，再尝试其他核数；跨核转换成功回放后才更新目标解。
    def offer(self, state, result, stage, *, emit=None, deadline=None):
        """接收所有合法候选，不受局部搜索是否接受该候选影响。
        
        先用来源分数排除无意义的回放。补空核或保持顺序的压缩只改变空闲核心，
        但仍必须在目标核数下成功评价后，才能更新目标最优方案。
        转换工作单独计数，其耗时计入调用搜索的预算。"""
        self.stats['observed_candidates'] += 1
        source_count = len(state.cores)
        active = sum(bool(row) for row in state.cores)
        # 先提交已经完成评分的来源方案，再执行可选的跨核数转换。
        targets = ([source_count] if source_count in self.counts else [])+[
            m for m in self.counts if m != source_count]
        for target in targets:
            if not self.sharing and target != source_count:
                continue
            if active > target:
                continue
            previous = self.entries.get(target)
            if previous is not None and score_key(result) >= score_key(previous['result']):
                continue
            converted = state.resized(target)
            fingerprint = converted.fingerprint()
            token = (target, fingerprint)
            if token in self.seen:
                continue
            if target == source_count:
                evaluated = result
            else:
                if deadline is not None and perf_counter() >= deadline:
                    break
                started = perf_counter()
                self.stats['conversion_evaluations'] += 1
                try:
                    evaluated = self.evaluator.evaluate(converted.plan(),deadline=deadline)
                except TimeoutError:
                    self.stats.setdefault('conversion_timeouts',0)
                    self.stats['conversion_timeouts'] += 1
                    break  # 不记录已尝试标记，后续时间片仍可重试该分数。
                except (ValueError, RuntimeError) as error:
                    self.stats['conversion_failures'] += 1
                    if emit:
                        emit({'event': 'core_pool_rejected', 'source_cores': source_count,
                              'target_cores': target, 'source_stage': stage,
                              'error': str(error)[:500]})
                    continue
                finally:
                    self.stats['conversion_seconds'] += perf_counter()-started
            self.seen.add(token)
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
                'reference_T1': self.reference_t1, 'stats': deepcopy(self.stats),
                'entries': {str(m): {'plan': e['state'].plan(),
                    'plan_sha256': digest(e['state'].plan()), 'score_key': list(score_key(e['result'])),
                    'result': deepcopy(e['result']), 'origin': deepcopy(e['origin'])}
                    for m, e in sorted(self.entries.items())}}

    # 恢复时重新建立并评价已存状态，防止仅凭旧的分数或完成标记接受方案。
    def restore(self, snapshot):
        if snapshot.get('format_version') != 1 or snapshot.get('identity') != self.identity:
            raise ValueError('core pool checkpoint belongs to different graph/hardware/counts')
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
        self.reference_t1 = snapshot['reference_T1']
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
        if phase_T is not None and report['replaced_by_core_pool']:
            report.setdefault('stages', []).append({
                'stage': 'core_pool', 'status': 'completed', 'before': phase_T, 'after': t,
                'source_cores': entry['origin']['source_cores'],
                'source_stage': entry['origin']['source_stage']})
        return state.plan(), report
