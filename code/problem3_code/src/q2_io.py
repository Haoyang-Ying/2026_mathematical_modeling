'提供可移植 JSON 输出、硬件配置读取及独立官方核验。'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from scene_b_internal import SceneBEvaluator
from evaluation_validation import read_evaluation_config
from multicore_cut_evaluate_problem_2 import evaluate_scene_b, read_scene_b_config
from singlecore_evaluate import evaluate_singlecore

ROOT = Path(__file__).resolve().parents[1]


def hardware(path=None):
    path = Path(path) if path else ROOT.parent/'100case_data/config.txt'
    settings = read_evaluation_config(path)
    settings['cross_core_copy_delay'] = read_scene_b_config(path)['cross_core_copy_delay_cycles']
    return settings


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True).encode()).hexdigest()


def compact(result, backend):
    fields = ('makespan', 'num_cores', 'task_count', 'cross_task_traffic',
              'data_movement_bytes', 'memory_peak_by_core', 'step3_by_core')
    out = {k: result[k] for k in fields}
    if backend == 'official':
        ops = [(c['core_id'], o['op_id'], o['start'], o['end'], o['subgraph_id'],
                o['pipe'], o['op']) for c in result['per_core_timeline'] for o in c['ops']]
        transfers = [(t['source_core'], t['source_copy_out_id'], t['target_core'],
                      t['target_copy_in_id'], t['copy_out_end'], t['copy_in_release'],
                      t['copy_in_start'], t['copy_in_end']) for t in result['cross_core_transfers']]
        ends = [max((o['end'] for o in c['ops']), default=0) for c in result['per_core_timeline']]
        groups = [(c['core_id'], g['subgraph_id'], g['start'], g['end'])
                  for c in result['per_core_timeline'] for g in c['subgraphs']]
    else:
        ops, transfers, ends = result['op_records'], result['transfer_records'], result['core_ends']
        group_cores = {row[4]: row[0] for row in ops if row[4] is not None}
        groups = [(group_cores[g], g, result['starts'][g], end) for g, end in result['ends'].items()]
    out.update(core_ends=ends, execution_ops=len(ops), cross_transfer_count=len(transfers),
               op_timing_sha256=digest(sorted(ops)), transfer_timing_sha256=digest(sorted(transfers)),
               subgraph_timing_sha256=digest(sorted(groups)))
    return json.loads(json.dumps(out))


def validate_plan(graph, plan, settings, *, official=True, expected_t1=None):
    t0 = perf_counter()
    evaluator = SceneBEvaluator(graph, **settings)
    internal = evaluator.evaluate(plan, capture=True)
    internal_seconds = perf_counter()-t0
    audit = compact(internal, 'internal')
    report = {'valid': True, 'internally_validated': True, 'officially_validated': False,
              'internal_seconds': internal_seconds, 'internal_audit': audit,
              'plan_sha256': digest(plan)}
    if official:
        t0 = perf_counter()
        external = evaluate_scene_b(graph, plan, **settings)
        report['official_seconds'] = perf_counter()-t0
        external_audit = compact(external, 'official')
        differences = [k for k in audit if audit[k] != external_audit[k]]
        report.update(officially_validated=True, exact_match=not differences,
                      different_fields=differences, official_audit=external_audit)
        if differences:
            raise AssertionError(f'official/internal mismatch: {differences}')
        t0 = perf_counter()
        reference = evaluate_singlecore(graph, bandwidth=settings['bandwidth'], capacity=settings['capacity'])
        report['reference_seconds'] = perf_counter()-t0
        report['official_singlecore_T1'] = reference['makespan']
        if expected_t1 is not None and expected_t1 != reference['makespan']:
            raise AssertionError(f'one-core reference mismatch: {expected_t1} != {reference["makespan"]}')
        report['k'] = reference['makespan']/audit['makespan'] if audit['makespan'] else None
        report['k_per_core'] = report['k']/audit['num_cores'] if report['k'] is not None else None
    return report
