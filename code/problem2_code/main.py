"""第2问调度算法入口：读取原始 case，计算并保存指定核数的调度方案。"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / '100case_data'
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def arguments():
    """只接受 case 名称和核数，输入位置固定为包内的原始数据目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', help='case_001 至 case_100，也可写为 001 或 case_001.json')
    parser.add_argument('m', type=int, choices=range(1, 6), help='核数：1、2、3、4、5')
    args = parser.parse_args()
    match = re.fullmatch(r'(?:case_)?([0-9]{1,3})(?:\.json)?', args.case)
    if match is None or not 1 <= int(match.group(1)) <= 100:
        parser.error('case 名称应为 case_001 至 case_100。')
    graph = DATA / f'case_{int(match.group(1)):03d}.json'
    if not graph.is_file() or not (DATA / 'config.txt').is_file():
        parser.error('请完整解压 code 目录，并保留 100case_data 中的原始文件。')
    return graph, args.m


def initialize():
    """固定解释器哈希种子，使同样输入采用相同的随机种子策略。"""
    if os.environ.get('PYTHONHASHSEED') != '1234':
        environment = dict(os.environ, PYTHONHASHSEED='1234')
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], environment)
    sys.path.insert(0, str(ROOT / 'src'))


def save_plan(graph, m, plan):
    """只保存本次计算得到的最终调度方案，完整写入后再替换目标文件。"""
    destination = ROOT / 'output' / graph.stem / f'm{m}' / f'{graph.stem}_multicore_res.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, destination)
    print(f'计算完成，调度方案：{destination}', flush=True)


def main():
    graph, m = arguments()
    initialize()
    from run import main as solve_main

    print(f'开始计算第二问：{graph.stem}，{m} 核。', flush=True)
    # 保留算法内部的跨核候选共享；所有候选均在本次调用中重新计算。
    counts = [1] if m == 1 else list(range(2, m + 1))
    with tempfile.TemporaryDirectory(prefix='problem2_compute_') as temporary:
        status = solve_main(['solve-many', '--case', str(graph),
                             '--cores', *map(str, counts), '--config', str(DATA / 'config.txt'),
                             '--out', temporary, '--memory-mb', '1536'])
        if status != 0:
            raise RuntimeError('本次计算未完成，未导出调度方案。')
        directory = Path(temporary) / f'm{m}'
        report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
        # 只有完成独立内部回放的最终方案才能写入输出目录。
        if report.get('status') != 'complete' or not report.get('independent_internal_replay_validated'):
            raise RuntimeError('最终方案未通过内部复核。')
        plan = json.loads((directory / 'schedule.json').read_text(encoding='utf-8'))
    save_plan(graph, m, plan)


if __name__ == '__main__':
    main()
