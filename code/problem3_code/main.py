"""第3问调度算法入口：读取原始 case，计算并保存指定核数的调度方案。"""
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
    from api_q3_full import solve_many
    from q3_time import TimePolicy

    print(f'开始计算第三问：{graph.stem}，{m} 核。', flush=True)
    # 从原始图构造初解，保留算法内部的跨核候选共享和只读缓存模拟。
    counts = [1] if m == 1 else list(range(2, m + 1))
    with tempfile.TemporaryDirectory(prefix='problem3_compute_') as temporary:
        results = solve_many(str(graph), counts, config=str(DATA / 'config.txt'),
                             out=temporary, resume=False, time_policy=TimePolicy(workers=1))
        plan, report = results[m]
        # 搜索和内部复核均完成后，仅导出目标核数的最终调度方案。
        if report.get('status') != 'complete' or not report.get('internal_recheck_passed'):
            raise RuntimeError('最终方案未通过内部复核。')
    save_plan(graph, m, plan)


if __name__ == '__main__':
    main()
