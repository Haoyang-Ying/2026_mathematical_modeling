"""第1问调度算法入口：读取原始 case，计算并保存指定核数的调度方案。"""
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
    from run import solve, single_plan

    print(f'开始计算第一问：{graph.stem}，{m} 核。', flush=True)
    if m == 1:
        # 单核采用完整图方案，并在输出前进行内部合法性检查和执行回放。
        plan = single_plan(graph)
    else:
        # 在同一次计算中生成低核候选，供目标核数继承；不读取任何已存方案。
        # 中间方案、搜索记录均放入临时目录，计算结束后自动清理。
        with tempfile.TemporaryDirectory(prefix='problem1_compute_') as temporary:
            solve(graph, list(range(2, m + 1)), temporary, mode='mgr')
            path = Path(temporary) / 'plans' / 'mgr' / f'm{m}' / graph.name
            plan = json.loads(path.read_text(encoding='utf-8'))
    save_plan(graph, m, plan)


if __name__ == '__main__':
    main()
