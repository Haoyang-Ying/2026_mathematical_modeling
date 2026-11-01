'所有受支持计算进程共用的随机种子策略。'
import os
import random
import sys

RANDOM_SEED = 1234
PACKAGE_REVISION = 'bspg_competitive_dualstart_seed1234_v2'


def child_environment(**updates):
    environment = dict(os.environ)
    environment.update(updates)
    environment['PYTHONHASHSEED'] = str(RANDOM_SEED)
    return environment


def initialize_seed(*, reexec_script=False):
    '设置 Python 随机种子；命令行重启确保解释器启动前设置哈希种子。\n\n通过 Python 接口调用时，应在启动宿主解释器前设置 PYTHONHASHSEED。调度代码只使用标准库，不依赖 NumPy。'
    if reexec_script and os.environ.get('PYTHONHASHSEED') != str(RANDOM_SEED):
        arguments = list(getattr(sys, 'orig_argv', [sys.executable, *sys.argv]))
        os.execve(sys.executable, [sys.executable, *arguments[1:]], child_environment())
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def seed_metadata():
    return {'random_seed': RANDOM_SEED,
            'python_hash_seed_environment': os.environ.get('PYTHONHASHSEED')}
