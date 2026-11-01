'主进程使用的轻量持久化工具，不导入图模型或模拟器。'
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def code_hash(root):
    files = sorted((root/'src').glob('*.py')) + sorted((root/'official_runtime').glob('*.py'))
    files += [root/name for name in ('run_q3_full.py','api_q3_full.py','run_q3.py','run_q3_batch.py','api_q3.py')]
    return digest({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


@contextmanager
def lock(path):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    stream = path.open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError('output is locked by an active coordinator: '+str(path)) from None
    try:
        yield
    finally:
        stream.close()
