'隔离执行一个核数的任务，全部评价使用内部实现。'
import os
from pathlib import Path
import sys
import threading
from time import perf_counter, sleep, time
import traceback

from q3_runtime_io import atomic, read


def main(request_path):
    request = read(request_path)
    end = request['deadline']
    done = threading.Event()
    def watch():
        while not done.wait(.1):
            if perf_counter() >= end:
                os._exit(124)
            try:
                owner = read(request['owner_path'])
                if owner.get('run_token') != request['run_token'] or time()-owner['heartbeat_epoch'] > 8:
                    os._exit(125)
                if Path(request['stop_path']).exists():
                    os._exit(130)
            except (OSError, ValueError, KeyError):
                # 即使暂时无法读取所属主进程的元数据，
                # 子进程的绝对截止时间仍然有效。
                pass
    threading.Thread(target=watch,daemon=True).start()
    try:
        if request['policy']['memory_mb']:
            try:
                import resource
                bound = request['policy']['memory_mb']*1024**2
                resource.setrlimit(resource.RLIMIT_AS,(bound,bound))
            except ImportError:
                pass
        from q3_time import commit_section
        from q3_timed_search import execute
        checkpoint = Path(request['checkpoint_path'])
        def save(entries,references):
            with commit_section():
                atomic(checkpoint,dict(format_version=2,identity=request['identity'],
                    entries=entries,references=references,attempt=request['attempt']))
        with Path(request['events_path']).open('a',encoding='utf-8') as stream:
            def emit(event):
                with commit_section():
                    event = dict(event,cores=request['cores'],attempt=request['attempt'])
                    stream.write(__import__('json').dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
                    stream.flush()
            result = execute(request,save,emit)
        result.update(identity=request['identity'],attempt=request['attempt'])
        atomic(request['result_path'],result)
        return 0
    except (TimeoutError, MemoryError) as exc:
        atomic(request['error_path'],dict(error=type(exc).__name__,message=str(exc),status='checkpoint_retained'))
        return 124
    except BaseException as exc:
        traceback.print_exc()
        atomic(request['error_path'],dict(error=type(exc).__name__,message=str(exc)))
        return 1
    finally:
        done.set()


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
