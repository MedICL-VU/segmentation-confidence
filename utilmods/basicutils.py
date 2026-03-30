# local imports

# standard library
import datetime
import importlib
import multiprocessing
import os
import subprocess
import time

# third party


class Mock:
    pass


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class Clock:

    def __init__(self):
        self.start = self.now = self.last = time.time()
        self._calculate()
        self._display()

    def __call__(self):
        self._update()
        self._calculate()
        self._display()

    def _update(self):
        self.last = self.now
        self.now = time.time()

    def _calculate(self):
        now = time.asctime(time.localtime(self.now)).split()[3]
        since_last = datetime.timedelta(seconds=round(self.now - self.last))
        since_start = datetime.timedelta(seconds=round(self.now - self.start))
        self.calculated = now, since_last, since_start

    def _display(self):
        now, since_last, since_start = self.calculated
        msg = f'\n[ {now} ]    '
        msg += f'since last: [ {since_last} ]    '
        msg += f'since start: [ {since_start} ]\n'
        print(msg)


def shell_run(args):
    pipe = subprocess.PIPE
    process = subprocess.run(args, stdout=pipe, stderr=pipe)

    code = process.returncode
    out = process.stdout.decode(errors='replace')
    err = process.stderr.decode(errors='replace')
    if code == 0:
        return out

    command = ' '.join(args)
    msg = [command, f'Shell error code: {code}']
    msg += ['-' * 66, out, '-' * 66, err]
    msg = '\n'.join(msg)
    raise RuntimeError(msg)


def beep(mode):
    fn = f'/home/baris/Music/{mode}.wav'
    args = ['paplay', fn]
    shell_run(args)


def get_info(obj):
    na = '.'
    name = getattr(type(obj), '__name__', na)
    length = len(obj) if hasattr(obj, '__len__') else na
    shape = getattr(obj, 'shape', na)
    dtype = getattr(obj, 'dtype', na)
    device = getattr(obj, 'device', na)
    info = f'[{name} {length} {shape} {dtype} {device}]'
    return info


def info(*objs):
    infos = [get_info(obj) for obj in objs]
    print('\n'.join(infos))


def raw_map_multicore(func, stream, n_cpus, chunksize):
    with multiprocessing.Pool(processes=n_cpus) as pool:
        try:
            results = pool.map_async(func, stream, chunksize=chunksize)
            results = results.get()
        except Exception as e:
            pool.terminate()
            raise RuntimeError(f'map_async failed:\n{e}') from e
    return results


def map_multicore(func, stream, n_cpus=None, chunksize=None):
    if n_cpus is None:
        n_cpus = max(1, os.cpu_count() // 2)
    if n_cpus == 1:
        return list(map(func, stream))

    if chunksize is None:
        length = len(stream)
        if length < n_cpus:
            n_cpus = length
            chunksize = 1
        else:
            chunksize = length // n_cpus
    return raw_map_multicore(func, stream, n_cpus, chunksize)


def dynamic_import(dotted_name):
    path, name = dotted_name.rsplit('.', maxsplit=1)
    module = importlib.import_module(path)
    return getattr(module, name)


def get_file_extension(fn, whitelist):
    for ext in whitelist:
        if fn.endswith(ext):
            break
    else:
        raise ValueError('unknown extension')
    return ext
