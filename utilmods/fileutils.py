# local imports
from . import basicutils as bu

# standard library
import json
import os
import pickle

# third party
import numpy as np
import pandas as pd


def save_pickle(fn, obj):
    protocol = pickle.HIGHEST_PROTOCOL
    with open(fn, 'wb') as file:
        pickle.dump(obj, file, protocol=protocol)


def load_pickle(fn):
    with open(fn, 'rb') as file:
        obj = pickle.load(file)
    return obj


def save_numpy(fn, obj):
    np.save(fn, obj)


def load_numpy(fn):
    return np.load(fn)


def save_csv(fn, obj):
    obj.to_csv(fn, index=False)


def load_csv(fn):
    return pd.read_csv(fn)


def save_json(fn, obj):
    with open(fn, 'w') as file:
        json.dump(obj, file, indent=4)


def load_json(fn):
    with open(fn, 'r') as file:
        obj = json.load(file)
    return obj


def save(fn, obj):
    func_map = {
        '.obj': save_pickle,
        '.npy': save_numpy,
        '.csv': save_csv,
        '.json': save_json,
    }
    ext = bu.get_file_extension(fn, func_map.keys())
    func = func_map[ext]

    func(fn, obj)
    info = bu.get_info(obj)
    print(f'[Saved] {fn} {info}')


def load(fn):
    func_map = {
        '.obj': load_pickle,
        '.npy': load_numpy,
        '.csv': load_csv,
        '.json': load_json,
    }
    ext = bu.get_file_extension(fn, func_map.keys())
    func = func_map[ext]

    obj = func(fn)
    info = bu.get_info(obj)
    print(f'[Loaded] {fn} {info}')
    return obj


def collect_fns(base, is_file=True, is_dir=False, recurse=False):
    fns = []
    paths = [base]
    while paths:
        path = paths.pop()
        with os.scandir(path) as entries:
            for entry in entries:
                if is_file and entry.is_file():
                    fns.append(entry.path)
                if is_dir and entry.is_dir():
                    fns.append(entry.path)
                if recurse and entry.is_dir():
                    paths.append(entry.path)
    return sorted(fns)
