"""Reading the per-document outputs written by hibert_extract.py.

Each split is a directory of `shard_XXXXX.pkl` files, each a list of per-document dicts in
file order (see the README of the GPU repository for the fields).
"""

import glob
import os
import pickle


def load_split(directory):
    records = []
    for path in sorted(glob.glob(os.path.join(directory, 'shard_*.pkl'))):
        with open(path, 'rb') as f:
            records.extend(pickle.load(f))
    ids = [r['doc_id'] for r in records]
    if ids != list(range(len(ids))):
        raise ValueError(f'{directory}: documents missing or out of order')
    return records
