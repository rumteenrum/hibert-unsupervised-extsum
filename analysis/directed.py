"""Directed attention centrality ("Directed STAS").

STAS ranks a sentence by the attention it receives from the other sentences, regardless of where
they are. Here the received attention is split by direction:

    F_i = sum_{j > i} G[j, i]      attention from later sentences
    B_i = sum_{j < i} G[j, i]      attention from earlier sentences
    D_i = lam_f * F_i + lam_b * B_i

G[j, i] is the attention sentence j pays to sentence i in STAS's ranking graph (rows sum to 1,
diagonal 0), as saved by the patched STAS run (stas_dump.<split>.npz). A sentence that later
sentences build on gets a high F; with lam_b < 0, attention from earlier sentences counts against it.
"""

import numpy as np


def load_graphs(npz_path):
    """Per-document graphs G [n, n] (float32), in document order."""
    d = np.load(npz_path)
    if list(d['doc_id']) != list(range(len(d['doc_id']))):
        raise ValueError(f'{npz_path}: documents not in order')
    graphs, offset = [], 0
    for n in d['nsents']:
        n = int(n)
        graphs.append(d['G'][offset:offset + n * n].astype(np.float32).reshape(n, n))
        offset += n * n
    return graphs


def received(G):
    """(F, B): attention each sentence receives from later and from earlier sentences."""
    later = np.tril(G, -1)    # entries G[j, i] with j > i
    earlier = np.triu(G, 1)   # entries G[j, i] with j < i
    return later.sum(axis=0), earlier.sum(axis=0)


def directed_degree(G, lam_f, lam_b):
    f, b = received(G)
    return lam_f * f + lam_b * b
