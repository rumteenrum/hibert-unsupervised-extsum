"""Shared protocol: evaluate settings on validation, pick one, score test once.

A setting is chosen by the mean of ROUGE-1, ROUGE-2 and ROUGE-L F1 on validation. Test is
scored only for the chosen setting (plus any fixed reference settings named explicitly), so
no test number influences any choice.

Results go to <out>/<system>.json (all validation scores, the choice, test scores) and
<out>/<system>.<split>.<setting>.npz (per-document scores and selected sentences), which the
comparison script reads.
"""

import json
import os
import re
from multiprocessing import Pool

import numpy as np

from analysis.rouge import rouge, summarize
from analysis.text import load_documents

SPLITS = ('valid', 'test')


def load_split(data_dir, split, n_docs=None):
    articles = load_documents(os.path.join(data_dir, f'{split}.article'))
    references = load_documents(os.path.join(data_dir, f'{split}.summary'))
    if n_docs is not None and n_docs < len(articles):
        print(f'| {split}: using the first {n_docs} of {len(articles)} documents')
        articles, references = articles[:n_docs], references[:n_docs]
    return articles, references


def _score(args):
    selections, articles, references = args
    summaries = [[a[i] for i in sel] for a, sel in zip(articles, selections)]
    return rouge(summaries, references)


def evaluate(settings, articles, references, jobs=4):
    """settings: {name: per-document lists of selected sentence indices}."""
    names = list(settings)
    work = [(settings[n], articles, references) for n in names]
    if jobs > 1 and len(work) > 1:
        with Pool(min(jobs, len(work))) as pool:
            scores = pool.map(_score, work)
    else:
        scores = [_score(w) for w in work]
    return dict(zip(names, scores))


def mean_rouge(corpus):
    return (corpus['rouge_1'] + corpus['rouge_2'] + corpus['rouge_l']) / 3


def file_safe(name):
    return re.sub(r'[^A-Za-z0-9.=_-]+', '_', name)


def save(out, system, split, name, scores, selections):
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f'{system}.{split}.{file_safe(name)}.npz'),
                        selections=np.array(selections, dtype=object), **scores)


def run(system, settings_for, data_dir, out, jobs=4, reference_settings=(), n_docs=None):
    """settings_for(split, articles) -> {name: selections} for that split.

    n_docs: optional {split: count} when the outputs cover only the first documents.
    """
    n_docs = n_docs or {}
    report = {'system': system}

    articles, references = load_split(data_dir, 'valid', n_docs.get('valid'))
    valid_sel = settings_for('valid', articles)
    valid = evaluate(valid_sel, articles, references, jobs)
    report['valid'] = {name: summarize(s) for name, s in valid.items()}
    chosen = max(report['valid'], key=lambda n: mean_rouge(report['valid'][n]))
    report['chosen'] = chosen
    save(out, system, 'valid', chosen, valid[chosen], valid_sel[chosen])
    print(f'| {system}: chosen on validation: {chosen}')

    articles, references = load_split(data_dir, 'test', n_docs.get('test'))
    test_sel = settings_for('test', articles)
    names = [chosen] + [n for n in reference_settings if n != chosen]
    test = evaluate({n: test_sel[n] for n in names}, articles, references, jobs)
    report['test'] = {name: summarize(s) for name, s in test.items()}
    for name, s in test.items():
        save(out, system, 'test', name, s, test_sel[name])

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f'{system}.json'), 'w') as f:
        json.dump(report, f, indent=2)
    for name, s in report['test'].items():
        tag = 'chosen' if name == chosen else 'reference'
        print(f'| {system} test ({tag}: {name}): ' + '  '.join(f'{k} {v:.2f}' for k, v in s.items()))
    return report
