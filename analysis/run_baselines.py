"""LEAD-3 and the label oracle, scored the same way as every system.

The oracle takes the sentences labelled T (up to three, in document order) and fills up
with the earliest F sentences, as STAS's evaluation code does for gold labels. There is
nothing to choose, so validation and test are both scored directly.

    python -m analysis.run_baselines --data DIR --out results
"""

import argparse
import json
import os

from analysis import experiment, selection
from analysis.rouge import summarize
from analysis.text import load_labels


def oracle(labels, k=3):
    chosen = [i for i, lbl in enumerate(labels) if lbl == 'T'][:k]
    backup = [i for i, lbl in enumerate(labels) if lbl == 'F']
    return sorted(chosen + backup[:max(0, k - len(chosen))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='has {valid,test}.{article,summary,label}')
    p.add_argument('--out', required=True)
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()

    report = {}
    for split in experiment.SPLITS:
        articles, references = experiment.load_split(a.data, split)
        labels = load_labels(os.path.join(a.data, f'{split}.label'))
        settings = {'lead3': [selection.lead(len(doc)) for doc in articles],
                    'oracle': [oracle(lbl) for lbl in labels]}
        scores = experiment.evaluate(settings, articles, references, a.jobs)
        for name, s in scores.items():
            experiment.save(a.out, name, split, name, s, settings[name])
            report.setdefault(name, {})[split] = summarize(s)
            print(f'| {name} {split}: ' + '  '.join(f'{k} {v:.2f}' for k, v in summarize(s).items()))
    with open(os.path.join(a.out, 'baselines.json'), 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
