"""Evaluate the released STAS model's sentence scores on our documents.

Reads the authors' output files `{k}.valid.txt` / `{k}.test.txt` (k = 0..19, see
analysis.stas) and turns each into summaries in both of the authors' selection modes:
top-3 in document order, and top-3 with trigram blocking. The setting is chosen on
validation; the authors' own released choice (k = 13 with trigram blocking) is also scored
on test for reference.

    python -m analysis.run_stas --stas-dir outputs/stas --data DIR --out results
"""

import argparse
import os

from analysis import experiment, selection, stas

MAX_SENTS = 30  # STAS scores the first 30 sentences of each document


def settings_for(stas_dir):
    def build(split, articles):
        settings = {}
        for k in range(len(stas.LAMBDAS) * stas.SLOTS):
            docs = stas.load_output(os.path.join(stas_dir, f'{k}.{split}.txt'))
            if len(docs) != len(articles):
                raise ValueError(f'{k}.{split}.txt has {len(docs)} documents, expected {len(articles)}')
            mismatched = sum(len(s) != min(MAX_SENTS, len(a)) for s, a in zip(docs, articles))
            if mismatched:
                # STAS drops sentences over 500 subwords, which would shift sentence indices
                raise ValueError(f'{k}.{split}.txt: {mismatched} documents with an unexpected number of scores')
            name = stas.setting_name(k)
            settings[f'{name}, in order'] = [selection.topk_in_order(s) for s in docs]
            settings[f'{name}, trigram blocking'] = [
                selection.topk_trigram_blocking(s, a) for s, a in zip(docs, articles)]
        return settings
    return build


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stas-dir', required=True)
    p.add_argument('--data', required=True, help='has {valid,test}.{article,summary}')
    p.add_argument('--out', required=True)
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    released = f'{stas.setting_name(13)}, trigram blocking'
    n_docs = {s: len(stas.load_output(os.path.join(a.stas_dir, f'0.{s}.txt'))) for s in experiment.SPLITS}
    experiment.run('stas', settings_for(a.stas_dir), a.data, a.out, a.jobs,
                   reference_settings=[released], n_docs=n_docs)


if __name__ == '__main__':
    main()
