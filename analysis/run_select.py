"""Sentence selection with relevance and redundancy on top of a fixed salience score.

Salience is taken at the ranking setting already chosen on validation for STAS and for
HIBERT-STAS (lambda1 = 0.88 after 2 updates, file index 9), so no new ranking choice is made
here. For each document the score of a sentence is

    z(salience) + lam_rel * z(relevance)

with z = per-document standardisation and relevance = HIBERT PMI with the document
(analysis.pmi.relevance). Redundancy is then controlled in one of three ways:

  trigram   trigram blocking on that score (the STAS default)
  cosine    greedy MMR, penalty lam_red * sum of word-count cosine with chosen sentences
  pmi       greedy MMR, penalty lam_red * sum of HIBERT PMI with chosen sentences
            (needs --pairs-dir from hibert_pairs.py)

Candidates are the sentences the salience covers: STAS's scored prefix, or HIBERT's first 30.
lam_rel and lam_red come from fixed grids and are chosen on validation; test is scored once.

    python -m analysis.run_select --salience stas --redundancy cosine \
        --stas-dir OUT/stas --hibert-dir OUT/hibert --data DIR --out results
"""

import argparse
import os

from analysis import experiment, hibert, pmi, selection, stas

SALIENCE_SETTING = 9  # lambda1 = 0.88 after 2 updates, chosen on validation for both systems
LAM_REL = (0.0, 0.25, 0.5, 1.0)
LAM_RED = (0.0, 0.5, 1.0, 2.0, 4.0)


def salience_for(kind, split, stas_dir, records):
    if kind == 'stas':
        return stas.load_output(os.path.join(stas_dir, f'{SALIENCE_SETTING}.{split}.txt'))
    return [stas.rank(stas.sentence_recovery(r['token_logprobs']),
                      stas.attention_graph(r['attn_masked_rows']))[SALIENCE_SETTING] for r in records]


def settings_for(kind, redundancy, stas_dir, hibert_dir, pairs_dir=None):
    def build(split, articles):
        records = hibert.load_split(os.path.join(hibert_dir, split))[:len(articles)]
        sal = salience_for(kind, split, stas_dir, records)
        pairs = hibert.load_split(os.path.join(pairs_dir, split))[:len(articles)] if redundancy == 'pmi' else None

        docs = []
        for d, (art, s, rec) in enumerate(zip(articles, sal, records)):
            n = len(s)
            if n > rec['n']:
                raise ValueError(f'{split} doc {d}: salience covers {n} sentences, HIBERT {rec["n"]}')
            red = None
            if redundancy == 'cosine':
                red = pmi.cosine_matrix(art[:n])
            elif redundancy == 'pmi':
                red = pmi.pmi_matrix(pairs[d])[:n, :n]
            docs.append((pmi.standardize(s), pmi.standardize(pmi.relevance(rec)[:n]), red, art))

        settings = {}
        for lr in LAM_REL:
            if redundancy == 'trigram':
                settings[f'rel={lr}, trigram blocking'] = [
                    selection.topk_trigram_blocking(zs + lr * zr, art) for zs, zr, _, art in docs]
                continue
            for ld in LAM_RED:
                settings[f'rel={lr}, red={ld}'] = [
                    selection.greedy(zs + lr * zr, red, ld) for zs, zr, red, _ in docs]
        return settings
    return build


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--salience', choices=['stas', 'hibert'], required=True)
    p.add_argument('--redundancy', choices=['trigram', 'cosine', 'pmi'], required=True)
    p.add_argument('--stas-dir', help='STAS output files (needed for --salience stas)')
    p.add_argument('--hibert-dir', required=True, help='hibert_extract.py outputs (for relevance)')
    p.add_argument('--pairs-dir', help='hibert_pairs.py outputs (needed for --redundancy pmi)')
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    if a.salience == 'stas' and not a.stas_dir:
        p.error('--salience stas needs --stas-dir')
    if a.redundancy == 'pmi' and not a.pairs_dir:
        p.error('--redundancy pmi needs --pairs-dir')

    n_docs = {s: len(hibert.load_split(os.path.join(a.hibert_dir, s))) for s in experiment.SPLITS}
    experiment.run(f'select_{a.salience}_{a.redundancy}',
                   settings_for(a.salience, a.redundancy, a.stas_dir, a.hibert_dir, a.pairs_dir),
                   a.data, a.out, a.jobs, n_docs=n_docs)


if __name__ == '__main__':
    main()
