"""Directed attention centrality on HIBERT's graph, at a fixed setting.

HIBERT with the STAS criteria, each document restricted to the sentences STAS scored (the
"prefix-graph" variant of run_hibert_stas), then combined with the directed degree of the same
graph: z(salience) + mu * z(D). With --full, HIBERT's first 30 sentences are used instead.
The directed setting is the one fixed on STAS; nothing is tuned here.

    python -m analysis.run_directed_hibert --hibert-dir OUT/hibert --stas-dir OUT/stas --data DIR --out results
"""

import argparse
import json
import os

from analysis import directed, experiment, hibert, positions, selection, stas, stats
from analysis.pmi import standardize
from analysis.rouge import summarize
from analysis.run_hibert_stas import truncate

# ranking setting chosen on validation for each variant: lambda1 0.92 after 3 updates (prefix-graph),
# lambda1 0.88 after 2 updates (full)
SALIENCE_SETTING = {'prefix': 14, 'full': 9}
FIXED = {'directed, lam_f=1, lam_b=-0.5, mu=1': (1.0, -0.5, 1.0),
         'undirected, lam_f=1, lam_b=1, mu=1': (1.0, 1.0, 1.0)}
BASELINE = 'baseline: HIBERT-STAS + trigram blocking'


def build_settings(hibert_dir, stas_dir, articles, split, variant='prefix'):
    records = hibert.load_split(os.path.join(hibert_dir, split))[:len(articles)]
    if variant == 'prefix':
        prefix = [len(s) for s in stas.load_output(os.path.join(stas_dir, f'0.{split}.txt'))]
    else:
        prefix = [r['n'] for r in records]
    if len(records) != len(articles) or len(prefix) != len(articles):
        raise ValueError('HIBERT records, STAS prefixes and articles differ in length')

    sal, graphs = [], []
    for rec, m in zip(records, prefix):
        rec = truncate(rec, min(m, rec['n']))
        g = stas.attention_graph(rec['attn_masked_rows'])
        sal.append(stas.rank(stas.sentence_recovery(rec['token_logprobs']), g)[SALIENCE_SETTING[variant]])
        graphs.append(g)

    pick = lambda scores: [selection.topk_trigram_blocking(sc, a) for sc, a in zip(scores, articles)]
    settings = {BASELINE: pick(sal)}
    for name, (lf, lb, mu) in FIXED.items():
        settings[name] = pick([standardize(s) + mu * standardize(directed.directed_degree(g, lf, lb))
                               for s, g in zip(sal, graphs)])
    return settings


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hibert-dir', required=True, help='has valid/ and test/ shard directories')
    p.add_argument('--stas-dir', help='STAS score files (for the per-document prefix)')
    p.add_argument('--full', action='store_true', help="use HIBERT's first 30 sentences, not STAS's prefix")
    p.add_argument('--data', required=True)
    p.add_argument('--split', default='valid')
    p.add_argument('--out', required=True)
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()

    articles, references = experiment.load_split(a.data, a.split)
    variant = 'full' if a.full else 'prefix'
    if variant == 'prefix' and not a.stas_dir:
        p.error('--stas-dir is needed unless --full is given')
    settings = build_settings(a.hibert_dir, a.stas_dir, articles, a.split, variant)
    scores = experiment.evaluate(settings, articles, references, a.jobs)
    table = {n: summarize(s) for n, s in scores.items()}
    per_doc = {n: (s['rouge_1'] + s['rouge_2'] + s['rouge_l']) / 3 for n, s in scores.items()}
    gains = {n: str(stats.paired(per_doc[n], per_doc[BASELINE])) for n in FIXED}
    pos = {n: positions.distribution(settings[n]).round(3).tolist()[:4] for n in settings}

    report = {'variant': variant, 'split': a.split, 'n_docs': len(articles), 'table': table,
              'minus_baseline_mean_rouge': gains, 'positions_first4': pos}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f'directed_hibert_{variant}.{a.split}.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print(f'| Directed HIBERT, {variant} ({a.split}, {len(articles)} docs)')
    for n, t in table.items():
        print(f'|   {experiment.mean_rouge(t):6.2f}  ({t["rouge_1"]:.2f} / {t["rouge_2"]:.2f} / {t["rouge_l"]:.2f})  {n}')
    for n, g in gains.items():
        print(f'| {n} − baseline (mean ROUGE, paired 95% CI): {g}')
    for n, d in pos.items():
        print(f'| positions 0/1/2/3: {d}  {n}')


if __name__ == '__main__':
    main()
