"""STAS ranking applied to HIBERT outputs.

Uses the per-sentence probabilities and attention saved by hibert_extract.py and the same
20 ranking settings and two selection modes as the released STAS model, with the same
protocol: choose on validation, score test once.

--graph masked     row i of the attention graph from the pass where sentence i is masked
                   (the authors' CNN/DailyMail setting); unmasked uses one unmasked pass.
--recovery true    r~ from the true tokens (as in STAS); top1 uses the probability of the
                   model's top prediction instead, which is what the thesis code computed.
--criteria both    r~ and attention, as in STAS. The two ablations of STAS's Table 2:
                   recovery-only ranks by r~ alone (their "r' = 0"; no lambda or update
                   choice, so only the two selection modes); attention-only replaces r~ with
                   a uniform vector (their "r~ = 1/|D|").

    python -m analysis.run_hibert_stas --hibert-dir outputs/hibert --data DIR --out results
"""

import argparse
import os

import numpy as np

from analysis import experiment, hibert, selection, stas


def document_scores(rec, lp_key, attn_key, criteria):
    """[(setting name, sentence scores)] for one document."""
    recovery = stas.sentence_recovery(rec[lp_key])
    if criteria == 'recovery-only':
        return [('recovery only', recovery)]
    if criteria == 'attention-only':
        recovery = np.full(len(recovery), 1.0 / len(recovery))
    ranks = stas.rank(recovery, stas.attention_graph(rec[attn_key]))
    return [(stas.setting_name(k), s) for k, s in enumerate(ranks)]


def truncate(rec, m):
    """The record restricted to its first m sentences."""
    out = dict(rec)
    for key in ('token_logprobs', 'top1_logprobs', 'uncond_logprobs', 'sent_tokens'):
        if key in rec:
            out[key] = rec[key][:m]
    for key in ('attn_masked_rows', 'attn_unmasked'):
        if key in rec:
            out[key] = rec[key][:, :m, :m]
    out['n'] = m
    return out


def settings_for(hibert_dir, graph, recovery, criteria='both', prefix_dir=None, prefix_mode='graph'):
    """prefix_dir: STAS output directory; restrict each document to the sentences STAS scored.
    prefix_mode 'graph' ranks within that prefix only; 'candidates' ranks all sentences but
    selects only within the prefix."""
    attn_key = 'attn_masked_rows' if graph == 'masked' else 'attn_unmasked'
    lp_key = 'token_logprobs' if recovery == 'true' else 'top1_logprobs'

    def build(split, articles):
        records = hibert.load_split(os.path.join(hibert_dir, split))
        if len(records) > len(articles):
            raise ValueError(f'{split}: {len(records)} HIBERT documents but {len(articles)} articles')
        prefix = None
        if prefix_dir:
            prefix = [len(s) for s in stas.load_output(os.path.join(prefix_dir, f'0.{split}.txt'))]
        per_setting = {}
        for d, (rec, art) in enumerate(zip(records, articles)):
            if rec['n_sents_total'] != len(art) or rec['n'] > len(art):
                raise ValueError(f'{split} doc {rec["doc_id"]}: {rec["n_sents_total"]} sentences in the '
                                 f'HIBERT input but {len(art)} in the text')
            m = rec['n'] if prefix is None else min(prefix[d], rec['n'])
            if prefix is not None and prefix_mode == 'graph':
                rec = truncate(rec, m)
            for name, s in document_scores(rec, lp_key, attn_key, criteria):
                per_setting.setdefault(name, []).append(s[:m])
        settings = {}
        for name, docs in per_setting.items():
            settings[f'{name}, in order'] = [selection.topk_in_order(s) for s in docs]
            settings[f'{name}, trigram blocking'] = [
                selection.topk_trigram_blocking(s, a) for s, a in zip(docs, articles)]
        return settings
    return build


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hibert-dir', required=True, help='has valid/ and test/ shard directories')
    p.add_argument('--data', required=True, help='has {valid,test}.{article,summary}')
    p.add_argument('--out', required=True)
    p.add_argument('--graph', choices=['masked', 'unmasked'], default='masked')
    p.add_argument('--recovery', choices=['true', 'top1'], default='true')
    p.add_argument('--criteria', choices=['both', 'recovery-only', 'attention-only'], default='both')
    p.add_argument('--prefix-from', default=None,
                   help='STAS output directory: use only the sentences STAS scored in each document')
    p.add_argument('--prefix-mode', choices=['graph', 'candidates'], default='graph')
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()

    n_docs = {s: len(hibert.load_split(os.path.join(a.hibert_dir, s))) for s in experiment.SPLITS}
    system = ('hibert_stas' + ('' if a.graph == 'masked' else '_unmasked')
              + ('' if a.recovery == 'true' else '_top1')
              + ('' if a.criteria == 'both' else '_' + a.criteria.replace('-', '_'))
              + ('' if not a.prefix_from else f'_prefix_{a.prefix_mode}'))
    experiment.run(system, settings_for(a.hibert_dir, a.graph, a.recovery, a.criteria,
                                        a.prefix_from, a.prefix_mode),
                   a.data, a.out, a.jobs, n_docs=n_docs)


if __name__ == '__main__':
    main()
