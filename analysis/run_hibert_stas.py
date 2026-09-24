"""STAS ranking applied to HIBERT outputs.

Uses the per-sentence probabilities and attention saved by hibert_extract.py and the same
20 ranking settings and two selection modes as the released STAS model, with the same
protocol: choose on validation, score test once.

--graph masked     row i of the attention graph from the pass where sentence i is masked
                   (the authors' CNN/DailyMail setting); unmasked uses one unmasked pass.
--recovery true    r~ from the true tokens (as in STAS); top1 uses the probability of the
                   model's top prediction instead, which is what the thesis code computed.

    python -m analysis.run_hibert_stas --hibert-dir outputs/hibert --data DIR --out results
"""

import argparse
import os

import numpy as np

from analysis import experiment, hibert, selection, stas


def settings_for(hibert_dir, graph, recovery):
    attn_key = 'attn_masked_rows' if graph == 'masked' else 'attn_unmasked'
    lp_key = 'token_logprobs' if recovery == 'true' else 'top1_logprobs'

    def build(split, articles):
        records = hibert.load_split(os.path.join(hibert_dir, split))
        if len(records) > len(articles):
            raise ValueError(f'{split}: {len(records)} HIBERT documents but {len(articles)} articles')
        per_setting = None
        for rec, art in zip(records, articles):
            if rec['n_sents_total'] != len(art) or rec['n'] > len(art):
                raise ValueError(f'{split} doc {rec["doc_id"]}: {rec["n_sents_total"]} sentences in the '
                                 f'HIBERT input but {len(art)} in the text')
            ranks = stas.rank(stas.sentence_recovery(rec[lp_key]), stas.attention_graph(rec[attn_key]))
            if per_setting is None:
                per_setting = [[] for _ in ranks]
            for k, s in enumerate(ranks):
                per_setting[k].append(s)
        settings = {}
        for k, docs in enumerate(per_setting):
            name = stas.setting_name(k)
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
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()

    n_docs = {s: len(hibert.load_split(os.path.join(a.hibert_dir, s))) for s in experiment.SPLITS}
    system = 'hibert_stas' + ('' if a.graph == 'masked' else '_unmasked') + ('' if a.recovery == 'true' else '_top1')
    experiment.run(system, settings_for(a.hibert_dir, a.graph, a.recovery), a.data, a.out, a.jobs,
                   n_docs=n_docs)


if __name__ == '__main__':
    main()
