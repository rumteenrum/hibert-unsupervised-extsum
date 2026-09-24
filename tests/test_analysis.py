import os
import pickle

import numpy as np
import pytest
import torch

from analysis import hibert, positions, selection, stas, stats, text


# --- STAS ranking: compare with the authors' code -------------------------------------

def authors_graph(attn_per_layer):
    """get_raw_matrix() from the authors' model, for one document (batch size 1)."""
    attn_head_avg_head = [weight.mean(dim=1) for weight in attn_per_layer]
    attn_weights_all_layer = sum(attn_head_avg_head)
    g = attn_weights_all_layer / len(attn_per_layer)
    e = torch.eye(g.size(-1)).bool().unsqueeze(0)
    g.masked_fill_(e, 0)
    return g / g.sum(-1, keepdim=True)


def authors_ranking(recovery, graph_matrix):
    """The score loop of the authors' forward(), batch size 1, one temperature."""
    para_list = []
    max_iter = 3
    for a in range(0, 10, 2):
        lam1 = 0.8 + a / 50.
        para_list.append([lam1, 1 - lam1])
    scores = [None] * (len(para_list) * (max_iter + 1))
    nsents = torch.tensor([[recovery.numel()]])
    recovery_score_all = recovery.view(1, -1)
    for j, (lam1, lam2) in enumerate(para_list):
        pr = recovery_score_all.unsqueeze(1)
        new_pr = pr.clone()
        new_pr.fill_(0)
        num_iter = 0
        while True:
            new_pr = pr * lam1 / nsents.unsqueeze(1).type_as(pr) + torch.bmm(pr, graph_matrix) * lam2
            new_pr = new_pr / new_pr.sum(dim=-1, keepdim=True)
            if num_iter < max_iter:
                scores[j * (max_iter + 1) + num_iter] = new_pr.squeeze(1)
            elif num_iter >= max_iter and ((pr - new_pr).abs().sum() < 1e-5 or num_iter > 20):
                break
            num_iter += 1
            pr = new_pr
        scores[j * (max_iter + 1) + max_iter] = new_pr.squeeze(1)
    return [s.view(-1).numpy() for s in scores]


@pytest.mark.parametrize('seed', range(5))
def test_rank_matches_authors_code(seed):
    rng = np.random.default_rng(seed)
    n, layers, heads = int(rng.integers(3, 30)), 6, 8
    per_head = [torch.softmax(torch.tensor(rng.normal(size=(1, heads, n, n))), dim=-1) for _ in range(layers)]
    token_logprobs = [np.log(rng.uniform(0.01, 1.0, int(rng.integers(3, 40)))) for _ in range(n)]

    ours = stas.rank(stas.sentence_recovery(token_logprobs),
                     stas.attention_graph(np.stack([h.mean(dim=1)[0].numpy() for h in per_head])))

    recovery = torch.tensor([np.exp(lp).mean() for lp in token_logprobs])
    theirs = authors_ranking(recovery / recovery.sum(), authors_graph(per_head))

    assert len(ours) == len(theirs) == 20
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


def test_rank_single_sentence():
    assert all(s.tolist() == [1.0] for s in stas.rank(np.array([1.0]), np.zeros((1, 1))))


def test_setting_names_follow_file_numbering():
    assert stas.setting_name(0) == 'lambda1=0.80, 1 update'
    assert stas.setting_name(13) == 'lambda1=0.92, 2 updates'
    assert stas.setting_name(19) == 'lambda1=0.96, converged'


def test_load_output(tmp_path):
    p = tmp_path / '13.test.txt'
    p.write_text('3\n<pad>\t0\nF\t1\nT\t2\n'
                 'True      Labels:\tF F F\nPredicted Labels:\tT F T\nScore:\t0.5 0.1 0.4\nPredicted Distri:\t\n'
                 'True      Labels:\tF F\nPredicted Labels:\tT T\nScore:\t0.6 0.4\nPredicted Distri:\t\n')
    docs = stas.load_output(p)
    assert [d.tolist() for d in docs] == [[0.5, 0.1, 0.4], [0.6, 0.4]]


# --- selection ---------------------------------------------------------------------------

def test_topk_in_order():
    assert selection.topk_in_order([0.1, 0.9, 0.5, 0.7], k=3) == [1, 2, 3]


def test_trigram_blocking_skips_repeats():
    sents = ['the cat sat on the mat', 'a dog sat on the mat today', 'birds sing', 'the cat sat quietly']
    assert selection.topk_trigram_blocking([0.9, 0.8, 0.1, 0.7], sents, k=2) == [0, 2]


def test_trigram_blocking_is_case_sensitive_like_the_original():
    sents = ['The cat sat here', 'the cat sat there']
    assert selection.topk_trigram_blocking([0.9, 0.8], sents, k=2) == [0, 1]


def test_lead_short_document():
    assert selection.lead(2) == [0, 1]


# --- text ----------------------------------------------------------------------------------

def test_remove_bpe_and_split():
    assert text.split_sentences('on a@@ ero@@ planes . <S_SEP> ok@@') == ['on aeroplanes .', 'ok']


# --- statistics ------------------------------------------------------------------------------

def test_paired_identical_systems():
    x = np.random.default_rng(0).uniform(size=200)
    c = stats.paired(x, x, n_boot=500, n_perm=500)
    assert c.mean_diff == 0 and c.ci_low == 0 and c.ci_high == 0 and c.p_value == 1


def test_paired_clear_difference():
    rng = np.random.default_rng(1)
    b = rng.uniform(0.2, 0.4, 2000)
    c = stats.paired(b + 0.01, b, n_boot=500, n_perm=500)
    assert c.mean_diff == pytest.approx(1.0) and c.ci_low == pytest.approx(1.0) and c.p_value < 0.01


# --- positions -----------------------------------------------------------------------------

def test_position_distribution_and_kl():
    p = positions.distribution([[0, 1, 2], [0, 1, 5]], n_positions=6)
    assert p.tolist() == pytest.approx([2 / 6, 2 / 6, 1 / 6, 0, 0, 1 / 6])
    assert positions.kl(p, p) == pytest.approx(0.0)


# --- HIBERT outputs --------------------------------------------------------------------------

def test_load_split_checks_order(tmp_path):
    for i, ids in enumerate([[0, 1], [2]]):
        with open(tmp_path / f'shard_{i:05d}.pkl', 'wb') as f:
            pickle.dump([{'doc_id': d} for d in ids], f)
    assert [r['doc_id'] for r in hibert.load_split(tmp_path)] == [0, 1, 2]
    os.remove(tmp_path / 'shard_00000.pkl')
    with pytest.raises(ValueError):
        hibert.load_split(tmp_path)


# --- ROUGE (needs the Perl script) -----------------------------------------------------------

@pytest.mark.skipif(not os.environ.get('ROUGE_HOME'), reason='ROUGE_HOME not set')
def test_rouge_per_document_scores():
    from analysis.rouge import rouge, summarize
    refs = [['the cat sat on the mat .', 'the cat was happy .'], ['the cat is quiet .']]
    sums = [refs[0], ['dogs bark loudly .']]
    s = rouge(sums, refs)
    assert s['rouge_1'].tolist() == [1.0, 0.0]
    assert summarize(s)['rouge_1'] == pytest.approx(50.0)
