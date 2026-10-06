import numpy as np
import pytest

from analysis.run_directed_hibert_residual import variant_inputs


def record():
    return {'doc_id': 0, 'n': 3, 'n_sents_total': 3,
            'token_logprobs': [np.array([-.1]), np.array([-.2]), np.array([-.3])],
            'attn_masked_rows': np.array([[[.1, .2, .7], [.3, .2, .5], [.4, .5, .1]]])}


def test_hibert_prefix_renormalizes_after_truncation():
    salience, graphs = variant_inputs([record()], [['a', 'b', 'c']], [2], 'prefix')
    np.testing.assert_allclose(graphs[0], [[0, 1], [1, 0]])
    assert len(salience[0]) == 2
    _, full = variant_inputs([record()], [['a', 'b', 'c']], [2], 'full')
    assert full[0].shape == (3, 3)
    np.testing.assert_allclose(full[0].sum(axis=1), [1, 1, 1])


def test_hibert_singleton_prefix_has_finite_zero_graph():
    salience, graphs = variant_inputs([record()], [['a', 'b', 'c']], [1], 'prefix')
    np.testing.assert_array_equal(salience[0], [1])
    np.testing.assert_array_equal(graphs[0], [[0]])


def test_hibert_text_alignment_is_checked():
    with pytest.raises(ValueError, match='alignment'):
        variant_inputs([record()], [['a', 'b']], [2], 'prefix')
