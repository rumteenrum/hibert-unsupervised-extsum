import numpy as np
import os
import pytest

from analysis.run_directed_residual import build_settings, residual_degree, uniform_degree
from analysis.run_directed_residual import score_without_settings_write


def test_uniform_attention_has_zero_residual_and_preserves_stas_selection():
    for n in (2, 3, 7, 30):
        g = (np.ones((n, n)) - np.eye(n)) / (n - 1)
        np.testing.assert_array_equal(residual_degree(g), np.zeros(n))
        s = np.arange(n, dtype=float)[::-1]
        settings = build_settings([s], [g], [[f'unique sentence {i}' for i in range(n)]])
        assert settings['residual'] == settings['stas']


def test_residual_matches_hand_calculation():
    g = np.array([[0, .7, .3], [.6, 0, .4], [.5, .5, 0]])
    np.testing.assert_allclose(uniform_degree(3), [1, .25, -.5])
    np.testing.assert_allclose(residual_degree(g), [.1, -.1, .15], atol=1e-14)


def test_single_sentence_has_no_bonus_even_with_nan_dump_graph():
    g = np.array([[np.nan]])
    np.testing.assert_array_equal(residual_degree(g), [0])
    settings = build_settings([np.ones(1)], [g], [['only sentence']])
    assert all(sel == [[0]] for sel in settings.values())


@pytest.mark.skipif(not os.environ.get('ROUGE_HOME'), reason='ROUGE_HOME not set')
def test_scoring_without_config_write_preserves_known_answer(tmp_path, monkeypatch):
    from pyrouge import Rouge155
    settings = tmp_path / 'settings.ini'
    settings.write_text('leave unchanged\n')
    monkeypatch.setattr(Rouge155, '_Rouge155__get_config_path', lambda self: str(settings))
    refs = [['the cat sat on the mat .'], ['the cat is quiet .']]
    scores = score_without_settings_write(([[0], [0]], [refs[0], ['dogs bark loudly .']], refs))
    assert scores['rouge_1'].tolist() == [1.0, 0.0]
    assert settings.read_text() == 'leave unchanged\n'
