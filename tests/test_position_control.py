import numpy as np

from analysis.pmi import standardize
from analysis.run_position_control import position_scores


def test_linear_bonus_matches_every_uniform_directed_null():
    n = 7
    sal = np.array([.1, .3, .2, .4, .1, .05, .2])
    i = np.arange(n)
    for lb in (0, -.5, -1, -2):
        uniform = ((n - 1 - i) + lb * i) / (n - 1)
        np.testing.assert_allclose(position_scores(sal, .5),
                                   standardize(sal) + .5 * standardize(uniform), atol=1e-14)


def test_position_bonus_breaks_uniform_salience_toward_earlier_sentences():
    scores = position_scores(np.ones(5), 1)
    assert np.all(np.diff(scores) < 0)
    np.testing.assert_array_equal(position_scores(np.ones(1), 1), [0])
