import os

import numpy as np
import pytest

from analysis.run_directed_residual import score_without_settings_write
from analysis.run_directed_test import score_fixed


@pytest.mark.skipif(not os.environ.get('ROUGE_HOME'), reason='ROUGE_HOME not set')
def test_reference_directory_cache_preserves_rouge_and_restores_lookup():
    from pyrouge import Rouge155
    original = Rouge155._Rouge155__get_model_filenames_for_id
    work = ([[0], [0], [0, 1]],
            [['the cat sat on the mat .'], ['dogs bark loudly .'],
             ['the first fact is important .', 'a second fact follows .']],
            [['the cat sat on the mat .'], ['the cat is quiet .'],
             ['the first fact matters .', 'the second fact follows .']])
    expected = score_without_settings_write(work)
    actual = score_fixed(work)
    for metric in expected:
        np.testing.assert_array_equal(actual[metric], expected[metric])
    assert Rouge155._Rouge155__get_model_filenames_for_id is original
