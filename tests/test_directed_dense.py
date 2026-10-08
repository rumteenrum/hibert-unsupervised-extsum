import os

import numpy as np
import pytest

from analysis.rouge import METRICS
from analysis.run_directed_dense import batch_work, pack_unique
from analysis.run_directed_test import score_fixed


@pytest.mark.skipif(not os.environ.get('ROUGE_HOME'), reason='ROUGE_HOME not set')
def test_unique_summary_scoring_preserves_document_references_and_order():
    # Same indices on two documents must not share a cache key. Reversed selection
    # order remains distinct, even though this particular ROUGE example may tie.
    articles = [['alpha beta gamma .', 'delta epsilon zeta .'],
                ['alpha beta gamma .', 'delta epsilon zeta .']]
    refs = [['alpha beta gamma .'], ['delta epsilon zeta .']]
    settings = {'a': [[0, 1], [0]], 'b': [[1, 0], [0]], 'c': [[0, 1], [1]]}
    summaries, unique_refs, mapping = pack_unique(settings, articles, refs)
    assert len(summaries) == 4
    assert mapping['a'][0] != mapping['b'][0]
    assert mapping['a'][0] != mapping['a'][1]
    unique = score_fixed(batch_work(summaries, unique_refs))
    for name, sels in settings.items():
        direct = score_fixed((sels, articles, refs))
        for metric in METRICS:
            np.testing.assert_array_equal(unique[metric][mapping[name]], direct[metric])
