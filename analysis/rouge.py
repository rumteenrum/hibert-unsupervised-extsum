"""ROUGE-1.5.5, run the way STAS and HIBERT report it.

Needs the original Perl script. Set ROUGE_HOME to the directory containing ROUGE-1.5.5.pl
and data/; the Perl modules XML::DOM and DB_File must be installed.

Settings: `-a -c 95 -m -n 2 -w 1.2`, full-length F1, one sentence per line in both the
summary and the reference. ROUGE-L here is therefore the summary-level LCS that other
toolkits call ROUGE-Lsum.
"""

import logging
import os
import re
import shutil
import tempfile

import numpy as np

ROUGE_ARGS = '-a -c 95 -m -n 2 -w 1.2'
METRICS = ('rouge_1', 'rouge_2', 'rouge_l')

_EVAL_LINE = re.compile(r'^\d+ ROUGE-(1|2|L) Eval (\d+)\.1 R:\S+ P:\S+ F:(\S+)$')
_NAME = {'1': 'rouge_1', '2': 'rouge_2', 'L': 'rouge_l'}


def rouge_home():
    home = os.environ.get('ROUGE_HOME')
    if not home or not os.path.exists(os.path.join(home, 'ROUGE-1.5.5.pl')):
        raise RuntimeError('set ROUGE_HOME to the directory containing ROUGE-1.5.5.pl')
    return home


def rouge(summaries, references):
    """Score summaries against references.

    Both arguments are lists (one entry per document) of lists of sentences.
    Returns {metric: array of per-document F1}, in document order. The corpus score
    reported by ROUGE-1.5.5 is the mean of these arrays.
    """
    from pyrouge import Rouge155
    assert len(summaries) == len(references)
    logging.disable(logging.INFO)  # pyrouge logs every file it converts

    home = rouge_home()
    tmp = tempfile.mkdtemp(prefix='rouge_')
    try:
        sys_dir, ref_dir = os.path.join(tmp, 'sys'), os.path.join(tmp, 'ref')
        os.makedirs(sys_dir)
        os.makedirs(ref_dir)
        width = len(str(len(summaries)))  # zero-padded so ROUGE's order is document order
        for i, (s, r) in enumerate(zip(summaries, references)):
            for d, ext, sents in ((sys_dir, 'test', s), (ref_dir, 'gold', r)):
                with open(os.path.join(d, f'{i:0{width}d}.{ext}'), 'w', encoding='utf8') as f:
                    f.write('\n'.join(sents) + '\n')

        r155 = Rouge155(home)
        r155.system_dir, r155.model_dir = sys_dir, ref_dir
        r155.system_filename_pattern = r'(\d+).test'
        r155.model_filename_pattern = '#ID#.gold'
        output = r155.convert_and_evaluate(rouge_args=f'{ROUGE_ARGS} -d -e {home}/data')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    scores = {m: np.full(len(summaries), np.nan) for m in METRICS}
    for line in output.splitlines():
        m = _EVAL_LINE.match(line.strip())
        if m:
            scores[_NAME[m.group(1)]][int(m.group(2)) - 1] = float(m.group(3))
    for name, arr in scores.items():
        if np.isnan(arr).any():
            raise RuntimeError(f'ROUGE returned no {name} score for {int(np.isnan(arr).sum())} documents')
    return scores


def summarize(scores):
    """Corpus scores in the usual 0-100 scale."""
    return {m: 100 * float(v.mean()) for m, v in scores.items()}
