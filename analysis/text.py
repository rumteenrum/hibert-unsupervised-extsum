"""Reading documents and summaries.

Files have one document per line with sentences separated by <S_SEP>. The text may carry
subword-nmt BPE markers ("a@@ ero@@ planes"); they are removed before evaluation so that
ROUGE sees whole words.
"""

import re

SENT_SEP = '<S_SEP>'


def remove_bpe(text):
    return re.sub(r'@@( |$)', '', text)


def split_sentences(line):
    return [s.strip() for s in remove_bpe(line.strip()).split(SENT_SEP)]


def load_documents(path):
    """One list of sentences per line of `path`."""
    with open(path, encoding='utf8') as f:
        return [split_sentences(line) for line in f]


def load_labels(path):
    """Extractive oracle labels, one 'T'/'F' per sentence."""
    with open(path, encoding='utf8') as f:
        return [line.split() for line in f]
