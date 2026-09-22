
import re
from .google_rouge import rouge
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
import sys

def _preprocess_line(line):
    line = re.sub(r"[^0-9a-z]", " ", line)
    line = re.sub(r"\s+", " ", line)
    words = line.strip().split()
    words = [stemmer.stem(w.lower()) for w in words]
    new_line = ' '.join(words)
    return new_line

def get_rouge_batch(hyps, refs, with_preprocess=True):
    if with_preprocess:
        hyps = list(map(_preprocess_line, hyps))
        refs = list(map(_preprocess_line, refs))
    return rouge(hyps, refs)

def get_rouge(hyp, ref, with_preprocess=True):
    return get_rouge_batch([hyp], [ref])

def get_rouge_file(hyp_file, ref_file, encoding='utf-8', with_preprocess=True, print_rouge=False):
    hyps = open(hyp_file, encoding=encoding).readlines()
    refs = open(ref_file, encoding=encoding).readlines()
    rouge = get_rouge_batch(hyps, refs)
    if print_rouge:
        r1, r2, rl = rouge['rouge_1/f_score'], rouge['rouge_2/f_score'], rouge['rouge_l/f_score']
        print('ROUGE-1 (F) %.2f | ROUGE-2 (F) %.2f | ROUGE-L (F) %.2f'%(r1*100, r2*100, rl*100))
        sys.stdout.flush()

    return rouge

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp_file')
    parser.add_argument('--ref_file')
    parser.add_argument('--nopreprocess', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    res = get_rouge_file(args.hyp_file, args.ref_file, with_preprocess=args.nopreprocess)
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(res)
