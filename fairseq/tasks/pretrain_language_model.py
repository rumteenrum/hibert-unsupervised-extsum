# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import torch
import numpy as np

from fairseq.data import (
    Dictionary, IndexedInMemoryDataset, IndexedRawTextDataset,
    FlexibleDictionary,
    MonolingualDataset, TokenBlockDataset, BERTDataset,
)

from . import FairseqTask, register_task


@register_task('pretrain_language_modeling')
class PretrainLanguageModelingTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('--sample-break-mode', metavar='VAL',
                            choices=['none', 'complete', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=1024, type=int, metavar='N',
                            help='max number of tokens per sample for LM dataset')
        parser.add_argument('--max-train-seq-length', default=512, type=int, metavar='N',
                            help='max length for a training sequence (two text spans)')
        parser.add_argument('--short-seq-prob', default=0.1, type=float, help='prob to shorten --max-train-seq-length')
        # masked_lm_prob, max_predictions_per_seq
        parser.add_argument('--masked-lm-prob', default=0.15, type=float, help='prob to predict masked lm')
        parser.add_argument('--max-predictions-per-seq', default=80, type=int, help='maximum length per sequence')
        parser.add_argument('--max-bert-batch-size', default=20, type=int, help='maximum batch size')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.max_src_size = 0
        self.max_tgt_size = 0
        self.rng = np.random.RandomState(args.seed)
        self.max_bert_batch_size = args.max_bert_batch_size

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
        # add special tokens for this task
        dictionary.add_symbol('<CLS>')
        dictionary.add_symbol('<MASK>')
        print('| dictionary: {} types'.format(len(dictionary)))
        print('<CLS> id is', dictionary.index('<CLS>'))
        print('<MASK> id is', dictionary.index('<MASK>'))
        return cls(args, dictionary)

    def load_dataset(self, split, shuffle=True):
        """Load a dataset split."""
        path = os.path.join(self.args.data, split)
        if self.args.raw_text and IndexedRawTextDataset.exists(path):
            ds = IndexedRawTextDataset(path, self.dictionary)
            tokens = ds.tokens_list
        elif not self.args.raw_text and IndexedInMemoryDataset.exists(path):
            ds = IndexedInMemoryDataset(path, fix_lua_indexing=True)
            tokens = ds.buffer
        else:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))

        dataset = ds

        self.datasets[split] = BERTDataset(dataset, dataset.sizes, self.dictionary, self.args.max_train_seq_length,
                                           self.args.short_seq_prob, self.args.masked_lm_prob, self.args.max_predictions_per_seq, shuffle=shuffle, rng=self.rng,
                                           max_batch_size=self.max_bert_batch_size)

    @property
    def target_dictionary(self):
        return self.dictionary

    def clear_cuda(self, sample):
        src_size = sample['net_input']['src_tokens'].numel()
        tgt_size = sample['target'].numel()
        if src_size > self.max_src_size or tgt_size > self.max_tgt_size:
            torch.cuda.empty_cache()
            if src_size > self.max_src_size:
                self.max_src_size = src_size
            if tgt_size > self.max_tgt_size:
                self.max_tgt_size = tgt_size
