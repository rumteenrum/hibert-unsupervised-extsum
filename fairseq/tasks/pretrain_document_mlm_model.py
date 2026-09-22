
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import numpy as np
import torch
from fairseq import options
from fairseq.data import (
    data_utils, Dictionary, FlexibleDictionary, LanguagePairDataset, ExtractSumDataset, IndexedInMemoryDataset,
    PretrainDocDataset,
    PretrainDocMLMDataset,
    IndexedRawTextDataset,
)

from . import FairseqTask, register_task


@register_task('pretrain_document_mlm_modeling')
class PretrainDocumentMLMModelingTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=40960, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=40960, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--ncpu-eval', default=2, type=int, metavar='N',
                            help='number of CPUs during rouge evaluation')
        parser.add_argument('--topk-sent-eval', default=3, type=int, metavar='N',
                            help='number of sentences selected during rouge evaluation')
        parser.add_argument('--raw-valid', default=None, metavar='RAW_VALID',
                            help='raw valid set')
        parser.add_argument('--raw-test', default=None, metavar='RAW_TEST',
                            help='raw test set')
        parser.add_argument('--max-sent-length', default=50, type=int, metavar='N',
                            help='max number of tokens a source document sentence can have')
        parser.add_argument('--max-doc-length', default=30, type=int, metavar='N',
                            help='max number of sentences a source document can have')

        parser.add_argument('--masked-sent-prob', default=0.15, type=float, help='prob to predict masked lm')
        parser.add_argument('--max-predictions-per-doc', default=5, type=int, help='maximum number of masked sentences per doc')

        # this part is for masked language model
        parser.add_argument('--masked-lm-prob', default=0.15, type=float, help='prob to predict masked lm')
        parser.add_argument('--max-predictions-per-seq', default=8, type=int, help='maximum length per sequence')

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.max_src_size = 0
        self.max_tgt_size = 0
        self.run_dummy_batch = True

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data)
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(args.source_lang)))
        src_dict.add_symbol('<WORD_MASK>')
        src_dict.add_symbol('<SENT_MASK>')
        print('<WORD_MASK> id is', src_dict.index('<WORD_MASK>'))
        print('<SENT_MASK> id is', src_dict.index('<SENT_MASK>'))

        tgt_dict = FlexibleDictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(args.target_lang)))
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def create_doc_size_file(self, doc_dataset, sent_sep_idx, doc_size_file):
        with open(doc_size_file, 'w', encoding='utf8') as fout:
            print('dataset size', len(doc_dataset))
            for i in range(len(doc_dataset)):
                src_doc = doc_dataset[i]
                # src_doc = self.src[index]

                istart = 0
                max_sent_len = 0
                doc_nsent = 0
                for i in range(len(src_doc)):
                    if src_doc[i] == sent_sep_idx or i == len(src_doc) - 1:
                        sent_len = i - istart
                        if src_doc[i] != sent_sep_idx:
                            sent_len += 1
                        max_sent_len = max(max_sent_len, sent_len)
                        istart = i+1
                        doc_nsent += 1

                fout.write('{}\t{}\n'.format(doc_nsent, max_sent_len))
                fout.flush()

    def load_doc_size_file(self, doc_size_file):
        doc_sizes = []
        for line in open(doc_size_file, encoding='utf8'):
            fds = line.strip().split()
            assert len(fds) == 2, 'size file MUST have two fileds'
            doc_sizes.append( (int(fds[0]), int(fds[1])) )
        print('load doc size done', len(doc_sizes))
        return doc_sizes

    def load_dataset(self, split, shuffle=True):
        """Load a dataset split."""

        def split_exists(src, tgt, lang):
            filename = os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedInMemoryDataset.exists(filename):
                return True
            return False

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang
        if split_exists(src, tgt, src):
            prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
        elif split_exists(tgt, src, src):
            prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, tgt, src))
        else:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedInMemoryDataset.exists(path):
                return IndexedInMemoryDataset(path, fix_lua_indexing=True)
            return None

        src_dataset = indexed_dataset(prefix + src, self.src_dict)
        tgt_dataset = indexed_dataset(prefix + tgt, self.tgt_dict)
        rng = np.random.RandomState(self.args.seed)

        # get doc size information
        assert isinstance(src_dataset, IndexedInMemoryDataset), 'currently only support IndexedInMemoryDataset'
        src_path = prefix + src
        src_doc_size_path = src_path + '.doc.size'
        if not os.path.exists(src_doc_size_path):
            print(src_doc_size_path, 'not exists!!!')
            SENT_SEP = '<S_SEP>'
            sent_sep_idx = self.src_dict.index(SENT_SEP)
            self.create_doc_size_file(src_dataset, sent_sep_idx, src_doc_size_path)
            print('create doc size file done!')
        doc_sizes = self.load_doc_size_file(src_doc_size_path)

        # need to be updated with extractive summarization dataset
        self.datasets[split] = PretrainDocMLMDataset(
            src_dataset, src_dataset.sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            shuffle=shuffle,
            max_sent_len=self.args.max_sent_length,
            max_doc_len=self.args.max_doc_length,
            masked_sent_prob=self.args.masked_sent_prob,
            max_predictions_per_doc=self.args.max_predictions_per_doc,
            masked_lm_prob=self.args.masked_lm_prob,
            max_predictions_per_seq=self.args.max_predictions_per_seq,
            rng=rng,
            doc_sizes=doc_sizes,
        )

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict

    def clear_cuda(self, sample):
        src_size = sample['net_input']['src_tokens'].numel()
        tgt_size = sample['target'].numel()
        if src_size > self.max_src_size or tgt_size > self.max_tgt_size:
            torch.cuda.empty_cache()
            if src_size > self.max_src_size:
                self.max_src_size = src_size
            if tgt_size > self.max_tgt_size:
                self.max_tgt_size = tgt_size
