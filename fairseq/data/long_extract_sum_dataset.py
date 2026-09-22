
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import data_utils, FairseqDataset

SENT_SEP = '<S_SEP>'

'''
def split_list(lst, key):
    istart = 0
    res = []
    sublist = []
    for i, v in enumerate(lst):
        sublist.append(v.item())
        if v == key:
            if len(sublist) > 0:
                res.append( sublist )
            sublist = []
    if len(sublist) > 0:
        res.append(sublist)

    return res

# right padding (easy to compute during training)
def docs2tensor(docs, max_nsent, max_sent_len, pad_idx, sep_idx):
    bsz = len(docs)
    src_tokens = torch.LongTensor(bsz, max_nsent, max_sent_len).fill_(pad_idx)
    src_tokens[:, :, -1] = sep_idx
    doc_pad_mask = torch.ByteTensor(bsz, max_nsent).fill_(1)
    for i, doc in enumerate(docs):
        for j, sent in enumerate(doc):
            doc_pad_mask[i, j] = 0
            sent_len = len(sent)
            src_tokens[i, j, -sent_len:] = torch.LongTensor(sent)

    return src_tokens, doc_pad_mask


def create_src_tok_batch(samples, sep_id, eos_idx, pad_idx):
    docs = []
    max_nsent = 0
    max_sent_len = 0
    for sample in samples:
        src = sample['source']
        sents = split_list(src, sep_id)
        if sents[-1][-1] != sep_id:
            sents[-1].append(sep_id)
        max_nsent = max(max_nsent, len(sents))
        cur_max_sent_len = max( map(len, sents) )
        max_sent_len = max( max_sent_len, cur_max_sent_len )
        docs.append(sents)

    return docs2tensor(docs, max_nsent, max_sent_len, pad_idx, sep_id)
'''

def find_sep(src, sep_id):
    sep_pos = []
    for i, v in enumerate(src):
        if sep_id == v:
            sep_pos.append(i)
    return sep_pos

# tokens are left-padding
def docs2tensor(docs, pad_idx):
    doc_sep_pos = map(lambda x: x[1], docs)
    max_nsent = max( map(len, doc_sep_pos) )
    srcs = map(lambda x: x[0], docs)
    max_seqlen = max( map(len, srcs) )
    bsz = len(docs)
    # print('max_nsent', max_nsent)
    # print('max_seqlen', max_seqlen)
    src_tokens = torch.LongTensor(bsz, max_seqlen).fill_(pad_idx)
    doc_pad_mask = torch.ByteTensor(bsz, max_nsent).fill_(1)
    src_sent_ends = torch.LongTensor(bsz, max_nsent).fill_(0) # assume default sentence ends (for padding) are 0s
    for i in range(bsz):
        src, sep_pos = docs[i]
        src_tokens[i, 0:len(src)] = src
        doc_pad_mask[i, 0:len(sep_pos)] = 0
        src_sent_ends[i, 0:len(sep_pos)] = torch.LongTensor(sep_pos)

    return src_tokens, doc_pad_mask, src_sent_ends


def create_src_tok_batch(samples, sep_id, eos_idx, pad_idx):
    docs = []
    for sample in samples:
        src = sample['source']
        if src[-1] != sep_id:
            src_len = src.size(0)
            new_src = src.new(src_len + 1)
            new_src[0:src_len] = src
            new_src[-1] = sep_id
            src = new_src

        sep_pos = find_sep(src, sep_id)
        docs.append((src, sep_pos))

    return docs2tensor(docs, pad_idx)


def create_target_batch(samples, pad_idx):
    maxlen = max( [len(s['target']) for s in samples] )
    bsz = len(samples)
    target = torch.LongTensor(bsz, maxlen).fill_(pad_idx)
    for i, s in enumerate(samples):
        tgt = s['target']
        tgt_len = len(tgt)
        target[i, 0:tgt_len] = tgt
    return target


def collate(samples, src_dict, tgt_dict, left_pad_source=True, left_pad_target=False):
    if len(samples) == 0:
        return {}

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens, doc_pad_mask, src_sent_ends = create_src_tok_batch(samples, src_dict.index(SENT_SEP), src_dict.eos(), src_dict.pad())

    # print('src_tokens', src_tokens.size())
    # print('doc_pad_mask', doc_pad_mask.size())
    # print( src_tokens[:, :, -1] )
    doc_pos_tok = torch.LongTensor( doc_pad_mask.size() ).fill_( src_dict.index(SENT_SEP) )
    doc_pos_tok[ doc_pad_mask ] = src_dict.pad()
    # print( '** doc_pos_tok **' )
    # print( doc_pos_tok )

    ntokens = sum(len(s['target']) for s in samples)
    target = create_target_batch(samples, tgt_dict.pad())

    # print('target', target.size())

    return {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_sent_ends': src_sent_ends,
            'doc_pad_mask': doc_pad_mask,
            'doc_pos_tok': doc_pos_tok,
        },
        'target': target,
    }


class LongExtractSumDataset(FairseqDataset):
    """A pair of torch.utils.data.Datasets."""

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True,
    ):
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle

        self.sent_sep_idx = self.src_dict.index(SENT_SEP)
        print(SENT_SEP, self.sent_sep_idx)

    def __getitem__(self, index):
        return {
            'id': index,
            'source': self.src[index],
            'target': self.tgt[index] if self.tgt is not None else None,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return collate(
            samples, self.src_dict, self.tgt_dict,
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        src_len, tgt_len = min(src_len, max_source_positions), min(tgt_len, max_target_positions)
        bsz = num_tokens // max(src_len, tgt_len)
        return self.collater([
            {
                'id': i,
                'source': self.src_dict.dummy_sentence(src_len),
                'target': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else None,
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Ordered indices for batching."""
        '''we need random order'''
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        '''
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        '''
        return indices

    def valid_size(self, index, max_positions):
        """Check if an example's size is valid according to max_positions."""
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        return (
            self.src_sizes[index] <= max_source_positions
            and (self.tgt_sizes is None or self.tgt_sizes[index] <= max_target_positions)
        )

    def _get_max_positions(self, max_positions):
        if max_positions is None:
            return self.max_source_positions, self.max_target_positions
        assert len(max_positions) == 2
        max_src_pos, max_tgt_pos = max_positions
        return min(self.max_source_positions, max_src_pos), min(self.max_target_positions, max_tgt_pos)
