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
SENT_MASK = '<SENT_MASK>'

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

def create_target_batch(samples, pad_idx):
    maxlen = max( [len(s['target']) for s in samples] )
    bsz = len(samples)
    target = torch.LongTensor(bsz, maxlen).fill_(pad_idx)
    for i, s in enumerate(samples):
        tgt = s['target']
        tgt_len = len(tgt)
        target[i, 0:tgt_len] = tgt
    return target

# split each doc into sentences
# sent sep included!
def get_docs(samples, vocab, maxlen=None):
    sep_id = vocab.index(SENT_SEP)
    docs = []
    for sample in samples:
        source = sample['source']
        if len(source) == 1:
            source = source[0]  # assume source is a list of one tensor

        if source[-1] == vocab.eos():
                source[-1] = sep_id

        doc = []
        istart = 0
        for i in range(len(source)):
            if source[i] == sep_id or i == len(source)-1:
                sent = source[istart:i+1]
                if source[i] != sep_id:
                    sent = torch.LongTensor(i+1-istart+1)
                    sent[0:-1] = source[istart:i+1]
                    sent[-1] = sep_id
                doc.append( sent )
                if maxlen and len(doc[-1]) > maxlen:
                    new_sent = doc[-1][0:maxlen]
                    new_sent[-1] = sep_id
                    doc[-1] = new_sent

                istart = i + 1
        docs.append(doc)

    return docs

class PretrainDocDataset(FairseqDataset):
    """A pair of torch.utils.data.Datasets."""

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=False, #####
        max_sent_len=None,
        max_doc_len=None,
        masked_sent_prob=None,
        max_predictions_per_doc=None,
        rng=None,
        doc_sizes=None,
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
        self.shuffle = False #####
        self.max_sent_len = max_sent_len
        self.max_doc_len = max_doc_len
        self.masked_sent_prob = masked_sent_prob
        self.max_predictions_per_doc = max_predictions_per_doc
        self.min_predictions_per_doc = 1
        self.rng = rng

        self.sent_sep_idx = self.src_dict.index(SENT_SEP)
        print(SENT_SEP, self.sent_sep_idx)
        self.sent_mask_idx = self.src_dict.index(SENT_MASK)

        # number of tokens in a doc: max_nsent x max_sent_len
        self.src_doc_sizes = doc_sizes

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
        return self.collate(
            samples, self.src_dict, self.tgt_dict,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            max_sent_len=self.max_sent_len,
        )

    def collate(self, samples, src_dict, tgt_dict, left_pad_source=True,
                left_pad_target=False, max_sent_len=None):
        if len(samples) == 0:
            return {}

        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens, doc_pad_mask, tgt_selected_indexes, tgt_input_masked_sents, tgt_masked_sents, source_docs = self.create_batch(samples, src_dict, max_sent_length=max_sent_len)

        # Flatten source_docs
        flattened_sources = []
        for doc in source_docs:
            flat_doc = []
            for sent in doc:
                flat_doc.extend(sent.tolist())
            flattened_sources.append(torch.LongTensor(flat_doc))

        doc_pos_tok = torch.LongTensor(doc_pad_mask.size()).copy_(src_tokens[:, :, -1])
        doc_pad_mask = doc_pos_tok.new_zeros(doc_pos_tok.size()).bool()
        doc_pos_tok[doc_pad_mask] = src_dict.pad()

        ntokens_label = sum(len(s['target']) for s in samples)
        target_label = create_target_batch(samples, tgt_dict.pad())
        ntokens = tgt_masked_sents.ne(self.src_dict.pad()).sum().item()

        return {
            'id': id,
            'ntokens_label': ntokens_label,
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'doc_pad_mask': doc_pad_mask,
                'doc_pos_tok': doc_pos_tok,
                'masked_sent_positions': tgt_selected_indexes,
                'prev_output_tokens': tgt_input_masked_sents,
            },
            'target_label': target_label,
            'target': tgt_masked_sents,
            'source': flattened_sources,  # Now contains flattened source documents
        }

    def doc2tensor(self, docs, vocab):
        max_sent_len, max_nsent = 0, 0
        # max_sent_len, max_nsent = 51, 30
        for doc in docs:
            max_nsent = max( len(doc), max_nsent )
            for sent in doc:
                max_sent_len = max( len(sent), max_sent_len )

        pad_idx = vocab.pad()
        sep_idx = self.sent_sep_idx
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

    '''def mask_sentences(self, index, docs, masked_sent_prob, max_predictions_per_doc, vocab):
        def get_rnd_sent(index, docs):
            rnd_idx = -1
            for i in range(10):
                rnd_idx = self.rng.randint(0, len(docs))
                if rnd_idx != index:
                    break
            sampled_doc = docs[rnd_idx]

            return sampled_doc[ self.rng.randint(0, len(sampled_doc)) ]

        doc = docs[index]
        candi_indexes = list(range(len(doc)))
        self.rng.shuffle(candi_indexes)
        num_pred = min( max(self.min_predictions_per_doc, int(len(candi_indexes) * masked_sent_prob)),
                        max_predictions_per_doc )

        assert len(candi_indexes[0:num_pred]) == len(set(candi_indexes[0:num_pred]))

        output_doc = list(doc)
        selected_indexes = candi_indexes[0:num_pred]
        selected_indexes.sort()
        masked_sents = []

        for i in selected_indexes:
            if self.rng.uniform() < 0.8:
                masked_sent = [ self.sent_mask_idx ] * len(output_doc[i])
                masked_sent[-1] = self.sent_sep_idx
                output_doc[i] = masked_sent
            else:
                if self.rng.uniform() < 0.5:
                    output_doc[i] = doc[i]
                else:
                    rnd_sent = get_rnd_sent(i, docs)
                    output_doc[i] = rnd_sent
            masked_sents.append( doc[i] )


        return output_doc, selected_indexes, masked_sents'''

    #####
    def mask_sentence_at_index(self, doc, sent_idx):
        """Mask only the sentence at the specified index."""
        if sent_idx >= len(doc):
            raise ValueError(f"Sentence index {sent_idx} out of range (doc has {len(doc)} sentences)")
    
        output_doc = list(doc)
        masked_sent = [self.sent_mask_idx] * len(output_doc[sent_idx])
        masked_sent[-1] = self.sent_sep_idx
        output_doc[sent_idx] = masked_sent
        return output_doc, [sent_idx], [doc[sent_idx]]

    # def mask_sentences_j(self, index, docs, masked_sent_prob, max_predictions_per_doc, vocab, j):
    def mask_sentences(self, index, docs, masked_sent_prob, max_predictions_per_doc, vocab):
        
        doc = docs[index]
        output_doc = list(doc)
        # selected_indexes = list(range(len(doc)))
        selected_indexes = [1]
        # selected_indexes = [j]
        masked_sents = []

        for i in selected_indexes:
            masked_sent = [ self.sent_mask_idx ] * len(output_doc[i])
            masked_sent[-1] = self.sent_sep_idx
            output_doc[i] = masked_sent
            
            masked_sents.append( doc[i] )

        return output_doc, selected_indexes, masked_sents

    # def mask_sentences(self, index, docs, masked_sent_prob, max_predictions_per_doc, vocab):
    #     # for j in range(len(docs[index])):
    #     #     if self.rng.uniform() < masked_sent_prob:
    #     j = 1
    #     return self.mask_sentences_j(index, docs, masked_sent_prob, max_predictions_per_doc, vocab, j)


    def masked_sents2tensor(self, docs_selected_indexes, docs_masked_sents):
        bsz = len(docs_selected_indexes)
        max_nsent = max( [len(sel_idxs) for sel_idxs in docs_selected_indexes] )
        tgt_selected_indexes = torch.LongTensor(bsz, max_nsent).fill_(0)
        for i, sel_idxs in enumerate(docs_selected_indexes):
            si_len = len(sel_idxs)
            tgt_selected_indexes[i, 0:si_len] = torch.LongTensor(sel_idxs)

        max_nsent2, max_sent_len = 0, 0
        for masked_sents in docs_masked_sents:
            max_nsent2 = max( max_nsent2, len(masked_sents) )
            local_max_sent_len = max( map(len, masked_sents) )
            max_sent_len = max(max_sent_len, local_max_sent_len)

        assert max_nsent == max_nsent2

        tgt_input_masked_sents = torch.LongTensor(bsz, max_nsent, max_sent_len).fill_(self.src_dict.pad())
        tgt_input_masked_sents[:, :, 0] = self.src_dict.eos()
        tgt_masked_sents = torch.LongTensor(bsz, max_nsent, max_sent_len).fill_(self.src_dict.pad())
        for i, masked_sents in enumerate(docs_masked_sents):
            for j, sent in enumerate(masked_sents):
                sent_len = len(sent)
                assert sent[-1] == self.sent_sep_idx
                sent[-1] = self.src_dict.eos()
                tgt_input_masked_sents[i, j, 1:sent_len] = torch.LongTensor(sent[0:-1])
                tgt_masked_sents[i, j, 0:sent_len] = torch.LongTensor(sent)

        return tgt_selected_indexes, tgt_input_masked_sents, tgt_masked_sents


    def create_batch(self, samples, vocab, max_sent_length=None):
        # split sample into documents
        docs = get_docs(samples, vocab, max_sent_length+1)
        # create masked sentence
        new_docs = []
        docs_selected_indexes = []
        docs_masked_sents = []
        for i in range(len(docs)):
            new_doc, selected_indexes, masked_sents = self.mask_sentences(i, docs,
                                                        self.masked_sent_prob,
                                                        self.max_predictions_per_doc,
                                                        vocab)
            new_docs.append(new_doc)
            docs_selected_indexes.append(selected_indexes)
            docs_masked_sents.append(masked_sents)

        # doc to tensor
        src_tokens, doc_pad_mask = self.doc2tensor(new_docs, vocab)

        # get masked sentences
        tgt_selected_indexes, tgt_input_masked_sents, tgt_masked_sents = self.masked_sents2tensor(docs_selected_indexes, docs_masked_sents)

        return src_tokens, doc_pad_mask, tgt_selected_indexes, tgt_input_masked_sents, tgt_masked_sents, docs


    def get_dummy_batch(self, num_docs, max_positions, src_len=128, tgt_len=128):
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        # src_len, tgt_len = min(src_len, max_source_positions), min(tgt_len, max_target_positions)
        bsz = num_docs

        def create_tgt():
            return torch.LongTensor([self.tgt_dict.index('F')] * self.max_doc_len)

        def create_src():
            doc = []
            for i in range(self.max_doc_len):
                for j in range(self.max_sent_len):
                    doc.append(self.src_dict.unk())
                if i != self.max_doc_len-1:
                    doc.append(self.sent_sep_idx)
                else:
                    doc.append(self.src_dict.eos())
            return torch.LongTensor(doc)

        orig_min_predictions_per_doc = self.min_predictions_per_doc
        self.min_predictions_per_doc = self.max_predictions_per_doc
        batch = self.collater([
            {
                'id': i,
                'source': create_src(),
                'target': create_tgt(),
            }
            for i in range(bsz)
        ])
        self.min_predictions_per_doc = orig_min_predictions_per_doc

        return batch


    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        # return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        nsent, max_sent_len = self.src_doc_sizes[index]
        return nsent * min(self.max_sent_len, max_sent_len)

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
