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
WORD_MASK = '<WORD_MASK>'

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

class PretrainDocMLMDataset(FairseqDataset):
    """A pair of torch.utils.data.Datasets."""

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True,
        max_sent_len=None,
        max_doc_len=None,
        masked_sent_prob=None,
        max_predictions_per_doc=None,
        masked_lm_prob=None,
        max_predictions_per_seq=None,
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
        self.shuffle = shuffle
        self.max_sent_len = max_sent_len
        self.max_doc_len = max_doc_len
        self.masked_sent_prob = masked_sent_prob
        self.max_predictions_per_doc = max_predictions_per_doc
        self.min_predictions_per_doc = 1

        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.min_predictions_per_seq = 1

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
        src_tokens, doc_pad_mask, tgt_selected_indexes, tgt_input_masked_sents, tgt_masked_sents, tgt_masked_words_indexes, tgt_masked_words = self.create_batch(samples, src_dict, max_sent_length=max_sent_len)

        doc_pos_tok = torch.LongTensor( doc_pad_mask.size() ).copy_(src_tokens[:, :, -1])
        doc_pos_tok[ doc_pad_mask ] = src_dict.pad()

        ntokens_label = sum(len(s['target']) for s in samples)
        target_label = create_target_batch(samples, tgt_dict.pad())
        ntokens = tgt_masked_sents.ne( self.src_dict.pad() ).sum().item()
        ntokens_word = tgt_masked_words.ne( self.src_dict.pad() ).sum().item()

        return {
            'id': id,
            'ntokens_label': ntokens_label,
            'ntokens': ntokens,
            'ntokens_word': ntokens_word,
            'net_input': {
                'src_tokens': src_tokens,
                'doc_pad_mask': doc_pad_mask,
                'doc_pos_tok': doc_pos_tok,
                'masked_sent_positions': tgt_selected_indexes,
                'prev_output_tokens': tgt_input_masked_sents,
                'masked_word_positions': tgt_masked_words_indexes,
            },
            'target_label': target_label,
            'target': tgt_masked_sents,
            'target_word': tgt_masked_words,
        }

    def doc2tensor(self, docs, vocab):
        max_sent_len, max_nsent = 0, 0
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

    def mask_words(self, doc, masked_lm_prob, max_predictions_per_seq, vocab):
        new_doc = []
        new_positions = []
        new_predicted_words = []
        for tokens in doc:
            if not isinstance(tokens, list):
                tokens = tokens.tolist()
            masked_tokens, positions, predicted_words = self.mask_words_in_sentence(tokens, masked_lm_prob, max_predictions_per_seq, vocab)
            new_doc.append( masked_tokens )
            new_positions.append( positions )
            new_predicted_words.append( predicted_words )

        return new_doc, new_positions, new_predicted_words

    # doc is a list of sentences; each sentence ends with sent_sep
    def mask_words_in_sentence(self, tokens, masked_lm_prob, max_predictions_per_seq, vocab):
        sep_id = vocab.index(SENT_SEP)
        mask_id = vocab.index(WORD_MASK)

        def is_masked_sent(tokens):
            masked_sent_id = vocab.index(SENT_MASK)
            # print('sent mask id ', masked_sent_id, tokens[0])
            return tokens[0] == masked_sent_id

        if is_masked_sent(tokens):
            # print('masked sent', tokens)
            return tokens, [], []

        candi_indexes = []
        for i, tok in enumerate(tokens):
            if tok != sep_id:
                candi_indexes.append(i)
        self.rng.shuffle(candi_indexes)
        num_pred = min( max(self.min_predictions_per_seq, int(len(candi_indexes) * masked_lm_prob)),
                        max_predictions_per_seq )

        assert len(candi_indexes[0:num_pred]) == len(set(candi_indexes[0:num_pred]))

        output_tokens = list(tokens)
        masked_lm = []
        sampled_indexes = candi_indexes[0:num_pred]
        for index in sampled_indexes:
            masked_token = None
            if self.rng.rand() < 0.8:
                masked_token = mask_id
            else:
                if self.rng.rand() < 0.5:
                    masked_token = tokens[index]
                else:
                    masked_token = self.rng.randint(0, len(vocab))

            output_tokens[index] = masked_token
            masked_lm.append( (index, tokens[index]) )

        masked_lm.sort(key=lambda x: x[0])
        positions = [m[0] for m in masked_lm]
        labels = [m[1] for m in masked_lm]

        assert len(positions) == len(labels)

        return output_tokens, positions, labels


    def mask_sentences(self, index, docs, masked_sent_prob, max_predictions_per_doc, vocab):
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

        return output_doc, selected_indexes, masked_sents


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
        # create masked words
        docs_selected_word_indexes = []
        docs_masked_words = []
        for i in range(len(docs)):
            new_doc, selected_indexes, masked_sents = self.mask_sentences(i, docs,
                                                        self.masked_sent_prob,
                                                        self.max_predictions_per_doc,
                                                        vocab)
            # apply masks on words in sentences
            new_doc_with_masked_words, new_doc_word_idxs, new_doc_pred_words = self.mask_words(new_doc, self.masked_lm_prob, self.max_predictions_per_seq, vocab)

            docs_selected_word_indexes.append( new_doc_word_idxs )
            docs_masked_words.append( new_doc_pred_words )

            # new_docs.append(new_doc)
            new_docs.append(new_doc_with_masked_words)
            docs_selected_indexes.append(selected_indexes)
            docs_masked_sents.append(masked_sents)

        # doc to tensor
        src_tokens, doc_pad_mask = self.doc2tensor(new_docs, vocab)

        # get masked sentences
        tgt_selected_indexes, tgt_input_masked_sents, tgt_masked_sents = self.masked_sents2tensor(docs_selected_indexes, docs_masked_sents)

        # get masked words
        tgt_masked_words_indexes, tgt_masked_words = self.masked_words2tensor(new_docs, docs_selected_word_indexes, docs_masked_words)

        return src_tokens, doc_pad_mask, tgt_selected_indexes, tgt_input_masked_sents, tgt_masked_sents, tgt_masked_words_indexes, tgt_masked_words


    def masked_words2tensor(self, new_docs, docs_selected_word_indexes, docs_masked_words):
        assert len(docs_selected_word_indexes) == len(docs_masked_words)
        bsz_ = len(docs_selected_word_indexes)
        bsz = len(new_docs)
        max_nsent_ = max([len(doc) for doc in new_docs])
        max_nwords = max([max(map(len, doc)) for doc in new_docs])

        assert bsz == bsz_, 'should be the same size'

        max_n_masked_words = 0
        max_nsent = 0
        for doc_word_indexes in docs_selected_word_indexes:
            max_nsent = max(max_nsent, len(doc_word_indexes))
            for words in doc_word_indexes:
                max_n_masked_words = max(max_n_masked_words, len(words))

        assert max_nsent_ == max_nsent

        tgt_masked_words_indexes = torch.LongTensor(bsz, max_nsent, max_n_masked_words).fill_(0)
        tgt_masked_words = torch.LongTensor(bsz, max_nsent, max_n_masked_words).fill_(self.src_dict.pad())

        for i in range(bsz):
            doc_selected_indexes = docs_selected_word_indexes[i]
            doc_masked_words = docs_masked_words[i]
            doc_size = len(doc_selected_indexes)
            for j in range(doc_size):
                selected_indexes = doc_selected_indexes[j]
                masked_words = doc_masked_words[j]
                assert len(selected_indexes) == len(masked_words)
                masked_words_len = len(masked_words)

                sent_len = len(new_docs[i][j])
                offset = max_nwords - sent_len
                selected_indexes_offset = [si + offset for si in selected_indexes]

                tgt_masked_words_indexes[i, j, 0:masked_words_len] = torch.LongTensor(selected_indexes_offset)
                tgt_masked_words[i, j, 0:masked_words_len] = torch.LongTensor(masked_words)

        return tgt_masked_words_indexes, tgt_masked_words


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

        orig_min_predictions_per_seq = self.min_predictions_per_seq
        self.min_predictions_per_seq = self.max_predictions_per_seq
        batch = self.collater([
            {
                'id': i,
                'source': create_src(),
                'target': create_tgt(),
            }
            for i in range(bsz)
        ])
        self.min_predictions_per_doc = orig_min_predictions_per_doc
        self.min_predictions_per_seq = orig_min_predictions_per_seq

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
