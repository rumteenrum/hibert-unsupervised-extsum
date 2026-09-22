# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import data_utils, FairseqDataset

S_SEP = '<S_SEP>'
S_CLS = '<CLS>'
S_MASK = '<MASK>'

# split each doc into sentences
def get_docs(samples, vocab):
    sep_id = vocab.index(S_SEP)
    docs = []
    for sample in samples:
        source = sample['source']
        if source[-1] == vocab.eos():
            source[-1] = sep_id
        doc = []
        istart = 0
        for i in range(len(source)):
            if source[i] == sep_id:
                doc.append( source[istart:i] )
                istart = i + 1
        docs.append(doc)

    return docs

# truncate two sequences to make sure their sum <= max_train_seq_length
def truncate_sequences(tokens_a, tokens_b, max_seq_length, rng):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_seq_length:
            break
        cur_seq = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(cur_seq) > 0
        # truncate from both the front and the back to avoid biases
        if rng.rand() < 0.5:
            del cur_seq[0]
        else:
            cur_seq.pop()

def create_token_and_span_ids(tokens_a, tokens_b, vocab):
    cls_id = vocab.index(S_CLS)
    sep_id = vocab.index(S_SEP)

    span_ids = [0]
    tokens = [cls_id]
    for tok in tokens_a:
        tokens.append(tok)
        span_ids.append(0)
    tokens.append(sep_id)
    span_ids.append(0)

    for tok in tokens_b:
        tokens.append(tok)
        span_ids.append(1)
    tokens.append(sep_id)
    span_ids.append(1)

    return tokens, span_ids

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab, rng):
    cls_id = vocab.index(S_CLS)
    sep_id = vocab.index(S_SEP)
    mask_id = vocab.index(S_MASK)

    candi_indexes = []
    for i, tok in enumerate(tokens):
        if tok != sep_id and tok != cls_id:
            candi_indexes.append(i)
    rng.shuffle(candi_indexes)
    num_pred = min( max(1, int(len(candi_indexes) * masked_lm_prob)),
                    max_predictions_per_seq )

    assert len(candi_indexes[0:num_pred]) == len(set(candi_indexes[0:num_pred]))

    output_tokens = list(tokens)
    masked_lm = []
    sampled_indexes = candi_indexes[0:num_pred]
    for index in sampled_indexes:
        masked_token = None
        if rng.rand() < 0.8:
            masked_token = mask_id
        else:
            if rng.rand() < 0.5:
                masked_token = tokens[index]
            else:
                masked_token = rng.randint(0, len(vocab))

        output_tokens[index] = masked_token
        masked_lm.append( (index, tokens[index]) )

    masked_lm.sort(key=lambda x: x[0])
    positions = [m[0] for m in masked_lm]
    labels = [m[1] for m in masked_lm]

    assert len(positions) == len(labels)

    return output_tokens, positions, labels

# usually short_seq_prob == 0.1
def create_training_instances(doc_id, docs, max_seq_length, short_seq_prob,
                              masked_lm_prob, max_predictions_per_seq, vocab,
                              rng):
    instances = []
    target_seq_length = max_seq_length
    if rng.rand() < short_seq_prob:
        target_seq_length = rng.randint(2, max_seq_length+1)

    doc = docs[doc_id]
    current_span = []
    current_span_length = 0
    i = 0
    while i < len(doc):
        current_span.append(doc[i])
        current_span_length += len(doc[i])
        if i == len(doc)-1 or current_span_length >= target_seq_length:
            a_end = 1
            if len(current_span) >= 2:
                # basically a will not include the last sentence
                a_end = rng.randint(1, len(current_span))

            tokens_a = []
            for j in range(a_end):
                tokens_a.extend(current_span[j].tolist())

            tokens_b = []
            # Random next
            is_random_next = False
            if len(current_span) == 1 or rng.rand() < 0.5:
                is_random_next = True
                target_b_length = target_seq_length - len(tokens_a)

                for _ in range(10):
                    rnd_doc_id = rng.randint(0, len(docs))
                    if rnd_doc_id != doc_id:
                        break

                rnd_doc = docs[rnd_doc_id]
                rnd_doc_start = rng.randint(0, len(rnd_doc))

                for j in range(rnd_doc_start, len(rnd_doc)):
                    tokens_b.extend(rnd_doc[j].tolist())
                    if len(tokens_b) >= target_b_length:
                        break

                num_remain_sent_in_doc = len(current_span) - a_end
                i -= num_remain_sent_in_doc
            else:
                is_random_next = False
                for j in range(a_end, len(current_span)):
                    tokens_b.extend(current_span[j].tolist())

            truncate_sequences(tokens_a, tokens_b, max_seq_length, rng)

            tokens, span_ids = create_token_and_span_ids(tokens_a, tokens_b, vocab)
            output_tokens, positions, labels = create_masked_lm_predictions(
                                                    tokens, masked_lm_prob, max_predictions_per_seq, vocab, rng)
            instances.append( dict(tokens=output_tokens, span_ids=span_ids,
                                   masked_positions=positions, masked_labels=labels,
                                   is_next=1 if is_random_next else 0) )

            current_span = []
            current_span_length = 0
        i += 1

    return instances


def instances2batch(instances, vocab):
    batch_size = len(instances)
    max_src_len = max( len(ins['tokens']) for ins in instances )
    max_span_ids_len = max( len(ins['span_ids']) for ins in instances )
    assert max_src_len == max_span_ids_len
    max_masked_label_len = max( len(ins['masked_labels']) for ins in instances )

    tokens = torch.LongTensor(batch_size, max_src_len).fill_(vocab.pad())
    span_ids = torch.LongTensor(batch_size, max_src_len).fill_(0)

    masked_positions = torch.LongTensor(batch_size, max_masked_label_len).fill_(0)
    masked_labels = torch.LongTensor(batch_size, max_masked_label_len).fill_(vocab.pad())
    is_next = torch.LongTensor(batch_size).fill_(0)

    for i, ins in enumerate(instances):
        tok_length = len(ins['tokens'])
        assert len(ins['tokens']) == len(ins['span_ids'])
        tokens[i, 0:tok_length] = torch.LongTensor(ins['tokens'])
        span_ids[i, 0:tok_length] = torch.LongTensor(ins['span_ids'])
        labels_length = len(ins['masked_labels'])
        masked_positions[i, 0:labels_length] = torch.LongTensor(ins['masked_positions'])
        masked_labels[i, 0:labels_length] = torch.LongTensor(ins['masked_labels'])
        is_next[i] = ins['is_next']

    return dict(tokens=tokens, span_ids=span_ids, masked_positions=masked_positions,
                masked_labels=masked_labels, is_next=is_next)

def pack_sentence_spans(samples, vocab, max_train_seq_length, short_seq_prob,
                        masked_lm_prob, max_predictions_per_seq, rng, max_batch_size):
    docs = get_docs(samples, vocab)
    sep_id = vocab.index(S_SEP)
    cls_id = vocab.index(S_CLS)
    mask_id = vocab.index(S_MASK)
    valid_max_seq_length = max_train_seq_length - 3 # CLS <sent> S_SEP <sent> S_SEP

    instances = []
    for i in range(len(docs)):
        instances.extend( create_training_instances(
                                i, docs, valid_max_seq_length, short_seq_prob,
                                masked_lm_prob, max_predictions_per_seq, vocab, rng) )

    if len(instances) > max_batch_size:
        rnd_range = len(instances) - max_batch_size + 1
        rnd_start = rng.randint(0, rnd_range)
        instances = instances[rnd_start: rnd_start+max_batch_size]

    batch = instances2batch(instances, vocab)
    ntokens = sum( len(ins['masked_labels']) for ins in instances )
    batch['ntokens'] = ntokens

    return batch

def collate(samples, vocab, max_train_seq_length, short_seq_prob, masked_lm_prob,
            max_predictions_per_seq, rng, max_batch_size):
    if len(samples) == 0:
        return {}

    batch = pack_sentence_spans(samples, vocab, max_train_seq_length, short_seq_prob,
                                masked_lm_prob, max_predictions_per_seq, rng, max_batch_size)

    return {
        'id': torch.LongTensor([s['id'] for s in samples]),
        'ntokens': batch['ntokens'],
        'net_input': {
            'src_tokens': batch['tokens'],
            'span_ids': batch['span_ids'],
            'masked_positions': batch['masked_positions'],
        },
        'target': batch['masked_labels'],
        'next_sent_target': batch['is_next'],
    }


class BERTDataset(FairseqDataset):
    """A wrapper around torch.utils.data.Dataset for monolingual data."""
    def __init__(self, dataset, sizes, vocab, max_train_seq_length, short_seq_prob,
                 masked_lm_prob, max_predictions_per_seq, shuffle, rng=None,
                 max_batch_size=None):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = vocab
        self.max_train_seq_length = max_train_seq_length
        self.short_seq_prob = short_seq_prob
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.shuffle = shuffle
        self.rng = rng
        self.max_batch_size = max_batch_size

    def __getitem__(self, index):
        source = self.dataset[index]

        return {'id': index, 'source': source}

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return collate(samples, self.vocab, self.max_train_seq_length, self.short_seq_prob, self.masked_lm_prob,
                       self.max_predictions_per_seq, self.rng, self.max_batch_size)

    def get_dummy_batch(self, num_tokens, max_positions, tgt_len=128):
        assert isinstance(max_positions, float) or isinstance(max_positions, int)
        tgt_len = min(tgt_len, max_positions)
        bsz = num_tokens // tgt_len
        target = self.vocab.dummy_sentence(tgt_len + 1)
        source, target = target[:-1], target[1:]
        return self.collater([
            {'id': i, 'source': source, 'target': target}
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        source = self.dataset[index]
        return len(source)

    def ordered_indices(self):
        """Ordered indices for batching."""
        if self.shuffle:
            order = np.random.permutation(len(self))
            # for test only
            # order = np.arange(len(self))
        else:
            order = np.arange(len(self))
        return order

    def valid_size(self, index, max_positions):
        """Check if an example's size is valid according to max_positions."""
        assert isinstance(max_positions, float) or isinstance(max_positions, int)
        return self.sizes[index] <= max_positions
