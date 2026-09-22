# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('pretrain_doc_mlm_loss')
class PretrainDocMLMLoss(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.tgt_padding_idx = task.target_dictionary.pad()
        self.src_padding_idx = task.source_dictionary.pad()
        self.masked_sent_loss_weight = args.masked_sent_loss_weight
        self.sent_label_weight = args.sent_label_weight
        self.masked_word_loss_weight = args.masked_word_loss_weight

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--masked-sent-loss-weight', default=1.0, type=float, metavar='D',
                            help='weight for masked sentence predition')
        parser.add_argument('--masked-word-loss-weight', default=1.0, type=float, metavar='D',
                            help='weight for masked word predition')
        parser.add_argument('--sent-label-weight', default=0.0, type=float, metavar='D',
                            help='weight for sentence label')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        # this is for the masked sentence loss
        lprobs = model.get_normalized_probs(net_output, log_probs=True, idx=0)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        # this is for padding mask
        non_pad_mask = target.ne(self.src_padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]

        # add sentence label loss
        lbl_lprobs = model.get_normalized_probs(net_output, log_probs=True, idx=1)
        lbl_lprobs = lbl_lprobs.view(-1, lbl_lprobs.size(-1))
        # print('lbl_lprobs', lbl_lprobs.size(), lbl_lprobs)
        lbl_target = model.get_targets(sample, net_output, 'target_label').view(-1, 1)
        # print('lbl_target', lbl_target.size(), lbl_target)
        lbl_non_pad_mask = lbl_target.ne(self.tgt_padding_idx)
        # print('lbl_non_pad_mask', lbl_non_pad_mask.size(), lbl_non_pad_mask)
        lbl_nll_loss = -lbl_lprobs.gather(dim=-1, index=lbl_target)[lbl_non_pad_mask]

        # add masked word loss
        masked_word_lprobs = model.get_normalized_probs(net_output, log_probs=True, idx=2)
        masked_word_lprobs = masked_word_lprobs.view(-1, masked_word_lprobs.size(-1))
        masked_word_target = model.get_targets(sample, net_output, 'target_word').view(-1, 1)
        masked_word_non_pad_mask = masked_word_target.ne(self.src_padding_idx)
        masked_word_nll_loss = -masked_word_lprobs.gather(dim=-1, index=masked_word_target)[masked_word_non_pad_mask]
        masked_word_smooth_loss = -masked_word_lprobs.sum(dim=-1, keepdim=True)[masked_word_non_pad_mask]

        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
            lbl_nll_loss = lbl_nll_loss.sum()
            masked_word_nll_loss = masked_word_nll_loss.sum()
            masked_word_smooth_loss = masked_word_smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        masked_word_loss = (1. - self.eps) * masked_word_nll_loss + eps_i * masked_word_smooth_loss
        assert reduce

        # ntokens here are number of masked sentence tokens
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        loss = loss * self.masked_sent_loss_weight * sample_size / sample['ntokens'] \
                + lbl_nll_loss * self.sent_label_weight * sample_size / sample['ntokens_label'] \
                + masked_word_loss * self.masked_word_loss_weight * sample_size / sample['ntokens_word']

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'lbl_nll_loss': utils.item(lbl_nll_loss.data) if reduce else lbl_nll_loss.data,
            'masked_word_loss': utils.item(masked_word_loss.data) if reduce else masked_word_loss.data,
            'masked_word_nll_loss': utils.item(masked_word_nll_loss.data) if reduce else masked_word_nll_loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'sample_size': sample_size,
        }
