# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('cross_entropy')
class CrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        lprobs_full = model.get_normalized_probs(net_output, log_probs=True)
        # lprobs_full is typically [batch_size, output_seq_len, num_classes]
        # e.g., [10, 1, 3] for this model

        targets_full = model.get_targets(sample, net_output)
        # targets_full is typically [batch_size, target_seq_len]
        # e.g., [10, 58] for this data

        # Handle cases where model output sequence length is 1 (e.g., [B, 1, C])
        # but target sequence length is > 1 (e.g., [B, S_target_len])
        # We'll use the first element of the target sequence in this case.
        if lprobs_full.dim() == 3 and lprobs_full.size(1) == 1 and \
           targets_full.dim() == 2 and lprobs_full.size(0) == targets_full.size(0) and \
           targets_full.size(1) > 1:
            lprobs = lprobs_full.squeeze(1)  # Shape: [batch_size, num_classes], e.g., [10, 3]
            target = targets_full[:, 0]     # Shape: [batch_size], e.g., [10]
        else:
            # Default behavior: flatten predictions and targets
            lprobs = lprobs_full.view(-1, lprobs_full.size(-1))
            target = targets_full.view(-1)

        loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx,
                          reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
