
import math

import torch

from fairseq import utils
from fairseq.models import FairseqIncrementalDecoder


class PointerGenerator(object):
    def __init__(
        self, models, src_dict, tgt_dict, beam_size=1, minlen=1, maxlen=None, stop_early=True,
        normalize_scores=True, len_penalty=1, unk_penalty=0, retain_dropout=False,
        sampling=False, sampling_topk=-1, sampling_temperature=1,
        dataset=None,
    ):
        """Generates translations of a given source sentence.
        Args:
            min/maxlen: The length of the generated output will be bounded by
                minlen and maxlen (not including the end-of-sentence marker).
            stop_early: Stop generation immediately after we finalize beam_size
                hypotheses, even though longer hypotheses might have better
                normalized scores.
            normalize_scores: Normalize scores by the length of the output.
        """
        self.models = models
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        self.minlen = minlen
        max_decoder_len = min(m.max_decoder_positions() for m in self.models)
        max_decoder_len -= 1  # we define maxlen not including the EOS marker
        self.maxlen = max_decoder_len if maxlen is None else min(maxlen, max_decoder_len)
        self.stop_early = stop_early
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.sampling = sampling
        self.sampling_topk = sampling_topk
        self.sampling_temperature = sampling_temperature
        self.dataset = dataset

        self.eos_mark = torch.ByteTensor()

    def cuda(self):
        for model in self.models:
            model.cuda()
        return self

    def get_original_text(self, sample, eos):
        orig_text = []
        for sample_id in sample['id'].data:
            orig = self.dataset.src.get_original_text(sample_id)
            words = orig.strip().split() + [eos]
            orig_text.append(words)
            # print( words )

        return orig_text

    def generate_batched_itr(
        self, data_itr, beam_size=None, maxlen_a=0.0, maxlen_b=None,
        cuda=False, timer=None, prefix_size=0,
    ):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda: use GPU for generation
            timer: StopwatchMeter for timing generations.
        """
        if maxlen_b is None:
            maxlen_b = self.maxlen

        # Note training examples are sorted
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if 'net_input' not in s:
                continue
            input = s['net_input']
            srclen = input['src_tokens'].size(1)
            if timer is not None:
                timer.start()

            # src_str = self.dataset.src.get_original_text(sample_id)

            # print('prefix size', prefix_size)
            # print('beam_size size', beam_size)
            with torch.no_grad():
                hypos = self.generate(
                    input['src_tokens'],
                    input['src_lengths'],
                    src_text=self.get_original_text(s, self.src_dict.eos_word) if self.dataset is not None else None,
                    beam_size=beam_size,
                    maxlen=int(maxlen_a*srclen + maxlen_b),
                    prefix_tokens=s['target'][:, :prefix_size] if prefix_size > 0 else None,
                )
            if timer is not None:
                timer.stop(sum(len(h[0]['tokens']) for h in hypos))
            for i, id in enumerate(s['id'].data):
                # remove padding
                src = utils.strip_pad(input['src_tokens'].data[i, :], self.pad)
                # ref = utils.strip_pad(s['target'].data[i, :], self.pad) if s['target'] is not None else None
                # note that this is a pointer net, and therefore it is different
                # ptr = utils.strip_pad(s['target'].data[i, :], self.pad) if s['target'] is not None else None
                # ptr = ' '.join( str(p.item()) for p in s['target'].data[i, :] )
                ptr = s['target'].data[i, :]
                pos = ptr.size(0) - 1
                while ptr[pos] == self.pad:
                    pos -= 1
                ptr = ptr[0:pos+1]
                ref = utils.strip_pad(s['target2'].data[i, :], self.pad) if s['target2'] is not None else None
                yield id, src, ref, ptr, hypos[i]

    def generate(self, src_tokens, src_lengths, src_text=None, beam_size=None, maxlen=None, prefix_tokens=None):
        """Generate a batch of translations."""
        with torch.no_grad():
            if beam_size == 1 or self.beam_size == 1:
                return self.greedy_search(src_tokens, src_lengths, src_text, maxlen, prefix_tokens)
            else:
                return self._generate(src_tokens, src_lengths, src_text, beam_size, maxlen, prefix_tokens)

    def greedy_search(self, src_tokens, src_lengths, src_text=None, maxlen=None, prefix_tokens=None):
        encoder_outs = []
        incremental_states = {}
        for model in self.models:
            if not self.retain_dropout:
                model.eval()
            if isinstance(model.decoder, FairseqIncrementalDecoder):
                incremental_states[model] = {}
            else:
                incremental_states[model] = None

            encoder_out = model.encoder(src_tokens, src_lengths)
            encoder_outs.append(encoder_out)

        bsz, srclen = src_tokens.size()

        tokens = src_tokens.data.new(bsz, maxlen + 2).fill_(self.pad)
        ptr_src_tokens = src_tokens.data.new(bsz, maxlen + 2).fill_(self.pad)
        pointers = src_tokens.data.new(bsz, maxlen + 2).fill_(-1)
        pointer_probs = encoder_outs[0]['encoder_out'].data.new(bsz, maxlen + 2).fill_(0)

        self.eos_mark.resize_(bsz).fill_(0)
        if src_tokens.data.is_cuda and (not self.eos_mark.is_cuda):
            self.eos_mark = self.eos_mark.cuda()

        tokens_buf = tokens.clone()
        tokens[:, 0] = self.eos

        step = 0
        # for step in range(maxlen + 1):
        attentions = [None]
        while step < maxlen + 1:
            probs, avg_attn_scores = self._decode(
                tokens[:, :step + 1], encoder_outs, incremental_states)
            # print('probs ', probs)
            # print('avg attn', avg_attn_scores)
            log_ptr_probs, selected_src_idxs = probs.max(1)
            pointers[:, step+1] = selected_src_idxs
            pointer_probs[:, step+1] = log_ptr_probs
            attentions.append( avg_attn_scores )
            selected_src_tokens = src_tokens.gather(1, selected_src_idxs.view(-1, 1))
            ptr_src_tokens[:, step+1] = selected_src_tokens.view(-1)
            for i in range( src_tokens.size(0) ):
                select_idx = selected_src_idxs[i].item()
                if (src_text is not None) and select_idx < len(src_text[i]):
                    src_word = src_text[i][select_idx]
                    tokens[i, step + 1] = self.tgt_dict.index( src_word )
                    # print('haha')
                else:
                    sel_tok = selected_src_tokens[i, 0].item()
                    tokens[i, step + 1] = self.tgt_dict.index( self.src_dict[ sel_tok ] )

            self.eos_mark = self.eos_mark | selected_src_tokens.view(-1).eq( self.src_dict.eos() )
            step += 1
            # print('step = ', step, 'maxlen = ', maxlen)
            if self.eos_mark.sum() == bsz:
                break
        # print( self.tgt_dict.string( tokens[:, 0:step+1] ) )

        hypos = []
        for i in range(bsz):
            length = min( src_lengths[i], tokens.size(1) )
            j = 1 # remove EOS
            score = 0.0
            attn = []
            p_scores = []
            # 'attention': hypo_attn,  # src_len x tgt_len
            while j < length:
                score += pointer_probs[i, j]
                p_scores.append( pointer_probs[i, j].item() )
                attn.append( attentions[j][i][0:src_lengths[i]].view(1, -1) )
                # print(attn[-1].size())
                if tokens[i, j] == self.tgt_dict.eos():
                    j += 1
                    break
                j += 1

            attention = torch.cat( attn, 0 ).t()
            _, alignment = attention.max(dim=0)
            # 'positional_scores': pos_scores[i],
            # print( p_scores )

            hypo = {
                'tokens': tokens[i, 1:j],
                'score': score,
                'pointers': pointers[i, 1:j],
                'attention': attention,
                'alignment': alignment,
                'positional_scores': pointer_probs.data.new( p_scores ),
            }
            # print( 'before hypo in ***** ', self.tgt_dict.string( hypo['tokens'] ) )
            hypos.append( [hypo] )

        return hypos


    def _generate(self, src_tokens, src_lengths, src_text=None, beam_size=None, maxlen=None, prefix_tokens=None):
        # in PtrNet, it is left padding
        bsz, srclen = src_tokens.size()
        maxlen = min(maxlen, self.maxlen) if maxlen is not None else self.maxlen

        # the max beam size is the dictionary size - 1, since we never select pad
        beam_size = beam_size if beam_size is not None else self.beam_size
        # beam_size = min(beam_size, self.vocab_size - 1)
        # note that Pointer Network is different
        beam_size = min(beam_size, src_tokens.size(1))

        encoder_outs = []
        incremental_states = {}
        for model in self.models:
            if not self.retain_dropout:
                model.eval()
            if isinstance(model.decoder, FairseqIncrementalDecoder):
                incremental_states[model] = {}
            else:
                incremental_states[model] = None

            # compute the encoder output for each beam
            encoder_out = model.encoder(
                src_tokens.repeat(1, beam_size).view(-1, srclen),
                src_lengths.expand(beam_size, src_lengths.numel()).t().contiguous().view(-1),
            )
            encoder_outs.append(encoder_out)

        # initialize buffers
        scores = src_tokens.data.new(bsz * beam_size, maxlen + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.data.new(bsz * beam_size, maxlen + 2).fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.eos
        attn = scores.new(bsz * beam_size, src_tokens.size(1), maxlen + 2)
        attn_buf = attn.clone()

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        worst_finalized = [{'idx': None, 'score': -math.inf} for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfinalized_scores=None):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                if self.stop_early or step == maxlen or unfinalized_scores is None:
                    return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                best_unfinalized_score = unfinalized_scores[sent].max()
                if self.normalize_scores:
                    best_unfinalized_score /= maxlen ** self.len_penalty
                if worst_finalized[sent]['score'] >= best_unfinalized_score:
                    return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores, unfinalized_scores=None):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.
            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.
            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
                unfinalized_scores: A vector containing scores for all
                    unfinalized hypotheses
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            tokens_clone[:, step] = self.eos
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step+2]

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step+1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                def get_hypo():

                    # remove padding tokens from attn scores
                    nonpad_idxs = src_tokens[sent].ne(self.pad)
                    hypo_attn = attn_clone[i][nonpad_idxs]
                    _, alignment = hypo_attn.max(dim=0)
                    # removing the <eos> position just do not help very much
                    # _, alignment = hypo_attn[0:-1, :].max(dim=0)

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': alignment,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())
                elif not self.stop_early and score > worst_finalized[sent]['score']:
                    # replace worst hypo for this sentence with new/better one
                    worst_idx = worst_finalized[sent]['idx']
                    if worst_idx is not None:
                        finalized[sent][worst_idx] = get_hypo()

                    # find new worst finalized hypo for this sentence
                    idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]['score'])
                    worst_finalized[sent] = {
                        'score': s['score'],
                        'idx': idx,
                    }

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfinalized_scores):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished

        reorder_state = None
        batch_idxs = None
        for step in range(maxlen + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                for i, model in enumerate(self.models):
                    if isinstance(model.decoder, FairseqIncrementalDecoder):
                        model.decoder.reorder_incremental_state(incremental_states[model], reorder_state)
                    encoder_outs[i] = model.encoder.reorder_encoder_out(encoder_outs[i], reorder_state)

            probs, avg_attn_scores = self._decode(
                tokens[:, :step + 1], encoder_outs, incremental_states)
            if step == 0:
                # at the first step all hypotheses are equally likely, so use
                # only the first beam
                probs = probs.unfold(0, 1, beam_size).squeeze(2).contiguous()
                scores = scores.type_as(probs)
                scores_buf = scores_buf.type_as(probs)
            elif not self.sampling:
                # make probs contain cumulative scores for each hypothesis
                probs.add_(scores[:, step - 1].view(-1, 1))

            probs[:, self.pad] = -math.inf  # never select pad
            probs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # Record attention scores
            attn[:, :, step + 1].copy_(avg_attn_scores)

            cand_scores = buffer('cand_scores', type_of=scores)
            cand_indices = buffer('cand_indices')
            cand_beams = buffer('cand_beams')
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)
            if step < maxlen:
                if prefix_tokens is not None and step < prefix_tokens.size(1):
                    probs_slice = probs.view(bsz, -1, probs.size(-1))[:, 0, :]
                    cand_scores = torch.gather(
                        probs_slice, dim=1,
                        index=prefix_tokens[:, step].view(-1, 1).data
                    ).expand(-1, cand_size)
                    cand_indices = prefix_tokens[:, step].view(-1, 1).expand(bsz, cand_size).data
                    cand_beams.resize_as_(cand_indices).fill_(0)
                elif self.sampling:
                    assert self.pad == 1, 'sampling assumes the first two symbols can be ignored'

                    if self.sampling_topk > 0:
                        values, indices = probs[:, 2:].topk(self.sampling_topk)
                        exp_probs = values.div_(self.sampling_temperature).exp()
                        if step == 0:
                            torch.multinomial(exp_probs, beam_size, replacement=True, out=cand_indices)
                        else:
                            torch.multinomial(exp_probs, 1, replacement=True, out=cand_indices)
                        torch.gather(exp_probs, dim=1, index=cand_indices, out=cand_scores)
                        torch.gather(indices, dim=1, index=cand_indices, out=cand_indices)
                        cand_indices.add_(2)
                    else:
                        exp_probs = probs.div_(self.sampling_temperature).exp_().view(-1, self.vocab_size)

                        if step == 0:
                            # we exclude the first two vocab items, one of which is pad
                            torch.multinomial(exp_probs[:, 2:], beam_size, replacement=True, out=cand_indices)
                        else:
                            torch.multinomial(exp_probs[:, 2:], 1, replacement=True, out=cand_indices)

                        cand_indices.add_(2)
                        torch.gather(exp_probs, dim=1, index=cand_indices, out=cand_scores)

                    cand_scores.log_()
                    cand_indices = cand_indices.view(bsz, -1).repeat(1, 2)
                    cand_scores = cand_scores.view(bsz, -1).repeat(1, 2)
                    if step == 0:
                        cand_beams = torch.zeros(bsz, cand_size).type_as(cand_indices)
                    else:
                        cand_beams = torch.arange(0, beam_size).repeat(bsz, 2).type_as(cand_indices)
                        # make scores cumulative
                        cand_scores.add_(
                            torch.gather(
                                scores[:, step - 1].view(bsz, beam_size), dim=1,
                                index=cand_beams,
                            )
                        )
                else:
                    # take the best 2 x beam_size predictions. We'll choose the first
                    # beam_size of these which don't predict eos to continue with.
                    torch.topk(
                        probs.view(bsz, -1),
                        k=min(cand_size, probs.view(bsz, -1).size(1) - 1),  # -1 so we never select pad
                        out=(cand_scores, cand_indices),
                    )
                    torch.div(cand_indices, self.vocab_size, out=cand_beams)
                    cand_indices.fmod_(self.vocab_size)
            else:
                # finalize all active hypotheses once we hit maxlen
                # pick the hypothesis with the highest prob of EOS right now
                torch.sort(
                    probs[:, self.eos],
                    descending=True,
                    out=(eos_scores, eos_bbsz_idx),
                )
                num_remaining_sent -= len(finalize_hypos(
                    step, eos_bbsz_idx, eos_scores))
                assert num_remaining_sent == 0
                break

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos)

            finalized_sents = set()
            if step >= self.minlen:
                # only consider eos when it's among the top beam_size indices
                torch.masked_select(
                    cand_bbsz_idx[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_bbsz_idx,
                )
                if eos_bbsz_idx.numel() > 0:
                    torch.masked_select(
                        cand_scores[:, :beam_size],
                        mask=eos_mask[:, :beam_size],
                        out=eos_scores,
                    )
                    finalized_sents = finalize_hypos(
                        step, eos_bbsz_idx, eos_scores, cand_scores)
                    num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < maxlen

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(bsz).type_as(cand_indices)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)

                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                attn_buf.resize_as_(attn)
                bsz = new_bsz
            else:
                batch_idxs = None

            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
            active_mask = buffer('active_mask')
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, _ignore = buffer('active_hypos'), buffer('_ignore')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(_ignore, active_hypos)
            )
            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 1],
            )
            torch.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            torch.index_select(
                attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                out=attn_buf[:, :, :step + 2],
            )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

        return finalized

    def _decode(self, tokens, encoder_outs, incremental_states):
        if len(self.models) == 1:
            return self._decode_one(tokens, self.models[0], encoder_outs[0], incremental_states, log_probs=True)

        avg_probs = None
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn = self._decode_one(tokens, model, encoder_out, incremental_states, log_probs=False)
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs.div_(len(self.models))
        avg_probs.log_()
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn

    '''
    def _decode_one(self, tokens, model, encoder_out, incremental_states, log_probs):
        with torch.no_grad():
            if incremental_states[model] is not None:
                decoder_out = list(model.decoder(tokens, encoder_out, incremental_states[model]))
            else:
                decoder_out = list(model.decoder(tokens, encoder_out))
            decoder_out[0] = decoder_out[0][:, -1, :]
            attn = decoder_out[1]
            if attn is not None:
                attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        return probs, attn
    '''

    def _decode_one(self, tokens, model, encoder_out, incremental_states, log_probs):
        with torch.no_grad():
            if incremental_states[model] is not None:
                decoder_out = list(model.decoder(tokens, encoder_out, incremental_states[model]))
            else:
                decoder_out = list(model.decoder(tokens, encoder_out))

            decoder_out[1] = decoder_out[1][:, -1, :]
            attn = model.get_normalized_probs(decoder_out, log_probs=False)
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        return probs, attn
