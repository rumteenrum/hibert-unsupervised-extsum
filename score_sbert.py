#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import collections
import itertools
import os, sys
import math
import torch
import numpy
from tqdm import tqdm
import tempfile
import pickle
import shutil
import numpy as np  # Add this
import heapq        # Add this
from collections import Counter  # Add this - fixes the Counter issue
import glob  # Add this

from fairseq import data, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data.pretrain_doc_dataset import get_docs, create_target_batch #, docs2tensor
from fairseq.fp16_trainer import FP16Trainer
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq.data.pretrain_doc_dataset import SENT_SEP




def main(args):
    # we should not do this!
    '''
    if args.max_tokens is None:
        args.max_tokens = 6000
    '''
    utils.xpprint(args)


    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)


    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)


    utils.xprintln('setup task done!')


    # Load dataset splits
    load_dataset_splits(args, task, ['train'])
    load_dataset_splits(args, task, ['valid', 'test'], shuffle=False)
    utils.xprintln('load dataset done!')


    if args.task == 'extractive_summarization':
        if distributed_utils.is_master(args):
            from sum_eval import MultiProcSumEval
            sum_eval_pool = MultiProcSumEval(args.ncpu_eval)
            sum_valid_pool_params = dict(article_file=args.raw_valid + '.article',
                                          summary_file=args.raw_valid + '.summary',
                                          entity_map_file=args.raw_valid + '.entity_map',
                                          length=-1, eval_type='predict',
                                          topk=args.topk_sent_eval, rerank=False, with_m=False,
                                          cmd='-a -c 95 -m -n 4 -w 1.2',
                                          trigram_block=args.trigram_block,)


            sum_test_pool_params = dict(article_file=args.raw_test + '.article',
                                          summary_file=args.raw_test + '.summary',
                                          entity_map_file=args.raw_test + '.entity_map',
                                          length=-1, eval_type='predict',
                                          topk=args.topk_sent_eval, rerank=False, with_m=False,
                                          cmd='-a -c 95 -m -n 4 -w 1.2',
                                          trigram_block=args.trigram_block,)
            sum_pool_params = dict(valid=sum_valid_pool_params, test=sum_test_pool_params)


            def make_params(default_dict, result_file, out_rouge_file, rerank=False, with_m=False):
                para_dict = dict(default_dict)
                para_dict['result_file'] = result_file
                para_dict['out_rouge_file'] = out_rouge_file
                para_dict['rerank'] = rerank
                para_dict['with_m'] = with_m
                return para_dict


    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))
    print(model)
    import sys
    sys.stdout.flush()


    # if summarization try to load pretrained model
    if args.task == 'extractive_summarization' or args.task == 'pretrain_document_modeling':
        # assume this is a single GPU program
        if args.init_from_pretrained_doc_model:
            task.load_pretrained_model(model, args.pretrained_doc_model_path)




    # Build trainer
    if args.fp16:
        trainer = FP16Trainer(args, task, model, criterion)
    else:
        if torch.cuda.get_device_capability(0)[0] >= 7:
            print('| NOTICE: your device may support faster training with --fp16')
        trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))


    # Initialize dataloader
    max_positions = trainer.get_model().max_positions()
    epoch_itr = data.EpochBatchIterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=1,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
    )


    # Load the latest checkpoint if one is available
    load_checkpoint(args, trainer, epoch_itr)
    # make sure training from a different checkpoint will use different random seed
    cur_dataset = task.dataset('train')
    if hasattr(cur_dataset, 'rng'):
        print('epoch ', epoch_itr.epoch)
        cur_dataset.rng = numpy.random.RandomState(args.seed+epoch_itr.epoch)


    if hasattr(task, 'run_dummy_batch') and task.run_dummy_batch:
        print('** run dummy batch **')
        # Send a dummy batch to warm the caching allocator
        dummy_batch = task.dataset('train').get_dummy_batch(args.max_tokens if args.max_tokens else args.max_sentences, max_positions)
        trainer.dummy_train_step(dummy_batch)


    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr)


        if epoch_itr.epoch % args.validate_interval == 0:
            if args.task == 'extractive_summarization':
                if distributed_utils.is_master(args):
                    validate_metric(args, trainer, task, epoch_itr, valid_subsets)
                    for subset in valid_subsets:
                        valid_result_file = os.path.join(args.save_dir, '{}.{}.txt'.format(epoch_itr.epoch, subset))
                        valid_out_file = os.path.join(args.save_dir, '{}.{}'.format(epoch_itr.epoch, subset))
                        sum_eval_pool.add_eval_job(**make_params(sum_pool_params[subset], valid_result_file, valid_out_file, False, False))
                        sum_eval_pool.add_eval_job(**make_params(sum_pool_params[subset], valid_result_file, valid_out_file, True, False))
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)


        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])


        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


    if args.task == 'extractive_summarization':
        if distributed_utils.is_master(args):
            sum_eval_pool.join()
            from sum_eval import summarize_rouge
            summarize_rouge(args.save_dir)




def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""


    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr()
    progress = progress_bar.build_progress_bar(args, itr, epoch_itr.epoch, no_progress_bar='simple')


    # update parameters every N batches
    if epoch_itr.epoch <= len(args.update_freq):
        update_freq = args.update_freq[epoch_itr.epoch - 1]
    else:
        update_freq = args.update_freq[-1]


    extra_meters = collections.defaultdict(lambda: AverageMeter())
    first_valid = args.valid_subset.split(',')[0]
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    num_batches = len(epoch_itr)
    for i, sample in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        if i < num_batches - 1 and (i + 1) % update_freq > 0:
            # buffer updates according to --update-freq
            trainer.train_step(sample, update_params=False)
            continue
        else:
            log_output = trainer.train_step(sample, update_params=True)


        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats)


        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()


        num_updates = trainer.get_num_updates()
        if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, [first_valid])
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])


            if args.task == 'extractive_summarization':
                if distributed_utils.is_master(args):
                    validate_metric(args, trainer, task, epoch_itr, valid_subsets)


        if num_updates >= max_update:
            break


        # this is for test
        '''
        if num_updates >= 15:
            break
        '''


    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)


    # reset training meters
    for k in ['train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'clip']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_lprobs(args, trainer, task, epoch_itr, cache_dir=None):
    raw_dataset = task.dataset(args.train_subset)
    print(f"Raw dataset size: {len(raw_dataset)}")
    
    # Set model to eval mode once
    trainer.model.eval()
    
    # Use persistent cache directory
    cache_dir = cache_dir or get_cache_dir(args)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")
    
    # Initialize doc cache mapping (reuse existing files if present)
    doc_cache_files = discover_cached_docs(cache_dir)
    
    # Get the dataset
    dataset = task.dataset(args.train_subset)
    
    # Process all doc ids
    doc_ids = list(range(len(raw_dataset)))
    print("Processing all samples (skipping those already cached)...")
    for doc_id in tqdm(doc_ids):
        doc_cache_file = os.path.join(cache_dir, f"doc_{doc_id}.pkl")
        if os.path.isfile(doc_cache_file):
            # Already cached, skip heavy compute
            doc_cache_files[doc_id] = doc_cache_file
            continue

        doc_data = {
            'sent_scores': [],
            'targets': [],
            'attentions': [],
            'sent_lengths': [],
            'correct_preds': 0,
            'total_preds': 0,
        }
        
        # Get the sample
        sample = raw_dataset[doc_id]
        sample['id'] = torch.LongTensor([doc_id])
        
        # Get document representation
        samples = [sample]
        docs = get_docs(samples, dataset.src_dict, maxlen=dataset.max_sent_len)
        
        # Process each sentence in the document
        num_sents = len(docs[0])
        
        # Process in smaller batches if document is large
        batch_size = 5
        for batch_start in range(0, num_sents, batch_size):
            batch_end = min(batch_start + batch_size, num_sents)
            
            # Process each sentence in this mini-batch
            for sent_idx in range(batch_start, batch_end):
                # Create masked version of the document
                new_doc, selected_indexes, masked_sents = dataset.mask_sentence_at_index(docs[0], sent_idx)
                
                # Prepare input for the model
                src_tokens, doc_pad_mask = dataset.doc2tensor([new_doc], dataset.src_dict)
                tgt_selected_indexes, tgt_input_masked_sents, tgt_masked_sents = dataset.masked_sents2tensor(
                    [selected_indexes], [masked_sents]
                )
                
                doc_pos_tok = torch.LongTensor(doc_pad_mask.size()).copy_(src_tokens[:, :, -1])
                doc_pad_mask = doc_pos_tok.new_zeros(doc_pos_tok.size()).bool()
                doc_pos_tok[doc_pad_mask] = dataset.src_dict.pad()
                
                # Create input sample
                current_sample = {
                    'net_input': {
                        'src_tokens': src_tokens,
                        'doc_pad_mask': doc_pad_mask,
                        'doc_pos_tok': doc_pos_tok,
                        'masked_sent_positions': tgt_selected_indexes,
                        'prev_output_tokens': tgt_input_masked_sents,
                    },
                    'target': tgt_masked_sents,
                    'id': torch.LongTensor([doc_id]),
                }
                
                # Move to GPU and get model output
                with torch.no_grad():
                    current_sample = utils.move_to_cuda(current_sample)
                    model_output = trainer.model(**current_sample['net_input'])
                    decoder_out, attention = model_output[0], model_output[1]
                    lprobs = trainer.model.get_normalized_probs(decoder_out, log_probs=True)

                # Directly calculate sentence score and accuracy to save space
                sent_probs = lprobs.squeeze(0)
                p_word = [float(sent_probs[i].max()) for i in range(sent_probs.size(0))]
                sent_score = np.mean(p_word) if p_word else 0.0
                doc_data['sent_scores'].append(sent_score)
                
                sent_length = lprobs.shape[1]
                doc_data['sent_lengths'].append(sent_length)

                target = current_sample['target']
                pred = lprobs.argmax(dim=-1)
                mask = target.ne(task.target_dictionary.pad())
                doc_data['correct_preds'] += (pred == target)[mask].sum().item()
                doc_data['total_preds'] += mask.sum().item()

                # Convert to numpy and store in document cache
                targets_np = current_sample['target'].detach().cpu().numpy()
                
                if isinstance(attention, torch.Tensor):
                    attention_np = attention.detach().cpu().numpy()
                else:
                    attention_np = [att.detach().cpu().numpy() for att in attention]

                doc_data['targets'].append(targets_np)
                doc_data['attentions'].append(attention_np)
            
            # Clear GPU cache after each batch
            torch.cuda.empty_cache()
        
        # Save document data to disk and store file path
        with open(doc_cache_file, 'wb') as f:
            pickle.dump(doc_data, f)
        doc_cache_files[doc_id] = doc_cache_file
        
        # Clear the doc_data from memory
        del doc_data
    
    return doc_cache_files, cache_dir


def get_cache_dir(args):
    """Stable cache dir inside save_dir per subset."""
    base = os.path.join(args.save_dir, "lprob_cache", args.train_subset)
    os.makedirs(base, exist_ok=True)
    return base


def discover_cached_docs(cache_dir):
    """Return {doc_id:int -> cache_file:str} for all cached docs in cache_dir."""
    mapping = {}
    for f in glob.glob(os.path.join(cache_dir, "doc_*.pkl")):
        try:
            fname = os.path.basename(f)
            doc_id = int(fname.split('_')[1].split('.')[0])
            mapping[doc_id] = f
        except Exception:
            continue
    return mapping


def load_doc_from_cache(cache_file):
    """Load a document's data from cache file"""
    with open(cache_file, 'rb') as f:
        return pickle.load(f)

def cleanup_cache(temp_dir):
    """Clean up temporary cache directory"""
    shutil.rmtree(temp_dir)
    print(f"Cleaned up cache directory: {temp_dir}")


def is_cache_complete(args, task):
    """Return (doc_cache_files:dict, cache_dir:str, complete:bool)."""
    cache_dir = get_cache_dir(args)
    doc_cache_files = discover_cached_docs(cache_dir)
    dataset_len = len(task.dataset(args.train_subset))
    complete = all(i in doc_cache_files for i in range(dataset_len))
    return doc_cache_files, cache_dir, complete


def main2(args):
    """Main training function that focuses on finetuning and then getting log probabilities."""
    utils.xpprint(args)

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Setup task
    task = tasks.setup_task(args)
    utils.xprintln('setup task done!')

    # Load dataset splits
    load_dataset_splits(args, task, ['train'], shuffle=False)
    load_dataset_splits(args, task, ['valid', 'test'], shuffle=False)

    # here
    # look here
    # look here for changing the subset
    args.train_subset = 'test'
    utils.xprintln('load dataset done!')

    # Dataset and RNG (needed for PMI and for consistent masking if we run the model)
    dataset = task.dataset(args.train_subset)
    if hasattr(dataset, 'rng'):
        dataset.rng = numpy.random.RandomState(args.seed)

    # Check persistent cache BEFORE building model
    doc_cache_files, cache_dir, cache_complete = is_cache_complete(args, task)
    if cache_complete:
        print(f"| Found complete cache for {args.train_subset} at: {cache_dir}")
        print("| Skipping model and using cached files only.")
    else:
        print("| Cache incomplete. Running model to compute missing docs...")

        # Build model and criterion (only if needed)
        model = task.build_model(args)
        criterion = task.build_criterion(args)
        print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
        print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))

        # Build trainer
        trainer = Trainer(args, task, model, criterion)
        print('| training on {} GPUs'.format(args.distributed_world_size))

        # Initialize dataloader
        max_positions = trainer.get_model().max_positions()
        epoch_itr = data.EpochBatchIterator(
            dataset=task.dataset(args.train_subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=max_positions,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
        )

        # Load checkpoint if available
        load_checkpoint(args, trainer, epoch_itr)

        # Re-seed masking
        dataset = task.dataset(args.train_subset)
        if hasattr(dataset, 'rng'):
            dataset.rng = numpy.random.RandomState(args.seed)

        print('| Getting log probabilities (filling persistent cache)...')
        doc_cache_files, cache_dir = get_lprobs(args, trainer, task, epoch_itr, cache_dir=cache_dir)

    L = 20  # Minimum sentence length to consider for ranking
    k = 3   # Number of sentences to select

    print("| Processing cached results...")

    r_doc = []
    r_doc_attn = []
    all_targets_for_pmi = []
    short_sent_indices = []

    # Use sorted doc ids to read cache
    cached_ids = sorted(doc_cache_files.keys())

    # First pass: collect data for ranking calculations
    for doc_id in cached_ids:
        doc_data = load_doc_from_cache(doc_cache_files[doc_id])

        r_doc.append(doc_data['sent_scores'])

        doc_short_indices = [i for i, length in enumerate(doc_data['sent_lengths']) if length < L]
        short_sent_indices.append(doc_short_indices)

        all_targets_for_pmi.append(doc_data['targets'])
        r_doc_attn.append([0] * len(doc_data['sent_scores']))

    # Define normalization function
    def normalize_scores(scores_list):
        normalized_list = []
        for doc_scores in scores_list:
            if len(doc_scores) > 1 and any(s != 0 for s in doc_scores):
                scores_array = np.array(doc_scores)
                min_score = np.min(scores_array)
                max_score = np.max(scores_array)
                if max_score > min_score:
                    normalized = (scores_array - min_score) / (max_score - min_score)
                else:
                    normalized = np.ones_like(scores_array) * 0.5
                normalized_list.append(normalized.tolist())
            else:
                normalized_list.append([0.5] * len(doc_scores))
        return normalized_list

    r_doc_norm = normalize_scores(r_doc)

    # Second pass: calculate attention scores properly
    for idx, doc_id in enumerate(cached_ids):
        doc_data = load_doc_from_cache(doc_cache_files[doc_id])
        r_sent_attn = []
        for sent_index in range(len(doc_data['attentions'])):
            attention_matrix = doc_data['attentions'][sent_index]
            if attention_matrix is not None:
                attention_tensor = torch.from_numpy(attention_matrix) if isinstance(attention_matrix, np.ndarray) else attention_matrix
                attention_avg = attention_tensor.mean(dim=0) if len(attention_tensor.shape) == 3 else attention_tensor
                attention_avg[sent_index][sent_index] = 0
                sent_attention_score = sum(attention_avg[sent_index] * r_doc_norm[idx][sent_index])
                r_sent_attn.append(float(sent_attention_score))
            else:
                r_sent_attn.append(0.0)
        r_doc_attn[idx] = r_sent_attn

    r_doc_attn_norm = normalize_scores(r_doc_attn)

    r_doc_pmi_relevance, r_doc_pmi_redundancy = calculate_pmi_and_redundancy(all_targets_for_pmi, dataset)

    rel_stats = [pmi for doc_pmis in r_doc_pmi_relevance for pmi in doc_pmis if pmi != 0]
    red_stats = [pmi for doc_pmis in r_doc_pmi_redundancy for pmi in doc_pmis if pmi != 0]
    if rel_stats:
        print(f"Relevance PMI stats: min={min(rel_stats):.4f}, max={max(rel_stats):.4f}, avg={sum(rel_stats)/len(rel_stats):.4f}")
    if red_stats:
        print(f"Redundancy PMI stats: min={min(red_stats):.4f}, max={max(red_stats):.4f}, avg={sum(red_stats)/len(red_stats):.4f}")

    r_doc_pmi_rel_norm = normalize_scores(r_doc_pmi_relevance)
    r_doc_pmi_red_norm = normalize_scores(r_doc_pmi_redundancy)

    for doc_idx, indices in enumerate(short_sent_indices):
        for sent_idx in indices:
            if sent_idx < len(r_doc_norm[doc_idx]):
                r_doc_norm[doc_idx][sent_idx] = -100
            if sent_idx < len(r_doc_attn_norm[doc_idx]):
                r_doc_attn_norm[doc_idx][sent_idx] = -100
            if sent_idx < len(r_doc_pmi_rel_norm[doc_idx]):
                r_doc_pmi_rel_norm[doc_idx][sent_idx] = -100
            if sent_idx < len(r_doc_pmi_red_norm[doc_idx]):
                r_doc_pmi_red_norm[doc_idx][sent_idx] = 100

    hyperparameter_sets = []
    for i in range(6):
        for j in range(6 - i):
            for n in range(6 - i - j):
                l = 5 - i - j - n
                alpha = i / 5.0
                beta = j / 5.0
                gamma_rel = n / 5.0
                gamma_red = l / 5.0
                hyperparameter_sets.append((alpha, beta, gamma_rel, gamma_red))

    print(f"\n| Starting hyperparameter search with {len(hyperparameter_sets)} combinations...")
    for alpha, beta, gamma_rel, gamma_red in hyperparameter_sets:
        print(f"| Evaluating with alpha={alpha:.1f}, beta={beta:.1f}, gamma_rel={gamma_rel:.1f}, gamma_red={gamma_red:.1f}")
        r_doc_final = []
        for d in range(len(r_doc)):
            combined = []
            for i in range(len(r_doc[d])):
                score = alpha * r_doc_norm[d][i]
                if d < len(r_doc_attn_norm) and i < len(r_doc_attn_norm[d]):
                    score += beta * r_doc_attn_norm[d][i]
                if d < len(r_doc_pmi_rel_norm) and i < len(r_doc_pmi_rel_norm[d]):
                    score += gamma_rel * r_doc_pmi_rel_norm[d][i]
                if d < len(r_doc_pmi_red_norm) and i < len(r_doc_pmi_red_norm[d]):
                    score -= gamma_red * r_doc_pmi_red_norm[d][i]
                combined.append(score)
            r_doc_final.append(combined)

        r_indexes = [[i for i, _ in heapq.nlargest(k, enumerate(row), key=lambda x: x[1])] for row in r_doc_final]

        if not args.no_save:
            save_dir = args.save_dir
            os.makedirs(save_dir, exist_ok=True)
            filename = f'r_indexes_alpha{alpha:.1f}_beta{beta:.1f}_gamma_rel{gamma_rel:.1f}_gamma_red{gamma_red:.1f}.txt'
            r_indexes_file = os.path.join(save_dir, filename)
            with open(r_indexes_file, 'w') as f:
                for doc_idx, indexes in enumerate(r_indexes):
                    f.write(f'Doc {doc_idx}: {indexes}\n')
            print(f'| --> Saved results to: {r_indexes_file}')

    # Accuracy using cached data
    correct = 0
    total = 0
    doc_accuracies = []
    for doc_id in cached_ids:
        doc_data = load_doc_from_cache(doc_cache_files[doc_id])
        doc_correct = doc_data['correct_preds']
        doc_total = doc_data['total_preds']
        doc_accuracy = doc_correct / doc_total if doc_total > 0 else 0
        doc_accuracies.append(doc_accuracy)
        correct += doc_correct
        total += doc_total

    accuracy = correct / total if total > 0 else 0
    print('| Accuracy: {:.2f}%'.format(accuracy * 100))
    print('| Per-document accuracy:')
    for i, acc in enumerate(doc_accuracies):
        print('  Doc {}: {:.2f}%'.format(i, acc * 100))

    if not getattr(args, 'no_save', False):
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        output_file = os.path.join(save_dir, 'lprobs_results.txt')
        with open(output_file, 'w') as f:
            f.write('Accuracy: {:.2f}%\n'.format(accuracy * 100))
            f.write('Total samples: {}\n'.format(total))
            f.write('Correct predictions: {}\n'.format(correct))
            f.write('Per-document accuracy:\n')
            for i, acc in enumerate(doc_accuracies):
                f.write('  Doc {}: {:.2f}%\n'.format(i, acc * 100))
        print('| Results saved to: {}'.format(output_file))

    return accuracy


def extract_tokens_from_target(sent_tensor_np, pad_idx):
    """Extract tokens from target tensor, removing padding"""
    tokens = sent_tensor_np.tolist()
    flat_tokens = []
    
    if isinstance(tokens, list):
        while isinstance(tokens, list) and len(tokens) == 1 and isinstance(tokens[0], list):
            tokens = tokens[0]
        
        for t in tokens:
            if isinstance(t, list):
                flat_tokens.extend([x for x in t if x != pad_idx])
            else:
                if t != pad_idx:
                    flat_tokens.append(t)
    
    return flat_tokens

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_pmi_and_redundancy(all_targets_for_pmi, dataset, trainer=None):
    """Calculate PMI using sentence embeddings instead of token overlap"""
    
    # Initialize sentence transformer model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    r_doc_pmi_relevance = []
    r_doc_pmi_redundancy = []
    
    for doc_idx, doc_targets in enumerate(all_targets_for_pmi):
        print(f"Processing PMI for document {doc_idx}")
        
        # Convert targets to text sentences
        doc_sentences_text = []
        for sent_idx, sent_tensor_np in enumerate(doc_targets):
            tokens = extract_tokens_from_target(sent_tensor_np, dataset.src_dict.pad())
            if tokens:
                # Convert token IDs back to text
                text = ' '.join([dataset.src_dict[token] for token in tokens if token < len(dataset.src_dict)])
                doc_sentences_text.append(text)
            else:
                doc_sentences_text.append("")
        
        if not any(doc_sentences_text):
            r_doc_pmi_relevance.append([0.0] * len(doc_targets))
            r_doc_pmi_redundancy.append([0.0] * len(doc_targets))
            continue
        
        # Get embeddings for all non-empty sentences
        valid_sentences = [s for s in doc_sentences_text if s.strip()]
        if not valid_sentences:
            r_doc_pmi_relevance.append([0.0] * len(doc_targets))
            r_doc_pmi_redundancy.append([0.0] * len(doc_targets))
            continue
            
        sentence_embeddings = embedding_model.encode(valid_sentences)
        
        # Calculate document centroid for relevance
        doc_centroid = np.mean(sentence_embeddings, axis=0)
        
        # Calculate RELEVANCE PMI using cosine similarity to document centroid
        sent_relevance_scores = []
        valid_idx = 0
        for sent_idx, sentence_text in enumerate(doc_sentences_text):
            if not sentence_text.strip():
                sent_relevance_scores.append(0.0)
                continue
            
            # Cosine similarity to document centroid as relevance proxy
            sent_embedding = sentence_embeddings[valid_idx].reshape(1, -1)
            doc_centroid_reshaped = doc_centroid.reshape(1, -1)
            relevance_score = cosine_similarity(sent_embedding, doc_centroid_reshaped)[0][0]
            
            # Convert to PMI-like score (log of probability ratio)
            relevance_pmi = math.log(max(relevance_score + 1, 1e-8))
            sent_relevance_scores.append(relevance_pmi)
            valid_idx += 1
        
        # Calculate REDUNDANCY PMI using similarity to previous sentences
        sent_redundancy_scores = []
        valid_idx = 0
        for sent_idx, sentence_text in enumerate(doc_sentences_text):
            if not sentence_text.strip() or sent_idx == 0:
                sent_redundancy_scores.append(0.0)
                if sentence_text.strip():
                    valid_idx += 1
                continue
            
            # Calculate similarity to all previous sentences
            sent_embedding = sentence_embeddings[valid_idx]
            prev_embeddings = sentence_embeddings[:valid_idx]
            
            if len(prev_embeddings) > 0:
                # Max similarity to any previous sentence as redundancy measure
                similarities = cosine_similarity(sent_embedding.reshape(1, -1), prev_embeddings)[0]
                max_similarity = np.max(similarities)
                
                # Convert to PMI-like score (higher similarity = higher redundancy)
                redundancy_pmi = math.log(max(max_similarity + 1, 1e-8))
            else:
                redundancy_pmi = 0.0
            
            sent_redundancy_scores.append(redundancy_pmi)
            valid_idx += 1
        
        # Pad scores to match expected length
        padded_relevance = [0.0] * len(doc_targets)
        padded_redundancy = [0.0] * len(doc_targets)
        
        for i in range(min(len(sent_relevance_scores), len(padded_relevance))):
            padded_relevance[i] = sent_relevance_scores[i]
            
        for i in range(min(len(sent_redundancy_scores), len(padded_redundancy))):
            padded_redundancy[i] = sent_redundancy_scores[i]
        
        r_doc_pmi_relevance.append(padded_relevance)
        r_doc_pmi_redundancy.append(padded_redundancy)
    
    return r_doc_pmi_relevance, r_doc_pmi_redundancy
def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = '{:.3f}'.format(trainer.get_meter('train_loss').avg)
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss').avg
        stats['nll_loss'] = '{:.3f}'.format(nll_loss)
    else:
        nll_loss = trainer.get_meter('train_loss').avg
    stats['ppl'] = get_perplexity(nll_loss)
    stats['wps'] = round(trainer.get_meter('wps').avg)
    stats['ups'] = '{:.1f}'.format(trainer.get_meter('ups').avg)
    stats['wpb'] = round(trainer.get_meter('wpb').avg)
    stats['bsz'] = round(trainer.get_meter('bsz').avg)
    stats['num_updates'] = trainer.get_num_updates()
    stats['valid_ppl'] = get_perplexity(nll_loss)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(save_checkpoint, 'best'):
        stats['best'] = min(save_checkpoint.best, stats['valid_loss'])
    return stats




def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')




def save_checkpoint(args, trainer, epoch_itr, val_loss):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()


    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
        end_of_epoch and not args.no_epoch_checkpoints and
        epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
        not end_of_epoch and args.save_interval_updates > 0 and
        updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
        val_loss is not None and
        (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink


    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'best': save_checkpoint.best,
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }


    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)


    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)




def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path)
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])


            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))


            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']




def load_dataset_splits(args, task, splits, shuffle=True):
    for split in splits:
        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            try:
                task.load_dataset(split_k, shuffle)
                print('| {} {} {} examples'.format(args.data, split_k, len(task.dataset(split_k))))
            except FileNotFoundError as e:
                if k > 0:
                    break
                raise e




if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)


    '''if args.distributed_port > 0 or args.distributed_init_method is not None:
        from distributed_train import main as distributed_main


        distributed_main(args)
    elif args.distributed_world_size > 1:
        from multiprocessing_train import main as multiprocessing_main


        multiprocessing_main(args)
    else:'''
    main2(args) #####


### /HiBERT_pretrained/hibert_m/cnndm/open-in-domain/checkpoint100.pt