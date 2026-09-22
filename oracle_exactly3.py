import os
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import sys

# --- Global Data ---
articles = []
summaries = []
# Rouge scorers initialized once per worker
scorer_lsum = None
scorer_all = None
# ---

def download_nltk_data():
    """
    Downloads the NLTK 'punkt' tokenizer if it's not already present.
    This is designed to be called by worker processes.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' tokenizer for worker process...")
        nltk.download('punkt', quiet=True)

def worker_init():
    """
    Initializer for each worker in the multiprocessing pool.
    Ensures NLTK data is available and initializes Rouge scorers once.
    """
    download_nltk_data()
    # Initialize scorers once per worker (much faster than per-call)
    global scorer_lsum, scorer_all
    scorer_lsum = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
    scorer_all = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

def read_data_into_globals(article_path, summary_path):
    """
    Reads article and summary files into the global lists.
    This is done once in the main process.
    """
    global articles, summaries
    print("Reading articles...")
    with open(article_path, 'r', encoding='utf-8') as f:
        articles = [line.strip().split(' <S_SEP> ') for line in f]
    print("Reading summaries...")
    with open(summary_path, 'r', encoding='utf-8') as f:
        summaries = [line.strip() for line in f]

def prepare_for_rouge_lsum(text):
    """
    Prepares text for ROUGE-L-SUM by splitting it into sentences,
    with each sentence on a new line.
    """
    sentences = sent_tokenize(text)
    return '\n'.join(sentences)

def calculate_rouge_scores_prepared(hyp_lsum, ref_lsum):
    """
    Scores already-prepared (newline-separated sentences) hyp/ref with all metrics.
    """
    # scorer_all is initialized in worker_init
    scores = scorer_all.score(ref_lsum, hyp_lsum)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeLsum': scores['rougeLsum'].fmeasure
    }

def calculate_rouge_lsum_only_prepared(hyp_lsum, ref_lsum):
    """
    Fast path for greedy selection: compute only ROUGE-Lsum on prepared strings.
    """
    return scorer_lsum.score(ref_lsum, hyp_lsum)['rougeLsum'].fmeasure

def calculate_rouge_scores(hypothesis, reference):
    """
    Backward-compatible helper if needed elsewhere.
    """
    hyp_lsum = prepare_for_rouge_lsum(hypothesis)
    ref_lsum = prepare_for_rouge_lsum(reference)
    return calculate_rouge_scores_prepared(hyp_lsum, ref_lsum)

def fast_greedy_oracle(article_segments, reference_lsum, num_sentences=3):
    """
    Finds the best combination of `num_sentences` from the article using a greedy approach.
    Uses only ROUGE-Lsum during selection for speed, and computes all metrics once at the end.
    article_segments are already sentence-split; hypotheses are joined with '\n' (no NLTK needed).
    """
    if not article_segments:
        return [0] * num_sentences, {'rouge1': 0.0, 'rouge2': 0.0, 'rougeLsum': 0.0}

    # Handle articles with fewer sentences than required (pad to exactly num_sentences)
    if len(article_segments) < num_sentences:
        indices = list(range(len(article_segments)))
        last_index = indices[-1] if indices else 0
        indices.extend([last_index] * (num_sentences - len(indices)))
        hyp_lsum = '\n'.join(article_segments[i] for i in indices)
        rouge_scores = calculate_rouge_scores_prepared(hyp_lsum, reference_lsum)
        return indices, rouge_scores

    selected_indices = []
    candidate_indices = list(range(len(article_segments)))

    for _ in range(num_sentences):
        best_score = -1.0
        best_candidate_pos = -1

        # Try adding each remaining sentence and keep the best by ROUGE-Lsum
        for i, idx in enumerate(candidate_indices):
            current_selection = selected_indices + [idx]
            hyp_lsum = '\n'.join(article_segments[j] for j in current_selection)
            score = calculate_rouge_lsum_only_prepared(hyp_lsum, reference_lsum)
            if score > best_score:
                best_score = score
                best_candidate_pos = i

        if best_candidate_pos != -1:
            selected_indices.append(candidate_indices.pop(best_candidate_pos))
        else:
            # Fallback (shouldn't trigger): pick the first available
            if candidate_indices:
                selected_indices.append(candidate_indices.pop(0))
            else:
                # Ensure exactly num_sentences
                selected_indices.append(selected_indices[-1])

    # Ensure exactly num_sentences (safety)
    if len(selected_indices) < num_sentences:
        selected_indices.extend([selected_indices[-1]] * (num_sentences - len(selected_indices)))
    elif len(selected_indices) > num_sentences:
        selected_indices = selected_indices[:num_sentences]

    selected_indices.sort()
    final_hyp_lsum = '\n'.join(article_segments[i] for i in selected_indices)
    final_rouge = calculate_rouge_scores_prepared(final_hyp_lsum, reference_lsum)
    return selected_indices, final_rouge

def process_single_document(doc_idx):
    """
    Worker function to process one document.
    """
    num_sentences = 3  # keep exactly 3 sentences
    article = articles[doc_idx]
    summary = summaries[doc_idx]

    if not article or not summary:
        return doc_idx, [0] * num_sentences, {'rouge1': 0.0, 'rouge2': 0.0, 'rougeLsum': 0.0}

    # Prepare reference once per document
    reference_lsum = prepare_for_rouge_lsum(summary)
    best_indices, rouge_scores = fast_greedy_oracle(article, reference_lsum, num_sentences)
    return doc_idx, best_indices, rouge_scores

def run_oracle_evaluation(article_path, summary_path, output_path, num_sentences=3):
    """
    Main function to orchestrate the oracle summary generation process.
    It loads data, sets up a multiprocessing pool, and aggregates results.
    """
    # 1. Load data into global variables before forking
    read_data_into_globals(article_path, summary_path)
    
    assert len(articles) == len(summaries), "Article and summary counts do not match."
    
    print(f"Loaded {len(articles)} articles and summaries.")
    print(f"Running greedy oracle to select exactly {num_sentences} sentences.")
    
    # Use a sensible number of workers
    max_workers = min(cpu_count(), 16)
    print(f"Using multiprocessing with {max_workers} workers.")
    
    doc_indices = list(range(len(articles)))
    results = []

    # 2. Create the pool with the initializer
    with Pool(processes=max_workers, initializer=worker_init) as pool:
        for result in tqdm(
            pool.imap_unordered(process_single_document, doc_indices),  # no chunksize
            total=len(doc_indices),
            desc="Processing documents",
            dynamic_ncols=True,
            mininterval=0.5,
            disable=not sys.stdout.isatty()
        ):
            results.append(result)
    
    # 3. Process results
    results.sort(key=lambda x: x[0])
    all_best_indices = [res[1] for res in results]
    all_rouge_scores = [res[2] for res in results]
    
    if not all_rouge_scores:
        print("No results were generated.")
        return
        
    # Calculate and print average scores
    avg_rouge1 = np.mean([s['rouge1'] for s in all_rouge_scores])
    avg_rouge2 = np.mean([s['rouge2'] for s in all_rouge_scores])
    avg_rougeLsum = np.mean([s['rougeLsum'] for s in all_rouge_scores])
    
    print("\n" + "="*30)
    print(f"ORACLE RESULTS ({num_sentences} sentences)")
    print("="*30)
    print(f"Average ROUGE-1:   {avg_rouge1:.4f}")
    print(f"Average ROUGE-2:   {avg_rouge2:.4f}")
    print(f"Average ROUGE-Lsum: {avg_rougeLsum:.4f}")
    print("="*30)
    
    # 4. Save the output indices
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_file = os.path.join(os.path.dirname(output_path), "oracle_results_fast_indices.txt")
    with open(output_file, 'w') as f:
        for indices in all_best_indices:
            f.write(' '.join(map(str, indices)) + '\n')
    print(f"Oracle indices saved to {output_file}")

if __name__ == "__main__":
    # Ensure NLTK data is available in the main process before starting
    print("Checking NLTK data in main process...")
    download_nltk_data()

    # Define file paths
    article_path = "/teamspace/studios/this_studio/data_after_bpe/full_data/test.article"
    summary_path = "/teamspace/studios/this_studio/data_after_bpe/full_data/test.summary"
    output_path = "/teamspace/studios/this_studio/data_after_bpe/oracle_results/"
    
    num_sentences_to_select = 3
    
    run_oracle_evaluation(
        article_path=article_path,
        summary_path=summary_path,
        output_path=output_path,
        num_sentences=num_sentences_to_select
    )