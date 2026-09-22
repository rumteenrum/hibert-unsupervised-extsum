import os
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
    Ensure NLTK punkt tokenizer is available.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' tokenizer for worker process...")
        nltk.download('punkt', quiet=True)

def worker_init():
    """
    Per-worker init: ensure NLTK data and create Rouge scorers once.
    """
    download_nltk_data()
    global scorer_lsum, scorer_all
    scorer_lsum = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
    scorer_all = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

def read_data_into_globals(article_path, summary_path):
    """
    Load articles (split by ' <S_SEP> ') and summaries.
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
    Prepare text for ROUGE-Lsum: one sentence per line.
    """
    sentences = sent_tokenize(text)
    return '\n'.join(sentences)

def calculate_rouge_scores_prepared(hyp_lsum, ref_lsum):
    """
    Compute rouge1/rouge2/rougeLsum on already-prepared strings.
    """
    scores = scorer_all.score(ref_lsum, hyp_lsum)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeLsum': scores['rougeLsum'].fmeasure
    }

def calculate_rouge_lsum_only_prepared(hyp_lsum, ref_lsum):
    """
    Fast path for greedy selection: ROUGE-Lsum only.
    """
    return scorer_lsum.score(ref_lsum, hyp_lsum)['rougeLsum'].fmeasure

def fast_greedy_oracle_up_to3(article_segments, reference_lsum, max_sentences=3, eps=1e-12):
    """
    Greedy selection of up to 3 sentences with early stopping if no ROUGE-Lsum gain.
    Picks at least 1 sentence if any exist.
    """
    if not article_segments:
        return [], {'rouge1': 0.0, 'rouge2': 0.0, 'rougeLsum': 0.0}

    selected_indices = []
    candidate_indices = list(range(len(article_segments)))
    current_score = 0.0

    for k in range(max_sentences):
        best_score = -1.0
        best_candidate_pos = -1

        for i, idx in enumerate(candidate_indices):
            hyp_lsum = '\n'.join(article_segments[j] for j in (selected_indices + [idx]))
            score = calculate_rouge_lsum_only_prepared(hyp_lsum, reference_lsum)
            if score > best_score + eps:
                best_score = score
                best_candidate_pos = i

        if best_candidate_pos == -1:
            break

        # Always take the first sentence; for subsequent ones require strict improvement
        if k == 0 or best_score > current_score + eps:
            selected_indices.append(candidate_indices.pop(best_candidate_pos))
            current_score = best_score
        else:
            break

    selected_indices.sort()
    if selected_indices:
        final_hyp_lsum = '\n'.join(article_segments[i] for i in selected_indices)
        final_rouge = calculate_rouge_scores_prepared(final_hyp_lsum, reference_lsum)
    else:
        final_rouge = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeLsum': 0.0}

    return selected_indices, final_rouge

def process_single_document(doc_idx):
    """
    Worker: select up to 3 sentences.
    """
    article = articles[doc_idx]
    summary = summaries[doc_idx]

    if not article or not summary:
        return doc_idx, [], {'rouge1': 0.0, 'rouge2': 0.0, 'rougeLsum': 0.0}

    reference_lsum = prepare_for_rouge_lsum(summary)
    best_indices, rouge_scores = fast_greedy_oracle_up_to3(article, reference_lsum, max_sentences=3)
    return doc_idx, best_indices, rouge_scores

def run_oracle_evaluation(article_path, summary_path, output_dir):
    """
    Orchestrate 'up to 3 sentences' oracle evaluation.
    """
    read_data_into_globals(article_path, summary_path)
    assert len(articles) == len(summaries), "Article and summary counts do not match."

    print(f"Loaded {len(articles)} articles and summaries.")
    print("Running greedy oracle to select up to 3 sentences.")

    max_workers = min(cpu_count(), 16)
    print(f"Using multiprocessing with {max_workers} workers.")

    doc_indices = list(range(len(articles)))
    results = []

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

    results.sort(key=lambda x: x[0])
    all_best_indices = [res[1] for res in results]
    all_rouge_scores = [res[2] for res in results]

    if not all_rouge_scores:
        print("No results were generated.")
        return

    avg_rouge1 = np.mean([s['rouge1'] for s in all_rouge_scores])
    avg_rouge2 = np.mean([s['rouge2'] for s in all_rouge_scores])
    avg_rougeLsum = np.mean([s['rougeLsum'] for s in all_rouge_scores])

    print("\n" + "="*30)
    print("ORACLE RESULTS (up to 3 sentences)")
    print("="*30)
    print(f"Average ROUGE-1:   {avg_rouge1:.4f}")
    print(f"Average ROUGE-2:   {avg_rouge2:.4f}")
    print(f"Average ROUGE-Lsum: {avg_rougeLsum:.4f}")
    print("="*30)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "oracle_results_up_to3_indices.txt")
    with open(output_file, 'w') as f:
        for indices in all_best_indices:
            f.write(' '.join(map(str, indices)) + '\n')
    print(f"Oracle indices saved to {output_file}")

if __name__ == "__main__":
    print("Checking NLTK data in main process...")
    download_nltk_data()

    article_path = "/teamspace/studios/this_studio/data_after_bpe/full_data/test.article"
    summary_path = "/teamspace/studios/this_studio/data_after_bpe/full_data/test.summary"
    output_dir = "/teamspace/studios/this_studio/data_after_bpe/oracle_results/"

    run_oracle_evaluation(article_path, summary_path, output_dir)