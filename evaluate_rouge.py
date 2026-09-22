import os
import sys
import re
import random
import nltk
from nltk.tokenize import sent_tokenize
import glob
import argparse
from rouge_score import rouge_scorer
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from pyrougex.rouge import get_rouge_batch

# >>> Add your selected weights here. Leave empty list [] to use CLI or all files.
# Each item is (alpha, beta, gamma_rel, gamma_red)
SELECTED_WEIGHTS = [
    (0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 0.4, 0.6),
    (0.0, 0.4, 0.0, 0.6),
    (0.4, 0.0, 0.0, 0.6)
]

'''
    (0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 0.2, 0.8),
    (0.0, 0.2, 0.0, 0.8),
    (0.2, 0.0, 0.0, 0.8),
    (0.0, 0.0, 0.4, 0.6),
    (0.0, 0.4, 0.0, 0.6),
    (0.4, 0.0, 0.0, 0.6),
    (0.2, 0.2, 0.0, 0.6),
    (0.2, 0.0, 0.2, 0.6),
    (0.0, 0.2, 0.2, 0.6)
    
'''

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def read_articles(file_path):
    """Read articles with segments separated by <S_SEP>"""
    with open(file_path, 'r', encoding='utf-8') as f:
        articles = [x.strip().split(' <S_SEP> ') for x in f.readlines()]
    return articles

def read_summaries(file_path):
    """Read reference summaries"""
    with open(file_path, 'r', encoding='utf-8') as f:
        summaries = [x.strip() for x in f.readlines()]
    return summaries

def read_indexes(file_path):
    """Read selected indices from index file"""
    indexes = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('Doc'):
                idxs = line.strip().split(':')[1].strip()
                idxs = idxs.strip('[]').split(',')
                idxs = [int(i.strip()) for i in idxs]
                indexes.append(idxs)
    return indexes

def normalize_text(text):
    """Apply text normalization for more accurate ROUGE calculation"""
    # Remove BPE artifacts
    text = text.replace('@@ ', '').replace('@@', '')
    
    # Convert to lowercase (makes matching case-insensitive)
    text = text.lower()
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalize punctuation (ensure consistent spacing)
    text = re.sub(r'\s+([,.!?:;])', r'\1', text)
    
    # Ensure spaces after sentence-ending punctuation
    text = re.sub(r'([.!?])\s*', r'\1 ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s{2,}', ' ', text)
    
    return text.strip()

def prepare_for_rouge_lsum(text):
    """Prepare text for ROUGE-L-SUM by ensuring proper sentence segmentation"""
    # Split into sentences
    sentences = sent_tokenize(text)
    # Join with newlines for ROUGE-L-SUM
    return '\n'.join(sentences)

def parse_hparams_from_filename(path):
    """Extract alpha/beta/gamma_rel/gamma_red from filenames like:
       r_indexes_alpha0.0_beta0.0_gamma_rel0.0_gamma_red1.0.txt
       Backward-compatible with: r_indexes_alpha0.3_beta0.7_gamma0.0.txt
       Returns a 4-tuple: (alpha, beta, gamma_rel, gamma_red)
    """
    name = os.path.basename(path)
    # New 4-weight pattern
    m4 = re.search(r'alpha([0-9.]+)_beta([0-9.]+)_gamma[_\-]?rel([0-9.]+)_gamma[_\-]?red([0-9.]+)', name)
    if m4:
        try:
            return float(m4.group(1)), float(m4.group(2)), float(m4.group(3)), float(m4.group(4))
        except ValueError:
            return None, None, None, None
    # Old 3-weight pattern (gamma -> gamma_rel; gamma_red defaults to 0.0)
    m3 = re.search(r'alpha([0-9.]+)_beta([0-9.]+)_gamma([0-9.]+)', name)
    if m3:
        try:
            return float(m3.group(1)), float(m3.group(2)), float(m3.group(3)), 0.0
        except ValueError:
            return None, None, None, None
    return None, None, None, None

# NEW: helpers to parse user-selected weights
def parse_weight_quad(s):
    """Parse a single 'alpha,beta,gamma_rel,gamma_red' (or with colons) to floats."""
    parts = re.split(r'[,:]', s.strip())
    if len(parts) != 4:
        raise ValueError(f"Invalid weight quad '{s}'. Expected 'alpha,beta,gamma_rel,gamma_red'.")
    return tuple(float(x) for x in parts)

def load_weights_from_file(path):
    """Load 4-number weight quads from a file. Lines like: '0.0,0.0,0.0,1.0' or '0.0 0.0 0.0 1.0'."""
    weights = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = re.split(r'[,\s]+', ln)
            if len(parts) != 4:
                raise ValueError(f"Invalid line in weights file (need 4 values): '{ln}'")
            weights.append(tuple(float(x) for x in parts))
    return weights

# NEW: helpers to build file names from weights
def _fmt_num(x: float) -> str:
    """Format floats for filenames: keep minimal decimals but always include one decimal place.
       Examples: 0    -> 0.0, 1 -> 1.0, 0.20 -> 0.2, 0.200000 -> 0.2
    """
    s = f"{float(x):.10f}".rstrip('0').rstrip('.')
    if '.' not in s:
        s += '.0'
    return s

def weight_to_index_filename(alpha, beta, gamma_rel, gamma_red) -> str:
    """Create a filename matching: r_indexes_alpha{a}_beta{b}_gamma_rel{gr}_gamma_red{gd}.txt"""
    return (
        f"r_indexes_alpha{_fmt_num(alpha)}"
        f"_beta{_fmt_num(beta)}"
        f"_gamma_rel{_fmt_num(gamma_rel)}"
        f"_gamma_red{_fmt_num(gamma_red)}.txt"
    )

def evaluate_rouge(article_path, summary_path, indexes_path, output_path=None, batch_size=100):
    """Evaluate ROUGE scores combining pyrougex for R1/R2 and rouge_score for RLSUM"""
    # Load data
    articles = read_articles(article_path)
    summaries = read_summaries(summary_path)
    indexes = read_indexes(indexes_path)
    
    # Validate data
    assert len(articles) == len(summaries) == len(indexes), (
        f"Mismatch in data lengths: articles={len(articles)}, "
        f"summaries={len(summaries)}, indexes={len(indexes)}"
    )
    
    hyps = []  # Extracted summaries
    refs = []  # Reference summaries
    
    # Initialize ROUGE-L-SUM scorer
    scorer_lsum = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
    rougeLsum_scores = []
    
    for doc_idx, (article, summary, idxs) in enumerate(zip(articles, summaries, indexes)):
        if not article or not summary:
            continue
            
        # Sort indices to maintain original document order
        sorted_idxs = sorted([i for i in idxs if i < len(article)])
        
        # Extract segments
        extracted_segments = [article[i] for i in sorted_idxs]
        
        # Strategy: Careful joining with punctuation handling
        extracted_text = ""
        for i, segment in enumerate(extracted_segments):
            segment = segment.strip()
            if i > 0:
                if not segment.startswith(('.', ',', '!', '?', ':', ';')) and not extracted_text.endswith(('.', ',', '!', '?', ':', ';')):
                    extracted_text += ' '
            extracted_text += segment
        
        # Normalize both hypothesis and reference
        hyp = normalize_text(extracted_text)
        ref = normalize_text(summary)
        
        # Prepare texts for ROUGE-L-SUM (with sentence segmentation)
        hyp_lsum = prepare_for_rouge_lsum(hyp)
        ref_lsum = prepare_for_rouge_lsum(ref)
        
        # Calculate ROUGE-L-SUM scores using rouge_score library
        scores_lsum = scorer_lsum.score(ref_lsum, hyp_lsum)
        rougeLsum_scores.append(scores_lsum['rougeLsum'].fmeasure)
        
        '''# Debug: Print sample examples
        if doc_idx < 1:
            # Show token overlap statistics
            ref_tokens = ref.split()
            hyp_tokens = hyp.split()
            overlap = set(ref_tokens) & set(hyp_tokens)
            print(f"Token overlap: {len(overlap)}/{len(ref_tokens)} reference tokens found in hypothesis")
        '''

        hyps.append(hyp)
        refs.append(ref)
    
    # Process ROUGE-1 and ROUGE-2 using pyrougex in batches
    all_rouge_scores = {}
    
    
    for i in range(0, len(hyps), batch_size):
        batch_hyps = hyps[i:i+batch_size]
        batch_refs = refs[i:i+batch_size]
        
        # Calculate ROUGE-1 and ROUGE-2 scores using pyrougex
        batch_scores = get_rouge_batch(batch_hyps, batch_refs)
        
        # Merge scores
        if not all_rouge_scores:
            all_rouge_scores = batch_scores
        else:
            # Aggregate scores
            batch_weight = len(batch_hyps) / len(hyps)
            for k in batch_scores:
                if k in all_rouge_scores:
                    all_rouge_scores[k] = all_rouge_scores[k] * (1 - batch_weight) + batch_scores[k] * batch_weight

    # Calculate average ROUGE-L-SUM
    avg_rougeLsum = sum(rougeLsum_scores) / len(rougeLsum_scores)

    # Find the correct keys for ROUGE-1 and ROUGE-2
    r1_key = next(k for k in all_rouge_scores.keys() if '1' in k and 'f' in k)
    r2_key = next(k for k in all_rouge_scores.keys() if '2' in k and 'f' in k)
    
    '''# Print results
    print("\n=== FINAL ROUGE SCORES ===")
    print(f"ROUGE-1 F1: {all_rouge_scores[r1_key]:.4f}")
    print(f"ROUGE-2 F1: {all_rouge_scores[r2_key]:.4f}")
    print(f"ROUGE-L-SUM F1: {avg_rougeLsum:.4f}")'''
    
    # Save results if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(f"ROUGE-1 F1: {all_rouge_scores[r1_key]:.4f}\n")
            f.write(f"ROUGE-2 F1: {all_rouge_scores[r2_key]:.4f}\n")
            f.write(f"ROUGE-L-SUM F1: {avg_rougeLsum:.4f}\n")
        print(f"Results saved to {output_path}")
    
    return all_rouge_scores, avg_rougeLsum

def run_hparam_grid(article_path, summary_path, indexes_dir, pattern, select_metric, topk, out_path, batch_size, selected_weights=None):
    """Run evaluation across multiple hyperparameter index files"""
    # If a list of weights is provided, construct the filenames directly from it.
    if selected_weights:
        pairs = []
        for w in selected_weights:
            a, b, gr, gd = w
            fname = weight_to_index_filename(a, b, gr, gd)
            fpath = os.path.join(indexes_dir, fname)
            pairs.append((fpath, (a, b, gr, gd)))

        existing = [(p, w) for (p, w) in pairs if os.path.exists(p)]
        missing = [(p, w) for (p, w) in pairs if not os.path.exists(p)]

        if missing:
            print("Warning: some constructed index files do not exist:")
            for p, w in missing:
                print(f"  - {os.path.basename(p)}  (weights={w})")
        if not existing:
            print("None of the constructed index files exist. Check SELECTED_WEIGHTS and indexes_dir.")
            return

        results = []
        print(f"Found {len(existing)} index files (from SELECTED_WEIGHTS). Evaluating with metric='{select_metric}'...")

        for idx_path, w in tqdm(existing):
            scores, rlsum = evaluate_rouge(article_path, summary_path, idx_path, output_path=None, batch_size=batch_size)

            r1_key = next(k for k in scores.keys() if '1' in k and 'f' in k)
            r2_key = next(k for k in scores.keys() if '2' in k and 'f' in k)

            r1 = float(scores[r1_key])
            r2 = float(scores[r2_key])
            a, b, gr, gd = w  # use the known weights we constructed with

            results.append({
                'file': os.path.basename(idx_path),
                'alpha': a, 'beta': b,
                'gamma_rel': gr, 'gamma_red': gd,
                'r1': r1, 'r2': r2, 'rlsum': float(rlsum),
            })

        # Sort by selected metric
        metric_key = select_metric.lower()
        if metric_key not in ('r1', 'r2', 'rlsum'):
            raise ValueError(f"Unsupported metric '{select_metric}'. Use one of: r1, r2, rlsum")

        results_sorted = sorted(results, key=lambda x: x[metric_key], reverse=True)
        top = results_sorted[:topk]

        # Log all results + top-k
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("file\talpha\tbeta\tgamma_rel\tgamma_red\trouge1_f\trouge2_f\trouge_lsum_f\n")
            for r in results_sorted:
                f.write(f"{r['file']}\t{r['alpha']}\t{r['beta']}\t{r['gamma_rel']}\t{r['gamma_red']}\t{r['r1']:.6f}\t{r['r2']:.6f}\t{r['rlsum']:.6f}\n")
            f.write(f"\nTOP {topk} by {metric_key}:\n")
            for i, r in enumerate(top, 1):
                f.write(
                    f"{i}. {r['file']} | alpha={r['alpha']} beta={r['beta']} "
                    f"gamma_rel={r['gamma_rel']} gamma_red={r['gamma_red']} | "
                    f"R1={r['r1']:.4f} R2={r['r2']:.4f} RLSUM={r['rlsum']:.4f}\n"
                )

        print(f"\nTop {topk} (by {metric_key}):")
        for i, r in enumerate(top, 1):
            print(
                f"{i}. {r['file']} | alpha={r['alpha']} beta={r['beta']} "
                f"gamma_rel={r['gamma_rel']} gamma_red={r['gamma_red']} | "
                f"R1={r['r1']:.4f} R2={r['r2']:.4f} RLSUM={r['rlsum']:.4f}"
            )
        print(f"\nFull results written to: {out_path}")
        return

    # Fallback: original directory scan (used only if no selected_weights provided)
    index_files = sorted(glob.glob(os.path.join(indexes_dir, pattern)))
    if not index_files:
        print(f"No index files found in {indexes_dir} with pattern '{pattern}'")
        return

    results = []
    print(f"Found {len(index_files)} index files. Evaluating with metric='{select_metric}'...")
    for idx_path in tqdm(index_files):
        scores, rlsum = evaluate_rouge(article_path, summary_path, idx_path, output_path=None, batch_size=batch_size)

        r1_key = next(k for k in scores.keys() if '1' in k and 'f' in k)
        r2_key = next(k for k in scores.keys() if '2' in k and 'f' in k)

        r1 = float(scores[r1_key])
        r2 = float(scores[r2_key])
        a, b, gr, gd = parse_hparams_from_filename(idx_path)

        results.append({
            'file': os.path.basename(idx_path),
            'alpha': a, 'beta': b,
            'gamma_rel': gr, 'gamma_red': gd,
            'r1': r1, 'r2': r2, 'rlsum': float(rlsum),
        })

    # Sort by selected metric
    metric_key = select_metric.lower()
    if metric_key not in ('r1', 'r2', 'rlsum'):
        raise ValueError(f"Unsupported metric '{select_metric}'. Use one of: r1, r2, rlsum")

    results_sorted = sorted(results, key=lambda x: x[metric_key], reverse=True)
    top = results_sorted[:topk]

    # Log all results + top-k
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("file\talpha\tbeta\tgamma_rel\tgamma_red\trouge1_f\trouge2_f\trouge_lsum_f\n")
        for r in results_sorted:
            f.write(f"{r['file']}\t{r['alpha']}\t{r['beta']}\t{r['gamma_rel']}\t{r['gamma_red']}\t{r['r1']:.6f}\t{r['r2']:.6f}\t{r['rlsum']:.6f}\n")
        f.write(f"\nTOP {topk} by {metric_key}:\n")
        for i, r in enumerate(top, 1):
            f.write(
                f"{i}. {r['file']} | alpha={r['alpha']} beta={r['beta']} "
                f"gamma_rel={r['gamma_rel']} gamma_red={r['gamma_red']} | "
                f"R1={r['r1']:.4f} R2={r['r2']:.4f} RLSUM={r['rlsum']:.4f}\n"
            )

    print(f"\nTop {topk} (by {metric_key}):")
    for i, r in enumerate(top, 1):
        print(
            f"{i}. {r['file']} | alpha={r['alpha']} beta={r['beta']} "
            f"gamma_rel={r['gamma_rel']} gamma_red={r['gamma_red']} | "
            f"R1={r['r1']:.4f} R2={r['r2']:.4f} RLSUM={r['rlsum']:.4f}"
        )
    print(f"\nFull results written to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ROUGE over multiple hyperparameter index files.")
    parser.add_argument("--article-path", type=str,
                        default="/teamspace/studios/this_studio/data_after_bpe/test.article")
    parser.add_argument("--summary-path", type=str,
                        default="/teamspace/studios/this_studio/data_after_bpe/test.summary")
    parser.add_argument("--indexes-dir", type=str,
                        default="/teamspace/studios/this_studio/data_after_bpe/TEST_pre",
                        help="Directory containing r_indexes_* files.")
    parser.add_argument("--pattern", type=str, default="r_indexes_*.txt",
                        help="Glob pattern to select index files.")
    parser.add_argument("--metric", type=str, default="rlsum",
                        choices=["r1", "r2", "rlsum"],
                        help="Metric to rank the top hyperparameters.")
    parser.add_argument("--topk", type=int, default=10, help="How many top setups to report.")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for ROUGE computation.")
    parser.add_argument("--out", type=str,
                        default="/teamspace/studios/this_studio/data_after_bpe/rouge_grid_results.tsv",
                        help="Path to write the full results table and top-k summary.")
    parser.add_argument("--single-index", type=str, default=None,
                        help="Optional: evaluate a single indexes file instead of grid.")
    # Keep CLI as optional fallback
    parser.add_argument(
        "--weight", action="append", default=None,
        help="Optional. Weight quad as 'alpha,beta,gamma_rel,gamma_red'. Ignored if SELECTED_WEIGHTS is set."
    )
    parser.add_argument(
        "--weights-file", type=str, default=None,
        help="Optional. File with 'alpha beta gamma_rel gamma_red' per line. Ignored if SELECTED_WEIGHTS is set."
    )

    args = parser.parse_args()

    # Build selected weights list (prefer the in-code list if provided)
    if SELECTED_WEIGHTS:
        selected_weights = [(float(a), float(b), float(gr), float(gd)) for (a, b, gr, gd) in SELECTED_WEIGHTS]
        print(f"Using {len(selected_weights)} in-code selected weights.")
    else:
        selected_weights = []
        if args.weight:
            for w in args.weight:
                selected_weights.append(parse_weight_quad(w))
        if args.weights_file:
            selected_weights.extend(load_weights_from_file(args.weights_file))
        if not selected_weights:
            selected_weights = None  # means evaluate all files matching pattern

    if args.single_index:
        evaluate_rouge(
            args.article_path,
            args.summary_path,
            args.single_index,
            output_path=args.out,
            batch_size=args.batch_size
        )
    else:
        run_hparam_grid(
            article_path=args.article_path,
            summary_path=args.summary_path,
            indexes_dir=args.indexes_dir,
            pattern=args.pattern,
            select_metric=args.metric,
            topk=args.topk,
            out_path=args.out,
            batch_size=args.batch_size,
            selected_weights=selected_weights
        )