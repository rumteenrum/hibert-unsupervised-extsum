import os
import sys
import re
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize
import nltk
from tqdm import tqdm

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

def read_labels(file_path):
    """Read ground truth labels (T/F for each sentence)"""
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split by space and convert T/F to boolean, then get indices of True values
                label_tokens = line.split()
                true_indices = [i for i, token in enumerate(label_tokens) if token == 'T']
                labels.append(true_indices)
    return labels

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
    sentences = sent_tokenize(text)
    return '\n'.join(sentences)

def join_segments_carefully(segments):
    """Join segments with careful punctuation handling"""
    extracted_text = ""
    for i, segment in enumerate(segments):
        segment = segment.strip()
        if i > 0:
            if not segment.startswith(('.', ',', '!', '?', ':', ';')) and not extracted_text.endswith(('.', ',', '!', '?', ':', ';')):
                extracted_text += ' '
        extracted_text += segment
    return extracted_text

def calculate_rouge_scores(hypothesis, reference):
    """Calculate ROUGE-1, ROUGE-2, and ROUGE-L-SUM scores"""
    # Normalize texts
    hyp = normalize_text(hypothesis)
    ref = normalize_text(reference)
    
    # Prepare for ROUGE-L-SUM
    hyp_lsum = prepare_for_rouge_lsum(hyp)
    ref_lsum = prepare_for_rouge_lsum(ref)
    
    # Initialize scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    
    # Calculate scores
    scores = scorer.score(ref_lsum, hyp_lsum)
    
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeLsum': scores['rougeLsum'].fmeasure
    }

def extract_summary_from_labels(article_segments, label_indices):
    """Extract summary using the ground truth label indices"""
    if not label_indices:
        # If no sentences are labeled as True, return the first sentence as fallback
        selected_segments = [article_segments[0]] if article_segments else [""]
        selected_indices = [0]
    else:
        # Use only valid indices (within bounds of article)
        valid_indices = [idx for idx in label_indices if idx < len(article_segments)]
        if not valid_indices:
            # Fallback if all indices are out of bounds
            selected_segments = [article_segments[0]] if article_segments else [""]
            selected_indices = [0]
        else:
            selected_segments = [article_segments[i] for i in valid_indices]
            selected_indices = valid_indices
    
    hypothesis = join_segments_carefully(selected_segments)
    return hypothesis, selected_indices

def run_oracle_with_labels(article_path, summary_path, label_path, output_path=None):
    """Run oracle evaluation using ground truth labels from test.label"""
    
    # Load data
    articles = read_articles(article_path)
    summaries = read_summaries(summary_path)
    labels = read_labels(label_path)
    
    # Validate data
    assert len(articles) == len(summaries) == len(labels), (
        f"Mismatch in data lengths: articles={len(articles)}, "
        f"summaries={len(summaries)}, labels={len(labels)}"
    )
    
    print(f"Loaded {len(articles)} articles, summaries, and label sets")
    
    all_selected_indices = []
    all_rouge_scores = []
    
    # Process each document
    for doc_idx, (article, summary, label_indices) in enumerate(tqdm(zip(articles, summaries, labels), total=len(articles))):
        if not article or not summary:
            # Handle edge case with empty data
            selected_indices = [0]
            rouge_breakdown = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeLsum': 0.0}
        else:
            # Extract summary using ground truth labels
            hypothesis, selected_indices = extract_summary_from_labels(article, label_indices)
            
            # Calculate ROUGE scores
            rouge_breakdown = calculate_rouge_scores(hypothesis, summary)
        
        all_selected_indices.append(selected_indices)
        all_rouge_scores.append(rouge_breakdown)
        
        # Print progress for first few documents
        if doc_idx < 5:
            print(f"\nDoc {doc_idx}:")
            print(f"  Label indices: {label_indices}")
            print(f"  Selected indices: {selected_indices} (count: {len(selected_indices)})")
            print(f"  ROUGE-1: {rouge_breakdown['rouge1']:.4f}")
            print(f"  ROUGE-2: {rouge_breakdown['rouge2']:.4f}")
            print(f"  ROUGE-L-SUM: {rouge_breakdown['rougeLsum']:.4f}")
            if doc_idx < 2:
                # Show the extracted text for first 2 documents
                extracted_text, _ = extract_summary_from_labels(article, label_indices)
                print(f"  Extracted: {extracted_text[:100]}...")
    
    # Calculate average scores
    avg_rouge1 = sum(scores['rouge1'] for scores in all_rouge_scores) / len(all_rouge_scores)
    avg_rouge2 = sum(scores['rouge2'] for scores in all_rouge_scores) / len(all_rouge_scores)
    avg_rougeLsum = sum(scores['rougeLsum'] for scores in all_rouge_scores) / len(all_rouge_scores)
    
    # Calculate statistics about sentence selection
    sentence_counts = [len(indices) for indices in all_selected_indices]
    avg_sentences = sum(sentence_counts) / len(sentence_counts)
    min_sentences = min(sentence_counts)
    max_sentences = max(sentence_counts)
    
    print(f"\n=== ORACLE RESULTS USING GROUND TRUTH LABELS ===")
    print(f"Average ROUGE-1: {avg_rouge1:.4f}")
    print(f"Average ROUGE-2: {avg_rouge2:.4f}")
    print(f"Average ROUGE-L-SUM: {avg_rougeLsum:.4f}")
    print(f"\n=== SENTENCE SELECTION STATISTICS ===")
    print(f"Average sentences per summary: {avg_sentences:.2f}")
    print(f"Min sentences: {min_sentences}")
    print(f"Max sentences: {max_sentences}")
    
    # Count distribution of sentence numbers
    from collections import Counter
    sentence_distribution = Counter(sentence_counts)
    print(f"Sentence count distribution: {dict(sorted(sentence_distribution.items()))}")
    
    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save average scores
        with open(output_path + '_ground_truth_scores.txt', 'w') as f:
            f.write("Ground Truth Oracle Results:\n")
            f.write(f"Average ROUGE-1: {avg_rouge1:.4f}\n")
            f.write(f"Average ROUGE-2: {avg_rouge2:.4f}\n")
            f.write(f"Average ROUGE-L-SUM: {avg_rougeLsum:.4f}\n")
            f.write(f"\nSentence Selection Statistics:\n")
            f.write(f"Average sentences per summary: {avg_sentences:.2f}\n")
            f.write(f"Min sentences: {min_sentences}\n")
            f.write(f"Max sentences: {max_sentences}\n")
            f.write(f"Sentence count distribution: {dict(sorted(sentence_distribution.items()))}\n")
        
        # Save selected indices
        with open(output_path + '_ground_truth_indices.txt', 'w') as f:
            for doc_idx, indices in enumerate(all_selected_indices):
                f.write(f'Doc {doc_idx}: {indices}\n')
        
        # Save detailed results
        with open(output_path + '_detailed_results.txt', 'w') as f:
            f.write("Doc_ID\tSelected_Indices\tNum_Sentences\tROUGE-1\tROUGE-2\tROUGE-L-SUM\n")
            for doc_idx, (indices, scores) in enumerate(zip(all_selected_indices, all_rouge_scores)):
                f.write(f"{doc_idx}\t{indices}\t{len(indices)}\t{scores['rouge1']:.4f}\t{scores['rouge2']:.4f}\t{scores['rougeLsum']:.4f}\n")
        
        print(f"Results saved to:")
        print(f"  - {output_path}_ground_truth_scores.txt")
        print(f"  - {output_path}_ground_truth_indices.txt") 
        print(f"  - {output_path}_detailed_results.txt")
    
    return avg_rouge1, avg_rouge2, avg_rougeLsum, all_selected_indices

def compare_with_fixed_sentences(article_path, summary_path, label_path, num_sentences_list=[1, 2, 3, 4, 5]):
    """Compare ground truth oracle with fixed sentence count oracles"""
    from oracle import greedy_oracle_selection  # Import from the previous oracle file
    
    # Load data
    articles = read_articles(article_path)
    summaries = read_summaries(summary_path)
    labels = read_labels(label_path)
    
    print("\n=== COMPARISON: Ground Truth vs Fixed Sentence Counts ===")
    
    # Ground truth results
    print("Ground Truth Oracle:")
    gt_r1, gt_r2, gt_rl, _ = run_oracle_with_labels(article_path, summary_path, label_path)
    
    # Fixed sentence count results
    for num_sentences in num_sentences_list:
        print(f"\nGreedy Oracle with {num_sentences} sentences:")
        total_r1, total_r2, total_rl = 0, 0, 0
        
        for article, summary in tqdm(zip(articles[:100], summaries[:100]), desc=f"Testing {num_sentences} sentences"):  # Test on first 100 for speed
            if not article or not summary:
                continue
            
            _, _, rouge_scores = greedy_oracle_selection(article, summary, num_sentences=num_sentences)
            total_r1 += rouge_scores['rouge1']
            total_r2 += rouge_scores['rouge2']
            total_rl += rouge_scores['rougeLsum']
        
        avg_r1 = total_r1 / 100
        avg_r2 = total_r2 / 100
        avg_rl = total_rl / 100
        
        print(f"  ROUGE-1: {avg_r1:.4f} (vs GT: {gt_r1:.4f})")
        print(f"  ROUGE-2: {avg_r2:.4f} (vs GT: {gt_r2:.4f})")
        print(f"  ROUGE-L-SUM: {avg_rl:.4f} (vs GT: {gt_rl:.4f})")

if __name__ == "__main__":
    # Paths based on your configuration
    article_path = "/teamspace/studios/this_studio/data_after_bpe/full_data/test.article"
    summary_path = "/teamspace/studios/this_studio/data_after_bpe/full_data/test.summary"
    label_path = "/teamspace/studios/this_studio/data_after_bpe/full_data/test.label"
    output_path = "/teamspace/studios/this_studio/data_after_bpe/oracle_results"
    
    # Run oracle evaluation using ground truth labels
    print("Running oracle evaluation with ground truth labels...")
    run_oracle_with_labels(
        article_path=article_path,
        summary_path=summary_path,
        label_path=label_path,
        output_path=output_path
    )
    
    # Optional: Compare with fixed sentence counts
    # Uncomment the line below if you want to compare with fixed sentence count oracles
    # compare_with_fixed_sentences(article_path, summary_path, label_path)