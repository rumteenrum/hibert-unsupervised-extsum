"""Turning sentence scores into extractive summaries.

The two modes follow STAS's evaluation code (sum_eval.py):

- `topk_in_order`: the k highest-scoring sentences, output in document order.
- `topk_trigram_blocking`: sentences in decreasing score order, skipping any sentence that
  shares a word trigram with those already chosen, until k are chosen. Output in the order
  chosen. Trigrams are over whitespace tokens, case-sensitive, as in the original.
"""


def lead(n_sents, k=3):
    return list(range(min(k, n_sents)))


def topk_in_order(scores, k=3):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return sorted(order[:k])


def trigrams(sentence):
    words = sentence.split()
    return [' '.join(words[i:i + 3]) for i in range(len(words) - 2)]


def topk_trigram_blocking(scores, sentences, k=3):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    chosen, seen = [], set()
    for i in order:
        tri = trigrams(sentences[i])
        if any(t in seen for t in tri):
            continue
        seen.update(tri)
        chosen.append(i)
        if len(chosen) >= k:
            break
    return chosen
