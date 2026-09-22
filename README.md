# hibert-unsupervised-extsum

Unsupervised extractive summarization on top of a pretrained hierarchical transformer
(HIBERT). This is the code from my MSc thesis at the University of Padova (2025), which I am
now reworking into a paper.

No summarization labels are used at any point. The model is only pretrained with masked
sentence prediction, and summaries are built by scoring sentences and picking the best ones.

## Method

Every sentence in a document gets four scores:

| | Criterion | Computed from |
|---|---|---|
| r1 | how well HIBERT predicts the sentence when it is masked | model log-probabilities (as in STAS) |
| r2 | how much attention the sentence receives from the others | sentence-level attention (as in STAS) |
| r3 | relevance to the document | a PMI-style score from token statistics, no language model |
| r4 | redundancy with the sentences before it | same kind of score, against preceding sentences |

The scores are min-max normalised within each document and combined as

```
score = α·r1 + β·r2 + γ_rel·r3 − γ_red·r4
```

Sentences shorter than `L` tokens are skipped, and the `k` highest-scoring sentences are
returned in their original order. The scoring scripts evaluate every weight combination on a
0.2 grid with α + β + γ_rel + γ_red = 1 (56 combinations) in a single pass, since the expensive
part, running HIBERT over each document, is cached and reused.

Besides the token-based r3/r4 above, the repository has variants that compute them with
sentence-BERT embeddings, n-gram overlap and TF-IDF.

## Status

Work in progress. The files are the thesis code, renamed to say what they do but otherwise not
yet refactored. The thesis results are being re-evaluated with a stricter protocol
(hyperparameters selected on the validation set only, confidence intervals, and a lead-3
baseline), so I am not reporting numbers here until that is done.

## Contents

| File | Purpose |
|---|---|
| `binarize_data.py` | converts the tokenized text into fairseq's binary format |
| `score_pmi.py` | main method: r1, r2 plus token-based r3/r4; caches model outputs per document |
| `score_pmi_tempcache.py` | same method, older version with a temporary cache |
| `score_sbert.py` | r3/r4 from sentence-BERT cosine similarity |
| `score_sbert_rel_pmi_red.py` | r3 from sentence-BERT, r4 token-based |
| `score_ngram.py` | r3/r4 from n-gram overlap (`n` is set inside the script) |
| `score_tfidf.py` | r3/r4 from TF-IDF cosine similarity |
| `evaluate_rouge.py` | ROUGE-1/2 and ROUGE-Lsum for the selected sentences |
| `oracle_exactly3.py`, `oracle_upto3.py` | greedy ROUGE oracles with exactly / up to 3 sentences |
| `oracle_from_labels.py` | oracle built from the extractive labels |
| `plot_grid.py` | plots of the weight grid results |
| `fairseq/` | modified copy of fairseq 0.5, as used by HIBERT |
| `pyrougex/` | ROUGE implementation used by `evaluate_rouge.py` |

## Setup

```bash
pip install -r requirements.txt
```

`fairseq/` is vendored and imported from the repository root, so run the scripts from there;
it does not need to be installed.

Not included:

- **CNN/DailyMail**, sentence-split and BPE-encoded as in HIBERT, as
  `{training,validation,test}.{article,summary,label}`, plus the HIBERT dictionary.
- **The pretrained HIBERT_M checkpoint** (open-domain + in-domain) from the HIBERT authors.

## Usage

1. Binarize the data:

   ```bash
   python binarize_data.py --source-lang article --target-lang label \
       --trainpref DATA/training --validpref DATA/validation --testpref DATA/test \
       --srcdict DICT --destdir DATA_BIN
   ```

2. Score sentences and write the selections for all 56 weight combinations. The scripts
   reuse fairseq's training entry point, so several optimizer flags are required even though
   nothing is trained:

   ```bash
   python score_pmi.py DATA_BIN \
       --task pretrain_document_modeling -a doc_pretrain_transformer_medium \
       --criterion pretrain_doc_loss --restore-file CHECKPOINT \
       --raw-valid DATA/validation --raw-test DATA/test --save-dir OUT \
       --optimizer adam --lr 0.0001 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
       --warmup-init-lr 1e-07 --min-lr 1e-09 --adam-betas '(0.9, 0.999)' \
       --weight-decay 0.01 --label-smoothing 0.1 --dropout 0.1 --relu-dropout 0.1 \
       --attention-dropout 0.1 --max-sentences 1 --max-sentences-valid 1 \
       --masked-sent-loss-weight 1 --sent-label-weight 0
   ```

   Output: one file per weight combination, `OUT/r_indexes_alpha*_beta*_gamma_rel*_gamma_red*.txt`,
   with a line `Doc i: [sentence indices]` per document. Model outputs are cached in
   `OUT/lprob_cache/`, so an interrupted run resumes where it stopped. The split, `k` and `L`
   are currently set inside `main2()`.

3. Compute ROUGE:

   ```bash
   python evaluate_rouge.py --article-path DATA/test.article --summary-path DATA/test.summary \
       --indexes-dir OUT --out results.tsv
   ```

   Note that the `SELECTED_WEIGHTS` list at the top of the file, when non-empty, overrides
   `--weight` and `--weights-file`.

## Known issues

- The oracle scripts and `plot_grid.py` contain hardcoded paths from the original environment.
- `oracle_from_labels.py` imports a module that is not part of this repository.
- The `score_*.py` scripts still contain fairseq's unused training loop alongside the scoring
  code in `main2()`.

## References

- Xingxing Zhang, Furu Wei, Ming Zhou. *HIBERT: Document Level Pre-training of Hierarchical
  Bidirectional Transformers for Document Summarization.* ACL 2019.
- Shusheng Xu, Xingxing Zhang, Yi Wu, Furu Wei, Ming Zhou. *Unsupervised Extractive
  Summarization by Pre-training Hierarchical Transformers.* Findings of EMNLP 2020. (STAS)

The thesis was supervised by Prof. Tomaso Erseghe.

## License

MIT (see `LICENSE`), except for code derived from other projects:

- `fairseq/`, `binarize_data.py` and the `score_*.py` scripts are based on fairseq and remain
  under its BSD license (`LICENSE-fairseq`, `PATENTS`).
- `pyrougex/google_rouge.py` is Google code under the Apache License 2.0.
