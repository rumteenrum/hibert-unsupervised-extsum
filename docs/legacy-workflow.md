# Legacy thesis workflow

These scripts preserve the original MSc thesis experiments. The current research workflow is described in the [main README](../README.md) and implemented in `analysis/`. Legacy score combinations and thesis tables are not verified evidence for the current method.

## Method

Every sentence in a document gets four scores:

| | Criterion | Computed from |
|---|---|---|
| r1 | how well HIBERT predicts the sentence when it is masked | model log-probabilities (as in STAS) |
| r2 | how much attention the sentence receives from the others | sentence-level attention (as in STAS) |
| r3 | relevance to the document | a PMI-style score from token statistics, no language model |
| r4 | intended redundancy penalty against preceding sentences | same kind of score, against preceding sentences |

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

