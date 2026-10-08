# Directed attention for unsupervised extractive summarization

This repository studies whether a hierarchical transformer's existing sentence-level attention can improve extractive summaries by distinguishing attention from earlier and later sentences. It retains the model's salience ranking and adds directional centrality, without training another sentence-similarity model.

The current workflow supports **STAS and HIBERT**, cached model outputs, controlled ranking comparisons and paired statistical evaluation. It builds on MSc thesis work at the University of Padova, supervised by Prof. Tomaso Erseghe. The original thesis scripts remain available; current research lives in [`analysis/`](analysis/).

## Method

For a document's sentence-attention graph `G`, `G[j, i]` is attention from source sentence `j` to candidate sentence `i`. The graph has a zero diagonal and normalized rows. Split incoming attention by sentence order:

```text
F_i = sum over j > i of G[j, i]     incoming attention from later sentences
B_i = sum over j < i of G[j, i]     incoming attention from earlier sentences
D_i = F_i + lambda_b * B_i

score_i = z(salience_i) + mu * z(D_i)
```

`salience` is the existing STAS-style ranking, computed from masked-sentence recovery and attention propagation. `z` standardizes scores within each document. The original directed reference uses `lambda_b = -0.5` and `mu = 1`. Summaries select up to three sentences using the same trigram-blocking rule as the comparison baseline.

Inference does not require summary labels or a separately trained summarization model. Reference summaries **are used for validation-based parameter selection and evaluation**.

Direction also introduces a preference for earlier sentences: they have more later sources and fewer earlier sources. A negative earlier-attention weight therefore does not, by itself, establish redundancy removal. Position-only and uniform-attention residual controls investigate this distinction.

## Current research scope

- Original directed STAS and HIBERT have received validation and fixed-setting test evaluation. HIBERT is compared both on a graph restricted to the STAS sentence prefix and on its full available graph.
- Residual-direction variants and a position-only control are separate experiments; they do not replace the original reference.
- A denser validation grid reuses scores for identical document/summary selections to avoid repeated ROUGE computation. It is exploratory work conducted after original test evaluation, not independent confirmation.
- Fresh validation controls, matching-preprocessing PACSUM comparisons and a second dataset remain unfinished. Superiority over prior directional methods and a redundancy mechanism are not established.

The original test set has already been inspected. New variants must not use it for tuning; previously inspected validation samples are not fresh confirmation.

## Repository layout

| Path | Purpose |
|---|---|
| [`analysis/stas.py`](analysis/stas.py), [`analysis/hibert.py`](analysis/hibert.py) | STAS ranking and cached HIBERT outputs |
| [`analysis/directed.py`](analysis/directed.py) | Earlier/later incoming attention |
| [`analysis/run_directed.py`](analysis/run_directed.py), [`analysis/run_directed_hibert.py`](analysis/run_directed_hibert.py) | Original directed validation workflows |
| [`analysis/run_directed_test.py`](analysis/run_directed_test.py) | Fixed original test comparisons with baseline reproduction checks |
| [`analysis/run_directed_residual.py`](analysis/run_directed_residual.py), [`analysis/run_directed_hibert_residual.py`](analysis/run_directed_hibert_residual.py) | Uniform-attention residual controls |
| [`analysis/run_position_control.py`](analysis/run_position_control.py) | Position-only tuning and locked holdout comparison |
| [`analysis/run_directed_dense.py`](analysis/run_directed_dense.py) | Manifest-checked dense grid with shared summary scoring |
| [`analysis/experiment.py`](analysis/experiment.py), [`analysis/rouge.py`](analysis/rouge.py), [`analysis/stats.py`](analysis/stats.py) | Shared evaluation, ROUGE and paired statistics |
| [`tests/`](tests/) | Ranking equivalence and experiment regressions |
| [`fairseq/`](fairseq/) | Vendored historical fairseq implementation used by HIBERT |
| [`docs/legacy-workflow.md`](docs/legacy-workflow.md) | Thesis-era scoring scripts and usage |

## Setup and required artifacts

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Analysis runs on CPU **after model outputs have been extracted**. Extracting masked-sentence probabilities and attention requires pretrained models and GPU inference. Models, datasets and extraction artifacts are not bundled.

Required inputs:

- Sentence-split CNN/DailyMail articles and reference summaries, with identical preprocessing across systems. Evaluation expects `valid.article`, `valid.summary`, `test.article` and `test.summary`; label-oracle evaluation also needs label files.
- STAS score files named `{setting}.{split}.txt`. Directed STAS additionally needs aligned `stas_dump.{split}.npz` attention dumps from the patched extraction workflow.
- HIBERT per-document recovery/attention shards under `valid/` and `test/`. Matching STAS score files define the sentence prefix when required.
- Official **ROUGE-1.5.5**, its data files and Perl modules `XML::DOM` and `DB_File`. Set `ROUGE_HOME` to the directory containing `ROUGE-1.5.5.pl`:

```bash
export ROUGE_HOME=/path/to/ROUGE-1.5.5
```

Vendored fairseq and thesis-era extraction scripts are historical code, not a promise of compatibility with current PyTorch. See the [legacy guide](docs/legacy-workflow.md).

## Evaluation examples

Baseline runners select settings on validation before scoring test:

```bash
python -m analysis.run_baselines --data DATA --out results
python -m analysis.run_stas --stas-dir STAS_OUT --data DATA --out results
python -m analysis.run_hibert_stas --hibert-dir HIBERT_OUT --data DATA --out results
python -m analysis.compare --results results --systems stas hibert_stas
```

Screen original directed STAS on validation using cached scores and attention:

```bash
python -m analysis.run_directed --dump-dir STAS_DUMP --data VALIDATION_DATA --out results/directed_validation
```

Use `--help` on each runner for arguments. Some experiments require previous reports, cached baseline scores or a frozen sample manifest. The dense runner also expects the original shared-workspace artifact layout. These are experiment-specific workflows, rather than end-to-end commands for an arbitrary dataset.

ROUGE uses the original Perl evaluator with `-a -c 95 -m -n 2 -w 1.2`, full-length F1 and one sentence per line. Parameter selection uses the mean of ROUGE-1, ROUGE-2 and ROUGE-L. Comparisons retain per-document selections/scores and report paired bootstrap confidence intervals, permutation tests and sentence-position diagnostics. Preserve sentence ordering and selection rules when reproducing a comparison.

Run tests after configuring ROUGE:

```bash
pytest -q
```

## Prior work and license

The work builds on HIBERT (Zhang, Wei and Zhou, ACL 2019), STAS (Xu et al., Findings of EMNLP 2020), and PMI-based summarization (Padmakumar and He, EACL 2021). PACSUM already uses preceding/following centrality; direction itself is not the novelty claimed here. The question is whether existing asymmetric model attention can supply a useful directional term while retaining its salience ranking.

MIT ([LICENSE](LICENSE)), except for inherited components: fairseq-derived code retains its BSD license ([LICENSE-fairseq](LICENSE-fairseq), [PATENTS](PATENTS)); `pyrougex/google_rouge.py` is Google code under Apache 2.0.
