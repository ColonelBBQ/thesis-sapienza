# Ambiguity-Aware Classification of Cloud Computing Requirements

Code and datasets for the MSc thesis
*Ambiguity-Aware Classification of Cloud Computing Requirements Using BERT and
Large Language Models* (Claudio Giannini, Sapienza University of Rome).

It studies whether *ambiguity* in natural-language cloud requirements affects
their automated classification, using the **AI-CRAS** dataset
(Casalicchio & Cotumaccio, 2015):

- **RQ1** — does ambiguity degrade a fine-tuned BERT classifier? (No measurable effect.)
- **RQ2** — does LLM-based disambiguation of ambiguous sentences improve BERT? (No.)
- **RQ3** — can LLMs used directly as classifiers match BERT? (No: fine-tuned BERT wins.)

## Repository layout

```
code/        # Reusable Python package (BERT pipeline, evaluation, experiments)
scripts/     # CLI tools (LLM annotation / agreement, LLM classification, disambiguation)
notebooks/   # run_final.py (BERT re-run) and colab_run.ipynb (GPU runner)
data/        # Datasets (see data/README.md for provenance)
results/     # Final experimental results (10 seeds per condition) + agreement report
paper/       # LaTeX thesis (compile with latexmk + biber)
tests/       # Pytest suite
references/  # Reference papers (PDF)
```

## Setup

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt        # runtime + test dependencies
pip install -e .                        # optional: install the `code` package
python -m pytest tests/                 # run the test suite
```

The BERT code runs on Apple Silicon (MPS), CUDA, or CPU and selects the best
available device automatically.

## Reproducing the experiments

### BERT experiments (RQ1/RQ2, the numbers reported in the thesis)

The canonical runner is `notebooks/run_final.py`. It runs one condition at a time
over 10 seeds, resumes after interruptions, and can back up each seed's results to
Google Drive when given `--drive-dir`:

```bash
python notebooks/run_final.py --condition baseline      --seeds 10
python notebooks/run_final.py --condition no_ambiguity --seeds 10
python notebooks/run_final.py --condition deambiguified --seeds 10
```

`--condition` is one of `baseline` (2,166 sentences), `no_ambiguity` (1,854, RQ1),
or `deambiguified` (2,166, RQ2). Results are written as `*_epoch_metrics.csv`
(accumulated across seeds), `*_results_validation.xlsx`, and `*_progress.txt`
into `--artifacts-dir` (default `artifacts/`).

To run on a GPU (e.g., Google Colaboratory), open `notebooks/colab_run.ipynb`,
upload the files it requests, and run its cells; `run_final.py` resumes any seeds
already completed.

The results currently reported in the thesis are in `results/`.

### LLM annotations and predictions (require an OpenAI API key)

The scripts below call the OpenAI API (they read `OPENAI_API_KEY` or
`SECRET_KEY_OPEN_AI` from the environment or a local `.env` file; see
`.env.example`):

```bash
# Ambiguity re-annotation + human-vs-LLM agreement (Chapter 3)
python scripts/annotate_ambiguity.py            # -> data/llm_ambiguity_annotation.csv
python scripts/analyze_annotation_agreement.py  # -> results/annotation_agreement_report.md

# LLM disambiguation of ambiguous sentences (RQ2 input)
python scripts/deambiguify_dataset.py --input data/dataframe.csv --output data/dataframe_deambiguified.csv

# LLM as a direct classifier (RQ3)
python scripts/classify_with_llm.py --mode zero-shot
python scripts/classify_with_llm.py --mode few-shot
```

All LLM runs use temperature 0.

## Data

The base corpus is derived from the AI-CRAS dataset (public RFP documents). See
`data/README.md` for the file-by-file provenance and notes on the annotation.

## Compiling the thesis

```bash
cd paper
latexmk -pdf main.tex     # requires a LaTeX distribution with biber
```

## License

Code is released under the MIT License (see `LICENSE`). The datasets and the
LaTeX thesis text are provided for academic use; please credit the original
AI-CRAS work (see `references/ai-cras.pdf`) if you reuse the corpus.
