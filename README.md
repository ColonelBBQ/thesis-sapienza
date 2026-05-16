# Thesis Project

Ambiguity-Aware Classification of Cloud Computing Requirements Using BERT and Large Language Models.

## Structure

```
├── paper/                     # LaTeX thesis document
│   ├── main.tex               # Master document
│   ├── main-frn.tex           # Frontispiece
│   ├── references.bib         # Bibliography
│   ├── settings/              # Document configuration
│   ├── frontmatter/           # Title page, abstract, quote
│   ├── chapters/              # Numbered thesis chapters
│   ├── backmatter/            # Appendix, acknowledgements
│   ├── figures/               # Figures and images
│   └── output/                # Compiled PDF (gitignored)
├── code/                      # Python source modules
│   ├── bert_pipeline.py       # Data, model, training pipeline
│   ├── bert_evaluation.py     # Evaluation metrics and export
│   ├── bert_experiments.py    # High-level experiment runners
│   └── metrics.py             # Shared metric computations
├── notebooks/                 # Jupyter notebooks
│   └── main.ipynb             # Main experiment notebook
├── scripts/                   # CLI tools
│   └── deambiguify_dataset.py # LLM-based disambiguation
├── data/                      # Datasets
├── artifacts/                 # Experiment outputs
├── references/                # Reference papers
└── tests/                     # Test suite
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Or with pyproject.toml
pip install -e .

# Run the main notebook
jupyter notebook notebooks/main.ipynb

# Disambiguate the dataset with an LLM
export OPENAI_API_KEY="your-key"
python scripts/deambiguify_dataset.py --input data/dataframe.csv --output data/dataframe_deambiguified.csv
```

## Experiments

The main workflow is driven by `notebooks/main.ipynb`, which:
1. Loads the AI-CRAS dataset with ambiguity annotations
2. Prepares subsets (baseline, no-ambiguity, per-ambiguity-type, deambiguified)
3. Trains and evaluates BERT classifiers on each subset
4. Exports results to `artifacts/`

The `code/` package provides reusable components:
- `BertExperimentConfig` — all training hyperparameters in one place
- `run_experiment()` — trains, evaluates, and exports results for a single dataset
- `run_experiment_suite()` — runs multiple experiments and produces a comparison summary

## Running Tests

```bash
pip install pytest
python -m pytest tests/
```
