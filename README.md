# Thesis Project

This repository contains the code used for a thesis on ambiguity in cloud requirement classification with BERT and LLM-based disambiguation.

## Main Files

- [`main.ipynb`](/Users/claudiogiannini/Desktop/MSc Data Science/thesis/main.ipynb): main experiment notebook for preprocessing, training, evaluation, and analysis.
- [`modules/bert_pipeline.py`](/Users/claudiogiannini/Desktop/MSc Data Science/thesis/modules/bert_pipeline.py): data preparation, tokenization, dataloaders, model definition, and training utilities.
- [`modules/bert_evaluation.py`](/Users/claudiogiannini/Desktop/MSc Data Science/thesis/modules/bert_evaluation.py): validation metrics, reports, ROC data, and export helpers.
- [`modules/bert_experiments.py`](/Users/claudiogiannini/Desktop/MSc Data Science/thesis/modules/bert_experiments.py): high-level experiment runners for comparing different dataframe subsets.

## Data

The main dataset is stored in [`data/dataframe.csv`](/Users/claudiogiannini/Desktop/MSc Data Science/thesis/data/dataframe.csv). Intermediate CSV files may be generated for inspection, but the notebook is designed to work mainly in memory.

## Goal

The workflow starts from a baseline BERT classifier, then supports experiments on:
- removing ambiguous sentences
- removing specific ambiguity categories
- testing LLM-disambiguated versions of the dataset

The modular structure is intended to make repeated experiments on different dataframe subsets easier and more consistent.
