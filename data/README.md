# Data

This folder contains the datasets used by the experiments in this thesis. All files
are stored as comma-separated values (CSV).

## Provenance

The base corpus is the **AI-CRAS** dataset of Casalicchio and Cotumaccio
(``AI-CRAS: AI-driven Cloud Service Requirement Analysis and Specification'', IEEE
EmpiRE 2015; see `references/ai-cras.pdf`). It is derived from publicly available
Request for Proposal (RFP) documents and supplementary synthetically generated
documents. The repository stores only the sentence-level extracts used here.

## Files

| File | Description |
|---|---|
| `dataframe.csv` | The base corpus (2,267 sentence instances; 2,166 unique after case-insensitive deduplication) with the AI-CRAS `goal`/category labels **and** the ambiguity annotation added by this thesis (`ambuiguity`, `ambiguity_type`). |
| `dataframe_deambiguified.csv` | Parallel corpus in which ambiguous sentences are replaced by LLM-written clarified rewrites (non-ambiguous sentences verbatim); used for RQ2. |
| `llm_ambiguity_annotation.csv` | Independent ambiguity annotation produced by an LLM (GPT-5.4-mini, temperature 0) over the same sentences; used for the reliability analysis in Chapter 3. |
| `llm_{zero,few}_shot_{model}_predictions.csv` | Category/requirement predictions of the LLMs used as direct classifiers (RQ3), aligned row-wise to the deduplicated corpus. |

## Notes

- The `ambuiguity` / `ambiguity_type` columns in `dataframe.csv` reflect the final
  annotation described in Chapter 3 (312 of the 2,166 unique sentences flagged
  ambiguous).
- LLM prediction and annotation files were produced with the OpenAI API at a fixed
  point in time (temperature 0). Re-running the corresponding scripts requires an
  API key (see `scripts/annotate_ambiguity.py`, `scripts/classify_with_llm.py`,
  `scripts/deambiguify_dataset.py`).
- If you redistribute the base sentence text, please also credit the original
  AI-CRAS work (see `references/ai-cras.pdf`).
