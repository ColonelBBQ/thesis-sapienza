from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

DEFAULT_HUMAN = "data/dataframe.csv"
DEFAULT_LLM = "data/llm_ambiguity_annotation.csv"
DEFAULT_OUTPUT = "artifacts/annotation_agreement_report.md"

TYPE_NAMES = {0: "none", 1: "lexical", 2: "syntactic", 3: "semantic", 4: "language_error", 5: "pragmatic"}


def kappa_interpretation(k: float) -> str:
    if k >= 0.81:
        return "almost perfect"
    if k >= 0.61:
        return "substantial"
    if k >= 0.41:
        return "moderate"
    if k >= 0.21:
        return "fair"
    if k >= 0.0:
        return "slight"
    return "poor"


def build_report(df: pd.DataFrame, output_path: Path) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Annotation Agreement Report (Human vs LLM)")
    add("")
    add(f"- Sentences annotated: **{len(df)}**")
    add(f"- Human ambiguous: **{int(df['human_ambuiguity'].sum())}** ({df['human_ambuiguity'].mean():.1%})")
    add(f"- LLM ambiguous: **{int(df['llm_is_ambiguous'].sum())}** ({df['llm_is_ambiguous'].mean():.1%})")
    add("")

    # Binary agreement
    human_bin = df["human_ambuiguity"].astype(int)
    llm_bin = df["llm_is_ambiguous"].astype(int)
    kappa_bin = cohen_kappa_score(human_bin, llm_bin)
    cm = confusion_matrix(human_bin, llm_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    add("## Binary ambiguity agreement")
    add("")
    add(f"- **Cohen's kappa: {kappa_bin:.3f}** ({kappa_interpretation(kappa_bin)})")
    add(f"- Raw agreement: {(human_bin == llm_bin).mean():.1%}")
    add("")
    add("| | LLM not ambiguous | LLM ambiguous |")
    add("|---|---|---|")
    add(f"| **Human not ambiguous** | {tn} (true neg) | {fp} (LLM over-flags) |")
    add(f"| **Human ambiguous** | {fn} (LLM under-flags) | {tp} (true pos) |")
    add("")

    # Six-way type agreement (only where both annotate; none=0 included)
    human_type = df["human_ambiguity_type"].astype(int)
    llm_type = df["llm_ambiguity_type"].astype(int)
    kappa_type = cohen_kappa_score(human_type, llm_type)
    kappa_type_aggressive = cohen_kappa_score(human_type, llm_type, weights="quadratic")

    add("## Six-way type agreement (0-5, including 'none')")
    add("")
    add(f"- **Cohen's kappa: {kappa_type:.3f}** ({kappa_interpretation(kappa_type)})")
    add(f"- Quadratic-weighted kappa: **{kappa_type_aggressive:.3f}**")
    add("")

    # Type confusion matrix
    labels = [0, 1, 2, 3, 4, 5]
    type_cm = confusion_matrix(human_type, llm_type, labels=labels)
    add("Human rows × LLM columns:")
    add("")
    add("| Human \\ LLM | " + " | ".join(TYPE_NAMES[l] for l in labels) + " |")
    add("|---|" + "---|" * len(labels))
    for i, row in enumerate(type_cm):
        add(f"| **{TYPE_NAMES[labels[i]]}** | " + " | ".join(str(v) for v in row) + " |")
    add("")

    # Among sentences BOTH agree are ambiguous: type agreement
    both_amb = df[(human_bin == 1) & (llm_bin == 1)]
    if len(both_amb) > 0:
        k = cohen_kappa_score(both_amb["human_ambiguity_type"].astype(int), both_amb["llm_ambiguity_type"].astype(int))
        add("## Type agreement (sentences both mark ambiguous)")
        add("")
        add(f"- n = {len(both_amb)}")
        add(f"- **Cohen's kappa: {k:.3f}** ({kappa_interpretation(k)})")
        add("")

    # Disagreement samples
    add("## Sample disagreements")
    add("")

    over = df[(human_bin == 0) & (llm_bin == 1)].head(8)
    add("### LLM over-flags (human: not ambiguous, LLM: ambiguous)")
    add("")
    for _, r in over.iterrows():
        rationale = r["llm_rationale"] if isinstance(r["llm_rationale"], str) else ""
        add(f"- \"{r['description'][:110]}\"")
        add(f"  - LLM type: {TYPE_NAMES.get(int(r['llm_ambiguity_type']), '?')} — {rationale}")
    add("")

    under = df[(human_bin == 1) & (llm_bin == 0)].head(8)
    add("### LLM under-flags (human: ambiguous, LLM: not)")
    add("")
    for _, r in under.iterrows():
        add(f"- \"{r['description'][:110]}\"")
        add(f"  - Human type: {TYPE_NAMES.get(int(r['human_ambiguity_type']), '?')}")
    add("")

    type_swap = df[(human_bin == 1) & (llm_bin == 1) & (human_type != llm_type)].head(8)
    add("### Type substitution (both ambiguous, disagree on type)")
    add("")
    for _, r in type_swap.iterrows():
        h = TYPE_NAMES.get(int(r["human_ambiguity_type"]), "?")
        l = TYPE_NAMES.get(int(r["llm_ambiguity_type"]), "?")
        add(f"- Human: **{h}** → LLM: **{l}** | \"{r['description'][:100]}\"")
    add("")

    return "\n".join(lines)


def _load_human(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["_norm"] = df["description"].str.strip().str.lower()
    df = df.loc[~df["_norm"].duplicated(keep="first")].copy()
    return df[["description", "_norm", "ambuiguity", "ambiguity_type"]]


def _load_llm(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["_norm"] = df["description"].str.strip().str.lower()
    return df[["description", "_norm", "is_ambiguous", "ambiguity_type", "rationale"]]


def merge_annotations(human_path: str | Path, llm_path: str | Path) -> pd.DataFrame:
    human = _load_human(human_path)
    llm = _load_llm(llm_path)
    merged = human.merge(llm, on="_norm", suffixes=("", "_llm"))
    return pd.DataFrame(
        {
            "description": merged["description"],
            "human_ambuiguity": merged["ambuiguity"].astype(int),
            "human_ambiguity_type": merged["ambiguity_type"].astype(int),
            "llm_is_ambiguous": merged["is_ambiguous"],
            "llm_ambiguity_type": merged["ambiguity_type_llm"],
            "llm_rationale": merged["rationale"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute human vs LLM ambiguity annotation agreement.")
    parser.add_argument("--human", default=DEFAULT_HUMAN, help="Human/final annotation CSV (dataframe) path")
    parser.add_argument("--llm", default=DEFAULT_LLM, help="LLM annotation CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Markdown report output path")
    args = parser.parse_args()

    df = merge_annotations(args.human, args.llm)
    report = build_report(df, Path(args.output))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
