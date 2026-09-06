#!/usr/bin/env python
"""Final BERT re-run: one condition per invocation, 10 seeds, crash-safe Drive backup."""
from __future__ import annotations

import argparse
import os
import shutil
import sys

CWD = os.getcwd()
if CWD not in sys.path:
    sys.path.insert(0, CWD)

CONDITIONS = ["baseline", "no_ambiguity", "deambiguified"]
N_RUNS = 10

REQUIRED = [
    ("data/dataframe.csv", "upload data/dataframe.csv"),
    ("data/dataframe_deambiguified.csv", "upload data/dataframe_deambiguified.csv"),
    ("code/bert_pipeline.py", "extract code.zip (it must contain code/bert_pipeline.py)"),
]


def build_condition_df(condition: str):
    import pandas as pd

    from code.bert_pipeline import DEFAULT_TARGET_LIST

    target_list = DEFAULT_TARGET_LIST.copy()

    df = pd.read_csv("data/dataframe.csv")
    df["description"] = df["description"].astype(str).str.strip()

    deam = pd.read_csv("data/dataframe_deambiguified.csv")
    deam["description"] = deam["description"].astype(str).str.strip()
    deam["deambiguified_description"] = deam["deambiguified_description"].astype(str).str.strip()

    df["_norm"] = df["description"].str.lower()
    mask = ~df["_norm"].duplicated(keep="first")
    df = df.loc[mask].reset_index(drop=True)
    deam = deam.loc[mask].reset_index(drop=True)
    df = df.drop(columns=["_norm"]).reset_index(drop=True)

    def prepare(d):
        drop_cols = ["source", "goal", "ambiguity_type", "ambuiguity", "comments"]
        out = d.drop(columns=drop_cols, errors="ignore").reset_index(drop=True).copy()
        out[target_list] = out[target_list].astype(int)
        return out

    if condition == "baseline":
        return prepare(df)
    if condition == "no_ambiguity":
        return prepare(df.loc[df["ambuiguity"].eq(0)].copy())
    # deambiguified
    dd = deam.copy()
    dd["description"] = dd["deambiguified_description"]
    dd = dd.drop(columns=["deambiguified_description"], errors="ignore")
    return prepare(dd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final BERT re-run, one condition at a time.")
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--seeds", type=int, default=N_RUNS)
    parser.add_argument("--drive-dir", default=None, help="Drive folder to back up each seed's results")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Where to write results (default: artifacts)")
    parser.add_argument("--dry-run", action="store_true", help="Build the dataset and exit (no training)")
    args = parser.parse_args()

    missing = [hint for path, hint in REQUIRED if not os.path.exists(path)]
    if missing:
        print("MISSING FILES. Please upload / extract these first:")
        for hint in missing:
            print("  -", hint)
        sys.exit(1)

    exp_df = build_condition_df(args.condition)
    print(f"condition={args.condition} rows={len(exp_df)}", flush=True)

    if args.dry_run:
        print("dry-run OK (no training)", flush=True)
        return

    import torch

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device, flush=True)

    from code.bert_experiments import run_experiment
    from code.bert_pipeline import DEFAULT_TARGET_LIST, BertExperimentConfig

    target_list = DEFAULT_TARGET_LIST.copy()

    artifacts_dir = args.artifacts_dir
    os.makedirs(artifacts_dir, exist_ok=True)
    progress_path = os.path.join(artifacts_dir, f"{args.condition}_progress.txt")
    csv_path = os.path.join(artifacts_dir, f"{args.condition}_epoch_metrics.csv")

    done: set[str] = set()
    if os.path.exists(progress_path):
        done = {line.strip() for line in open(progress_path) if line.strip()}
    if done:
        print("resuming, seeds already done:", sorted(done, key=int), flush=True)

    for seed in range(args.seeds):
        if str(seed) in done:
            print(f"seed {seed} already done - skipping", flush=True)
            continue

        run_config = BertExperimentConfig(
            max_len=300,
            train_batch_size=16,
            valid_batch_size=16,
            epochs=5,
            learning_rate=3e-5,
            threshold=0.20,
            seed=seed,
        )
        run_experiment(
            df=exp_df,
            experiment_name=args.condition,
            device=device,
            config=run_config,
            target_list=target_list,
            artifacts_dir=artifacts_dir,
            export_validation_results_flag=True,
            runs_log_path=None,
        )
        with open(progress_path, "a") as f:
            f.write(f"{seed}\n")

        if args.drive_dir:
            os.makedirs(args.drive_dir, exist_ok=True)
            shutil.copy(csv_path, os.path.join(args.drive_dir, f"{args.condition}_epoch_metrics.csv"))
            shutil.copy(progress_path, os.path.join(args.drive_dir, f"{args.condition}_progress.txt"))
        print(f"seed {seed} done [condition={args.condition}]", flush=True)

    print(f"CONDITION '{args.condition}' DONE ({args.seeds} seeds).", flush=True)


if __name__ == "__main__":
    main()
