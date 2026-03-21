from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import torch

from modules.bert_evaluation import build_validation_results_dataframes, evaluate_model, export_validation_results
from modules.bert_pipeline import (
    BertExperimentConfig,
    DEFAULT_TARGET_LIST,
    build_dataloaders,
    build_model_and_optimizer,
    compute_distribution_classes,
    get_tokenizer,
    train_model,
)


def sanitize_experiment_name(experiment_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", experiment_name.strip())
    return sanitized.strip("._-") or "experiment"


def compute_precision_recall_from_confusion_df(confusion_matrix_df: pd.DataFrame) -> dict[str, float]:
    total_tp = confusion_matrix_df["TP"].sum()
    total_fp = confusion_matrix_df["FP"].sum()
    total_fn = confusion_matrix_df["FN"].sum()
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
    }


def run_experiment(
    df: pd.DataFrame,
    experiment_name: str,
    device: torch.device,
    config: BertExperimentConfig | None = None,
    target_list: list[str] | None = None,
    artifacts_dir: str | Path = "artifacts",
    export_validation_results: bool = True,
) -> dict[str, object]:
    config = config or BertExperimentConfig()
    target_list = target_list or DEFAULT_TARGET_LIST

    safe_name = sanitize_experiment_name(experiment_name)
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_path / f"{safe_name}_best_model.pt"
    results_path = artifacts_path / f"{safe_name}_results_validation.xlsx"

    tokenizer = get_tokenizer(config)
    train_df, val_df, train_loader, val_loader = build_dataloaders(
        df,
        tokenizer=tokenizer,
        config=config,
        target_list=target_list,
    )

    train_distribution, val_distribution = compute_distribution_classes(train_df, val_df, target_list)
    model, optimizer = build_model_and_optimizer(config, num_labels=len(target_list), device=device)
    model, training_losses, validation_losses, epochs_list = train_model(
        n_epochs=config.epochs,
        training_loader=train_loader,
        validation_loader=val_loader,
        model=model,
        optimizer=optimizer,
        best_model_path=str(model_path),
        device=device,
    )

    threshold = 0.20
    evaluation = evaluate_model(
        model,
        val_loader,
        device,
        threshold,
        target_list,
    )

    results_df, confusion_matrix_df = build_validation_results_dataframes(
        val_df,
        model,
        tokenizer,
        config.max_len,
        threshold,
        device,
        target_list,
    )

    if export_validation_results:
        export_summary = export_validation_results_to_disk(
            results_df=results_df,
            confusion_matrix_df=confusion_matrix_df,
            output_path=results_path,
        )
    else:
        export_summary = compute_precision_recall_from_confusion_df(confusion_matrix_df)

    return {
        "experiment_name": experiment_name,
        "safe_name": safe_name,
        "model_path": str(model_path),
        "results_path": str(results_path) if export_validation_results else None,
        "config": config,
        "target_list": target_list,
        "tokenizer": tokenizer,
        "model": model,
        "train_df": train_df,
        "val_df": val_df,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_distribution": train_distribution,
        "val_distribution": val_distribution,
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "epochs_list": epochs_list,
        "evaluation": evaluation,
        "results_df": results_df,
        "confusion_matrix_df": confusion_matrix_df,
        "export_summary": export_summary,
    }


def export_validation_results_to_disk(
    results_df: pd.DataFrame,
    confusion_matrix_df: pd.DataFrame,
    output_path: str | Path,
) -> dict[str, float]:
    return export_validation_results(results_df, confusion_matrix_df, str(output_path))


def run_experiment_suite(
    experiments: dict[str, pd.DataFrame],
    device: torch.device,
    config: BertExperimentConfig | None = None,
    target_list: list[str] | None = None,
    artifacts_dir: str | Path = "artifacts",
    export_validation_results: bool = True,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    results: list[dict[str, object]] = []

    for experiment_name, df in experiments.items():
        summary = run_experiment(
            df=df,
            experiment_name=experiment_name,
            device=device,
            config=config,
            target_list=target_list,
            artifacts_dir=artifacts_dir,
            export_validation_results=export_validation_results,
        )
        results.append(summary)

    summary_rows = []
    for result in results:
        evaluation = result["evaluation"]
        export_summary = result["export_summary"]
        summary_rows.append(
            {
                "experiment_name": result["experiment_name"],
                "train_size": len(result["train_df"]),
                "val_size": len(result["val_df"]),
                "exact_match_accuracy": evaluation["exact_match_accuracy"],
                "hamming_loss": evaluation["hamming_loss"],
                "jaccard_score": evaluation["jaccard_score"],
                "precision": export_summary["precision"],
                "recall": export_summary["recall"],
                "best_model_path": result["model_path"],
                "validation_results_path": result["results_path"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return results, summary_df
