from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import pandas as pd
import torch

from code.bert_evaluation import build_validation_results_dataframes, evaluate_model, export_validation_results
from code.bert_pipeline import (
    BertExperimentConfig,
    DEFAULT_TARGET_LIST,
    build_dataloaders,
    build_model_and_optimizer,
    compute_distribution_classes,
    get_tokenizer,
    set_seed,
    train_model,
)
from code.metrics import compute_precision_recall_from_confusion_df


def sanitize_experiment_name(experiment_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", experiment_name.strip())
    return sanitized.strip("._-") or "experiment"


def _append_run_to_log(row: dict, runs_log_path: Path, config: BertExperimentConfig) -> None:
    flat_config = {
        "config_epochs": config.epochs,
        "config_learning_rate": config.learning_rate,
        "config_train_batch_size": config.train_batch_size,
        "config_max_len": config.max_len,
        "config_test_size": config.test_size,
        "config_seed": config.seed,
        "config_model_name": config.model_name,
    }
    log_row = {"run_timestamp": datetime.now().isoformat(), **row, **flat_config}
    log_df = pd.DataFrame([log_row])
    write_header = not runs_log_path.exists()
    log_df.to_csv(runs_log_path, mode="a", header=write_header, index=False)


def run_experiment(
    df: pd.DataFrame,
    experiment_name: str,
    device: torch.device,
    config: BertExperimentConfig | None = None,
    target_list: list[str] | None = None,
    artifacts_dir: str | Path = "artifacts",
    export_validation_results_flag: bool = True,
    runs_log_path: str | Path | None = "data/runs_log.csv",
) -> dict[str, object]:
    config = config or BertExperimentConfig()
    target_list = target_list or DEFAULT_TARGET_LIST

    set_seed(config.seed)

    safe_name = sanitize_experiment_name(experiment_name)
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_path / f"{safe_name}_best_model.pt"
    results_path = artifacts_path / f"{safe_name}_results_validation.xlsx"
    epoch_metrics_path = artifacts_path / f"{safe_name}_epoch_metrics.csv"

    tokenizer = get_tokenizer(config)
    train_df, val_df, train_loader, val_loader = build_dataloaders(
        df,
        tokenizer=tokenizer,
        config=config,
        target_list=target_list,
    )

    train_distribution, val_distribution = compute_distribution_classes(train_df, val_df, target_list)
    model, optimizer = build_model_and_optimizer(config, num_labels=len(target_list), device=device)
    model, training_losses, validation_losses, epochs_list, epoch_metrics = train_model(
        n_epochs=config.epochs,
        training_loader=train_loader,
        validation_loader=val_loader,
        model=model,
        optimizer=optimizer,
        best_model_path=str(model_path),
        device=device,
        threshold=config.threshold,
        target_list=target_list,
    )

    threshold = config.threshold
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

    if export_validation_results_flag:
        export_summary = export_validation_results(
            results_df=results_df,
            confusion_matrix_df=confusion_matrix_df,
            output_path=str(results_path),
        )
    else:
        export_summary = compute_precision_recall_from_confusion_df(
            total_tp=float(confusion_matrix_df["TP"].sum()),
            total_fp=float(confusion_matrix_df["FP"].sum()),
            total_fn=float(confusion_matrix_df["FN"].sum()),
        )

    epoch_metrics_df = pd.DataFrame(epoch_metrics)
    epoch_metrics_df["seed"] = config.seed
    epoch_metrics_df["experiment_name"] = experiment_name
    epoch_metrics_df["run_timestamp"] = datetime.now().isoformat()
    write_header = not epoch_metrics_path.exists()
    epoch_metrics_df.to_csv(epoch_metrics_path, mode="a", header=write_header, index=False)

    if runs_log_path is not None:
        summary_row = {
            "experiment_name": experiment_name,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "exact_match_accuracy": evaluation["exact_match_accuracy"],
            "hamming_loss": evaluation["hamming_loss"],
            "jaccard_score": evaluation["jaccard_score"],
            "precision": export_summary["precision"],
            "recall": export_summary["recall"],
            "best_model_path": str(model_path),
            "validation_results_path": str(results_path) if export_validation_results_flag else None,
            "epoch_metrics_path": str(epoch_metrics_path),
        }
        _append_run_to_log(summary_row, Path(runs_log_path), config)

    return {
        "experiment_name": experiment_name,
        "safe_name": safe_name,
        "model_path": str(model_path),
        "results_path": str(results_path) if export_validation_results_flag else None,
        "epoch_metrics_path": str(epoch_metrics_path),
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
        "epoch_metrics": epoch_metrics,
        "epoch_metrics_df": epoch_metrics_df,
        "evaluation": evaluation,
        "results_df": results_df,
        "confusion_matrix_df": confusion_matrix_df,
        "export_summary": export_summary,
    }


def run_experiment_suite(
    experiments: dict[str, pd.DataFrame],
    device: torch.device,
    config: BertExperimentConfig | None = None,
    target_list: list[str] | None = None,
    artifacts_dir: str | Path = "artifacts",
    export_validation_results_flag: bool = True,
    runs_log_path: str | Path | None = "data/runs_log.csv",
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
            export_validation_results_flag=export_validation_results_flag,
            runs_log_path=runs_log_path,
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
                "epoch_metrics_path": result["epoch_metrics_path"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return results, summary_df
