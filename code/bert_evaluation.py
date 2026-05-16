from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from code.bert_pipeline import test_model, validate_multilabel
from code.metrics import (
    compute_precision_recall_from_confusion_df,
    evaluate_predictions,
)


def get_display_target_list(target_list: list[str]) -> list[str]:
    return [label.replace("_", " ") for label in target_list]


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float,
    target_list: list[str],
) -> dict[str, object]:
    targets, probabilities = validate_multilabel(model, data_loader, device)
    return evaluate_predictions(targets, probabilities, threshold, target_list)


def binary_presence_elements(true_labels: list[str], predicted_labels: list[str]) -> tuple[int, int, int, int]:
    if len(predicted_labels) > 0:
        if len(true_labels) > 0:
            return 1, 0, 0, 0
        return 0, 0, 1, 0
    if len(true_labels) > 0:
        return 0, 0, 0, 1
    return 0, 1, 0, 0


def build_validation_results_dataframes(
    df: pd.DataFrame,
    model: torch.nn.Module,
    tokenizer: BertTokenizer,
    max_len: int,
    threshold: float,
    device: torch.device,
    target_list: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, object]] = []

    for _, row in df.iterrows():
        sentence = row["description"]
        true_labels = [label for label in target_list if row[label] == 1]
        predicted_labels, probabilities = test_model(
            sentence,
            model,
            tokenizer,
            max_len,
            threshold,
            device,
            target_list,
        )
        results.append(
            {
                "sentence": sentence,
                "true_label": get_display_target_list(true_labels),
                "predicted": get_display_target_list(predicted_labels),
                "correctness": true_labels == predicted_labels,
                "probabilities": [[round(prob, 3) for prob in sublist] for sublist in probabilities],
            }
        )

    results_df = pd.DataFrame(results)
    confusion_matrix_rows: list[dict[str, object]] = []

    for _, row in results_df.iterrows():
        tp, tn, fp, fn = binary_presence_elements(row["true_label"], row["predicted"])
        confusion_matrix_rows.append(
            {
                "predicted": 1 if len(row["predicted"]) > 0 else 0,
                "true_label": 1 if len(row["true_label"]) > 0 else 0,
                "text": row["sentence"],
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
            }
        )

    confusion_matrix_df = pd.DataFrame(confusion_matrix_rows)
    return results_df, confusion_matrix_df


def export_validation_results(
    results_df: pd.DataFrame,
    confusion_matrix_df: pd.DataFrame,
    output_path: str,
) -> dict[str, float]:
    with pd.ExcelWriter(output_path) as writer:
        results_df.to_excel(writer, sheet_name="Results", index=False)
        confusion_matrix_df.to_excel(writer, sheet_name="Confusion Matrix", index=False)

    return compute_precision_recall_from_confusion_df(
        total_tp=float(confusion_matrix_df["TP"].sum()),
        total_fp=float(confusion_matrix_df["FP"].sum()),
        total_fn=float(confusion_matrix_df["FN"].sum()),
    )
