from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, hamming_loss, jaccard_score, roc_curve
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from modules.bert_pipeline import test_model, validate_multilabel


def get_display_target_list(target_list: list[str]) -> list[str]:
    return [label.replace("_", " ") for label in target_list]


def build_prediction_arrays(
    targets: list[list[int]],
    probabilities: list[list[float]],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets_array = np.array(targets)
    probs_array = np.array(probabilities)
    outputs_array = (probs_array >= threshold).astype(int)
    return targets_array, probs_array, outputs_array


def compute_binary_requirement_metrics(
    targets_array: np.ndarray,
    outputs_array: np.ndarray,
) -> dict[str, object]:
    is_requirement_true = targets_array.sum(axis=1) > 0
    is_requirement_pred = outputs_array.sum(axis=1) > 0

    return {
        "true": is_requirement_true,
        "pred": is_requirement_pred,
        "classification_report": classification_report(is_requirement_true, is_requirement_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(is_requirement_true, is_requirement_pred),
    }


def compute_per_label_confusion_matrices(
    targets_array: np.ndarray,
    outputs_array: np.ndarray,
    target_list: list[str],
) -> dict[str, np.ndarray]:
    return {
        label: confusion_matrix(targets_array[:, idx], outputs_array[:, idx])
        for idx, label in enumerate(target_list)
    }


def compute_per_label_accuracies(
    targets_array: np.ndarray,
    outputs_array: np.ndarray,
    target_list: list[str],
) -> dict[str, float]:
    return {
        label: float(np.mean(targets_array[:, idx] == outputs_array[:, idx]))
        for idx, label in enumerate(target_list)
    }


def compute_roc_data(
    targets_array: np.ndarray,
    probs_array: np.ndarray,
    target_list: list[str],
) -> dict[str, object]:
    n_classes = targets_array.shape[1]
    fpr: dict[object, np.ndarray] = {}
    tpr: dict[object, np.ndarray] = {}
    roc_auc: dict[object, float] = {}
    all_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(all_fpr)

    for idx, label in enumerate(target_list):
        fpr[label], tpr[label], _ = roc_curve(targets_array[:, idx], probs_array[:, idx])
        roc_auc[label] = auc(fpr[label], tpr[label])
        mean_tpr += np.interp(all_fpr, fpr[label], tpr[label])

    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    fpr["micro"], tpr["micro"], _ = roc_curve(targets_array.ravel(), probs_array.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
    }


def evaluate_predictions(
    targets: list[list[int]],
    probabilities: list[list[float]],
    threshold: float,
    target_list: list[str],
) -> dict[str, object]:
    display_target_list = get_display_target_list(target_list)
    targets_array, probs_array, outputs_array = build_prediction_arrays(targets, probabilities, threshold)
    binary_metrics = compute_binary_requirement_metrics(targets_array, outputs_array)

    return {
        "threshold": threshold,
        "targets": targets,
        "probabilities": probabilities,
        "targets_array": targets_array,
        "probabilities_array": probs_array,
        "outputs_array": outputs_array,
        "outputs": outputs_array.tolist(),
        "display_target_list": display_target_list,
        "binary_metrics": binary_metrics,
        "exact_match_accuracy": float(accuracy_score(targets_array, outputs_array)),
        "classification_report": classification_report(
            targets_array,
            outputs_array,
            target_names=display_target_list,
            zero_division=0,
        ),
        "hamming_loss": float(hamming_loss(targets_array, outputs_array)),
        "jaccard_score": float(jaccard_score(targets_array, outputs_array, average="macro")),
        "per_label_confusion_matrices": compute_per_label_confusion_matrices(targets_array, outputs_array, target_list),
        "per_label_accuracies": compute_per_label_accuracies(targets_array, outputs_array, target_list),
        "roc_data": compute_roc_data(targets_array, probs_array, target_list),
    }


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float,
    target_list: list[str],
) -> dict[str, object]:
    targets, probabilities = validate_multilabel(model, data_loader, device)
    return evaluate_predictions(targets, probabilities, threshold, target_list)


def calculate_confusion_matrix_elements(true_labels: list[str], predicted_labels: list[str]) -> tuple[int, int, int, int]:
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
        tp, tn, fp, fn = calculate_confusion_matrix_elements(row["true_label"], row["predicted"])
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

    total_tp = confusion_matrix_df["TP"].sum()
    total_fp = confusion_matrix_df["FP"].sum()
    total_fn = confusion_matrix_df["FN"].sum()
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
    }
