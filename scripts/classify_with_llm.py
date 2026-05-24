from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

# --- Constants ---

DEFAULT_INPUT = "data/dataframe.csv"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
DEFAULT_OUTPUT_DIR = "data"
DEFAULT_CHUNK_SIZE = 10

TARGET_LIST = [
    "compute",
    "data_handling",
    "network",
    "security_compliance",
    "management_monitoring",
    "cloud_service_essentials",
]

DEFAULT_DROP_COLUMNS = ["source", "goal", "ambiguity_type", "comments"]

CLS_DROP_COLUMNS = ["source", "goal", "ambiguity_type", "ambuiguity", "comments"]

CATEGORY_DEFINITIONS = {
    "compute": "Processing resources: virtual machines, container orchestration, serverless computing, CPU/GPU and memory specifications.",
    "data_handling": "Data storage and management: databases (SQL, NoSQL), storage systems, caching and data optimisation.",
    "network": "Connectivity and traffic management: firewalls, load balancers, VPNs, DNS, latency and bandwidth specifications.",
    "security_compliance": "Protection and regulatory adherence: encryption, identity management, certifications (ISO 27001, GDPR), access control.",
    "management_monitoring": "Operational oversight: dashboards, logging, alerting, cost management, auditing.",
    "cloud_service_essentials": "General cloud properties: elasticity, multitenancy, pay-per-use billing, availability zones, SLAs.",
}


# --- Data Loading ---

def load_and_prepare_data(input_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    df["_temp_lower"] = df["description"].str.lower()
    dedup_mask = ~df["_temp_lower"].duplicated(keep="first")
    df = df.loc[dedup_mask].copy()
    df = df.drop(columns=["_temp_lower"]).reset_index(drop=True)

    baseline_df = df.drop(columns=DEFAULT_DROP_COLUMNS, errors="ignore").reset_index(drop=True).copy()
    baseline_df[TARGET_LIST] = baseline_df[TARGET_LIST].astype(int)
    return baseline_df


# --- Few-Shot Example Selection ---

def _label_count(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_LIST].sum(axis=1).astype(int)


def _best_candidates(candidates: pd.DataFrame, used: set[int], n: int, seed: int) -> pd.DataFrame:
    available = candidates[~candidates.index.isin(used)].copy()
    if len(available) == 0:
        return available.iloc[:0]
    available["_n_labels"] = _label_count(available)
    available = available.sort_values("_n_labels")
    n_pick = min(n, len(available))
    if n_pick <= 1:
        return available.head(n_pick).drop(columns=["_n_labels"])
    return available.head(n_pick * 3).sample(n=n_pick, random_state=seed).drop(columns=["_n_labels"])


def select_few_shot_examples(
    train_df: pd.DataFrame,
    examples_per_category: int = 2,
    seed: int = 42,
) -> list[dict]:
    selected: list[dict] = []
    used_indices: set[int] = set()
    has_ambiguity = "ambuiguity" in train_df.columns

    # Compute non-req proportion from the pool and match it
    pool_label_counts = _label_count(train_df)
    non_req_rate = float((pool_label_counts == 0).mean())
    total_labeled = examples_per_category * len(TARGET_LIST)
    non_req_count = max(2, round(total_labeled * non_req_rate / (1 - non_req_rate)))
    non_req_count = min(non_req_count, (pool_label_counts == 0).sum())

    # Among labeled sentences in pool, compute how many labeled examples should be single-label
    labeled_mask = pool_label_counts > 0
    single_label_rate = float((pool_label_counts[labeled_mask] == 1).mean()) if labeled_mask.any() else 0.5
    total_single_labeled = round(total_labeled * single_label_rate)

    # Tier 1: pick 1 clean single-label prototype per category
    picked_per_category: dict[str, pd.DataFrame] = {}
    for label in TARGET_LIST:
        candidates = train_df[train_df[label] == 1]
        if has_ambiguity:
            unambiguous = candidates[candidates["ambuiguity"] == 0]
            single = unambiguous[_label_count(unambiguous) == 1]
            picked = _best_candidates(single, used_indices, 1, seed)
        else:
            picked = _best_candidates(candidates, used_indices, 1, seed)
        picked_per_category[label] = picked

    # Track which indices have been used by Tier 1
    for picked in picked_per_category.values():
        for _, row in picked.iterrows():
            used_indices.add(row.name)

    # Tier 2: fill remaining slots, matching the pool's single/multi ratio
    remaining_single = total_single_labeled - len(TARGET_LIST)  # subtract 6 already picked
    remaining_multi = total_labeled - total_single_labeled
    tier2_labels: list[tuple[str, str]] = []
    for label in TARGET_LIST:
        if remaining_single > 0:
            tier2_labels.append((label, "single"))
            remaining_single -= 1
        else:
            tier2_labels.append((label, "multi"))
            remaining_multi -= 1

    for label, kind in tier2_labels:
        candidates = train_df[train_df[label] == 1]
        if has_ambiguity:
            unambiguous = candidates[candidates["ambuiguity"] == 0]
            if kind == "single":
                pool_for_tier2 = unambiguous[_label_count(unambiguous) == 1]
            else:
                pool_for_tier2 = unambiguous[_label_count(unambiguous) > 1]
                if len(pool_for_tier2) == 0:
                    pool_for_tier2 = unambiguous  # fallback
            extra = _best_candidates(pool_for_tier2, used_indices, 1, seed)
        else:
            extra = _best_candidates(candidates, used_indices, 1, seed)

        if len(extra) > 0:
            picked_per_category[label] = pd.concat([picked_per_category[label], extra])
            for _, row in extra.iterrows():
                used_indices.add(row.name)

    for label in TARGET_LIST:
        picked = picked_per_category[label]
        for _, row in picked.iterrows():
            labels = [l for l in TARGET_LIST if row[l] == 1]
            selected.append({"sentence": row["description"], "labels": labels})
            used_indices.add(row.name)

    # Non-requirement examples in proportion to pool's distribution
    if non_req_count > 0 and has_ambiguity:
        non_req = train_df[_label_count(train_df) == 0]
        non_req = non_req[non_req["ambuiguity"] == 0]
        non_req_available = non_req[~non_req.index.isin(used_indices)]
        n_pick = min(non_req_count, len(non_req_available))
        if n_pick > 0:
            picked_nr = non_req_available.sample(n=n_pick, random_state=seed) if n_pick > 1 else non_req_available.iloc[[0]]
            for _, row in picked_nr.iterrows():
                selected.append({"sentence": row["description"], "labels": []})
                used_indices.add(row.name)

    return selected


# --- Prompt Building ---

def _format_category_definitions() -> str:
    lines = []
    for label in TARGET_LIST:
        display = label.replace("_", " ").title()
        lines.append(f"- {display}: {CATEGORY_DEFINITIONS[label]}")
    return "\n".join(lines)


def _format_few_shot_examples(examples: list[dict]) -> str:
    if not examples:
        return ""
    lines = ["LABELED EXAMPLES:"]
    for ex in examples:
        labels_str = json.dumps(ex["labels"])
        lines.append(f'  Sentence: "{ex["sentence"]}"')
        lines.append(f"  Labels: {labels_str}")
        is_req = len(ex["labels"]) > 0
        lines.append(f"  Is Requirement: {json.dumps(is_req)}")
        lines.append("")
    return "\n".join(lines)


def _build_sentences_json(chunk: pd.DataFrame) -> str:
    items = []
    for idx, (_, row) in enumerate(chunk.iterrows()):
        items.append({"id": idx, "sentence": row["description"]})
    return json.dumps(items, ensure_ascii=False, indent=2)


def build_zero_shot_prompt(chunk: pd.DataFrame) -> str:
    sentences_json = _build_sentences_json(chunk)

    return (
        "You are a cloud requirement classifier. Your task is to classify each sentence "
        "into zero or more of the following cloud service categories.\n\n"
        "CATEGORIES:\n"
        f"{_format_category_definitions()}\n\n"
        "RULES:\n"
        "- For each sentence, determine which categories apply (can be zero, one, or multiple).\n"
        "- A sentence is a 'requirement' if it gets at least one label; otherwise it is a 'non-requirement'.\n"
        "- If a sentence does not clearly belong to any category, return an empty labels list.\n"
        "- Output MUST be valid JSON only. No explanation, no markdown, no extra text.\n\n"
        "OUTPUT FORMAT:\n"
        "Return a JSON array with one object per sentence:\n"
        '  {"id": 0, "labels": ["compute", "network"], "is_requirement": true}\n'
        '  {"id": 1, "labels": [], "is_requirement": false}\n\n'
        "The id field must match the id in the input.\n\n"
        "SENTENCES TO CLASSIFY:\n"
        f"{sentences_json}"
    )


def build_few_shot_prompt(chunk: pd.DataFrame, examples: list[dict]) -> str:
    sentences_json = _build_sentences_json(chunk)

    return (
        "You are a cloud requirement classifier. Below are labeled examples showing how to classify sentences "
        "into cloud service categories.\n\n"
        f"{_format_few_shot_examples(examples)}\n"
        "CATEGORIES:\n"
        f"{_format_category_definitions()}\n\n"
        "RULES:\n"
        "- For each sentence, determine which categories apply (can be zero, one, or multiple).\n"
        "- A sentence is a 'requirement' if it gets at least one label; otherwise it is a 'non-requirement'.\n"
        "- If a sentence does not clearly belong to any category, return an empty labels list.\n"
        "- Output MUST be valid JSON only. No explanation, no markdown, no extra text.\n\n"
        "OUTPUT FORMAT:\n"
        "Return a JSON array with one object per sentence:\n"
        '  {"id": 0, "labels": ["compute", "network"], "is_requirement": true}\n'
        '  {"id": 1, "labels": [], "is_requirement": false}\n\n'
        "The id field must match the id in the input.\n\n"
        "SENTENCES TO CLASSIFY:\n"
        f"{sentences_json}"
    )


def build_repair_prompt(input_sentences: str, previous_output: str, error_message: str) -> str:
    return (
        "You previously returned an invalid JSON batch. Fix it now.\n\n"
        f"Validation error: {error_message}\n\n"
        "Return ONLY valid JSON. The output must be a JSON array with one object per sentence.\n"
        'Each object: {"id": <int>, "labels": [<string>, ...], "is_requirement": <bool>}\n\n'
        "Input sentences (JSON):\n"
        f"{input_sentences}\n\n"
        "Your previous output:\n"
        f"{previous_output}"
    )


# --- API Calling ---

def call_openai_chat_completion(system_prompt: str, user_prompt: str, model: str, api_key: str, timeout: int = 120) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed: {error.code} {error.reason}\n{details}") from error

    try:
        return response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"Unexpected API response structure: {response_data}") from error


# --- Label Normalization ---

_LABEL_ALIASES: dict[str, str] = {}
for _label in TARGET_LIST:
    _LABEL_ALIASES[_label] = _label
    _LABEL_ALIASES[_label.replace("_", " ")] = _label
    _LABEL_ALIASES[_label.replace("_", " ").title()] = _label
    _LABEL_ALIASES[_label.replace("_", " ").lower()] = _label
    _LABEL_ALIASES[_label.replace("_", " ").upper()] = _label
    _LABEL_ALIASES[_label.replace("_", "-")] = _label


def _normalize_label(lab: str) -> str:
    result = lab.strip()
    if result in _LABEL_ALIASES:
        return _LABEL_ALIASES[result]
    lowered = result.lower()
    if lowered in _LABEL_ALIASES:
        return _LABEL_ALIASES[lowered]
    raise ValueError(f"Unknown label: {lab!r}")


# --- JSON Parsing ---

def normalize_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[7:]
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
    if stripped.endswith("```"):
        stripped = stripped[:-3].strip()
    return stripped.strip()


def parse_output_json(text: str, expected_count: int) -> list[dict]:
    normalized = normalize_json_text(text)
    parsed = json.loads(normalized)

    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON array, got {type(parsed).__name__}")

    if len(parsed) != expected_count:
        raise ValueError(f"Expected {expected_count} objects, got {len(parsed)}")

    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError(f"Expected dict items, got {type(item).__name__}")
        if "id" not in item:
            raise ValueError("Missing required field: id")
        if "labels" not in item:
            raise ValueError("Missing required field: labels")
        if "is_requirement" not in item:
            raise ValueError("Missing required field: is_requirement")
        if not isinstance(item["labels"], list):
            raise ValueError(f"labels must be a list, got {type(item['labels']).__name__}")
        item["labels"] = [_normalize_label(l) for l in item["labels"]]

    return parsed


def get_valid_json_from_model(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    expected_count: int,
    max_retries: int,
    retry_sleep_seconds: float,
    input_sentences_json: str,
) -> list[dict]:
    raw_output = ""
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_output = call_openai_chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                api_key=api_key,
            )
            return parse_output_json(raw_output, expected_count=expected_count)
        except Exception as error:
            last_error = error
            if attempt < max_retries:
                print(f"  Retry {attempt}/{max_retries}: {error}", flush=True)
                time.sleep(retry_sleep_seconds * attempt)
                user_prompt = build_repair_prompt(
                    input_sentences=input_sentences_json,
                    previous_output=raw_output,
                    error_message=str(error),
                )
            else:
                raise RuntimeError(f"Chunk failed after {max_retries} attempts") from error

    raise RuntimeError("Chunk failed unexpectedly") from last_error


# --- Classification Pipeline ---

def classify_batch(
    chunk: pd.DataFrame,
    model: str,
    api_key: str,
    mode: str,
    examples: list[dict] | None = None,
    max_retries: int = 3,
    retry_sleep_seconds: float = 2.0,
) -> list[dict]:
    system_prompt = "You output only valid JSON. No explanations."

    if mode == "few-shot" and examples:
        user_prompt = build_few_shot_prompt(chunk, examples)
    else:
        user_prompt = build_zero_shot_prompt(chunk)

    input_json = _build_sentences_json(chunk)
    return get_valid_json_from_model(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        api_key=api_key,
        expected_count=len(chunk),
        max_retries=max_retries,
        retry_sleep_seconds=retry_sleep_seconds,
        input_sentences_json=input_json,
    )


def predictions_to_df(chunk: pd.DataFrame, predictions: list[dict]) -> pd.DataFrame:
    rows = []

    for pred in predictions:
        pred_id = pred["id"]
        row = chunk.iloc[pred_id]

        pred_labels = pred["labels"]
        pred_is_req = pred["is_requirement"]
        true_labels = [l for l in TARGET_LIST if row[l] == 1]
        true_is_req = len(true_labels) > 0

        label_vector = [1 if l in pred_labels else 0 for l in TARGET_LIST]
        true_vector = [int(row[l]) for l in TARGET_LIST]

        rows.append({
            "description": row["description"],
            "true_labels": json.dumps(true_labels),
            "predicted_labels": json.dumps(pred_labels),
            "true_is_requirement": true_is_req,
            "predicted_is_requirement": pred_is_req,
            **{f"true_{l}": true_vector[i] for i, l in enumerate(TARGET_LIST)},
            **{f"pred_{l}": label_vector[i] for i, l in enumerate(TARGET_LIST)},
        })

    return pd.DataFrame(rows)


def run_classification(
    df: pd.DataFrame,
    model: str,
    api_key: str,
    mode: str,
    output_path: str | Path,
    examples: list[dict] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_retries: int = 3,
    retry_sleep_seconds: float = 2.0,
    limit_chunks: int | None = None,
) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_predictions: list[pd.DataFrame] = []
    total_sentences = len(df)
    total_chunks = min(limit_chunks, (total_sentences + chunk_size - 1) // chunk_size) if limit_chunks else (total_sentences + chunk_size - 1) // chunk_size

    start_time = time.monotonic()
    sentences_done = 0

    for chunk_index, chunk in enumerate(df.groupby(df.index // chunk_size), start=1):
        chunk = chunk[1].reset_index(drop=True)

        if limit_chunks is not None and chunk_index > limit_chunks:
            break

        elapsed = time.monotonic() - start_time
        pct = sentences_done / total_sentences * 100

        if chunk_index > 1 and sentences_done > 0:
            avg_per_chunk = elapsed / (chunk_index - 1)
            eta = avg_per_chunk * (total_chunks - chunk_index + 1)
            eta_str = f" | ~{eta:.0f}s remaining"
        else:
            eta_str = ""

        print(f"Chunk {chunk_index}/{total_chunks} ({len(chunk)} sentences) | {sentences_done}/{total_sentences} ({pct:.0f}%) | {elapsed:.1f}s elapsed{eta_str}", flush=True)

        predictions = classify_batch(
            chunk=chunk,
            model=model,
            api_key=api_key,
            mode=mode,
            examples=examples,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )

        pred_df = predictions_to_df(chunk, predictions)
        all_predictions.append(pred_df)
        sentences_done += len(chunk)

    total_time = time.monotonic() - start_time
    result = pd.concat(all_predictions, ignore_index=True)
    result.to_csv(output_path, index=False)
    print(f"\nTotal time: {total_time:.0f}s | {len(all_predictions)}/{total_chunks} chunks | {len(result)} predictions saved to {output_path}")
    return result


# --- Evaluation ---

def evaluate_llm_predictions(predictions_df: pd.DataFrame) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        hamming_loss,
        jaccard_score,
    )

    targets = predictions_df[[f"true_{l}" for l in TARGET_LIST]].values
    outputs = predictions_df[[f"pred_{l}" for l in TARGET_LIST]].values

    # Binary requirement metrics
    true_req = predictions_df["true_is_requirement"].values
    pred_req = predictions_df["predicted_is_requirement"].values
    cm = confusion_matrix(true_req, pred_req)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    req_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    req_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    req_f1 = 2 * req_precision * req_recall / (req_precision + req_recall) if (req_precision + req_recall) > 0 else 0.0

    # Multi-label metrics
    exact_match = float(accuracy_score(targets, outputs))
    hl = float(hamming_loss(targets, outputs))
    js = float(jaccard_score(targets, outputs, average="macro", zero_division=0))

    # Micro
    micro_tp = int((outputs * targets).sum())
    micro_fp = int((outputs * (1 - targets)).sum())
    micro_fn = int(((1 - outputs) * targets).sum())
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Macro (per-label)
    per_label = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    for i, label in enumerate(TARGET_LIST):
        tp_i = int((outputs[:, i] * targets[:, i]).sum())
        fp_i = int((outputs[:, i] * (1 - targets[:, i])).sum())
        fn_i = int(((1 - outputs[:, i]) * targets[:, i]).sum())
        p = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0.0
        r = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_label[label] = {"precision": p, "recall": r, "f1": f1}
        macro_precision += p
        macro_recall += r
        macro_f1 += f1

    n = len(TARGET_LIST)
    macro_precision /= n
    macro_recall /= n
    macro_f1 /= n

    return {
        "n_samples": len(predictions_df),
        "exact_match_accuracy": exact_match,
        "hamming_loss": hl,
        "jaccard_score": js,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "requirement_precision": req_precision,
        "requirement_recall": req_recall,
        "requirement_f1": req_f1,
        "requirement_confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "per_label": per_label,
    }


def print_evaluation(eval_result: dict) -> None:
    print("\n" + "=" * 60)
    print("LLM CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated: {eval_result['n_samples']}")
    print(f"  Exact Match Accuracy: {eval_result['exact_match_accuracy']:.4f}")
    print(f"  Hamming Loss:         {eval_result['hamming_loss']:.4f}")
    print(f"  Jaccard Score:        {eval_result['jaccard_score']:.4f}")
    print(f"  Micro F1:             {eval_result['micro_f1']:.4f}")
    print(f"  Macro F1:             {eval_result['macro_f1']:.4f}")
    print(f"  Requirement F1:       {eval_result['requirement_f1']:.4f}")
    print()
    print("  Per-label F1:")
    for label in TARGET_LIST:
        p = eval_result["per_label"][label]
        display = label.replace("_", " ").title()
        print(f"    {display:<30} F1={p['f1']:.4f}")
    print("=" * 60)


def sanitize_model_name(model: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip()).strip("._-") or "unknown"


# --- CLI ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-based cloud requirement classification (RQ3).")
    parser.add_argument("--mode", required=True, choices=["zero-shot", "few-shot"], help="Classification mode")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output", default=None, help="Output CSV path for predictions (default: data/llm_{mode}_{model}_predictions.csv)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Sentences per API call")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per chunk")
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0, help="Backoff between retries")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Process only the first N chunks")
    parser.add_argument("--example-pool-size", type=int, default=200, help="Number of rows reserved for few-shot example selection (first N rows)")
    parser.add_argument("--eval-only", action="store_true", help="Skip classification, evaluate existing predictions CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.eval_only:
        raise RuntimeError("OPENAI_API_KEY environment variable is required")

    safe_model = sanitize_model_name(args.model)
    if args.output is None:
        args.output = str(Path(DEFAULT_OUTPUT_DIR) / f"llm_{args.mode.replace('-', '_')}_{safe_model}_predictions.csv")

    if args.eval_only:
        if not Path(args.output).exists():
            raise FileNotFoundError(f"Predictions file not found: {args.output}")
        predictions_df = pd.read_csv(args.output)
        eval_result = evaluate_llm_predictions(predictions_df)
        print_evaluation(eval_result)
        return

    print(f"Loading data from {args.input}...")
    df = load_and_prepare_data(args.input)
    print(f"Loaded {len(df)} sentences")

    if args.mode == "few-shot":
        pool_size = min(args.example_pool_size, len(df))
        example_pool_df = df.iloc[:pool_size]
        classify_df = df.iloc[pool_size:].reset_index(drop=True)
        if len(classify_df) == 0:
            raise RuntimeError(f"Example pool ({pool_size}) covers the entire dataset — nothing left to classify. Reduce --example-pool-size.")
        examples = select_few_shot_examples(example_pool_df, examples_per_category=2)
        print(f"Example pool: {len(example_pool_df)} rows | Selected {len(examples)} few-shot examples")
    else:
        classify_df = df
        examples = []

    classify_df = classify_df.drop(columns=["ambuiguity"], errors="ignore")
    total_chunks = (len(classify_df) + args.chunk_size - 1) // args.chunk_size

    print(f"\nClassifying ({args.mode}): {len(classify_df)} sentences in {total_chunks} chunk(s) of up to {args.chunk_size}")
    print(f"Model: {args.model} | Output: {args.output}")
    print()

    predictions_df = run_classification(
        df=classify_df,
        model=args.model,
        api_key=api_key,
        mode=args.mode,
        output_path=args.output,
        examples=examples,
        chunk_size=args.chunk_size,
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        limit_chunks=args.limit_chunks,
    )

    eval_result = evaluate_llm_predictions(predictions_df)
    print_evaluation(eval_result)


if __name__ == "__main__":
    main()
