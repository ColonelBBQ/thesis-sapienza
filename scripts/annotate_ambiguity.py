from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

# --- Constants ---

DEFAULT_INPUT = "data/dataframe.csv"
DEFAULT_OUTPUT = "data/llm_ambiguity_annotation.csv"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
DEFAULT_CHUNK_SIZE = 15

TYPES = {
    1: "lexical",
    2: "syntactic",
    3: "semantic",
    4: "language_error",
    5: "pragmatic",
}

TYPE_RUBRIC = """\
CLASSIFICATION RULES
====================
For EVERY sentence, decide:
  1) is_ambiguous: true ONLY IF the sentence is open to more than one plausible interpretation,
     i.e., you genuinely cannot determine its intended meaning from the sentence itself.
  2) ambiguity_type: the DOMINANT type, using the taxonomy below. Use 0 if is_ambiguous is false.
     Each sentence receives EXACTLY ONE type; if several apply, pick the most salient one.

CORE RULE — EVALUATE THE SENTENCE IN ISOLATION
----------------------------------------------
- Each sentence is annotated ON ITS OWN, with no surrounding document or contract context.
- A sentence is ambiguous ONLY IF it admits more than one plausible interpretation — you cannot
  tell what it means or what it requires.
- If the meaning can be clearly INFERRED from the sentence itself, it is NOT ambiguous, even when
  some operational details (exact numbers, durations, thresholds, values) are not specified.
- Missing numeric detail is NOT ambiguity when the intent is clear.
  Example: "The CSP shall monitor the environment" is NOT ambiguous — the intent (monitoring is
  required) is clear, even though exact metrics are not given.
- Only flag a sentence when a reasonable reader would be genuinely unsure WHAT is being asked.

TAXONOMY (with verbatim examples from the dataset)
--------------------------------------------------
1. LEXICAL ambiguity: a word or phrase conveys multiple meanings depending on context.
   This includes vague, qualitative terms without a quantifiable interpretation:
   "high speed", "adequate", "better performance", "timely", "high availability", "scalable".
   Example: "The database server storage has to be provided on high speed disks (SSD's) for better performance"
   -> "high speed" and "better performance" are qualitative and admit no single quantitative reading.

2. SYNTACTIC ambiguity: the grammatical structure admits more than one valid parse.
   Example: "The solution needs to provide the ability for IBM IT Administrators to automatically provision
   the services via a Web Portal (Self Provisioning), provide metering and billing to provide service
   assurance for maintenance & operations activities"
   -> "provide metering and billing" has no explicit subject; it can attach to the main clause or to the
   subordinate clause, yielding different scopes of obligation.

3. SEMANTIC ambiguity: the logical content supports multiple interpretations even when the syntax is clear.
   Example: "It is the prime responsibility of CSP to ensure continuity of service at all times of the
   Agreement including exit management period (may be one month)"
   -> "at all times" conflicts with "may be one month" without saying which constraint takes precedence.

4. LANGUAGE-ERROR ambiguity: grammatical mistakes or typographical errors obscure the intended meaning.
   Example: "All the equipment's/Devices in the path have to be in HA mode"
   -> the spurious apostrophe in "equipment's" suggests a typo, and the acronym "HA" is undefined.

5. PRAGMATIC ambiguity: the meaning depends on implicit context NOT present in the text such that
   the sentence has multiple plausible readings.
   IMPORTANT: missing operational detail (duration, frequency, limits) is NOT by itself pragmatic
   ambiguity. Only flag as pragmatic if you genuinely cannot determine what the requirement asks for.
   Example: "This will include system maintenance windows"
   -> NOT ambiguous: the intent is clear (maintenance windows must be included). The exact schedule
   is unspecified, but the meaning can be inferred, so this is type 0.

DECISION NOTES
--------------
- LEXICAL vs PRAGMATIC: lexical ambiguity is about WHAT A WORD MEANS (fixed with a number/glossary);
  pragmatic ambiguity is about WHAT THE REQUIREMENT ACTUALLY ASKS FOR (fixed with extra context).
- An undefined acronym like "HA" should be treated as LANGUAGE-ERROR if it obscures the meaning.
- Vague magnitude words ("fast", "high", "adequate", "scalable", "large") without any number are LEXICAL.
- If the sentence reads clearly and specifies concrete, verifiable values, it is NOT ambiguous (type 0).

DO NOT OVER-FLAG
----------------
- The mere presence of a cloud/domain keyword (backup, monitoring, security, firewall, storage,
  load balancer, SLA, encryption, etc.) does NOT make a sentence ambiguous. Those are normal
  technical terms with clear meanings.
- Example: "The proposed backup strategy should be Disk-to-Disk" is NOT ambiguous. The meaning is
  clear: it names a concrete backup strategy. Do not flag it just because it contains "backup".
- A terse but clear requirement is still unambiguous. Only flag sentences that genuinely admit more
  than one interpretation, contain undefined or contradictory constraints, or are missing essential
  context needed to be verifiable.
- When in doubt between flagging and not flagging, prefer NOT flagging (under-flag rather than over-flag).

CALIBRATION
-----------
- Base rate: in this corpus, only about ONE THIRD of the sentences are genuinely ambiguous.
  Do not flag more than roughly that proportion.
- Decision test: a sentence is ambiguous ONLY IF you cannot determine its MEANING, SCOPE, or
  CONSTRAINTS from the sentence alone. If the intent is clear — even without exact numbers —
  it is NOT ambiguous.
- The following sentences are concrete, verifiable requirements and are NOT ambiguous:
  * "The Bidder will be responsible for provisioning of required IT infrastructure as IaaS & PaaS for hosting NRC Applications"
  * "The above environments are to be deployed on the Virtual Private Cloud/Government Community Cloud"
  * "Each of the environments mentioned above should be logically isolated, i.e., separate from the production environment"
  * "Additional charges for Data Ingress or Egress will not be paid by NRC/ Department"
  * "Manage the instances of storage, compute instances, and network environments"
  * "The proposed backup strategy should be Disk-to-Disk"
- Vague magnitude words ("fast", "high", "adequate", "scalable", "large") WITHOUT any number
  are LEXICAL and DO count as ambiguous.
- An undefined acronym that obscures meaning is LANGUAGE-ERROR.
- When in doubt, choose NOT ambiguous.
"""


def load_and_dedup(input_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df["_temp_lower"] = df["description"].str.strip().str.lower()
    df = df.loc[~df["_temp_lower"].duplicated(keep="first")].copy()
    df = df.drop(columns=["_temp_lower"]).reset_index(drop=True)
    if "ambuiguity" not in df.columns:
        raise KeyError("Input CSV is missing the 'ambuiguity' column")
    return df


def _resolve_api_key() -> str:
    for var in ("OPENAI_API_KEY", "SECRET_KEY_OPEN_AI"):
        value = os.environ.get(var)
        if value:
            return value.strip()
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key in ("OPENAI_API_KEY", "SECRET_KEY_OPEN_AI") and value:
                    return value
    raise RuntimeError(
        "OpenAI API key not found. Set OPENAI_API_KEY or SECRET_KEY_OPEN_AI in the environment "
        "or in the project .env file."
    )


# --- Prompt Building ---

def build_chunk_prompt(chunk: pd.DataFrame) -> str:
    sentences = json.dumps(
        [{"id": idx, "sentence": row["description"]} for idx, (_, row) in enumerate(chunk.iterrows())],
        ensure_ascii=False,
        indent=2,
    )
    return (
        "You are an expert annotator of ambiguity in cloud-computing requirement sentences.\n"
        "Your task is to annotate each sentence for ambiguity following a precise taxonomy.\n\n"
        f"{TYPE_RUBRIC}\n\n"
        "OUTPUT FORMAT (MANDATORY)\n"
        "--------------------------\n"
        "Return a JSON array with one object per sentence, in the same order as the input:\n"
        '  {"id": 0, "is_ambiguous": true, "ambiguity_type": 1, "rationale": "brief reason"}\n'
        '  {"id": 1, "is_ambiguous": false, "ambiguity_type": 0, "rationale": ""}\n\n'
        "RULES:\n"
        "- id must match the input id.\n"
        "- ambiguity_type must be an integer in 0-5.\n"
        "- If is_ambiguous is false, ambiguity_type MUST be 0.\n"
        "- If is_ambiguous is true, ambiguity_type MUST be 1-5.\n"
        "- rationale: one short sentence (max ~15 words) naming the ambiguous term or construction.\n"
        "- Output ONLY valid JSON. No explanations, no markdown.\n\n"
        "SENTENCES TO ANNOTATE:\n"
        f"{sentences}"
    )


def build_repair_prompt(input_sentences_json: str, previous_output: str, error_message: str) -> str:
    return (
        "You previously returned an invalid JSON batch. Fix it now.\n\n"
        f"Validation error: {error_message}\n\n"
        "Return ONLY valid JSON: a JSON array with one object per sentence, in input order:\n"
        '  {"id": <int>, "is_ambiguous": <bool>, "ambiguity_type": <int 0-5>, "rationale": <string>}\n'
        "If is_ambiguous is false, ambiguity_type MUST be 0.\n\n"
        "Input sentences (JSON):\n"
        f"{input_sentences_json}\n\n"
        "Your previous output:\n"
        f"{previous_output}"
    )


# --- API Calling ---

def call_openai_chat_completion(prompt: str, model: str, api_key: str, timeout: int = 120) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You output only valid JSON. No explanations."},
            {"role": "user", "content": prompt},
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
        if "id" not in item or not isinstance(item["id"], int):
            raise ValueError("Missing or invalid field: id (must be int)")
        if "is_ambiguous" not in item or not isinstance(item["is_ambiguous"], bool):
            raise ValueError("Missing or invalid field: is_ambiguous (must be boolean)")
        if "ambiguity_type" not in item or not isinstance(item["ambiguity_type"], int):
            raise ValueError("Missing or invalid field: ambiguity_type (must be int)")
        if item["ambiguity_type"] not in TYPES and item["ambiguity_type"] != 0:
            raise ValueError(f"ambiguity_type must be in 0-5, got {item['ambiguity_type']}")
        if item["is_ambiguous"] and item["ambiguity_type"] == 0:
            raise ValueError("is_ambiguous is true but ambiguity_type is 0")
        if not item["is_ambiguous"] and item["ambiguity_type"] != 0:
            raise ValueError("is_ambiguous is false but ambiguity_type is non-zero")
        if "rationale" not in item or not isinstance(item["rationale"], str):
            raise ValueError("Missing or invalid field: rationale (must be string)")

    return parsed


def get_valid_json_from_model(
    prompt: str,
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
            raw_output = call_openai_chat_completion(prompt=prompt, model=model, api_key=api_key)
            return parse_output_json(raw_output, expected_count=expected_count)
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt < max_retries:
                print(f"    Retry {attempt}/{max_retries}: {error}", flush=True)
                time.sleep(retry_sleep_seconds * attempt)
                prompt = build_repair_prompt(
                    input_sentences_json=input_sentences_json,
                    previous_output=raw_output,
                    error_message=str(error),
                )
            else:
                raise RuntimeError(f"Chunk failed after {max_retries} attempts") from error

    raise RuntimeError("Chunk failed unexpectedly") from last_error


# --- Annotation Pipeline ---

def annotate_chunk(chunk: pd.DataFrame, model: str, api_key: str, max_retries: int, retry_sleep_seconds: float) -> list[dict]:
    sentences_json = json.dumps(
        [{"id": idx, "sentence": row["description"]} for idx, (_, row) in enumerate(chunk.iterrows())],
        ensure_ascii=False,
    )
    prompt = build_chunk_prompt(chunk)
    return get_valid_json_from_model(
        prompt=prompt,
        model=model,
        api_key=api_key,
        expected_count=len(chunk),
        max_retries=max_retries,
        retry_sleep_seconds=retry_sleep_seconds,
        input_sentences_json=sentences_json,
    )


def run_annotation(
    df: pd.DataFrame,
    model: str,
    api_key: str,
    output_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_retries: int = 3,
    retry_sleep_seconds: float = 2.0,
    limit_chunks: int | None = None,
) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_descriptions: set[str] = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        done_descriptions = set(existing["description"].astype(str))
        print(f"Resuming: {len(done_descriptions)} sentences already annotated in {output_path}")

    total_sentences = len(df)
    total_chunks = (total_sentences + chunk_size - 1) // chunk_size
    if limit_chunks is not None:
        total_chunks = min(limit_chunks, total_chunks)

    rows: list[dict] = []
    start_time = time.monotonic()
    sentences_done = len(done_descriptions)
    new_write = not output_path.exists()

    def flush(chunk_rows: list[dict]) -> None:
        nonlocal new_write
        if not chunk_rows:
            return
        pd.DataFrame(chunk_rows).to_csv(
            output_path,
            index=False,
            mode="w" if new_write else "a",
            header=new_write,
        )
        new_write = False

    for chunk_index, start in enumerate(range(0, total_sentences, chunk_size), start=1):
        if limit_chunks is not None and chunk_index > total_chunks:
            break
        chunk = df.iloc[start : start + chunk_size].reset_index(drop=True)

        pending = chunk[~chunk["description"].astype(str).isin(done_descriptions)]
        if len(pending) == 0:
            continue

        elapsed = time.monotonic() - start_time
        pct = sentences_done / total_sentences * 100
        if sentences_done > 0:
            avg_per_chunk = elapsed / max(sentences_done - (len(done_descriptions) or 0), 1)
            eta = avg_per_chunk * (total_sentences - sentences_done)
            eta_str = f" | ~{eta:.0f}s remaining"
        else:
            eta_str = ""

        print(
            f"Chunk {chunk_index}/{total_chunks} ({len(pending)} sentences) | "
            f"{sentences_done}/{total_sentences} ({pct:.0f}%) | {elapsed:.1f}s elapsed{eta_str}",
            flush=True,
        )

        predictions = annotate_chunk(
            chunk=pending,
            model=model,
            api_key=api_key,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )

        chunk_rows: list[dict] = []
        for pred in predictions:
            pred_id = pred["id"]
            row = pending.iloc[pred_id]
            chunk_rows.append(
                {
                    "description": row["description"],
                    "human_ambuiguity": int(row["ambuiguity"]),
                    "human_ambiguity_type": int(row["ambiguity_type"]) if "ambiguity_type" in df.columns else "",
                    "llm_is_ambiguous": bool(pred["is_ambiguous"]),
                    "llm_ambiguity_type": int(pred["ambiguity_type"]),
                    "llm_rationale": pred["rationale"],
                }
            )
        flush(chunk_rows)
        rows.extend(chunk_rows)
        done_descriptions.update(cr["description"] for cr in chunk_rows)
        sentences_done += len(chunk_rows)

    total_time = time.monotonic() - start_time
    result = pd.DataFrame(rows)
    print(f"\nTotal time: {total_time:.0f}s | {len(result)} new annotations appended to {output_path}")
    return result


# --- CLI ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-based ambiguity re-annotation to measure agreement with the manual annotation."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path for annotations")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Sentences per API call")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per chunk")
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0, help="Backoff between retries")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Process only the first N chunks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = _resolve_api_key()

    print(f"Loading data from {args.input}...")
    df = load_and_dedup(args.input)
    print(f"Loaded {len(df)} deduplicated sentences")
    print(f"  human-annotated ambiguous: {int(df['ambuiguity'].sum())}")

    total_chunks = (len(df) + args.chunk_size - 1) // args.chunk_size
    if args.limit_chunks is not None:
        total_chunks = min(total_chunks, args.limit_chunks)

    print(f"\nAnnotating: {min(len(df), args.limit_chunks * args.chunk_size) if args.limit_chunks else len(df)} sentences "
          f"in {total_chunks} chunk(s) of up to {args.chunk_size}")
    print(f"Model: {args.model} | Output: {args.output}")
    print()

    run_annotation(
        df=df,
        model=args.model,
        api_key=api_key,
        output_path=args.output,
        chunk_size=args.chunk_size,
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        limit_chunks=args.limit_chunks,
    )


if __name__ == "__main__":
    main()
