from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "data/dataframe.csv"
DEFAULT_OUTPUT = "data/dataframe_deambiguified.csv"
DEFAULT_CHUNK_SIZE = 10
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")


def detect_columns(df: pd.DataFrame, text_column: str | None, ambiguity_column: str | None) -> tuple[str, str]:
    if text_column and ambiguity_column:
        return text_column, ambiguity_column

    candidate_text_columns = ["description", "original", "sentence", "text"]
    candidate_ambiguity_columns = ["ambuiguity", "ambiguity", "ambiguity_score", "score"]

    resolved_text = text_column
    resolved_ambiguity = ambiguity_column

    if resolved_text is None:
        for column in candidate_text_columns:
            if column in df.columns:
                resolved_text = column
                break

    if resolved_ambiguity is None:
        for column in candidate_ambiguity_columns:
            if column in df.columns:
                resolved_ambiguity = column
                break

    if resolved_text is None:
        raise KeyError(f"Could not find a text column. Available columns: {list(df.columns)}")
    if resolved_ambiguity is None:
        raise KeyError(f"Could not find an ambiguity column. Available columns: {list(df.columns)}")

    return resolved_text, resolved_ambiguity


def build_chunk_prompt(chunk: pd.DataFrame) -> str:
    rows = build_input_rows(chunk)

    return (
        "You are processing a batch of rows from data/dataframe.csv.\n\n"
        "Each row contains a sentence and an ambiguity score.\n\n"
        "========================\n"
        "TASK\n"
        "========================\n"
        "For EVERY row where ambiguity == 1:\n"
        '- Rewrite the sentence into a new field called "deambiguified_description"\n\n'
        "For rows where ambiguity != 1:\n"
        '- Copy the original sentence unchanged into "deambiguified_description" with no grammar fixes, no paraphrasing, and no added context\n\n'
        "========================\n"
        "CORE RULES\n"
        "========================\n"
        "1. Preserve the EXACT original meaning\n"
        "   - Do NOT change intent, claims, or tone\n"
        "   - Only clarify ambiguity when ambiguity == 1\n"
        "   - If the source is ungrammatical or missing necessary context, correct grammar or add the minimum context needed to preserve the intended meaning\n\n"
        "2. Remove ambiguity by ADDING a small, precise clarification\n"
        "   - Prefer adding details in parentheses\n\n"
        "3. Every ambiguous sentence MUST include:\n"
        "   - measurable values OR\n"
        "   - technical specifics OR\n"
        "   - concrete constraints\n\n"
        "========================\n"
        "ALLOWED CLARIFICATIONS (EXAMPLES)\n"
        "========================\n"
        "Use realistic, generic values where needed:\n\n"
        '- "high speed" → "(e.g., 100 MB/s throughput)"\n'
        '- "low latency" → "(e.g., <10 ms response time)"\n'
        '- "fast response" → "(e.g., responds within 50 ms)"\n'
        '- "scalable" → "(e.g., supports up to 1,000 concurrent users)"\n'
        '- "high availability" → "(e.g., 99.99% uptime SLA)"\n'
        '- "secure" → "(e.g., AES-256 encryption at rest, TLS 1.3 in transit)"\n'
        '- "real-time" → "(e.g., <1 second processing delay)"\n'
        '- "large scale" → "(e.g., handles millions of requests per day)"\n'
        '- "accurate" → "(e.g., 95% precision on validation data)"\n'
        '- "reliable" → "(e.g., less than 0.1% failure rate)"\n'
        '- "robust" → "(e.g., remains stable under 10,000 requests per minute)"\n'
        '- "quickly" → "(e.g., within 2 seconds)"\n'
        '- "small" → "(e.g., under 50 MB in size)"\n'
        '- "efficient" → "(e.g., reduces CPU usage by 30%)"\n\n'
        "Grammar / context handling:\n"
        "- If the sentence is grammatically incorrect, fix only the grammar needed for clarity\n"
        "- If the sentence is missing context, add only the smallest context needed to keep the original meaning intact\n"
        "- Do not expand the meaning beyond what the original sentence implies\n\n"
        "Grammar / context examples:\n"
        '- "No ambiguity" → "No ambiguity"\n'
        '- "Language errors ambiguity is a grammatically wrong construction that is nevertheless understood and possibly in different ways due to the error." → "Language errors ambiguity is a grammatically incorrect construction that is nevertheless understood and may be interpreted in different ways because of the error."\n'
        '- "Pragmatic Ambiguity is concerned with the relationship between the interpretations of a sentence and its context." → "Pragmatic Ambiguity is concerned with the relationship between the interpretations of a sentence and its context (e.g., the speaker’s intended meaning and the surrounding situation)."\n'
        '- "Syntactic ambiguity or structural ambiguity in a sentence arises when the sentence can be parsed in more than one way." → "Syntactic ambiguity or structural ambiguity in a sentence arises when the sentence can be parsed in more than one way, producing more than one grammatical structure."\n'
        '- "Semantic ambiguity arises in a sentence when predicate logic of a sentence can lead to multiple interpretations." → "Semantic ambiguity arises in a sentence when the predicate logic of the sentence can lead to multiple interpretations."\n\n'
        "========================\n"
        "STRICT ANTI-LAZINESS RULES\n"
        "========================\n"
        "- NEVER skip a row\n"
        "- NEVER output vague phrases like:\n"
        '  ❌ "better performance"\n'
        '  ❌ "more efficient"\n'
        '  ❌ "high quality"\n'
        "- EVERY rewritten ambiguous sentence must include a concrete clarification\n"
        "- Keep additions minimal and clean\n"
        "- DO NOT repeat the same exact numbers for every row unless appropriate\n"
        "- Use varied but realistic values\n\n"
        "========================\n"
        "OUTPUT FORMAT (MANDATORY)\n"
        "========================\n"
        "Return ONLY valid CSV. No explanations.\n\n"
        "Columns:\n"
        "original,deambiguified_description\n\n"
        "Rules:\n"
        "- Keep original text EXACTLY as given\n"
        "- If ambiguity != 1, deambiguified_description must be identical to original\n"
        "- Maintain row order\n"
        "- Quote every field\n\n"
        "Example:\n"
        '"fast system","fast system (e.g., processes 10,000 requests per second)"\n'
        '"secure platform","secure platform (e.g., AES-256 encryption at rest and TLS 1.3 in transit)"\n'
        '"real-time monitoring","real-time monitoring (e.g., updates within 1 second)"\n'
        '"scalable service","scalable service (e.g., supports 1,000 concurrent users)"\n\n'
        "========================\n"
        "BATCH DISCIPLINE\n"
        "========================\n"
        "- Treat this as part of a larger dataset (thousands of rows)\n"
        "- Be consistent across all rows\n"
        "- Do not degrade quality over the batch\n\n"
        "========================\n"
        "FINAL SELF-CHECK (REQUIRED)\n"
        "========================\n"
        "Before outputting, verify:\n"
        "1. Did I process EVERY row?\n"
        "2. Did I rewrite EVERY ambiguity == 1 row?\n"
        "3. Does EACH rewritten sentence include a measurable or technical clarification?\n"
        "4. Did I preserve the original meaning exactly?\n"
        "5. Is the output valid CSV with no extra text?\n\n"
        "If ANY check fails -> fix it before output.\n\n"
        "========================\n"
        "IMPORTANT\n"
        "========================\n"
        "Output ONLY the CSV. No comments. No explanations.\n\n"
        "Input rows (JSON):\n"
        f"{json.dumps(rows, ensure_ascii=False)}"
    )


def build_input_rows(chunk: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for row_id, (_, row) in enumerate(chunk.iterrows(), start=1):
        rows.append(
            {
                "row_id": row_id,
                "original": str(row["original"]),
                "ambiguity": int(row["ambiguity"]),
            }
        )
    return rows


def build_repair_prompt(input_rows: list[dict[str, object]], previous_output: str, error_message: str) -> str:
    return (
        "You previously returned an invalid CSV batch. Fix it now.\n\n"
        "The output MUST contain exactly one CSV row per input row, in the same order.\n"
        "Do not omit, merge, or duplicate any rows.\n"
        "Return ONLY valid CSV with columns exactly: original,deambiguified_description\n"
        "Keep original text exactly as given.\n"
        "If ambiguity != 1, copy the original sentence unchanged.\n"
        "If ambiguity == 1, rewrite only with a minimal concrete clarification.\n\n"
        f"Validation error: {error_message}\n\n"
        "Input rows (JSON):\n"
        f"{json.dumps(input_rows, ensure_ascii=False)}\n\n"
        "Your previous CSV output:\n"
        f"{previous_output}"
    )


def call_openai_chat_completion(prompt: str, model: str, api_key: str, timeout: int = 120) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You output only valid CSV."},
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


def normalize_csv_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
    return stripped.strip()


def parse_output_csv(text: str, expected_rows: int) -> pd.DataFrame:
    normalized = normalize_csv_text(text)
    output = pd.read_csv(io.StringIO(normalized))

    expected_columns = ["original", "deambiguified_description"]
    if list(output.columns) != expected_columns:
        raise ValueError(f"Unexpected output columns: {list(output.columns)}")
    if len(output) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(output)}")

    return output


def get_valid_output_from_model(
    prompt: str,
    model: str,
    api_key: str,
    expected_rows: int,
    max_retries: int,
    retry_sleep_seconds: float,
    input_rows: list[dict[str, object]],
) -> pd.DataFrame:
    raw_csv = ""
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_csv = call_openai_chat_completion(prompt=prompt, model=model, api_key=api_key)
            return parse_output_csv(raw_csv, expected_rows=expected_rows)
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds * attempt)
                prompt = build_repair_prompt(input_rows=input_rows, previous_output=raw_csv, error_message=str(error))
            else:
                raise RuntimeError(f"Chunk failed after {max_retries} attempts") from error

    raise RuntimeError("Chunk failed unexpectedly") from last_error


def process_file(
    input_path: str | Path,
    output_path: str | Path,
    chunk_size: int,
    model: str,
    api_key: str,
    text_column: str | None = None,
    ambiguity_column: str | None = None,
    max_retries: int = 3,
    retry_sleep_seconds: float = 2.0,
    limit_chunks: int | None = None,
    dry_run: bool = False,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_write = True

    for chunk_index, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size), start=1):
        if limit_chunks is not None and chunk_index > limit_chunks:
            break

        resolved_text_column, resolved_ambiguity_column = detect_columns(chunk, text_column, ambiguity_column)
        prepared = chunk[[resolved_text_column, resolved_ambiguity_column]].copy()
        prepared.columns = ["original", "ambiguity"]

        input_rows = build_input_rows(prepared)
        prompt = build_chunk_prompt(prepared)
        parsed = get_valid_output_from_model(
            prompt=prompt,
            model=model,
            api_key=api_key,
            expected_rows=len(prepared),
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
            input_rows=input_rows,
        )

        if list(parsed["original"].astype(str)) != list(prepared["original"].astype(str)):
            repair_prompt = (
                "Your previous CSV preserved the wrong original text order or changed originals.\n"
                "Return ONLY valid CSV with columns exactly: original,deambiguified_description\n"
                "Keep each original value exactly as provided in the input rows JSON and keep row order.\n"
                f"Input rows (JSON):\n{json.dumps(input_rows, ensure_ascii=False)}\n\n"
                f"Previous CSV output:\n{parsed.to_csv(index=False)}"
            )
            parsed = get_valid_output_from_model(
                prompt=repair_prompt,
                model=model,
                api_key=api_key,
                expected_rows=len(prepared),
                max_retries=max_retries,
                retry_sleep_seconds=retry_sleep_seconds,
                input_rows=input_rows,
            )

        if not dry_run:
            parsed.to_csv(
                output_path,
                index=False,
                mode="w" if first_write else "a",
                header=first_write,
                quoting=csv.QUOTE_ALL,
            )
        first_write = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deambiguify a CSV file in 10-row chunks using an agent.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Rows per agent call")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--text-column", default="description", help="Text column name if auto-detection is wrong")
    parser.add_argument("--ambiguity-column", default="ambuiguity", help="Ambiguity column name if auto-detection is wrong")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per chunk")
    parser.add_argument("--retry-sleep-seconds", type=float, default=2.0, help="Backoff between retries")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Process only the first N chunks")
    parser.add_argument("--dry-run", action="store_true", help="Validate chunk output without writing a file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    process_file(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        model=args.model,
        api_key=api_key,
        text_column=args.text_column,
        ambiguity_column=args.ambiguity_column,
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        limit_chunks=args.limit_chunks,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
