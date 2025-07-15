from __future__ import annotations

"""evaluation_refinement.py
=======================================
Refined evaluation script for Pragmarag vs LightRAG on *graphragbench_medical*.
It reads the original `evaluation.csv`, calls Gemini with a progressive
"information‑overlap" prompt and appends structured results in
`evaluation_refined.csv`. The script is **resume‑safe** and parallel.

CLI example
-----------
```bash
python evaluation/exp_7_13/graphrag_medical/evaluation_refinement.py \
    --in_csv  C:/Users/paolo/Desktop/Ontology-Induction/outputs/exp_7_13/graphragbench_medical/evaluation.csv \
    --out_csv C:/Users/paolo/Desktop/Ontology-Induction/outputs/exp_7_13/evaluation/evaluation_refined.csv \
    --prompt_path evaluation/exp_7_13/graphrag_medical/informativity_evaluation_prompt.txt \
    --model gemini-1.5-pro --workers 4
```
"""

import argparse
import csv
import json
import logging
import queue
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Thread
from typing import List

from google import genai
from pydantic import BaseModel, Field, ValidationError
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Attempt to import the organisation‑internal helper utilities
# ---------------------------------------------------------------------------
try:
    from llm.llm_keys import GEMINI_KEY  # noqa: E402
    from llm.call_llm import call_gemini  # noqa: E402
except ImportError as err:  # pragma: no cover – clearer error for users
    raise SystemExit("❌  Cannot import llm helpers – are you in the right env?") from err

# ---------------------------------------------------------------------------
# Pydantic schema returned by the LLM
# ---------------------------------------------------------------------------

class InfoEvalResponse(BaseModel):
    """Schema capturing informational overlap analysis."""

    required_information: List[str] = Field(..., description="Key facts needed to answer the question")
    gold_information: List[str] = Field(..., description="Facts contained in the gold answer")
    gold_information_coverage: int = Field(..., ge=0, le=5)
    lightrag_information: List[str] = Field(...)
    lightrag_information_coverage: int = Field(..., ge=0, le=5)
    pragmarag_information: List[str] = Field(...)
    pragmarag_information_coverage: int = Field(..., ge=0, le=5)

# ---------------------------------------------------------------------------
# Default system prompt (written if missing)
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """
You are an expert evaluator of medical question‑answering systems. Your task is
**information‑centric**: judge how well candidate answers cover the key facts
needed for the question.

**Internal reasoning protocol (do NOT reveal):**
1. Identify the list of *required information* to fully answer the question.
2. Extract the information actually present in the *gold* answer and rate its
   coverage from 0‑5.
3. For each candidate answer (LightRAG and Pragmarag) list the information it
   contains and rate coverage of the *same* required facts, 0‑5.
4. Return ONLY valid JSON conforming to the provided schema.

Ignore stylistic elements (e.g., apologies, disclaimers). Focus exclusively on
informational content and correctness.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_prompt_file(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_DEFAULT_SYSTEM_PROMPT, encoding="utf-8")
        logging.info("✍️  Created default system prompt at %s", path)


def build_user_prompt(question: str, gold: str, light: str, prag: str) -> str:
    return (
        f"QUESTION:\n{question.strip()}\n\n"
        f"GOLD ANSWER (reference):\n{gold.strip()}\n\n"
        f"MODEL ANSWER 1 – LightRAG:\n{light.strip()}\n\n"
        f"MODEL ANSWER 2 – Pragmarag:\n{prag.strip()}\n\n"
        "Follow the evaluation protocol from the system message and output only valid JSON."
    )


def writer_loop(out_path: Path, fieldnames: List[str], delimiter: str, q: "queue.Queue[dict | None]") -> None:  # noqa: D401
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=delimiter)
        if is_new:
            writer.writeheader()
            fout.flush()
        while True:
            item = q.get()
            if item is None:
                break
            writer.writerow(item)
            fout.flush()
            q.task_done()


def evaluate_row(row: dict[str, str], gemini: genai.Client, system_prompt: str, model: str) -> dict[str, str]:
    user_prompt = build_user_prompt(
        question=row["question"],
        gold=row["gold_answer"],
        light=row["lightrag_answer"],
        prag=row["pragmarag_prompt_answer"],
    )
    try:
        resp: InfoEvalResponse = call_gemini(
            gemini_client=gemini,
            model_name=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=InfoEvalResponse,  # type: ignore[arg-type]
            thinking_budget=512,
            verbose=False,
        )
    except (ValidationError, Exception) as exc:  # pragma: no cover
        logging.error("Gemini failure for id %s – %s", row.get("id", "?"), exc)
        resp = InfoEvalResponse(
            required_information=[],
            gold_information=[],
            gold_information_coverage=0,
            lightrag_information=[],
            lightrag_information_coverage=0,
            pragmarag_information=[],
            pragmarag_information_coverage=0,
        )

    out = row.copy()
    out.update(
        {
            "question_information": " | ".join(resp.required_information),
            "gold_answer_information": " | ".join(resp.gold_information),
            "gold_answer_information_coverage": resp.gold_information_coverage,
            "lightrag_information": " | ".join(resp.lightrag_information),
            "lightrag_information_coverage": resp.lightrag_information_coverage,
            "pragmarag_information": " | ".join(resp.pragmarag_information),
            "pragmarag_information_coverage": resp.pragmarag_information_coverage,
        }
    )
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    in_csv = "C:/Users/paolo/Desktop/Ontology-Induction/outputs/exp_7_13/graphragbench_medical/evaluation/evaluation.csv"
    out_csv = "C:/Users/paolo/Desktop/Ontology-Induction/outputs/exp_7_13/graphragbench_medical/evaluation/evaluation_refined.csv"
    prompt_path = "C:/Users/paolo/Desktop/Ontology-Induction/evaluation/exp_7_13/graphrag_medical/informativity_evaluation_prompt.txt"
    model = "gemini-2.5-flash"
    workers = 12
    delimiter = ","

    parser = argparse.ArgumentParser("Refine evaluation with Gemini")
    parser.add_argument("--in_csv", default=in_csv)
    parser.add_argument("--out_csv", default=out_csv)
    parser.add_argument("--prompt_path", default=prompt_path)
    parser.add_argument("--model", default=model)
    parser.add_argument("--workers", type=int, default=workers)
    parser.add_argument("--delimiter", default=",", help="CSV delimiter (default ',')")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    in_path = Path(args.in_csv).expanduser()
    out_path = Path(args.out_csv).expanduser()
    prompt_path = Path(args.prompt_path).expanduser()

    ensure_prompt_file(prompt_path)
    system_prompt = prompt_path.read_text(encoding="utf-8")

    gemini_client = genai.Client(api_key=GEMINI_KEY)

    # Load already processed ids for resume capability
    processed_ids: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as fprev:
            processed_ids = {row["id"] for row in csv.DictReader(fprev, delimiter=args.delimiter)}
            logging.info("Loaded %d processed rows", len(processed_ids))

    # Read input
    with in_path.open("r", encoding="utf-8") as fin:
        reader_in = csv.DictReader(fin, delimiter=args.delimiter)
        rows = [row for row in reader_in if row["id"] not in processed_ids]
        if not rows:
            logging.info("All rows already processed – exiting.")
            return

    # Prepare output header
    base_fields = list(rows[0].keys())
    extra_fields = [
        "question_information",
        "gold_answer_information",
        "gold_answer_information_coverage",
        "lightrag_information",
        "lightrag_information_coverage",
        "pragmarag_information",
        "pragmarag_information_coverage",
    ]
    fieldnames_out = base_fields + extra_fields

    # Queue & writer thread
    q: "queue.Queue[dict | None]" = queue.Queue(maxsize=args.workers * 2)
    writer = Thread(target=writer_loop, args=(out_path, fieldnames_out, args.delimiter, q), daemon=True)
    writer.start()

    # Parallel evaluation
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(evaluate_row, row, gemini_client, system_prompt, args.model): row["id"] for row in rows
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating", unit="row"):
            q.put(fut.result())

    q.put(None)  # sentinel
    writer.join()
    logging.info("✅  Evaluation refinement completed – %s", out_path)


if __name__ == "__main__":
    main()
