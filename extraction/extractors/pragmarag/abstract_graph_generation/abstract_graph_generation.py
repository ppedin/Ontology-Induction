from __future__ import annotations

"""abstract_graph_generation.py

Pydantic schemas and utilities to generate *abstract graphs* from questions and
persist them fault‑tolerantly. Now includes a tqdm progress bar.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field
from tqdm.auto import tqdm  # progress bar

from llm.call_llm import call_gemini

__all__ = [
    "ReasoningStep",
    "EntityType",
    "RelationshipType",
    "AbstractGraphGenerationResponse",
    "generate_abstract_subgraphs",
]

# --------------------------------------------------------------------------- #
# Pydantic models (reasoning_steps optional)
# --------------------------------------------------------------------------- #

class ReasoningStep(BaseModel):
    entity: str = Field(..., description="Starting entity label.")
    entity_type: str = Field(..., description="Abstract class of the entity.")
    relation: str = Field(..., description="Relation traversed (lower_snake_case).")


class EntityType(BaseModel):
    name: str = Field(..., description="Canonical name of the entity type.")
    description: str = Field(..., description="Short description of the entity type.")


class RelationshipType(BaseModel):
    name: str = Field(..., description="Canonical name of the relation type.")
    description: str = Field(..., description="Short description of the relation type.")


class AbstractGraphGenerationResponse(BaseModel):
    reasoning_steps: Optional[List[ReasoningStep]] = Field(
        default=None,
        description="Ordered chain of reasoning as edge traversals (optional).",
    )
    entity_types: List[EntityType]
    relationship_types: List[RelationshipType]


# --------------------------------------------------------------------------- #
# Graph generation utility
# --------------------------------------------------------------------------- #

_TEMPLATE_FILE = Path(__file__).with_name("abstract_graph_generation_from_question_prompt_template.txt")
_OUTPUT_DIR = Path("outputs/graphragbench_medical/extraction/pragmarag")
_OUTPUT_FILE = _OUTPUT_DIR / "abstract_questions_subgraphs.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_system_prompt(corpus_type: str) -> str:
    text = _TEMPLATE_FILE.read_text(encoding="utf-8")
    return text.replace("{CORPUS_TYPE}", corpus_type)


def _invoke_llm(
    qid_question: Tuple[str, str],
    *,
    system_prompt: str,
    gemini_client,
    model_name: str,
    retry: int = 3,
) -> Optional[dict]:
    qid, question = qid_question
    backoff = 2.0
    for attempt in range(retry):
        try:
            resp: AbstractGraphGenerationResponse = call_gemini(
                gemini_client=gemini_client,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=question,
                response_schema=AbstractGraphGenerationResponse,
                thinking_budget=512,
                verbose=False,
            )
            return {"id": qid, **resp.model_dump()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM call failed for %s (attempt %d/%d): %s", qid, attempt + 1, retry, exc)
            time.sleep(backoff)
            backoff *= 2
    logger.error("LLM call ultimately failed for %s after %d attempts", qid, retry)
    return None


def generate_abstract_subgraphs(
    question_iter: Iterable[dict],
    gemini_client,
    model_name: str,
    *,
    output_path: Path | None = None,
    num_workers: int = 4,
    corpus_type: str = "NCCN guidelines",
) -> None:
    """Generate abstract subgraphs for questions and save to JSONL.

    Uses a thread pool and a tqdm progress bar for feedback.
    """

    out_path = Path(output_path or _OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Already processed IDs
    processed_ids: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    processed_ids.add(json.loads(line)["id"])
                except Exception:
                    continue
        logger.info("Resuming – %d questions already processed", len(processed_ids))

    # Pending questions
    pending: List[Tuple[str, str]] = []
    for rec in question_iter:
        qid = rec.get("Metadata", {}).get("Id") or rec.get("Metadata", {}).get("id")
        if qid is None or qid in processed_ids:
            continue
        pending.append((qid, rec["Question"]))

    if not pending:
        logger.info("Nothing new to process.")
        return

    system_prompt = _load_system_prompt(corpus_type)
    logger.info("Processing %d new questions with %d workers…", len(pending), num_workers)

    with ThreadPoolExecutor(max_workers=num_workers) as pool, out_path.open("a", encoding="utf-8") as fh:
        futures = [
            pool.submit(
                _invoke_llm,
                item,
                system_prompt=system_prompt,
                gemini_client=gemini_client,
                model_name=model_name,
            )
            for item in pending
        ]

        with tqdm(total=len(futures), desc="Abstract graphs", unit="q") as pbar:
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fh.flush()
                pbar.update(1)

    logger.info("Done. Total processed: %d", len(processed_ids) + len(pending))


if __name__ == "__main__":
    from llm.llm_keys import GEMINI_KEY
    from datasets.graphragbench_medical.graphragbench_medical_questions.graphragbench_medical_questions_generator import generate_questions
    from google import genai

    questions = generate_questions(split="train", fraction=0.1, seed=42)
    client = genai.Client(api_key=GEMINI_KEY)

    generate_abstract_subgraphs(
        question_iter=questions,
        gemini_client=client,
        model_name="models/gemini-2.5-flash-lite-preview-06-17",
        num_workers=4,
    )