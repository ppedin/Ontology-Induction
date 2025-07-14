from __future__ import annotations

"""abstract_graph_generation.py

Pydantic schemas **and** utilities for *PragmaRAG* abstract‑graph generation:

1. **Abstract subgraph creation**: generate a per‑question hypothetical graph
   via Gemini (`generate_abstract_subgraphs`).
2. **Graph summarisation / abstraction**: iteratively merge batches of k
   subgraphs into higher‑level schemas (`summarise_subgraphs`).

Both functions are thread‑safe, fault‑tolerant (append‑only JSONL) and show a
`tqdm` progress bar.
"""

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from llm.call_llm import call_gemini

__all__ = [
    # Schemas
    "ReasoningStep",
    "EntityType",
    "RelationshipType",
    "AbstractGraphGenerationResponse",
    # Utilities
    "generate_abstract_subgraphs",
    "summarise_subgraphs",
]

# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #

class ReasoningStep(BaseModel):
    entity: str = Field(..., description="Starting entity label.")
    entity_type: str = Field(..., description="Abstract class of the entity.")
    relation: str = Field(..., description="Relation traversed (lower_snake_case).")


class EntityType(BaseModel):
    name: str = Field(..., description="Canonical entity‑type name.")
    description: str = Field(..., description="Short human‑readable description.")


class RelationshipType(BaseModel):
    name: str = Field(..., description="Canonical relation‑type name.")
    description: str = Field(..., description="Short human‑readable description.")


class AbstractGraphGenerationResponse(BaseModel):
    """Container for Gemini responses.

    `reasoning_steps` is optional: present when generated from a question,
    absent after summarisation.
    """

    reasoning_steps: Optional[List[ReasoningStep]] = Field(default=None)
    entity_types: List[EntityType]
    relationship_types: List[RelationshipType]


# Discursive summary schema
class GraphDescriptionResponse(BaseModel):
    text: str = Field(..., description="Narrative description of the abstract graph.")

# --------------------------------------------------------------------------- #
# Paths & logging
# --------------------------------------------------------------------------- #

_THIS_DIR = Path(__file__).parent
_TEMPLATE_Q_PATH = _THIS_DIR / "abstract_graph_generation_from_question_prompt_template.txt"
_TEMPLATE_SUM_PATH = _THIS_DIR / "graph_summarizer_prompt_template.txt"
_OUT_DIR = Path("outputs/exp_7_13/graphragbench_medical/pragmarag_prompt/extraction")
_OUT_DIR.mkdir(parents=True, exist_ok=True)
_TEMPLATE_DISC_PATH = _THIS_DIR / "discoursive_summarizer_prompt_template.txt"


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("abstract_graph_generation")

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _retry_call(
    *,
    fn,
    max_attempts: int = 3,
    backoff: float = 2.0,
    **kwargs,
):
    """Simple retry wrapper with exponential back‑off."""

    attempt = 0
    delay = backoff
    while attempt < max_attempts:
        try:
            return fn(**kwargs)
        except Exception as exc:  # noqa: BLE001
            attempt += 1
            if attempt >= max_attempts:
                raise
            time.sleep(delay)
            delay *= 2
            logger.warning("Retry %d/%d after error: %s", attempt, max_attempts, exc)

# --------------------------------------------------------------------------- #
# 1. Per‑question abstract graph generation
# --------------------------------------------------------------------------- #

def _invoke_question_llm(
    qid_question: Tuple[str, str],
    *,
    system_prompt: str,
    gemini_client,
    model_name: str,
) -> Optional[dict]:
    qid, question = qid_question
    try:
        resp: AbstractGraphGenerationResponse = _retry_call(
            fn=call_gemini,
            gemini_client=gemini_client,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=question,
            response_schema=AbstractGraphGenerationResponse,
            thinking_budget=512,
            verbose=False,
        )
        return {"id": qid, **resp.dict()}
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM call failed permanently for %s: %s", qid, exc)
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
    """Generate hypothetical subgraphs for each question (threaded, resumable)."""

    out_path = Path(output_path or _OUT_DIR / "abstract_questions_subgraphs.jsonl")
    processed_ids: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as fh:
            processed_ids.update(json.loads(l)["id"] for l in fh if l.strip())
        logger.info("Resuming – %d questions already done", len(processed_ids))

    pending: List[Tuple[str, str]] = []
    for rec in question_iter:
        qid = rec.get("Metadata", {}).get("Id") or rec.get("Metadata", {}).get("id")
        if qid and qid not in processed_ids:
            pending.append((qid, rec["Question"]))

    if not pending:
        logger.info("No new questions to process.")
        return

    system_prompt = _load_template(_TEMPLATE_Q_PATH).replace("{CORPUS_TYPE}", corpus_type)
    logger.info("Generating %d abstract graphs…", len(pending))

    with ThreadPoolExecutor(max_workers=num_workers) as pool, out_path.open("a", encoding="utf-8") as fh, tqdm(total=len(pending), desc="Q→Graph", unit="q") as pbar:

        futures = [
            pool.submit(
                _invoke_question_llm,
                item,
                system_prompt=system_prompt,
                gemini_client=gemini_client,
                model_name=model_name,
            )
            for item in pending
        ]
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                fh.flush()
            pbar.update(1)

    logger.info("Finished per‑question graph generation.")

# --------------------------------------------------------------------------- #
# 2. Summariser – merge k subgraphs at a time
# --------------------------------------------------------------------------- #

def _format_graph_for_prompt(graph: dict, idx: int) -> str:
    """Return a human‑readable string for a single graph."""
    ent = json.dumps(graph["entity_types"], ensure_ascii=False)
    rel = json.dumps(graph["relationship_types"], ensure_ascii=False)
    return f"Graph {idx}:\nEntity Types: {ent}\nRelation Types: {rel}\n"


def _invoke_summariser_llm(
    batch_ids: List[str],
    prompt_str: str,
    *,
    system_prompt: str,
    gemini_client,
    model_name: str,
) -> Optional[dict]:
    try:
        resp: AbstractGraphGenerationResponse = _retry_call(
            fn=call_gemini,
            gemini_client=gemini_client,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=prompt_str,
            response_schema=AbstractGraphGenerationResponse,
            thinking_budget=512,
            verbose=False,
        )
        merged_id = "+".join(batch_ids)
        return {"id": merged_id, **resp.model_dump()}
    except Exception as exc:  # noqa: BLE001
        logger.error("Summariser LLM failed for batch [%s]: %s", ",".join(batch_ids), exc)
        return None


def summarise_subgraphs(
    *,
    input_path: Path | None = None,
    output_path: Path | None = None,
    k: int = 10,
    gemini_client,
    model_name: str,
    num_workers: int = 4,
) -> None:
    """Merge batches of *k* abstract graphs into higher‑level schemas.

    Can be run iteratively to obtain successively coarser abstractions.
    """

    in_path = Path(input_path or _OUT_DIR / "abstract_questions_subgraphs.jsonl")
    out_path = Path(output_path or _OUT_DIR / "abstract_questions_subgraph_abstraction1.jsonl")

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    # Load existing output IDs for resuming
    done_ids: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as fh:
            done_ids.update(json.loads(l)["id"] for l in fh if l.strip())
        logger.info("Resuming summarisation – %d batches already done", len(done_ids))

    # Load all input graphs
    graphs: List[dict] = []
    with in_path.open("r", encoding="utf-8") as fh:
        graphs = [json.loads(l) for l in fh if l.strip()]

    # Build batches of k graphs
    batches: List[List[dict]] = []
    for i in range(0, len(graphs), k):
        batch = graphs[i : i + k]
        batch_ids = [g["id"] for g in batch]
        merged_id = "+".join(batch_ids)
        if merged_id in done_ids:
            continue  # already processed
        batches.append(batch)

    if not batches:
        logger.info("No new batches to summarise.")
        return

    system_prompt = _load_template(_TEMPLATE_SUM_PATH)
    logger.info("Summarising %d batches of size ≤ %d", len(batches), k)

    with ThreadPoolExecutor(max_workers=num_workers) as pool, out_path.open("a", encoding="utf-8") as fh, tqdm(total=len(batches), desc="Summarising", unit="batch") as pbar:

        futures = []
        for batch in batches:
            batch_ids = [g["id"] for g in batch]
            prompt_parts = [
                _format_graph_for_prompt(g, idx=i + 1) for i, g in enumerate(batch)
            ]
            user_prompt = "\n".join(prompt_parts)
            futures.append(
                pool.submit(
                    _invoke_summariser_llm,
                    batch_ids,
                    user_prompt,
                    system_prompt=system_prompt,
                    gemini_client=gemini_client,
                    model_name=model_name,
                )
            )

        for fut in as_completed(futures):
            result = fut.result()
            if result:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                fh.flush()
            pbar.update(1)

    logger.info("Summarisation complete – results in %s", out_path)


def describe_subgraphs_discursively(
    *,
    input_path: Path | None = None,
    output_path: Path | None = None,
    gemini_client,
    model_name: str,
) -> None:
    """Generate a plain‑language narrative for a (single) abstract graph.

    Parameters
    ----------
    input_path : Path or None
        JSONL file containing ONE line with keys ``entity_types`` and
        ``relationship_types``. If multiple lines exist, they are concatenated
        and summarised together.
    output_path : Path or None
        Destination ``.txt`` file (overwritten).
    gemini_client / model_name : LLM access.
    """

    in_path = Path(input_path or _OUT_DIR / "abstract_questions_subgraph_abstraction3.jsonl")
    out_path = Path(output_path or _OUT_DIR / "abstract_questions_subgraph_description.txt")

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    # Read the whole file and prepare the user prompt
    with in_path.open("r", encoding="utf-8") as fh:
        graphs_raw = [json.loads(l) for l in fh if l.strip()]

    if not graphs_raw:
        logger.warning("Input JSONL is empty – nothing to describe.")
        return

    # Format for the LLM (reuse compact formatter)
    def _format_graph(g):
        ent = json.dumps(g["entity_types"], ensure_ascii=False)
        rel = json.dumps(g["relationship_types"], ensure_ascii=False)
        return f"Entity Types: {ent}\nRelation Types: {rel}"

    user_prompt = "\n\n".join(_format_graph(g) for g in graphs_raw)

    system_prompt = _load_template(_TEMPLATE_DISC_PATH)

    logger.info("Requesting discursive summary from LLM…")
    with tqdm(total=1, desc="LLM", unit="call") as pbar:
        response: GraphDescriptionResponse = call_gemini(
            gemini_client=gemini_client,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=GraphDescriptionResponse,
            thinking_budget=512,
            verbose=False,
        )
        pbar.update(1)

    # Save
    out_path.write_text(response.text.strip() + "\n", encoding="utf-8")
    logger.info("Discursive description saved to %s", out_path)






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
    summarise_subgraphs(
        input_path=Path("outputs/exp_7_13/graphragbench_medical/pragmarag_prompt/extraction/abstract_questions_subgraphs_abstraction2.jsonl"),
        output_path=Path("outputs/exp_7_13/graphragbench_medical/pragmarag_prompt/extraction/abstract_questions_subgraphs_abstraction3.jsonl"),
        gemini_client=client,
        model_name="models/gemini-2.5-flash-lite-preview-06-17",
        num_workers=4,
    )
    """
    describe_subgraphs_discursively(
        input_path=Path("outputs/graphragbench_medical/extraction/pragmarag/abstract_questions_subgraph_abstraction3.jsonl"),
        output_path=Path("outputs/graphragbench_medical/extraction/pragmarag/abstract_questions_subgraph_description.txt"),
        gemini_client=client,
        model_name="models/gemini-2.5-flash-lite-preview-06-17",
    )
    """