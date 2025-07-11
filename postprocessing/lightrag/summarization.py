from __future__ import annotations

import json
import logging
import os
import queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from google import genai

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from llm.call_llm import call_gemini
from llm.llm_keys import GEMINI_KEY  # noqa: WPS433 – external secret holder

# ---------------------------------------------------------------------------#
# LLM prompt templates
# ---------------------------------------------------------------------------#

SYSTEM_PROMPT = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
You will be given one entity (or relationship), and a set of descriptions, all related to the same element. 
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the element name so we do have full context. 
Use English as output language.
"""

USER_PROMPT_TEMPLATE = """Type: {kind}
Name: {name}
Description: {description}
"""

# ---------------------------------------------------------------------------#
# Pydantic response schema expected from `call_gemini`
# ---------------------------------------------------------------------------#


class SummaryResponse(BaseModel):
    """Minimal schema used by `call_gemini` for Gemini responses."""

    response: str


# ---------------------------------------------------------------------------#
# Internal helpers
# ---------------------------------------------------------------------------#


def _ensure_dirs(*dirs: Path) -> None:
    """Create directories (recursively) if they do not yet exist."""
    for _dir in dirs:
        _dir.mkdir(parents=True, exist_ok=True)


def _read_done_set(done_file: Path) -> set[str]:
    """Return the set of *already processed* IDs (empty if file is absent)."""
    if not done_file.exists():
        return set()
    with done_file.open(encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip()}


def _append_done(done_file: Path, element_id: str) -> None:
    """Append a *single* processed ID to the done‑file (thread‑safe)."""
    # low‑contention, short file; rely on OS atomic append
    with done_file.open("a", encoding="utf-8") as fh:
        fh.write(f"{element_id}\n")


def _write_error(log_file: Path, ref_id: str, exc: BaseException) -> None:
    """Write a single error line to the error log."""
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {ref_id}: {exc}\n")


def _flush_to_parquet(tmp_path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Append `rows` (a list of dicts) to `tmp_path` in Parquet format.

    Because *appending* to Parquet is not natively supported, read‑modify‑write
    is used.  Files are typically small (< few MB) so the overhead is negligible
    and the simplicity ↑ reliability.
    """
    df_new = pd.DataFrame(rows)
    if tmp_path.exists():
        df_old = pd.read_parquet(tmp_path, engine="pyarrow")
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_parquet(tmp_path, engine="pyarrow", index=False)


def _aggregate_descriptions(
    df: pd.DataFrame,
    key_col: str,
    desc_col: str,
    extra_cols: Sequence[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Collapse multiple rows referring to the *same* element.

    Returns:
        mapping  element‑id -> {"descriptions": List[str], **extra_fields}
    """
    extra_cols = extra_cols or []
    grouped = df.groupby(key_col)
    out: Dict[str, Dict[str, Any]] = {}
    for key, sub in grouped:
        descriptions = sub[desc_col].dropna().astype(str).tolist()
        extra = {col: sub[col].iloc[0] for col in extra_cols if col in sub}
        out[key] = {"descriptions": descriptions, **extra}
    return out


# ---------------------------------------------------------------------------#
# Worker
# ---------------------------------------------------------------------------#

# sentinel used internally by the queue
_STOP = object()


def _worker(
    worker_id: int,
    kind: str,
    work_q: "queue.Queue[Tuple[str, Dict[str, Any]]]",
    tmp_dir: Path,
    done_file: Path,
    error_log: Path,
    model_name: str,
    batch_size: int = 16,
    max_retries: int = 3,
    retry_backoff: float = 2.0,
) -> None:  # noqa: WPS231, WPS234 – length OK in this context
    """
    Repeatedly consume tasks from `work_q` until sentinel is seen.

    Every `batch_size` successful items are flushed to
    `kind_worker_<id>.tmp.parquet` so that partial work survives crashes.
    """
    logger = logging.getLogger(f"worker-{worker_id}-{kind}")
    logger.info("Starting worker %s (%s)", worker_id, kind)

    tmp_path = tmp_dir / f"{kind}_worker_{worker_id}.tmp.parquet"

    # initialise Gemini *inside* the thread; avoids pickling client objects
    gemini_client = genai.Client(api_key=GEMINI_KEY)

    rows_buffer: List[Dict[str, Any]] = []

    while True:
        task = work_q.get()
        if task is _STOP:
            break

        element_id, meta = task
        descriptions: List[str] = meta["descriptions"]

        user_prompt = USER_PROMPT_TEMPLATE.format(
            kind="Entity" if kind == "entities" else "Relationship",
            name=element_id,
            description=" ".join(descriptions),
        )

        # --- call Gemini with exponential back‑off ------------------------
        last_exc: BaseException | None = None
        for attempt in range(1, max_retries + 1):
            try:
                response: SummaryResponse = call_gemini(
                    gemini_client=gemini_client,
                    model_name=model_name,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    response_schema=SummaryResponse,
                    thinking_budget=512,
                    verbose=False,
                )
                summary_text = response.response
                break
            except Exception as exc:  # noqa: BLE001 – any Gemini error
                last_exc = exc
                sleep_for = retry_backoff**attempt
                logger.warning(
                    "Error calling Gemini for %s (attempt %s/%s): %s; retrying in %.1f s",
                    element_id,
                    attempt,
                    max_retries,
                    exc,
                    sleep_for,
                )
                time.sleep(sleep_for)
        else:
            _write_error(error_log, element_id, last_exc or RuntimeError("Unknown"))
            continue  # skip, but do not crash worker

        rows_buffer.append(
            {
                # naming kept generic to merge later
                "name": element_id,
                "type": kind,
                "description": " ".join(descriptions),
                "summary": summary_text.strip(),
            },
        )
        _append_done(done_file, element_id)

        # flush periodically
        if len(rows_buffer) >= batch_size:
            _flush_to_parquet(tmp_path, rows_buffer)
            rows_buffer.clear()
            logger.info("Flushed %s rows to %s", batch_size, tmp_path.name)

    # final flush
    if rows_buffer:
        _flush_to_parquet(tmp_path, rows_buffer)
        logger.info("Flushed final %s rows to %s", len(rows_buffer), tmp_path.name)

    logger.info("Worker %s finished", worker_id)


# ---------------------------------------------------------------------------#
# Main orchestration
# ---------------------------------------------------------------------------#


def _start_pool(  # noqa: WPS231 – clarity > length
    kind: str,
    tasks: Dict[str, Dict[str, Any]],
    output_dir: Path,
    num_workers: int,
    model_name: str = "gemini-2.5-flash-lite-preview-06-17",
) -> None:
    """
    Spin up a thread‑pool and process all `tasks`.

    Parameters
    ----------
    kind
        Either ``"entities"`` or ``"relationships"``.
    tasks
        Mapping ``element_id -> metadata`` (must include *descriptions* list).
    output_dir
        Base directory for outputs (`logs` will be created inside).
    """
    logger = logging.getLogger(kind)
    if not tasks:
        logger.info("Nothing to summarise for %s", kind)
        return
    

    logs_dir = output_dir / "logs"
    _ensure_dirs(logs_dir)
    done_file = logs_dir / f"processed_{kind[:-1]}_ids.txt"
    error_log = logs_dir / "error_log.txt"

    # -------------------------------------------------------------------#
    # prepare queue and launch workers
    # -------------------------------------------------------------------#
    work_q: "queue.Queue[Any]" = queue.Queue(maxsize=4 * num_workers)

    tmp_dir = output_dir  # tmp parquet files live alongside final ones

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for wid in range(num_workers):
            pool.submit(
                _worker,
                wid,
                kind,
                work_q,
                tmp_dir,
                done_file,
                error_log,
                model_name,
            )
        for element_id, meta in tasks.items():
            work_q.put((element_id, meta))
        for _ in range(num_workers):
            work_q.put(_STOP)

    logger.info("All workers for %s finished", kind)


def _merge_tmp_parquets(
    original_df: pd.DataFrame,
    tmp_files: List[Path],
    kind: str,
) -> pd.DataFrame:
    """
    Merge the *original* DataFrame with summaries collected in `tmp_files`.

    The column used for joining is ``'entity_name'`` / ``'relation_keywords'``
    depending on *kind*.
    """
    if not tmp_files:
        raise FileNotFoundError(
            f"No temporary Parquet files produced for {kind}; "
            "did the workers run?",
        )

    map_df = pd.concat(
        [pd.read_parquet(f, engine="pyarrow") for f in tmp_files],
        ignore_index=True,
    ).drop_duplicates(subset="name", keep="last")

    join_col = "entity_name" if kind == "entities" else "relation_keywords"

    merged = original_df.merge(
        map_df.rename(columns={"name": join_col}),
        on=join_col,
        how="left",
    )
    return merged


# ---------------------------------------------------------------------------#
# PUBLIC FUNCTION
# ---------------------------------------------------------------------------#


def summarize_graph(input_dir: Path, output_dir: Path, num_workers: int = 4) -> None:  # noqa: WPS231 – clarity > length
    """
    Read the *deduplicated* Parquet files in ``input_dir``, generate summaries
    for each entity and relationship (in parallel), and write enriched Parquet
    files plus fault‑tolerance artefacts under ``output_dir``.

    Parameters
    ----------
    input_dir
        Directory containing ``entities.parquet`` and ``relationships.parquet``.
    output_dir
        Target directory (will be created if missing).
    num_workers
        Parallelism level (defaults to 4). Thread‑based workers are used because
        Gemini calls are I/O bound.
    """
    _ensure_dirs(output_dir)
    logs_dir = output_dir / "logs"
    _ensure_dirs(logs_dir)

    # -------------------------------------------------------------------#
    # Load data
    # -------------------------------------------------------------------#
    entities_path = input_dir / "entities.parquet"
    relations_path = input_dir / "relationships.parquet"

    if not entities_path.exists() or not relations_path.exists():
        raise FileNotFoundError(
            f"Input Parquet files not found in {input_dir} "
            "(expected entities.parquet and relationships.parquet).",
        )

    df_entities = pd.read_parquet(entities_path, engine="pyarrow")
    df_rel = pd.read_parquet(relations_path, engine="pyarrow")

    # -------------------------------------------------------------------#
    # Aggregate descriptions (one task = one element)
    # -------------------------------------------------------------------#
    agg_entities = _aggregate_descriptions(
        df_entities,
        key_col="entity_name",
        desc_col="entity_description",
        extra_cols=["entity_type"],
    )
    agg_rel = _aggregate_descriptions(
        df_rel,
        key_col="relation_keywords",
        desc_col="relation_description",
    )

    # -------------------------------------------------------------------#
    # Remove already processed IDs
    # -------------------------------------------------------------------#
    done_entities = _read_done_set(logs_dir / "processed_entity_ids.txt")
    done_rel = _read_done_set(logs_dir / "processed_relation_ids.txt")

    tasks_entities = {k: v for k, v in agg_entities.items() if k not in done_entities}
    tasks_rel = {k: v for k, v in agg_rel.items() if k not in done_rel}

    logging.info(
        "Entities: %s total, %s pending | Relationships: %s total, %s pending",
        len(agg_entities),
        len(tasks_entities),
        len(agg_rel),
        len(tasks_rel),
    )

    # -------------------------------------------------------------------#
    # Parallel processing
    # -------------------------------------------------------------------#
    _start_pool("entities", tasks_entities, output_dir, num_workers)
    _start_pool("relationships", tasks_rel, output_dir, num_workers)

    # -------------------------------------------------------------------#
    # Merge temp results with original data
    # -------------------------------------------------------------------#
    tmp_entities = list(output_dir.glob("entities_worker_*.tmp.parquet"))
    tmp_rel = list(output_dir.glob("relationships_worker_*.tmp.parquet"))

    enriched_entities = _merge_tmp_parquets(df_entities, tmp_entities, "entities")
    enriched_rel = _merge_tmp_parquets(df_rel, tmp_rel, "relationships")

    enriched_entities.to_parquet(output_dir / "entities.parquet", index=False)
    enriched_rel.to_parquet(output_dir / "relationships.parquet", index=False)

    logging.info(
        "✅ Summarisation complete. Outputs saved to %s",
        output_dir.resolve(),
    )


# ---------------------------------------------------------------------------#
# Convenience: enable `python summarization.py <input_dir> <output_dir>`
# ---------------------------------------------------------------------------#
if __name__ == "__main__":  # pragma: no cover – used only when launched as a script
    in_dir = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/graphragbench_medical/postprocessing/deduplication").expanduser().resolve()
    out_dir = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/graphragbench_medical/postprocessing/summarization").expanduser().resolve()
    workers = 3

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    summarize_graph(in_dir, out_dir, workers)
