from __future__ import annotations

import csv
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from google import genai
from tqdm.auto import tqdm

# ------------------------------------------------------------------#
# Percorsi base
# ------------------------------------------------------------------#
OUTPUT_PATH = Path(
    "C:/Users/paolo/Desktop/Ontology-Induction/outputs/exp_7_13/graphragbench_medical"
)
EVAL_CSV = OUTPUT_PATH / "evaluation" / "evaluation.csv"
EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)

# prompt template comuni
KEYWORD_PROMPT = Path(
    "C:/Users/paolo/Desktop/Ontology-Induction/generation/lightrag/keyword_extraction_prompt.txt"
)
RAG_TEMPLATE = Path(
    "C:/Users/paolo/Desktop/Ontology-Induction/generation/lightrag/rag_response_prompt.txt"
)

# ------------------------------------------------------------------#
# Import funzioni pipeline
# ------------------------------------------------------------------#
from generation.lightrag.generation import (  # noqa: E402
    keyword_extraction,
    retrieve_context,
    retrieve_graph_context,
    rag_generate_response,
)
from evaluation.coherence_evaluation import evaluate_coherence  # noqa: E402

# ------------------------------------------------------------------#
# Gemini client
# ------------------------------------------------------------------#
from llm.llm_keys import GEMINI_KEY  # noqa: E402
from llm.call_llm import call_gemini  # noqa: E402

gemini_client = genai.Client(api_key=GEMINI_KEY)

# ------------------------------------------------------------------#
# Question generator
# ------------------------------------------------------------------#
from datasets.graphragbench_medical.graphragbench_medical_questions.graphragbench_medical_questions_generator import (  # noqa: E402, E501
    generate_questions,
)

# ------------------------------------------------------------------#
# Logging
# ------------------------------------------------------------------#
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluation")

# ------------------------------------------------------------------#
# Helper: aggiorna CSV in append (header se manca)
# ------------------------------------------------------------------#
def _append_eval_row(row: Dict[str, str]) -> None:
    header_exists = EVAL_CSV.exists()
    with EVAL_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "id",
                "question",
                "question_type",          # ← virgola e colonna ora presenti
                "gold_answer",
                "lightrag_answer",
                "pragmarag_answer",
                "lightrag_coherence",
                "pragmarag_coherence",
            ],
        )
        if not header_exists:
            writer.writeheader()
        writer.writerow(row)


def _read_processed_ids() -> set[str]:
    if not EVAL_CSV.exists():
        return set()
    with EVAL_CSV.open(encoding="utf-8") as fh:
        return {row["id"] for row in csv.DictReader(fh)}


# ------------------------------------------------------------------#
# Worker: processa una singola domanda
# ------------------------------------------------------------------#
def _process_question(
    qid: str,
    question_text: str,
    gold_answer: str,
    question_type: str,
) -> Tuple[str, Dict[str, str]]:
    """Ritorna (id, row) ; row è dict per CSV."""
    # 1. keyword extraction
    k_resp = keyword_extraction(question_text, prompt_path=KEYWORD_PROMPT)

    # ---- paths comuni -------------------------------------------------- #
    idx_base_lrt = OUTPUT_PATH / "lightrag" / "postprocessing" / "indexing"
    idx_base_prg = OUTPUT_PATH / "pragmarag_prompt" / "postprocessing" / "indexing"

    # 2. retrieval a) LightRAG
    low_ctx_lrt, high_ctx_lrt = retrieve_context(
        high=k_resp.high_level_keywords,
        low=k_resp.low_level_keywords,
        entities_index=idx_base_lrt / "entities.index",
        entities_meta=idx_base_lrt / "entity_metadata.pkl",
        rel_index=idx_base_lrt / "relationships.index",
        rel_meta=idx_base_lrt / "relationship_metadata.pkl",
        top_k=2,
    )

    #    b) PragmaRAG-prompt
    low_ctx_prg, high_ctx_prg = retrieve_context(
        high=k_resp.high_level_keywords,
        low=k_resp.low_level_keywords,
        entities_index=idx_base_prg / "entities.index",
        entities_meta=idx_base_prg / "entity_metadata.pkl",
        rel_index=idx_base_prg / "relationships.index",
        rel_meta=idx_base_prg / "relationship_metadata.pkl",
        top_k=2,
    )

    # 3. graph context
    graph_ctx_lrt = retrieve_graph_context(
        low_entities=[n for n, _ in low_ctx_lrt],
        high_relations=[n for n, _ in high_ctx_lrt],
        graph_path=OUTPUT_PATH
        / "lightrag"
        / "postprocessing"
        / "graph_builder"
        / "graphragbench_medical.igraph",
    )

    graph_ctx_prg = retrieve_graph_context(
        low_entities=[n for n, _ in low_ctx_prg],
        high_relations=[n for n, _ in high_ctx_prg],
        graph_path=OUTPUT_PATH
        / "pragmarag_prompt"
        / "postprocessing"
        / "graph_builder"
        / "graphragbench_medical.igraph",
    )

    # 4. RAG answers
    lrt_answer = rag_generate_response(
        question_text,
        graph_ctx_lrt,
        rag_template_path=RAG_TEMPLATE,
    )
    prg_answer = rag_generate_response(
        question_text,
        graph_ctx_prg,
        rag_template_path=RAG_TEMPLATE,
    )

    # 5. Coherence
    lrt_coh = evaluate_coherence(
        gemini_client=gemini_client,
        model_name="models/gemini-2.5-flash-lite-preview-06-17",
        question=question_text,
        gold_answer=gold_answer,
        model_answer=lrt_answer,
        judge_name="LightRAG-Gemini",
    ).coherence_score

    prg_coh = evaluate_coherence(
        gemini_client=gemini_client,
        model_name="models/gemini-2.5-flash-lite-preview-06-17",
        question=question_text,
        gold_answer=gold_answer,
        model_answer=prg_answer,
        judge_name="PragmaRAG-Gemini",
    ).coherence_score

    row = {
        "id": qid,
        "question": question_text,
        "question_type": question_type,
        "gold_answer": gold_answer,
        "lightrag_answer": lrt_answer,
        "pragmarag_answer": prg_answer,
        "lightrag_coherence": f"{lrt_coh:.2f}",
        "pragmarag_coherence": f"{prg_coh:.2f}",
    }
    return qid, row


# ------------------------------------------------------------------#
# MAIN
# ------------------------------------------------------------------#
def main() -> None:
    logger.info("Preparing test set …")
    # --- 1. build exclusion list ------------------------------------- #
    train_ids = [
        rec["Metadata"]["Id"]
        for rec in generate_questions(
            split="train", fraction=0.1, seed=42
        )
    ]

    # --- 2. generator ALL, filter not in train & not processed ------- #
    processed_ids = _read_processed_ids()

    pending: List[Tuple[str, str, str, str]] = []  # (id, question, answer, question_type)
    for rec in generate_questions(split="all", fraction=1, seed=42):
        rid = rec["Metadata"]["Id"]
        if rid in train_ids or rid in processed_ids:
            continue
        pending.append((rid, rec["Question"], rec["Answer"], rec["Metadata"]["Question_Type"]))

    if not pending:
        logger.info("Nothing new to evaluate – all done.")
        return

    logger.info("Pending questions: %d", len(pending))

    # --- 3. Parallel processing with 5 workers ----------------------- #
    with ThreadPoolExecutor(max_workers=4) as pool, tqdm(
        total=len(pending), desc="Evaluating", unit="Q"
    ) as pbar:
        fut_to_id = {
            pool.submit(_process_question, qid, qtxt, ans, qtype): qid
            for qid, qtxt, ans, qtype in pending
        }
        for fut in as_completed(fut_to_id):
            try:
                _, row = fut.result()
                _append_eval_row(row)
            except Exception as exc:  # pragma: no cover
                logger.error("Error processing %s: %s", fut_to_id[fut], exc)
            pbar.update(1)

    logger.info("✅ Evaluation finished – results in %s", EVAL_CSV.resolve())


if __name__ == "__main__":
    main()
