"""
Generation – LightRAG end‑to‑end demo (with question generator)
===============================================================

Pipeline
--------
1. Prende **una sola domanda** dal *generator* del dataset medicale
   (`generate_questions`) e ne estrae il campo "Question".
2. Estrae le keyword (high‑level / low‑level) con Gemini
   (prompt: `generation/lightrag/keyword_extraction_prompt.txt`).
3. Esegue retrieval su due indici FAISS (entità + relazioni) per costruire
   un contesto (nome | summary) e lo stampa.

Percorsi hard‑coded (puoi modificarli in cima al file):
    • Dataset JSON        ➜  C:/Users/paolo/Desktop/Ontology-Induction/datasets/graphragbench_medical/graphragbench_medical_questions/graphragbench_medical_questions.json
    • Prompt system       ➜  generation/lightrag/keyword_extraction_prompt.txt
    • Indici FAISS        ➜  outputs/graphragbench_medical/postprocessing/indexing/

Requisiti:
    pip install pydantic==1.* sentence-transformers faiss-cpu google-generativeai pandas numpy
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import faiss  # type: ignore
import numpy as np
import pickle
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config – modifica se i percorsi cambiano
# ---------------------------------------------------------------------------

QUESTIONS_JSON = Path(
    r"C:/Users/paolo/Desktop/Ontology-Induction/datasets/graphragbench_medical/graphragbench_medical_questions/graphragbench_medical_questions.json"
)
PROMPT_PATH = Path("C:/Users/paolo/Desktop/Ontology-Induction/generation/lightrag/keyword_extraction_prompt.txt")

INDEX_DIR = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/graphragbench_medical/postprocessing/indexing")
ENTITIES_INDEX = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/graphragbench_medical/postprocessing/indexing/entities.index")
ENTITIES_META = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/graphragbench_medical/postprocessing/indexing/entity_metadata.pkl")
REL_INDEX = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/graphragbench_medical/postprocessing/indexing/relationships.index")
REL_META = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/graphragbench_medical/postprocessing/indexing/relationship_metadata.pkl")
RAG_TEMPLATE = Path(
    r"C:/Users/paolo/Desktop/Ontology-Induction/generation/lightrag/rag_response_prompt.txt"
)


TOP_K = 2  # risultati per keyword

# ---------------------------------------------------------------------------
# Schema Pydantic per la risposta di Gemini
# ---------------------------------------------------------------------------

class KeywordExtractionResponse(BaseModel):
    high_level_keywords: List[str] = Field(default_factory=list)
    low_level_keywords: List[str] = Field(default_factory=list)

class SimpleResponse(BaseModel):
    response: str


# ---------------------------------------------------------------------------
# Import LLM utilities (Google Gemini)
# ---------------------------------------------------------------------------
try:
    from google import genai  # type: ignore
except ImportError as exc:
    raise SystemExit("google-generativeai non installato – pip install google-generativeai") from exc

try:
    from llm.call_llm import call_gemini  # noqa: WPS433
    from llm.llm_keys import GEMINI_KEY  # noqa: WPS433 – external secret
    gemini_client = genai.Client(api_key=GEMINI_KEY)
except ImportError as exc:
    raise SystemExit(
        "Impossibile importare llm.call_llm o llm.llm_keys – verifica PYTHONPATH"
    ) from exc

# ---------------------------------------------------------------------------
# Import question generator
# ---------------------------------------------------------------------------
try:
    from datasets.graphragbench_medical.graphragbench_medical_questions.graphragbench_medical_questions_generator import (
        generate_questions,
    )
except ImportError as exc:
    raise SystemExit(
        "Impossibile importare il question generator – assicurati che i pacchetti \
        'datasets' abbiano gli __init__.py e che il root progetto sia nel PYTHONPATH."
    ) from exc

# ---------------------------------------------------------------------------
# Lazy‑load embedder e FAISS
# ---------------------------------------------------------------------------
_embedder: SentenceTransformer | None = None
_entities_index: faiss.Index | None = None
_entities_meta: list[dict] | None = None
_rel_index: faiss.Index | None = None
_rel_meta: list[dict] | None = None

def _lazy_load():
    global _embedder, _entities_index, _entities_meta, _rel_index, _rel_meta  # noqa: WPS420
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    if _entities_index is None:
        _entities_index = faiss.read_index(str(ENTITIES_INDEX))
        with open(ENTITIES_META, "rb") as fh:
            _entities_meta = pickle.load(fh)
    if _rel_index is None:
        _rel_index = faiss.read_index(str(REL_INDEX))
        with open(REL_META, "rb") as fh:
            _rel_meta = pickle.load(fh)

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_context(high: List[str], low: List[str]):
    """Restituisce due liste (entità, relazioni) di tuple (name, summary)."""
    _lazy_load()
    assert _embedder and _entities_index and _rel_index
    print("Retrieving context from index...")

    low_ctx: list[tuple[str, str]] = []
    for kw in low:
        vec = _embedder.encode(kw, convert_to_numpy=True).astype(np.float32)
        vec = _l2_normalize(vec)
        _, I = _entities_index.search(vec.reshape(1, -1), TOP_K)
        for idx in I[0]:
            meta = _entities_meta[idx]
            low_ctx.append((meta["name"], meta["summary"]))

    high_ctx: list[tuple[str, str]] = []
    for kw in high:
        vec = _embedder.encode(kw, convert_to_numpy=True).astype(np.float32)
        vec = _l2_normalize(vec)
        _, I = _rel_index.search(vec.reshape(1, -1), TOP_K)
        for idx in I[0]:
            meta = _rel_meta[idx]
            high_ctx.append((meta["name"], meta["summary"]))

    return low_ctx, high_ctx

# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def keyword_extraction(question: str) -> KeywordExtractionResponse:
    print("Keyword extraction...")
    with open(PROMPT_PATH, "r", encoding="utf-8") as fh:
        system_prompt = fh.read()
    return call_gemini(
        gemini_client=gemini_client,
        model_name="models/gemini-2.5-flash-lite-preview-06-17",
        system_prompt=system_prompt,
        user_prompt=question,
        response_schema=KeywordExtractionResponse,
        thinking_budget=512,
        verbose=False,
    )

GRAPH_PATH = Path(
    r"C:/Users/paolo/Desktop/Ontology-Induction/outputs/graphragbench_medical/postprocessing/graph_builder/medical_graph.igraph"
)

_graph = None  # sarà un igraph.Graph

def _lazy_load_graph():
    """Carica medical_graph.igraph solo alla prima chiamata."""
    global _graph  # noqa: WPS420
    if _graph is None:
        try:
            import igraph as ig
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("pip install igraph") from exc
        _graph = ig.Graph.Read_Pickle(str(GRAPH_PATH))


# ---------------------------------------------------------------------------
# (NUOVA) retrieve_graph_context
# ---------------------------------------------------------------------------
def retrieve_graph_context(
    low_entities: list[str],
    high_relations: list[str],
) -> str:
    """
    Costruisce un contesto testuale navigando il grafo iGraph.

    • Per ogni low-level entity:
        – summary del nodo
        – tutti i vicini 1-hop con descrizione
    • Per ogni high-level relation:
        – descrizione della relazione
        – tutte le triple che la usano
    """
    print("Retrieving graph context...")
    _lazy_load_graph()
    ctx_lines: list[str] = []

    # --- nodi / low-level -------------------------------------------------- #
    for ent in low_entities:
        try:
            v = _graph.vs.find(name=ent)
        except ValueError:
            continue  # nodo non presente

        ctx_lines.append(f"[ENTITY] {v['name']} – {v['description']}")
        for n_idx in _graph.neighbors(v, mode="ALL"):
            n_v = _graph.vs[n_idx]
            ctx_lines.append(f"  ↳ neighbor: {n_v['name']} – {n_v['description']}")

    # --- relazioni / high-level ------------------------------------------- #
    for rel in high_relations:
        edges = _graph.es.select(keyword_eq=rel)
        if not edges:
            continue

        ctx_lines.append(f"[RELATION] {rel} – {edges[0]['description']}")
        for e in edges:
            head = _graph.vs[e.source]["name"]
            tail = _graph.vs[e.target]["name"]
            ctx_lines.append(f"  ↳ triple: ({head}) -[{rel}]-> ({tail})")

    return "\n".join(ctx_lines)


# ---------------------------------------------------------------------------
# Generazione risposta RAG
# ---------------------------------------------------------------------------
def rag_generate_response(question: str, context: str) -> str:
    """Costruisce prompt RAG e chiama Gemini, restituisce la risposta."""
    with open(RAG_TEMPLATE, "r", encoding="utf-8") as fh:
        sys_prompt_template = fh.read()

    system_prompt = sys_prompt_template.replace("{context_data}", context)

    resp: SimpleResponse = call_gemini(
        gemini_client=gemini_client,
        model_name="models/gemini-2.5-flash-lite-preview-06-17",
        system_prompt=system_prompt,
        user_prompt=question,
        response_schema=SimpleResponse,
        thinking_budget=512,
        verbose=False,
    )
    return resp.response.strip()



# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    # 1. domanda
    first_record = next(generate_questions(path=QUESTIONS_JSON))
    question_text = first_record["Question"]
    print(f"Question: {question_text}\n")

    # 2. estrazione keyword
    k_resp = keyword_extraction(question_text)
    #  print(f"High-level keywords: {k_resp.high_level_keywords}")
    # print(f"Low-level keywords : {k_resp.low_level_keywords}\n")

    # 3. retrieval su FAISS
    low_ctx, high_ctx = retrieve_context(
        k_resp.high_level_keywords, k_resp.low_level_keywords
    )

    # 4. costruzione contesto dal grafo
    graph_ctx = retrieve_graph_context(
        low_entities=[name for name, _ in low_ctx],
        high_relations=[name for name, _ in high_ctx],
    )

    # 5. generazione risposta RAG
    rag_response = rag_generate_response(question_text, graph_ctx)
    print(f"RAG response: {rag_response}")

if __name__ == "__main__":
    main()