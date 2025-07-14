"""
lightrag_pipeline.py  –  keyword-to-graph RAG pipeline (stateless + configurable)

Funzioni pubbliche principali
-----------------------------
• keyword_extraction(question, prompt_path, gemini_model) -> KeywordExtractionResponse
• retrieve_context(high, low, *, entities_index, entities_meta, rel_index, rel_meta, top_k=2)
• retrieve_graph_context(low_entities, high_relations, *, graph_path) -> str
• rag_generate_response(question, context, rag_template_path, gemini_model) -> str

Gli oggetti (embedder, indici FAISS, grafo) sono cache-ati per path, quindi
richiamare più volte con gli stessi file non ricarica da disco.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------#
# Imports standard
# ---------------------------------------------------------------------------#
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from datasets.graphragbench_medical.graphragbench_medical_questions.graphragbench_medical_questions_generator import generate_questions




# ---------------------------------------------------------------------------#
# LLM (Gemini) – lazily imported per evitare dipendenza forte
# ---------------------------------------------------------------------------#
try:
    from google import genai  # type: ignore
    from llm.call_llm import call_gemini  # noqa: WPS433
    from llm.llm_keys import GEMINI_KEY  # noqa: WPS433
    _GEMINI_CLIENT = genai.Client(api_key=GEMINI_KEY)
except Exception as exc:  # pragma: no cover
    raise SystemExit("Impossibile importare Google Gemini; verifica l'installazione.") from exc

# ---------------------------------------------------------------------------#
# Pydantic response schemas
# ---------------------------------------------------------------------------#
class KeywordExtractionResponse(BaseModel):
    high_level_keywords: List[str] = Field(default_factory=list)
    low_level_keywords: List[str] = Field(default_factory=list)


class SimpleResponse(BaseModel):
    response: str


# ---------------------------------------------------------------------------#
# Lazy caches
# ---------------------------------------------------------------------------#
_EMBEDDER: SentenceTransformer | None = None
_INDEX_CACHE: dict[str, "faiss.Index"] = {}
_META_CACHE: dict[str, list[dict]] = {}
_GRAPH_CACHE: dict[str, "igraph.Graph"] = {}

# ---------------------------------------------------------------------------#
# Helpers
# ---------------------------------------------------------------------------#
def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        # forza il caricamento su CPU (o "cuda" se la tua installazione GPU è a posto)
        _EMBEDDER = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
    return _EMBEDDER


def _load_index(idx_path: Path, meta_path: Path) -> Tuple["faiss.Index", list[dict]]:
    import faiss  # local import to avoid mandatory dependency if not used

    key = str(idx_path)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = faiss.read_index(str(idx_path))
        with open(meta_path, "rb") as fh:
            _META_CACHE[key] = pickle.load(fh)
    return _INDEX_CACHE[key], _META_CACHE[key]


def _load_graph(graph_path: Path):
    import igraph as ig  # local import

    key = str(graph_path)
    if key not in _GRAPH_CACHE:
        _GRAPH_CACHE[key] = ig.Graph.Read_Pickle(str(graph_path))
    return _GRAPH_CACHE[key]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


# ---------------------------------------------------------------------------#
# 1. Keyword extraction (LLM)
# ---------------------------------------------------------------------------#
def keyword_extraction(
    question: str,
    prompt_path: Path,
    gemini_model: str = "models/gemini-2.5-flash-lite-preview-06-17",
) -> KeywordExtractionResponse:
    """
    Estrae high/low-level keywords da una domanda via Gemini.
    `prompt_path` deve contenere il system prompt da usare.
    """
    system_prompt = Path(prompt_path).read_text(encoding="utf-8")
    return call_gemini(
        gemini_client=_GEMINI_CLIENT,
        model_name=gemini_model,
        system_prompt=system_prompt,
        user_prompt=question,
        response_schema=KeywordExtractionResponse,
        thinking_budget=512,
        verbose=False,
    )


# ---------------------------------------------------------------------------#
# 2. Retrieval da indici FAISS
# ---------------------------------------------------------------------------#
def retrieve_context(
    *,
    high: List[str],
    low: List[str],
    entities_index: Path,
    entities_meta: Path,
    rel_index: Path,
    rel_meta: Path,
    top_k: int = 2,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Restituisce (low_ctx, high_ctx) ciascuna lista di tuple (name, summary).

    • `entities_*` → per i low-level keywords
    • `rel_*`      → per i high-level keywords
    """
    import faiss  # type: ignore – local import

    embedder = _get_embedder()
    ent_idx, ent_meta = _load_index(entities_index, entities_meta)
    rel_idx, rel_meta = _load_index(rel_index, rel_meta)

    low_ctx: list[tuple[str, str]] = []
    for kw in low:
        vec = _l2_normalize(embedder.encode(kw, convert_to_numpy=True).astype(np.float32))
        _, I = ent_idx.search(vec.reshape(1, -1), top_k)
        low_ctx.extend((ent_meta[i]["name"], ent_meta[i]["summary"]) for i in I[0])

    high_ctx: list[tuple[str, str]] = []
    for kw in high:
        vec = _l2_normalize(embedder.encode(kw, convert_to_numpy=True).astype(np.float32))
        _, I = rel_idx.search(vec.reshape(1, -1), top_k)
        high_ctx.extend((rel_meta[i]["name"], rel_meta[i]["summary"]) for i in I[0])

    return low_ctx, high_ctx


# ---------------------------------------------------------------------------#
# 3. Retrieval dal grafo iGraph
# ---------------------------------------------------------------------------#
def retrieve_graph_context(
    *,
    low_entities: List[str],
    high_relations: List[str],
    graph_path: Path,
) -> str:
    """
    Costruisce un contesto testuale navigando il grafo specificato.
    Ritorna una stringa multilinea.
    """
    g = _load_graph(graph_path)
    ctx_lines: List[str] = []

    # nodi
    for ent in low_entities:
        try:
            v = g.vs.find(name=ent)
        except ValueError:
            continue
        ctx_lines.append(f"[ENTITY] {v['name']} – {v['description']}")
        for nbr_idx in g.neighbors(v, mode="ALL"):
            n = g.vs[nbr_idx]
            ctx_lines.append(f"  ↳ neighbor: {n['name']} – {n['description']}")

    # relazioni
    for rel in high_relations:
        edges = g.es.select(keyword_eq=rel)
        if not edges:
            continue
        ctx_lines.append(f"[RELATION] {rel} – ")
        for e in edges:
            head = g.vs[e.source]["name"]
            tail = g.vs[e.target]["name"]
            ctx_lines.append(f"  ↳ triple: ({head}) -[{rel}]-> ({tail})")

    return "\n".join(ctx_lines)


# ---------------------------------------------------------------------------#
# 4. RAG generation
# ---------------------------------------------------------------------------#
def rag_generate_response(
    question: str,
    context: str,
    rag_template_path: Path,
    gemini_model: str = "models/gemini-2.5-flash-lite-preview-06-17",
) -> str:
    """
    Genera risposta RAG data la domanda `question` e il `context` testuale.
    Il file `rag_template_path` deve contenere il prompt con il placeholder
    `{context_data}`.
    """
    sys_prompt_template = Path(rag_template_path).read_text(encoding="utf-8")
    system_prompt = sys_prompt_template.replace("{context_data}", context)

    resp: SimpleResponse = call_gemini(
        gemini_client=_GEMINI_CLIENT,
        model_name=gemini_model,
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
    BASE_PATH = Path("C:/Users/paolo/Desktop/Ontology-Induction/datasets")
    QUESTIONS_JSON = BASE_PATH / "graphragbench_medical" / "graphragbench_medical_questions" / "graphragbench_medical_questions.json"
    OUTPUT_PATH = Path("C:/Users/paolo/Desktop/Ontology-Induction/outputs/exp_7_13/graphragbench_medical")
    first_record = next(generate_questions(
        path=QUESTIONS_JSON,
        split="test",
        fraction=0.01,
        seed=42,
    ))
    question_text = first_record["Question"]
    print(f"Question: {question_text}\n")

    # 2. estrazione keyword
    k_resp = keyword_extraction(question_text, prompt_path=Path("C:/Users/paolo/Desktop/Ontology-Induction/generation/lightrag/keyword_extraction_prompt.txt"))
    #  print(f"High-level keywords: {k_resp.high_level_keywords}")
    # print(f"Low-level keywords : {k_resp.low_level_keywords}\n")

    # 3. retrieval su FAISS
    low_ctx, high_ctx = retrieve_context(
        high=k_resp.high_level_keywords,
        low=k_resp.low_level_keywords,
        entities_index=OUTPUT_PATH / "pragmarag_prompt" / "postprocessing" / "indexing" / "entities.index",
        entities_meta=OUTPUT_PATH / "pragmarag_prompt" / "postprocessing" / "indexing" / "entity_metadata.pkl",
        rel_index=OUTPUT_PATH / "pragmarag_prompt" / "postprocessing" / "indexing" / "relationships.index",
        rel_meta=OUTPUT_PATH / "pragmarag_prompt" / "postprocessing" / "indexing" / "relationship_metadata.pkl",
        top_k=2,
    )

    # 4. costruzione contesto dal grafo
    graph_ctx_lightrag = retrieve_graph_context(
        low_entities=[name for name, _ in low_ctx],
        high_relations=[name for name, _ in high_ctx],
        graph_path=OUTPUT_PATH / "lightrag" / "postprocessing" / "graph_builder" / "graphragbench_medical.igraph",
    )
    print(graph_ctx)

    # 5. generazione risposta RAG
    rag_response = rag_generate_response(question_text, graph_ctx, 
                                         rag_template_path=Path("C:/Users/paolo/Desktop/Ontology-Induction/generation/lightrag/rag_response_prompt.txt"))
    print(f"RAG response: {rag_response}")
    """
    # 6-bis. Coherence evaluation   
    gold_answer = first_record["Answer"]
    coherence = evaluate_coherence(
        gemini_client=gemini_client,
        model_name="models/gemini-2.5-flash-lite-preview-06-17",
        question=question_text,
        gold_answer=gold_answer,
        model_answer=rag_response,
        judge_name="GraphRAG-Gemini",
    )
    """

    # 7. Output finale
    print("\n========================")
    print("QUESTION:")
    print(question_text)
    print("\nANSWER (GraphRAG-Gemini):")
    print(rag_response)


if __name__ == "__main__":
    main()