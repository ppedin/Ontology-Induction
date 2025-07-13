"""
LLM-as-a-Judge – Coherence Evaluation
=====================================
Valuta quanto la risposta del modello è coerente con la risposta di
riferimento (“gold”) per la stessa domanda.

• coherence_score : intero 0-10
    0   = totalmente incoerente / contraddittoria
    10  = perfettamente coerente (nessuna discrepanza fattuale o logica)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from llm.call_llm import call_gemini  # noqa: WPS433


class CoherenceResponse(BaseModel):
    coherence_score: int = Field(
        ge=0,
        le=10,
        description="Integer 0–10 measuring answer coherence with gold answer",
    )


SYSTEM_PROMPT = """
You are an expert evaluator. Your task is to judge **coherence** between
a candidate answer provided by an AI model and a gold reference answer
for the same question.

* “Coherence” measures factual alignment, logical consistency, and
  completeness **relative to the reference**, NOT fluency or style.
* Score **10** if the candidate matches the reference in all factual
  points, with no contradictions or omissions.
* Score **0** if the candidate contradicts, ignores, or is unrelated to
  the reference answer.
* Use ONLY integers 0–10, no text, no decimals.

Return a JSON object:
{ "coherence_score": <integer 0-10> }
"""


def evaluate_coherence(
    gemini_client: "genai.Client",  # type: ignore
    model_name: str,
    question: str,
    gold_answer: str,
    model_answer: str,
    judge_name: str,
) -> CoherenceResponse:
    """
    Invoca Gemini per ottenere uno score di coerenza (0-10).

    Parameters
    ----------
    gemini_client : genai.Client
    model_name    : quale modello Gemini usare
    question      : la domanda originale
    gold_answer   : risposta di riferimento (ground truth)
    model_answer  : risposta generata dal sistema da valutare
    judge_name    : nome/etichetta del sistema valutato (incluso nel prompt)

    Returns
    -------
    CoherenceResponse
        oggetto Pydantic con campo ``coherence_score``.
    """
    user_prompt = (
        f"### Question\n{question}\n\n"
        f"### Gold Answer\n{gold_answer}\n\n"
        f"### {judge_name} Answer\n{model_answer}\n\n"
        "Provide only the JSON with the coherence_score."
    )

    return call_gemini(
        gemini_client=gemini_client,
        model_name=model_name,
        system_prompt=SYSTEM_PROMPT.strip(),
        user_prompt=user_prompt,
        response_schema=CoherenceResponse,
        thinking_budget=512,
        verbose=False,
    )
