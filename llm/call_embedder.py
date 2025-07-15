from __future__ import annotations

from typing import Iterable, List

import numpy as np
from openai import OpenAI
from openai._types import NOT_GIVEN

from llm.llm_keys import OPENAI_KEY  # type: ignore

__all__ = ["get_embeddings_sync"]

# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "text-embedding-3-small"
_CLIENT = OpenAI(api_key=OPENAI_KEY)


def get_embeddings_sync(
    texts: List[str],
    model: str = _DEFAULT_MODEL,
    client: OpenAI | None = None,
) -> np.ndarray:
    """Embedding sincrono (batch unico). Restituisce np.ndarray (float32)."""
    if client is None:
        client = _CLIENT
    resp = client.embeddings.create(model=model, input=texts)
    vectors = [d.embedding for d in resp.data]
    return np.asarray(vectors, dtype="float32")


if __name__ == "__main__":
    texts = ["Hello, world!", "This is a test", "Another test"]
    embeddings = get_embeddings_sync(texts)
    print(embeddings)