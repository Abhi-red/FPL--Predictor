"""Retrieve the most relevant recent news chunks for a player.

    from rag.retrieve import retrieve
    retrieve("Bukayo Saka", k=5)  # -> list of chunk dicts, best first

Uses the FAISS index written by src/rag/embed.py when it (and the faiss package)
is available, otherwise a brute-force cosine over the ``news_vectors.npy`` matrix
that embed.py always writes. The searcher and embedding model are cached at
module level. Raises FileNotFoundError if embed.py hasn't run yet.
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_connection  # noqa: E402
from rag.embed import (  # noqa: E402
    IDS_PATH,
    INDEX_PATH,
    VECTORS_PATH,
    _load_model,
    embed_texts,
)


class _NumpySearcher:
    """Brute-force inner-product search over the normalised embedding matrix."""

    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix

    def search(self, query: np.ndarray, k: int):
        sims = self._matrix @ query[0]
        top = np.argsort(-sims)[:k]
        return sims[top][None, :], top[None, :]


@lru_cache(maxsize=1)
def _searcher():
    if not IDS_PATH.exists():
        raise FileNotFoundError(
            f"{IDS_PATH} not found - run `python src/rag/embed.py` first"
        )
    ids = np.load(IDS_PATH)
    if INDEX_PATH.exists():
        try:
            import faiss

            return faiss.read_index(str(INDEX_PATH)), ids
        except Exception:  # noqa: BLE001 - fall through to numpy
            pass
    if not VECTORS_PATH.exists():
        raise FileNotFoundError(
            f"neither {INDEX_PATH} nor {VECTORS_PATH} is usable - re-run embed.py"
        )
    return _NumpySearcher(np.load(VECTORS_PATH).astype("float32")), ids


@lru_cache(maxsize=1)
def _model():
    return _load_model()


def _surname(player_name: str) -> str:
    parts = [p for p in player_name.replace(".", " ").split() if len(p) >= 3]
    return parts[-1].lower() if parts else player_name.lower()


def retrieve(player_name: str, k: int = 5) -> list[dict]:
    """Top-k recent chunks for `player_name`, re-ranked to prefer surname mentions."""
    searcher, ids = _searcher()
    query = f"{player_name} injury team news fitness lineup availability"
    vector = embed_texts(_model(), [query])
    scores, positions = searcher.search(vector, min(max(k * 4, 20), len(ids)))

    hit_ids = [str(ids[p]) for p in positions[0] if p >= 0]
    if not hit_ids:
        return []
    placeholders = ",".join("?" * len(hit_ids))
    with get_connection() as conn:
        rows = {
            r["chunk_id"]: dict(r)
            for r in conn.execute(
                f"SELECT * FROM news_chunks WHERE chunk_id IN ({placeholders})", hit_ids
            )
        }

    surname = _surname(player_name)
    results: list[dict] = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue
        chunk = rows.get(str(ids[position]))
        if chunk is None:
            continue
        chunk["score"] = float(score) + (0.15 if surname in chunk["text"].lower() else 0.0)
        results.append(chunk)

    results.sort(key=lambda c: c["score"], reverse=True)
    return results[:k]


if __name__ == "__main__":
    name = " ".join(sys.argv[1:]) or "Bukayo Saka"
    for chunk in retrieve(name):
        print(f"[{chunk['score']:.3f}] {chunk['source']} - {chunk['title']}")
        print(f"    {chunk['text'][:160]}...")
