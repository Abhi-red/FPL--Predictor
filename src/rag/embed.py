"""Scrape a few public football-news feeds, chunk + embed them, build the index.

Sources (all free, low-friction RSS; a failing source is logged and skipped,
never fatal):
  * BBC Sport - Football
  * Sky Sports - Football
  * The Guardian - Football

Chunks older than NEWS_MAX_AGE_DAYS are aged out of ``news_chunks`` before the
vector store is rebuilt from every surviving row, so retrieval stays recent.

Embeddings use ``sentence-transformers`` (EMBED_MODEL). If that package can't be
imported in the current environment we fall back to a dependency-light
scikit-learn HashingVectorizer so the pipeline still runs (see DECISIONS.md).

Artifacts under ``data/faiss/``:
  * ``news_vectors.npy`` — normalised embedding matrix (always written)
  * ``news_ids.npy``     — chunk_id per row, in order
  * ``news.index``       — FAISS IndexFlatIP (written when faiss is importable)

Run:
    python src/rag/embed.py
"""

import hashlib
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import (  # noqa: E402
    EMBED_MODEL,
    FPL_HEADERS,
    MAX_ATTEMPTS,
    NEWS_MAX_AGE_DAYS,
    REQUEST_TIMEOUT,
)
from db import get_connection, init_db  # noqa: E402

FAISS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "faiss"
INDEX_PATH = FAISS_DIR / "news.index"
VECTORS_PATH = FAISS_DIR / "news_vectors.npy"
IDS_PATH = FAISS_DIR / "news_ids.npy"

RSS_SOURCES: dict[str, str] = {
    "BBC Sport": "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "Sky Sports": "https://www.skysports.com/rss/12040",
    "The Guardian": "https://www.theguardian.com/football/rss",
}

CHUNK_CHARS = 600
CHUNK_OVERLAP = 100
EMBED_DIM = 384  # all-MiniLM-L6-v2, and the hashing-fallback width

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text or "")).strip()


def _get(url: str) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(url, headers=FPL_HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.RequestException as error:
            last_error = error
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not fetch {url}") from last_error


def _struct_to_iso(parsed) -> str:
    if parsed:
        return datetime(*parsed[:6], tzinfo=timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def fetch_rss(source: str, url: str) -> list[dict]:
    resp = _get(url)
    # Pass the HTTP headers so feedparser reads the real charset (some feeds,
    # e.g. the Guardian, otherwise get mis-decoded and mangle accented names).
    feed = feedparser.parse(resp.content, response_headers=dict(resp.headers))
    articles = []
    for entry in feed.entries:
        text = _clean(f"{entry.get('title', '')}. {entry.get('summary', '')}")
        if len(text) < 40:
            continue
        articles.append(
            {
                "source": source,
                "url": entry.get("link", url),
                "title": _clean(entry.get("title", ""))[:300],
                "published_at": _struct_to_iso(entry.get("published_parsed")),
                "text": text,
            }
        )
    return articles


def collect_articles() -> list[dict]:
    articles: list[dict] = []
    for source, url in RSS_SOURCES.items():
        try:
            found = fetch_rss(source, url)
            print(f"  {source}: {len(found)} articles")
            articles.extend(found)
        except Exception as error:  # noqa: BLE001 - one bad source must not abort
            print(f"  {source}: FAILED ({error})", file=sys.stderr)

    by_url: dict[str, dict] = {}
    for article in articles:
        by_url.setdefault(article["url"], article)
    return list(by_url.values())


def chunk_article(article: dict) -> list[dict]:
    text = article["text"]
    step = CHUNK_CHARS - CHUNK_OVERLAP
    pieces = [text[i : i + CHUNK_CHARS] for i in range(0, max(len(text), 1), step)]
    now = datetime.now(timezone.utc).isoformat()
    chunks = []
    for index, piece in enumerate(pieces):
        if not piece.strip():
            continue
        chunk_id = hashlib.sha1(f"{article['url']}#{index}".encode()).hexdigest()
        chunks.append(
            {
                "chunk_id": chunk_id,
                "source": article["source"],
                "url": article["url"],
                "title": article["title"],
                "published_at": article["published_at"],
                "text": piece.strip(),
                "ingested_at": now,
            }
        )
    return chunks


UPSERT_CHUNK = """
INSERT INTO news_chunks (chunk_id, source, url, title, published_at, text, ingested_at)
VALUES (:chunk_id, :source, :url, :title, :published_at, :text, :ingested_at)
ON CONFLICT(chunk_id) DO UPDATE SET
    title = excluded.title, text = excluded.text, ingested_at = excluded.ingested_at
"""


# --------------------------------------------------------------------------- #
# Embedding backend
# --------------------------------------------------------------------------- #
class _HashingEmbedder:
    """Fallback when sentence-transformers is unavailable: L2-normalised hashing."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vec = HashingVectorizer(
            n_features=EMBED_DIM, alternate_sign=False, norm="l2", stop_words="english"
        )

    def encode(self, texts: list[str], **_) -> np.ndarray:
        return self._vec.transform(texts).toarray().astype("float32")


def _load_model():
    """A SentenceTransformer if importable, else the hashing fallback."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBED_MODEL)
    except Exception as error:  # noqa: BLE001
        print(
            f"  sentence-transformers unavailable ({error}); "
            f"using hashing-vectorizer fallback",
            file=sys.stderr,
        )
        return _HashingEmbedder()


def embed_texts(model, texts: list[str]) -> np.ndarray:
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-8, None)


def rebuild_index() -> int:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT chunk_id, text FROM news_chunks ORDER BY chunk_id"
        ).fetchall()
    if not rows:
        print("  no chunks to index", file=sys.stderr)
        return 0

    model = _load_model()
    vectors = embed_texts(model, [r["text"] for r in rows])

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_PATH, vectors)
    np.save(IDS_PATH, np.array([r["chunk_id"] for r in rows], dtype="<U40"))
    try:
        import faiss

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(INDEX_PATH))
    except Exception as error:  # noqa: BLE001
        INDEX_PATH.unlink(missing_ok=True)
        print(f"  faiss unavailable ({error}); retrieval will use numpy", file=sys.stderr)
    return len(rows)


def main() -> None:
    init_db()
    print("Collecting articles")
    articles = collect_articles()

    chunks = [chunk for article in articles for chunk in chunk_article(article)]
    with get_connection() as conn:
        conn.executemany(UPSERT_CHUNK, chunks)

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=NEWS_MAX_AGE_DAYS)
    ).isoformat()
    with get_connection() as conn:
        deleted = conn.execute(
            "DELETE FROM news_chunks WHERE published_at < ?", (cutoff,)
        ).rowcount

    indexed = rebuild_index()
    print(
        f"Upserted {len(chunks)} chunks from {len(articles)} articles; "
        f"aged out {deleted}; store now holds {indexed} chunks"
    )


if __name__ == "__main__":
    main()
