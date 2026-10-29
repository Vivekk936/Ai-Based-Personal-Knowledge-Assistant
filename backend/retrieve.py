"""Hybrid retrieval: dense vectors for meaning, BM25 for exact strings, fused with
Reciprocal Rank Fusion.

Dense-only retrieval fails silently and confidently on exactly the queries an enterprise
cares about. Ask for "DWG-11482 rev C" and an embedding model — which was trained to
generalise across surface form — happily returns DWG-11483, because to it those strings
are near-identical. It always returns *something*, so the user gets a fluent wrong answer
rather than a miss, which is the worst failure mode a document assistant can have.
BM25 is the half that matches literal tokens.

Fusion is RRF rather than weighted score blending because BM25 scores and cosine
distances don't share an axis; normalising them into one requires per-corpus tuning we
have no labelled data to do. RRF only needs rank order, so it has nothing to tune.
"""
import logging

import chromadb

from config import embed_texts
import store
from chunker import embed_input

log = logging.getLogger("ada.retrieve")

# ponytail: Chroma holds the whole index in RSS (~6.5GB at 1M chunks). Fine for a granted
# folder or two; swap to LanceDB if a customer indexes a corpus past ~500k chunks.
_CHROMA_PATH = str(store.DATA_DIR / "vectors")
_client = None
_collection = None

RRF_K = 60  # standard constant; damps the influence of any single list's top rank


def collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=_CHROMA_PATH)
        _collection = _client.get_or_create_collection("chunks")
    return _collection


def index_chunks(rows: list[dict]) -> int:
    """Writes chunks to both halves of the index. SQLite first: it owns the text, and a
    vector with no text behind it is unciteable, whereas text with no vector is still
    reachable by BM25. Ordering the failure that way keeps a crash recoverable."""
    if not rows:
        return 0
    path = rows[0]["path"]
    store.replace_chunks(path, rows)

    vectors = embed_texts(
        [embed_input(r["filename"], r["section"], r["page"], r["text"]) for r in rows]
    )
    col = collection()
    try:
        col.upsert(
            ids=[r["id"] for r in rows],
            embeddings=vectors,
            metadatas=[
                {"doc_id": r["doc_id"], "filename": r["filename"], "page": r["page"], "path": r["path"]}
                for r in rows
            ],
        )
    except Exception as e:
        if "dimension" in str(e).lower():
            log.warning("Embedding dimension mismatch (%s); recreating Chroma collection", e)
            global _collection
            if _client:
                try:
                    _client.delete_collection("chunks")
                except Exception:
                    pass
                _collection = _client.get_or_create_collection("chunks")
                _collection.upsert(
                    ids=[r["id"] for r in rows],
                    embeddings=vectors,
                    metadatas=[
                        {"doc_id": r["doc_id"], "filename": r["filename"], "page": r["page"], "path": r["path"]}
                        for r in rows
                    ],
                )
        else:
            raise
    return len(rows)


def drop_document(path: str) -> None:
    try:
        collection().delete(where={"path": path})
    except Exception as e:
        log.warning("vector delete failed for %s: %s", path, e)
    store.replace_chunks(path, [])


def search_dense(query: str, k: int = 30) -> list[tuple[str, float]]:
    try:
        [vec] = embed_texts([query], input_type="query")
    except Exception as e:
        log.warning("query embedding failed: %s", e)
        return []
    col = collection()
    if col.count() == 0:
        return []
    res = col.query(query_embeddings=[vec], n_results=min(k, col.count()))
    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    return list(zip(ids, dists))


def search(query: str, k: int = 6, candidates: int = 30) -> list[dict]:
    """Returns fused hits, best first. Each carries `rrf` (fusion score) and the ranks it
    came from, because `dense_rank is None` is a useful signal on its own — it means the
    only reason this hit surfaced is a literal string match."""
    dense = search_dense(query, candidates)
    lexical = store.search_lexical(query, candidates)
    if not dense and not lexical:
        return []

    scores: dict[str, float] = {}
    dense_rank = {cid: i for i, (cid, _) in enumerate(dense)}
    lex_rank = {cid: i for i, (cid, _) in enumerate(lexical)}
    for ranks in (dense_rank, lex_rank):
        for cid, rank in ranks.items():
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

    top = sorted(scores, key=scores.get, reverse=True)[:k]
    records = store.get_chunks(top)
    dist = dict(dense)

    hits = []
    for cid in top:
        rec = records.get(cid)
        if not rec:
            continue  # vector index outlived its text (interrupted re-index); skip, don't crash
        hits.append(
            {
                **rec,
                "rrf": round(scores[cid], 5),
                "dense_rank": dense_rank.get(cid),
                "lexical_rank": lex_rank.get(cid),
                "distance": dist.get(cid),
                "agreed": cid in dense_rank and cid in lex_rank,
            }
        )
    return hits


# Both retrievers independently ranking a chunk highly is the strongest cheap signal that
# it is genuinely relevant. Neither putting anything near the top is the strongest signal
# that the corpus simply doesn't contain the answer — which is when the agent must say so
# instead of writing something plausible.
def confidence(hits: list[dict]) -> str:
    if not hits:
        return "none"
    best = hits[0]
    if best["agreed"]:
        return "high"
    if best["dense_rank"] is not None and (best["distance"] or 1.0) < 0.45:
        return "medium"
    if best["lexical_rank"] is not None and best["lexical_rank"] < 3:
        return "medium"
    return "low"


def demo() -> None:
    """Exercises fusion arithmetic without needing a model or a populated index."""
    d = [("a", 0.1), ("b", 0.2), ("c", 0.9)]
    lx = [("c", -5.0), ("a", -1.0)]
    scores: dict[str, float] = {}
    for ranks in ({cid: i for i, (cid, _) in enumerate(d)}, {cid: i for i, (cid, _) in enumerate(lx)}):
        for cid, rank in ranks.items():
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    order = sorted(scores, key=scores.get, reverse=True)
    assert order[0] == "a", f"a is top-1 dense and top-2 lexical, should win: {order}"
    assert "c" in order and "b" in order
    assert scores["c"] > scores["b"], "c appears in both lists, b in one — c must rank higher"

    assert confidence([]) == "none"
    assert confidence([{"agreed": True, "dense_rank": 0, "lexical_rank": 0, "distance": 0.3}]) == "high"
    assert confidence([{"agreed": False, "dense_rank": 0, "lexical_rank": None, "distance": 0.2}]) == "medium"
    assert confidence([{"agreed": False, "dense_rank": 9, "lexical_rank": None, "distance": 0.8}]) == "low"
    print("PASS retrieve — RRF ordering and confidence tiers")


if __name__ == "__main__":
    demo()
