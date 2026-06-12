"""
embed_and_retrieve.py
Milestone 4 — Embedding + Vector Store + Retrieval

Pipeline stage (from planning.md architecture):

  chunks.json  (Milestone 3 output)
       │
       ▼
  ┌─────────────────────────────┐
  │  Embedding + Vector Store   │
  │  • all-MiniLM-L6-v2         │
  │  • ChromaDB                 │
  └──────────────┬──────────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │         Retrieval           │
  │  Semantic + keyword match   │
  │  Tool: ChromaDB             │
  └─────────────────────────────┘

Usage
-----
# One-time ingestion — builds (or refreshes) the vector store on disk:
    python embed_and_retrieve.py --ingest

# Query the store:
    python embed_and_retrieve.py --query "Is CSE 373 good for software engineers?"

# Both in one go:
    python embed_and_retrieve.py --ingest --query "How hard is MATH 126?"

Dependencies
------------
    pip install sentence-transformers chromadb
"""

import argparse
import json
import os
import re
from pathlib import Path

# ── third-party (install before running) ────────────────────────────────────
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ── Paths ────────────────────────────────────────────────────────────────────
CHUNKS_FILE   = Path(__file__).parent / "chunks.json"
CHROMA_DIR    = Path(__file__).parent / "chroma_store"
COLLECTION_NAME = "course_reviews"

# ── Retrieval config (from planning.md) ─────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # local model, no API key needed
TOP_K           = 6                     # planning.md: "Top-k: 5 to 6 chunks"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Embedding model
# ─────────────────────────────────────────────────────────────────────────────

def load_embedding_model() -> SentenceTransformer:
    """
    Load all-MiniLM-L6-v2 from sentence-transformers.
    The first call downloads the model (~22 MB) and caches it locally.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings.
    Returns a list of float vectors, one per input text.
    batch_size=64 keeps memory use reasonable on CPU.
    """
    vectors = model.encode(texts, batch_size=64, show_progress_bar=True)
    return vectors.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 2. ChromaDB setup
# ─────────────────────────────────────────────────────────────────────────────

def get_chroma_collection(persist_dir: Path) -> chromadb.Collection:
    """
    Create (or open) a persistent ChromaDB collection.
    Uses the new chromadb >=0.4 PersistentClient API.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    # get_or_create is idempotent — safe to call on reruns
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )
    return collection


# ─────────────────────────────────────────────────────────────────────────────
# 3. Metadata flattening
#    ChromaDB metadata values must be str | int | float | bool — no dicts.
#    The nested `ratings` dict is flattened to individual keys.
# ─────────────────────────────────────────────────────────────────────────────

def flatten_metadata(chunk: dict) -> dict:
    """
    Convert a chunk dict into a flat metadata dict safe for ChromaDB.
    None values become empty strings so ChromaDB never receives null.
    """
    ratings = chunk.get("ratings") or {}
    return {
        "course":            chunk.get("course") or "",
        "source_url":        chunk.get("source_url") or "",
        "professor":         chunk.get("professor") or "",
        "term":              chunk.get("term") or "",
        "grade":             chunk.get("grade") or "",
        "workload":          chunk.get("workload") or "",
        "textbook":          chunk.get("textbook") or "",
        "tokens":            chunk.get("tokens") or 0,
        "rating_overall":    ratings.get("overall") or 0,
        "rating_difficulty": ratings.get("difficulty") or 0,
        "rating_interest":   ratings.get("interest") or 0,
        "rating_usefulness": ratings.get("usefulness") or 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ingestion — load chunks → embed → upsert into ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def ingest(chunks_file: Path, persist_dir: Path) -> None:
    """
    Load chunks.json, embed every chunk's text with all-MiniLM-L6-v2,
    and upsert into ChromaDB with full source metadata.

    Upsert (not add) means re-running this is safe — existing entries
    are overwritten rather than duplicated.
    """
    # --- load chunks --------------------------------------------------------
    print(f"\n[1/4] Loading chunks from {chunks_file}")
    with open(chunks_file, encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)
    print(f"      {len(chunks)} chunks loaded")

    # --- embed --------------------------------------------------------------
    print(f"\n[2/4] Embedding with {EMBEDDING_MODEL}")
    model = load_embedding_model()
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(model, texts)
    print(f"      {len(embeddings)} embeddings produced")

    # --- prepare ChromaDB inputs --------------------------------------------
    print("\n[3/4] Preparing ChromaDB documents")
    ids        = [c["chunk_id"]          for c in chunks]
    metadatas  = [flatten_metadata(c)    for c in chunks]
    documents  = texts                   # ChromaDB stores the raw text too

    # --- upsert into collection ---------------------------------------------
    print(f"\n[4/4] Upserting into ChromaDB collection '{COLLECTION_NAME}'")
    collection = get_chroma_collection(persist_dir)

    # Upsert in batches of 50 to stay well within ChromaDB's default limits
    BATCH = 50
    for start in range(0, len(ids), BATCH):
        end = start + BATCH
        collection.upsert(
            ids        = ids[start:end],
            embeddings = embeddings[start:end],
            documents  = documents[start:end],
            metadatas  = metadatas[start:end],
        )
    print(f"      Collection now has {collection.count()} documents")
    print(f"      Persisted to: {persist_dir.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
    top_k: int = TOP_K,
    filter_course: str | None = None,
    filter_professor: str | None = None,
) -> list[dict]:
    """
    Semantic retrieval: embed the query and return the top-k most similar
    chunks from ChromaDB, optionally filtered by course or professor name.

    Parameters
    ----------
    query            : natural-language question
    model            : loaded SentenceTransformer (reuse across calls)
    collection       : open ChromaDB collection
    top_k            : number of results (planning.md: 5-6)
    filter_course    : e.g. "CSE 373" — restricts search to one course
    filter_professor : e.g. "Kasey Champion" — restricts to one professor

    Returns
    -------
    List of result dicts, each with keys:
        chunk_id, text, distance, course, professor, term,
        source_url, grade, workload, textbook,
        rating_overall, rating_difficulty, rating_interest, rating_usefulness
    """
    # Build optional ChromaDB `where` filter
    where: dict | None = None
    if filter_course and filter_professor:
        where = {"$and": [
            {"course":    {"$eq": filter_course}},
            {"professor": {"$eq": filter_professor}},
        ]}
    elif filter_course:
        where = {"course": {"$eq": filter_course}}
    elif filter_professor:
        where = {"professor": {"$eq": filter_professor}}

    # Embed the query (single string → 1-element list → unwrap)
    query_vector = model.encode([query]).tolist()[0]

    # Query ChromaDB
    kwargs = dict(
        query_embeddings = [query_vector],
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    # Unpack and return as a clean list of dicts
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "chunk_id":          meta.get("course", "?") + " | " + meta.get("professor", "?"),
            "text":              doc,
            "distance":          round(dist, 4),
            "course":            meta.get("course"),
            "professor":         meta.get("professor"),
            "term":              meta.get("term"),
            "source_url":        meta.get("source_url"),
            "grade":             meta.get("grade"),
            "workload":          meta.get("workload"),
            "textbook":          meta.get("textbook"),
            "rating_overall":    meta.get("rating_overall"),
            "rating_difficulty": meta.get("rating_difficulty"),
            "rating_interest":   meta.get("rating_interest"),
            "rating_usefulness": meta.get("rating_usefulness"),
        })
    return output


def print_results(query: str, results: list[dict]) -> None:
    """Pretty-print retrieval results to the terminal."""
    print(f"\n{'='*64}")
    print(f"Query: {query!r}")
    print(f"Top {len(results)} results (cosine distance — lower = more similar):")
    print(f"{'='*64}")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['course']} — {r['professor']} ({r['term']})")
        print(f"    Distance  : {r['distance']}")
        print(f"    Ratings   : overall={r['rating_overall']}  "
              f"difficulty={r['rating_difficulty']}  "
              f"interest={r['rating_interest']}  "
              f"usefulness={r['rating_usefulness']}")
        print(f"    Grade     : {r['grade']}  |  Workload: {r['workload']}")
        print(f"    Source    : {r['source_url']}")
        print(f"    Text      : {r['text'][:300]}{'...' if len(r['text']) > 300 else ''}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Embed course review chunks into ChromaDB and/or query them."
    )
    p.add_argument(
        "--ingest", action="store_true",
        help="Load chunks.json, embed with all-MiniLM-L6-v2, upsert into ChromaDB."
    )
    p.add_argument(
        "--query", type=str, default=None,
        help="Natural-language query to retrieve relevant chunks."
    )
    p.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Number of results to return (default: {TOP_K})."
    )
    p.add_argument(
        "--course", type=str, default=None,
        help="Filter results to a specific course, e.g. 'CSE 373'."
    )
    p.add_argument(
        "--professor", type=str, default=None,
        help="Filter results to a specific professor name."
    )
    p.add_argument(
        "--chunks-file", type=Path, default=CHUNKS_FILE,
        help=f"Path to chunks.json (default: {CHUNKS_FILE})."
    )
    p.add_argument(
        "--chroma-dir", type=Path, default=CHROMA_DIR,
        help=f"Directory for the ChromaDB store (default: {CHROMA_DIR})."
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not args.ingest and not args.query:
        print("Nothing to do — pass --ingest, --query, or both. Use --help for options.")
        return

    if args.ingest:
        ingest(args.chunks_file, args.chroma_dir)

    if args.query:
        print(f"\nLoading model and collection for query…")
        model      = load_embedding_model()
        collection = get_chroma_collection(args.chroma_dir)

        if collection.count() == 0:
            print("ERROR: ChromaDB collection is empty. Run with --ingest first.")
            return

        results = retrieve(
            query            = args.query,
            model            = model,
            collection       = collection,
            top_k            = args.top_k,
            filter_course    = args.course,
            filter_professor = args.professor,
        )
        print_results(args.query, results)


if __name__ == "__main__":
    main()
