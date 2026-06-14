"""
generate_and_run.py
Milestone 5 — Generation + Interface

Completes the RAG pipeline from planning.md:

  ChromaDB
     │
     ▼
  ┌─────────────────────────────┐
  │         Generation          │
  │  Grounded answer with       │
  │  citations to the original  │
  │  review page                │
  │  LLM: Groq                  │
  │  (llama-3.3-70b-versatile)  │
  └─────────────────────────────┘

Grounding guarantee — three-layer enforcement
─────────────────────────────────────────────
1. SYSTEM PROMPT  — hard rules (not suggestions): the LLM is told it has
   no knowledge of its own about UW courses; it MUST cite every claim with
   a [SOURCE N] tag; it MUST NOT use URLs or names in citations; it MUST
   reply with ONLY_CONTEXT_INSUFFICIENT if the context is not enough.

2. CONTEXT FORMAT — each retrieved chunk is labelled [SOURCE N] in the
   prompt. The LLM can only reference labels that exist.

3. POST-PROCESSOR — after generation, Python:
     a) Parses every [SOURCE N] tag actually present in the LLM output.
     b) Rejects any N that is out of range (hallucinated source number).
     c) Builds the Sources section programmatically from metadata —
        the LLM never writes a URL; Python does.
     d) If zero valid citations remain, the answer is rejected and a
        canned "insufficient context" reply is returned instead.

Usage
─────
    python generate_and_run.py                            # interactive REPL
    python generate_and_run.py --query "How is CSE 373?"  # single query
    python generate_and_run.py --query "..." --course "CSE 373"
    python generate_and_run.py --query "..." --professor "Kasey Champion"

Dependencies
────────────
    pip install sentence-transformers chromadb groq
    export GROQ_API_KEY="your-key-here"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

# ── load .env before anything else touches os.environ ────────────────────────
# Reads GROQ_API_KEY from a .env file in the project root automatically.
# This means you never need to run `export` or `set` manually.
# On Windows (PowerShell/CMD), `export` is not a valid command — use .env instead.
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass  # python-dotenv not installed; falls back to os.environ

# ── third-party ──────────────────────────────────────────────────────────────
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb

# ── sibling modules ───────────────────────────────────────────────────────────
from embed_and_retrieve import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    get_chroma_collection,
    load_embedding_model,
    retrieve,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

GROQ_MODEL        = "llama-3.3-70b-versatile"
MAX_TOKENS        = 1024

# Sentinel the LLM must emit when context is truly insufficient.
# Using an unlikely token sequence avoids false positives.
INSUFFICIENT_SENTINEL = "ONLY_CONTEXT_INSUFFICIENT"

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
#
# Design principles:
#   • Opens with an identity constraint ("You are a retrieval-grounded
#     assistant…") so the model cannot lean on parametric knowledge.
#   • Every rule is a MUST / MUST NOT, not a "try to" or "prefer".
#   • The citation format is concrete and machine-parseable: [SOURCE N].
#   • The sentinel is explicit and unambiguous.
#   • Output format is specified so post-processing is deterministic.
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a retrieval-grounded question-answering assistant for University of \
Washington course and professor reviews. You have NO knowledge of your own \
about these courses or professors. Every fact you state MUST come directly \
from the numbered sources provided in the user message.

CITATION RULES — these are requirements, not suggestions:
1. Every factual claim in your answer MUST be followed immediately by one or \
more citation tags in the form [SOURCE N], where N is the number of the source \
you are drawing from.
2. You MUST NOT cite a source number that was not provided to you.
3. You MUST NOT include URLs, professor names from memory, or any other \
attribution in your answer — use ONLY [SOURCE N] tags.
4. If the provided sources contain enough information to answer the question, \
produce a clear, concise answer with [SOURCE N] citations after each claim.
5. If the provided sources do NOT contain enough information to answer the \
question, you MUST reply with exactly the single word: """ + INSUFFICIENT_SENTINEL + """

OUTPUT FORMAT:
<answer>
Your answer here, with [SOURCE N] citations inline after each claim.
</answer>

Do not include anything outside the <answer> tags. Do not add a Sources \
section — that is handled separately. Do not add preamble or closing remarks.\
"""

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Source:
    """One cited source, ready to render in the output."""
    index:      int
    course:     str
    professor:  str
    term:       str
    source_url: str

    def render(self) -> str:
        return (
            f"[{self.index}] {self.course} — {self.professor} ({self.term})\n"
            f"    {self.source_url}"
        )


@dataclass
class RAGResponse:
    """
    The fully assembled, grounded response.

    answer_text   : LLM answer with [SOURCE N] inline
    sources       : only the sources actually cited (deduplicated, ordered)
    grounded      : False if the LLM signalled insufficient context or
                    produced zero valid citations — never silently pass-through
    raw_llm_output: the unmodified LLM string, for debugging
    """
    answer_text:    str
    sources:        list[Source]
    grounded:       bool
    raw_llm_output: str = field(repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────────────────────

def build_context_block(chunks: list[dict]) -> str:
    """
    Render retrieved chunks as a numbered [SOURCE N] block for the prompt.
    The numbering is the single source of truth — the post-processor uses
    the same indices to look up metadata.

    Format per source:
        [SOURCE 1]
        Course: CSE 373 | Professor: Kasey Champion | Term: Spring 2021
        Review: <text>
    """
    lines = ["The following sources are the ONLY information you may use:\n"]
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[SOURCE {i}]")
        lines.append(
            f"Course: {chunk.get('course', 'Unknown')} | "
            f"Professor: {chunk.get('professor', 'Unknown')} | "
            f"Term: {chunk.get('term', 'Unknown')}"
        )
        lines.append(f"Review: {chunk['text']}")
        lines.append("")   # blank line between sources
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Post-processor
# ─────────────────────────────────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[SOURCE\s+(\d+)\]", re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer_block(raw: str) -> str:
    """
    Pull text from inside <answer>...</answer>.
    Falls back to the full string if the tags are absent (defensive).
    """
    m = _ANSWER_TAG_RE.search(raw)
    return m.group(1).strip() if m else raw.strip()


def parse_and_validate_citations(
    answer_text: str,
    chunks: list[dict],
) -> tuple[str, list[Source]]:
    """
    Walk every [SOURCE N] tag in answer_text.

    - N within range  → kept; metadata looked up from chunks list.
    - N out of range  → replaced with [SOURCE ?] in the answer text
                        and excluded from the sources list.

    Returns (cleaned_answer_text, list_of_valid_Source_objects).
    The sources list is deduplicated and ordered by first appearance.
    """
    valid_range  = set(range(1, len(chunks) + 1))
    seen_indices: list[int] = []          # preserves first-appearance order
    invalid_tags: set[int]  = set()

    for m in _CITATION_RE.finditer(answer_text):
        n = int(m.group(1))
        if n in valid_range:
            if n not in seen_indices:
                seen_indices.append(n)
        else:
            invalid_tags.add(n)

    # Replace out-of-range tags in the answer text
    def replace_tag(m: re.Match) -> str:
        n = int(m.group(1))
        return "[SOURCE ?]" if n in invalid_tags else m.group(0)

    cleaned_answer = _CITATION_RE.sub(replace_tag, answer_text)

    # Build Source objects in citation order
    sources = []
    for n in seen_indices:
        chunk = chunks[n - 1]          # 1-indexed → 0-indexed
        sources.append(Source(
            index      = n,
            course     = chunk.get("course", "Unknown"),
            professor  = chunk.get("professor", "Unknown"),
            term       = chunk.get("term", "Unknown"),
            source_url = chunk.get("source_url", ""),
        ))

    return cleaned_answer, sources


def post_process(
    raw_llm_output: str,
    chunks: list[dict],
) -> RAGResponse:
    """
    Full post-processing pipeline:
      1. Extract the <answer> block.
      2. Detect the insufficient-context sentinel.
      3. Validate and clean citations.
      4. Reject the answer if zero valid citations remain.

    This is the grounding guarantee — the LLM cannot smuggle in uncited
    claims because the source list is built entirely from citation tags
    that Python validates against the actual retrieved chunks.
    """
    answer_text = extract_answer_block(raw_llm_output)

    # Check for sentinel — LLM signalled it cannot answer from context
    if INSUFFICIENT_SENTINEL in answer_text or INSUFFICIENT_SENTINEL in raw_llm_output:
        return RAGResponse(
            answer_text    = (
                "The retrieved reviews do not contain enough information "
                "to answer this question."
            ),
            sources        = [],
            grounded       = False,
            raw_llm_output = raw_llm_output,
        )

    # Validate citations and build source list
    cleaned_answer, sources = parse_and_validate_citations(answer_text, chunks)

    # Hard rejection: answer with no valid citations is ungrounded
    if not sources:
        return RAGResponse(
            answer_text    = (
                "The system could not produce a grounded answer for this question. "
                "The response contained no valid source citations."
            ),
            sources        = [],
            grounded       = False,
            raw_llm_output = raw_llm_output,
        )

    return RAGResponse(
        answer_text    = cleaned_answer,
        sources        = sources,
        grounded       = True,
        raw_llm_output = raw_llm_output,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_response(response: RAGResponse, query: str) -> str:
    """
    Produce the final "answer + source list" string for display.

    Format (from planning.md):
        Answer
        ──────
        <answer text with inline [SOURCE N] citations>

        Sources
        ───────
        [1] COURSE — Professor (Term)
            https://...
        [2] ...
    """
    sep = "─" * 60
    lines = [
        f"\nQuery: {query}",
        sep,
        "Answer",
        sep,
    ]

    if response.grounded:
        # Wrap long lines for terminal readability
        wrapped = textwrap.fill(
            response.answer_text,
            width=80,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.append(wrapped)
        lines.append("")
        lines.append("Sources")
        lines.append(sep)
        for source in response.sources:
            lines.append(source.render())
    else:
        lines.append(response.answer_text)

    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    query: str,
    chunks: list[dict],
    groq_client: Groq,
) -> RAGResponse:
    """
    Call the Groq API with a grounded context prompt and post-process
    the response to guarantee source attribution.

    Parameters
    ----------
    query       : the user's question
    chunks      : retrieved chunks from embed_and_retrieve.retrieve()
    groq_client : authenticated Groq client

    Returns
    -------
    RAGResponse with grounded answer text and validated Source objects
    """
    if not chunks:
        return RAGResponse(
            answer_text    = "No relevant reviews were found for this query.",
            sources        = [],
            grounded       = False,
            raw_llm_output = "",
        )

    context_block = build_context_block(chunks)

    user_message = (
        f"{context_block}\n"
        f"Question: {query}\n\n"
        "Remember: cite every claim with [SOURCE N]. "
        f"If the sources are insufficient, reply only with: {INSUFFICIENT_SENTINEL}"
    )

    completion = groq_client.chat.completions.create(
        model      = GROQ_MODEL,
        max_tokens = MAX_TOKENS,
        temperature = 0.0,     # deterministic — grounding, not creativity
        messages   = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    raw = completion.choices[0].message.content or ""
    return post_process(raw, chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline — wire Retrieval → Generation together
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Single entry point for the complete pipeline:
      query → retrieve → generate → grounded RAGResponse

    Holds loaded models so they are not reloaded between turns in the REPL.
    """

    def __init__(
        self,
        chroma_dir:  Path = CHROMA_DIR,
        top_k:       int  = TOP_K,
    ) -> None:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set.\n"
                "Add it to your .env file in the project root:\n"
                "  GROQ_API_KEY=your-key-here\n"
                "Get a free key at https://console.groq.com"
            )

        print(f"Initialising pipeline…")
        print(f"  Embedding model : {EMBEDDING_MODEL}")
        print(f"  LLM             : {GROQ_MODEL} (via Groq)")
        print(f"  Vector store    : {chroma_dir}")
        print(f"  Top-k           : {top_k}")

        self.embed_model  = load_embedding_model()
        self.collection   = get_chroma_collection(chroma_dir)
        self.groq_client  = Groq(api_key=groq_api_key)
        self.top_k        = top_k

        doc_count = self.collection.count()
        if doc_count == 0:
            raise RuntimeError(
                "ChromaDB collection is empty. "
                "Run  python embed_and_retrieve.py --ingest  first."
            )
        print(f"  Collection docs : {doc_count}\n")

    def ask(
        self,
        query:            str,
        filter_course:    str | None = None,
        filter_professor: str | None = None,
    ) -> RAGResponse:
        """
        Full pipeline: retrieve → generate → return grounded RAGResponse.
        """
        # Retrieve
        chunks = retrieve(
            query            = query,
            model            = self.embed_model,
            collection       = self.collection,
            top_k            = self.top_k,
            filter_course    = filter_course,
            filter_professor = filter_professor,
        )

        # Generate
        return generate(
            query       = query,
            chunks      = chunks,
            groq_client = self.groq_client,
        )



# ─────────────────────────────────────────────────────────────────────────────
# Public ask() function — used by app.py (Gradio interface)
# ─────────────────────────────────────────────────────────────────────────────

# Module-level singleton so the pipeline is only initialised once when
# app.py imports this module (avoids reloading the embedding model on
# every button click).
_pipeline: "RAGPipeline | None" = None


def get_pipeline() -> RAGPipeline:
    """Return the module-level pipeline, initialising it on first call."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def ask(
    question: str,
    filter_course: "str | None" = None,
    filter_professor: "str | None" = None,
) -> dict:
    """
    Public interface used by app.py and any external caller.

    Returns a dict with keys:
        "answer"  : str  -- answer text with inline [SOURCE N] citations
        "sources" : list[str] -- formatted source lines, one per cited chunk
        "grounded": bool -- False if context was insufficient or no citations
    """
    response = get_pipeline().ask(
        query            = question,
        filter_course    = filter_course,
        filter_professor = filter_professor,
    )
    return {
        "answer":   response.answer_text,
        "sources":  [s.render() for s in response.sources],
        "grounded": response.grounded,
    }


def run_repl(pipeline: RAGPipeline) -> None:
    """
    Simple read-eval-print loop.
    Commands:
        /course <name>      — set course filter (e.g.  /course CSE 373)
        /professor <name>   — set professor filter
        /filter             — show active filters
        /clear              — clear all filters
        /quit               — exit
    """
    print("=" * 60)
    print("  UW Course Review RAG — Interactive Mode")
    print("  Type a question, or /help for commands.")
    print("=" * 60)

    filter_course    = None
    filter_professor = None

    while True:
        try:
            raw_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────
        if raw_input.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye.")
            break

        if raw_input.lower() == "/help":
            print(
                "  /course <name>      Filter to a specific course  (e.g. /course CSE 373)\n"
                "  /professor <name>   Filter to a specific professor\n"
                "  /filter             Show active filters\n"
                "  /clear              Remove all filters\n"
                "  /quit               Exit"
            )
            continue

        if raw_input.lower().startswith("/course "):
            filter_course = raw_input[8:].strip()
            print(f"Course filter set to: {filter_course!r}")
            continue

        if raw_input.lower().startswith("/professor "):
            filter_professor = raw_input[11:].strip()
            print(f"Professor filter set to: {filter_professor!r}")
            continue

        if raw_input.lower() == "/filter":
            print(f"  course    : {filter_course!r}")
            print(f"  professor : {filter_professor!r}")
            continue

        if raw_input.lower() == "/clear":
            filter_course = filter_professor = None
            print("Filters cleared.")
            continue

        # ── Query ─────────────────────────────────────────────────────────
        print("  …retrieving and generating…")
        response = pipeline.ask(
            query            = raw_input,
            filter_course    = filter_course,
            filter_professor = filter_professor,
        )
        print(render_response(response, raw_input))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "UW course review RAG — generation + interface layer.\n"
            "Runs as an interactive REPL by default, or answers a single "
            "query with --query."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--query", "-q", type=str, default=None,
        help="Single query to answer, then exit.",
    )
    p.add_argument(
        "--course", "-c", type=str, default=None,
        help="Restrict retrieval to one course, e.g. 'CSE 373'.",
    )
    p.add_argument(
        "--professor", "-p", type=str, default=None,
        help="Restrict retrieval to one professor.",
    )
    p.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Number of chunks to retrieve (default: {TOP_K}).",
    )
    p.add_argument(
        "--chroma-dir", type=Path, default=CHROMA_DIR,
        help=f"ChromaDB store directory (default: {CHROMA_DIR}).",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Print the raw LLM output before post-processing.",
    )
    return p


def main() -> None:
    args   = build_parser().parse_args()
    pipeline = RAGPipeline(chroma_dir=args.chroma_dir, top_k=args.top_k)

    if args.query:
        response = pipeline.ask(
            query            = args.query,
            filter_course    = args.course,
            filter_professor = args.professor,
        )
        if args.debug:
            print("\n── Raw LLM output ──")
            print(response.raw_llm_output)
            print("── End raw output ──\n")
        print(render_response(response, args.query))
    else:
        run_repl(pipeline)


if __name__ == "__main__":
    main()
