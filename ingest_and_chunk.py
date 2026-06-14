"""
ingest_and_chunk.py
Milestone 3 — Ingestion and Chunking

Loads all course review .txt files, cleans them, parses each review
into structured records, and splits them into token-bounded chunks
(~250 tokens, 0-1 sentence overlap) ready for embedding.

Output
------
chunks.json   — list of chunk dicts, each with:
    {
        "chunk_id":   "CSE_142-r0-c0",
        "course":     "CSE 142",
        "source_url": "https://www.ratemycourses.io/uw/course/cse142",
        "professor":  "Stuart Reges",
        "term":       "Winter 2021",
        "ratings": {
            "overall": 4, "difficulty": 2,
            "interest": 5, "usefulness": 5
        },
        "grade":      "A+",
        "workload":   "10hrs/week",
        "textbook":   "Optional",
        "text":       "<chunk text>",
        "tokens":     183
    }
"""

import re
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — mirrors planning.md
# ---------------------------------------------------------------------------
CHUNK_SIZE_TOKENS = 250   # target max tokens per chunk
OVERLAP_SENTENCES = 1     # 0 or 1 sentence overlap between consecutive chunks

DATA_DIR = Path(__file__).parent / "documents"
OUTPUT_FILE = Path(__file__).parent / "chunks.json"

# Map filename stem → (display name, source URL)
COURSE_META = {
    "CSE 142 Reviews":  ("CSE 142",   "https://www.ratemycourses.io/uw/course/cse142"),
    "CSE 143 Reviews":  ("CSE 143",   "https://www.ratemycourses.io/uw/course/cse143"),
    "CSE 373 Reviews":  ("CSE 373",   "https://www.ratemycourses.io/uw/course/cse373"),
    "ECON 200 Reviews": ("ECON 200",  "https://www.ratemycourses.io/uw/course/econ200"),
    "ENGL 131 Reviews": ("ENGL 131",  "https://www.ratemycourses.io/uw/course/engl131"),
    "MATH 125 Reviews": ("MATH 125",  "https://www.ratemycourses.io/uw/course/math125"),
    "MATH 126 Reviews": ("MATH 126",  "https://www.ratemycourses.io/uw/course/math126"),
    "PHYS 121 Reviews": ("PHYS 121",  "https://www.ratemycourses.io/uw/course/phys121"),
    "PSYCH 210 Reviews":("PSYCH 210", "https://www.ratemycourses.io/uw/course/psych210"),
    "STAT 311 Reviews": ("STAT 311",  "https://www.ratemycourses.io/uw/course/stat311"),
}

# Rating label → field name
RATING_LABELS = {
    "bad class": "overall", "ok class": "overall",
    "good class": "overall", "amazing class": "overall",
    "awful class": "overall",
    "very hard": "difficulty", "hard": "difficulty",
    "avg. difficulty": "difficulty", "easy": "difficulty",
    "very easy": "difficulty",
    "very boring": "interest", "boring": "interest",
    "kinda interesting": "interest", "interesting": "interest",
    "very interesting": "interest",
    "barely useful": "usefulness", "kinda useful": "usefulness",
    "useful": "usefulness", "very useful": "usefulness",
}

# Section headers to strip from text body
SECTION_HEADERS = re.compile(
    r"^(comments on the (course|professor)|advice|suggest a professor|"
    r"course content|delivery:.*|assignment heavy|exam heavy|"
    r"quiz heavy|participation heavy)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Trailing metadata line: "Grade: A+Workload: 10hrs/weekTextbook Use: Optional"
META_LINE = re.compile(
    r"(?:Grade:\s*(?P<grade>[^\s]+))?"
    r"(?:Workload:\s*(?P<workload>[^\s]+(?:\s*/\s*week)?))?"
    r"(?:Textbook Use:\s*(?P<textbook>.+))?",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Tokenisation (whitespace-based approximation, ~1 token ≈ 1 word for English)
# A full tiktoken import is not required at this stage; swap in tiktoken if
# you want exact GPT token counts before embedding.
# ---------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    """Approximate token count using whitespace splitting."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def clean_ratings_block(lines: list[str]) -> tuple[dict, list[str]]:
    """
    Parse the 'Class Ratings / Professor Rating' header block.
    Returns (ratings_dict, remaining_lines).

    Raw format looks like:
        Class Ratings
        2Bad Class
        3Avg. Difficulty
        5Very Interesting
        4Useful
        Professor Rating   ← optional
        3OK Prof
    """
    ratings = {"overall": None, "difficulty": None, "interest": None, "usefulness": None}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Rating line: digit followed immediately by text, e.g. "4Good Class"
        m = re.match(r"^([1-5])(.+)$", line)
        if m:
            score, label = int(m.group(1)), m.group(2).strip().lower()
            field = RATING_LABELS.get(label)
            if field:
                ratings[field] = score
            i += 1
            continue
        # Stop when we hit a non-rating line that isn't a header
        if line.lower() in ("class ratings", "professor rating", ""):
            i += 1
            continue
        break  # reached the prof/date line
    return ratings, lines[i:]


def parse_metadata_footer(line: str) -> dict:
    """Extract grade, workload, textbook from the compact footer line."""
    meta = {"grade": None, "workload": None, "textbook": None}
    # Grade
    m = re.search(r"Grade:\s*(\S+?)(?=Workload|Textbook|$)", line, re.IGNORECASE)
    if m:
        meta["grade"] = m.group(1).strip()
    # Workload
    m = re.search(r"Workload:\s*([^T]+?)(?=Textbook|$)", line, re.IGNORECASE)
    if m:
        meta["workload"] = m.group(1).strip()
    # Textbook
    m = re.search(r"Textbook Use:\s*(.+)", line, re.IGNORECASE)
    if m:
        meta["textbook"] = m.group(1).strip()
    return meta


def clean_body(text: str) -> str:
    """Remove section headers, stray dates, and normalise whitespace."""
    # Remove section header lines
    text = SECTION_HEADERS.sub("", text)
    # Remove date lines like "Aug 24, 2021"
    text = re.sub(r"^\w+ \d{1,2}, \d{4}\s*$", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace
    return text.strip()


# ---------------------------------------------------------------------------
# Review parsing
# ---------------------------------------------------------------------------

def split_reviews(raw: str) -> list[str]:
    """Split a file into individual review blocks on 'Class Ratings' headings."""
    blocks = re.split(r"(?=^Class Ratings\s*$)", raw, flags=re.MULTILINE)
    return [b.strip() for b in blocks if b.strip()]


def parse_review(block: str, course: str, source_url: str, review_idx: int) -> dict | None:
    """Parse one review block into a structured dict."""
    lines = block.splitlines()

    # 1. Ratings block
    ratings, lines = clean_ratings_block(lines)

    # 2. Prof / term line: "Prof: Stuart Reges / Winter 2021"
    professor, term = None, None
    if lines and lines[0].startswith("Prof:"):
        prof_line = lines.pop(0).strip()
        m = re.match(r"Prof:\s*(.+?)\s*/\s*(.+)", prof_line)
        if m:
            professor = m.group(1).strip()
            term = m.group(2).strip()

    # 3. Join remaining lines for body processing
    remaining = "\n".join(lines)

    # 4. Separate footer metadata line (last line with "Grade:" or "Workload:")
    footer_meta = {"grade": None, "workload": None, "textbook": None}
    footer_match = re.search(
        r"((?:Grade:[^\n]+)?(?:Workload:[^\n]+)?(?:Textbook Use:[^\n]+)?)$",
        remaining, re.IGNORECASE
    )
    if footer_match and any(k in footer_match.group(1) for k in ("Grade:", "Workload:", "Textbook")):
        footer_meta = parse_metadata_footer(footer_match.group(1))
        remaining = remaining[:footer_match.start()].strip()

    # 5. Clean narrative body
    body = clean_body(remaining)
    if not body:
        return None

    return {
        "course": course,
        "source_url": source_url,
        "professor": professor,
        "term": term,
        "ratings": ratings,
        "grade": footer_meta["grade"],
        "workload": footer_meta["workload"],
        "textbook": footer_meta["textbook"],
        "body": body,
        "review_idx": review_idx,
    }


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter on '.', '!', '?' followed by whitespace/EOL."""
    # Split while keeping the delimiter with the sentence
    parts = re.split(r"(?<=[.!?])\s+", text)
    # Also split on newlines (paragraph breaks count as boundaries)
    sentences = []
    for part in parts:
        sub = [s.strip() for s in part.split("\n") if s.strip()]
        sentences.extend(sub)
    return sentences


def chunk_review(review: dict, chunk_size: int, overlap: int) -> list[dict]:
    """
    Split review body into token-bounded chunks.

    overlap=0  → no repeated sentences
    overlap=1  → last sentence of chunk N is the first sentence of chunk N+1
    """
    sentences = split_sentences(review["body"])
    chunks = []
    i = 0
    chunk_idx = 0

    while i < len(sentences):
        current_sentences = []
        current_tokens = 0

        j = i
        while j < len(sentences):
            s = sentences[j]
            t = count_tokens(s)
            if current_tokens + t > chunk_size and current_sentences:
                break
            current_sentences.append(s)
            current_tokens += t
            j += 1

        if not current_sentences:
            # Single sentence exceeds chunk size — include it anyway
            current_sentences = [sentences[i]]
            current_tokens = count_tokens(current_sentences[0])
            j = i + 1

        chunk_text = " ".join(current_sentences)
        chunk_id = f"{review['course'].replace(' ', '_')}-r{review['review_idx']}-c{chunk_idx}"

        chunks.append({
            "chunk_id":   chunk_id,
            "course":     review["course"],
            "source_url": review["source_url"],
            "professor":  review["professor"],
            "term":       review["term"],
            "ratings":    review["ratings"],
            "grade":      review["grade"],
            "workload":   review["workload"],
            "textbook":   review["textbook"],
            "text":       chunk_text,
            "tokens":     current_tokens,
        })

        chunk_idx += 1
        # Advance: back up by `overlap` sentences for the next chunk
        i = j - overlap if j - overlap > i else j

    return chunks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def ingest_and_chunk() -> list[dict]:
    all_chunks = []

    for stem, (course_name, url) in COURSE_META.items():
        filepath = DATA_DIR / f"{stem}.txt"
        if not filepath.exists():
            print(f"  [WARN] File not found: {filepath}")
            continue

        raw = filepath.read_text(encoding="utf-8")
        review_blocks = split_reviews(raw)

        file_chunks = []
        for idx, block in enumerate(review_blocks):
            review = parse_review(block, course_name, url, idx)
            if review is None:
                continue
            chunks = chunk_review(review, CHUNK_SIZE_TOKENS, OVERLAP_SENTENCES)
            file_chunks.extend(chunks)

        print(f"  {course_name:10s}  {len(review_blocks):2d} reviews → {len(file_chunks):3d} chunks")
        all_chunks.extend(file_chunks)

    return all_chunks


if __name__ == "__main__":
    print("=== Ingestion & Chunking Pipeline ===\n")
    chunks = ingest_and_chunk()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\nTotal chunks produced: {len(chunks)}")
    print(f"Output written to:     {OUTPUT_FILE}")

    # --- Spot-check: print the first two chunks --------------------------
    print("\n--- Sample chunks ---")
    for c in chunks[:2]:
        print(f"\nchunk_id : {c['chunk_id']}")
        print(f"course   : {c['course']}  |  prof: {c['professor']}  |  term: {c['term']}")
        print(f"ratings  : {c['ratings']}")
        print(f"grade    : {c['grade']}  |  workload: {c['workload']}  |  textbook: {c['textbook']}")
        print(f"tokens   : {c['tokens']}")
        print(f"text     :\n{c['text']}\n")
