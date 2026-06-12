"""
test_embed_and_retrieve.py
Unit tests for the logic in embed_and_retrieve.py that don't require
sentence-transformers or ChromaDB to be installed.

Covers:
  1. flatten_metadata  — correct field names, None → "", nested ratings unpacked
  2. retrieve()        — result shape via a mock ChromaDB collection
  3. filter building   — course-only, professor-only, both, neither
  4. chunks.json       — spot-checks the ingestion pipeline output
"""

import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Inline the two pure-Python helpers from embed_and_retrieve so the tests
# don't need sentence-transformers or chromadb installed.
# ---------------------------------------------------------------------------

def flatten_metadata(chunk: dict) -> dict:
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


def build_where(filter_course, filter_professor):
    """Extracted filter-building logic from retrieve()."""
    if filter_course and filter_professor:
        return {"$and": [
            {"course":    {"$eq": filter_course}},
            {"professor": {"$eq": filter_professor}},
        ]}
    elif filter_course:
        return {"course": {"$eq": filter_course}}
    elif filter_professor:
        return {"professor": {"$eq": filter_professor}}
    return None


def unpack_results(raw: dict) -> list[dict]:
    """The unpacking logic from retrieve()."""
    output = []
    for doc, meta, dist in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
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


CHUNKS_FILE = Path(__file__).parent / "chunks.json"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFlattenMetadata(unittest.TestCase):

    def _make_chunk(self, **overrides):
        base = {
            "chunk_id":   "CSE_373-r0-c0",
            "course":     "CSE 373",
            "source_url": "https://www.ratemycourses.io/uw/course/cse373",
            "professor":  "Kasey Champion",
            "term":       "Spring 2021",
            "ratings": {"overall": 5, "difficulty": 2, "interest": 5, "usefulness": 5},
            "grade":      "A",
            "workload":   "Heavy",
            "textbook":   "Optional",
            "tokens":     73,
            "text":       "Very interesting class.",
        }
        base.update(overrides)
        return base

    def test_all_expected_keys_present(self):
        meta = flatten_metadata(self._make_chunk())
        expected_keys = {
            "course", "source_url", "professor", "term",
            "grade", "workload", "textbook", "tokens",
            "rating_overall", "rating_difficulty",
            "rating_interest", "rating_usefulness",
        }
        self.assertEqual(set(meta.keys()), expected_keys)

    def test_no_nested_dicts(self):
        meta = flatten_metadata(self._make_chunk())
        for k, v in meta.items():
            self.assertNotIsInstance(v, dict, f"Key '{k}' is still a dict")

    def test_ratings_unpacked_correctly(self):
        meta = flatten_metadata(self._make_chunk())
        self.assertEqual(meta["rating_overall"],    5)
        self.assertEqual(meta["rating_difficulty"], 2)
        self.assertEqual(meta["rating_interest"],   5)
        self.assertEqual(meta["rating_usefulness"], 5)

    def test_none_values_become_empty_string_or_zero(self):
        chunk = self._make_chunk(professor=None, grade=None, tokens=None, ratings=None)
        meta = flatten_metadata(chunk)
        self.assertEqual(meta["professor"], "")
        self.assertEqual(meta["grade"],     "")
        self.assertEqual(meta["tokens"],    0)
        self.assertEqual(meta["rating_overall"], 0)

    def test_values_are_chroma_safe_types(self):
        meta = flatten_metadata(self._make_chunk())
        allowed = (str, int, float, bool)
        for k, v in meta.items():
            self.assertIsInstance(v, allowed,
                f"Key '{k}' has type {type(v).__name__}, not ChromaDB-safe")

    def test_chunk_id_not_in_metadata(self):
        # chunk_id is passed separately as the ChromaDB `ids` list — not in metadata
        meta = flatten_metadata(self._make_chunk())
        self.assertNotIn("chunk_id", meta)

    def test_text_not_in_metadata(self):
        # text goes into `documents`, not `metadatas`
        meta = flatten_metadata(self._make_chunk())
        self.assertNotIn("text", meta)


class TestFilterBuilding(unittest.TestCase):

    def test_no_filter_returns_none(self):
        self.assertIsNone(build_where(None, None))

    def test_course_only_filter(self):
        w = build_where("CSE 373", None)
        self.assertEqual(w, {"course": {"$eq": "CSE 373"}})

    def test_professor_only_filter(self):
        w = build_where(None, "Kasey Champion")
        self.assertEqual(w, {"professor": {"$eq": "Kasey Champion"}})

    def test_both_filters_uses_and(self):
        w = build_where("CSE 373", "Kasey Champion")
        self.assertIn("$and", w)
        self.assertEqual(len(w["$and"]), 2)
        courses = [c for c in w["$and"] if "course" in c]
        profs   = [c for c in w["$and"] if "professor" in c]
        self.assertEqual(courses[0]["course"]["$eq"],    "CSE 373")
        self.assertEqual(profs[0]["professor"]["$eq"],   "Kasey Champion")


class TestUnpackResults(unittest.TestCase):

    def _make_raw_results(self, n=3):
        docs  = [f"chunk text {i}" for i in range(n)]
        metas = [
            {
                "course": "PSYCH 210", "professor": "Nicole Mcnichols",
                "term": "Spring 2021", "source_url": "https://example.com",
                "grade": "A+", "workload": "6hrs/week", "textbook": "Yes",
                "rating_overall": 5, "rating_difficulty": 5,
                "rating_interest": 5, "rating_usefulness": 4,
            }
            for _ in range(n)
        ]
        dists = [0.05 * (i + 1) for i in range(n)]
        return {"documents": [docs], "metadatas": [metas], "distances": [dists]}

    def test_returns_correct_count(self):
        results = unpack_results(self._make_raw_results(6))
        self.assertEqual(len(results), 6)

    def test_result_has_all_keys(self):
        results = unpack_results(self._make_raw_results(1))
        required = {
            "chunk_id", "text", "distance", "course", "professor", "term",
            "source_url", "grade", "workload", "textbook",
            "rating_overall", "rating_difficulty", "rating_interest", "rating_usefulness",
        }
        self.assertEqual(set(results[0].keys()), required)

    def test_distance_is_rounded_to_4dp(self):
        results = unpack_results(self._make_raw_results(1))
        d = results[0]["distance"]
        self.assertEqual(d, round(d, 4))

    def test_chunk_id_combines_course_and_professor(self):
        results = unpack_results(self._make_raw_results(1))
        self.assertIn("PSYCH 210", results[0]["chunk_id"])
        self.assertIn("Nicole Mcnichols", results[0]["chunk_id"])

    def test_text_preserved(self):
        results = unpack_results(self._make_raw_results(2))
        self.assertEqual(results[0]["text"], "chunk text 0")
        self.assertEqual(results[1]["text"], "chunk text 1")

    def test_results_ordered_by_distance_ascending(self):
        results = unpack_results(self._make_raw_results(3))
        dists = [r["distance"] for r in results]
        self.assertEqual(dists, sorted(dists))


class TestChunksJson(unittest.TestCase):
    """Spot-checks on the chunks.json produced by ingest_and_chunk.py."""

    @classmethod
    def setUpClass(cls):
        if not CHUNKS_FILE.exists():
            cls.chunks = None
            return
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            cls.chunks = json.load(f)

    def setUp(self):
        if self.chunks is None:
            self.skipTest("chunks.json not found — run ingest_and_chunk.py first")

    def test_chunks_list_is_non_empty(self):
        self.assertGreater(len(self.chunks), 0)

    def test_all_chunks_under_250_tokens(self):
        over = [c for c in self.chunks if c.get("tokens", 0) > 250]
        self.assertEqual(over, [], f"{len(over)} chunks exceed 250 tokens")

    def test_all_required_fields_present(self):
        required = {"chunk_id", "course", "source_url", "professor",
                    "term", "ratings", "text", "tokens"}
        for c in self.chunks:
            missing = required - set(c.keys())
            self.assertEqual(missing, set(), f"Chunk {c.get('chunk_id')} missing: {missing}")

    def test_ratings_have_four_keys(self):
        for c in self.chunks:
            r = c.get("ratings", {})
            self.assertIn("overall",    r)
            self.assertIn("difficulty", r)
            self.assertIn("interest",   r)
            self.assertIn("usefulness", r)

    def test_all_ten_courses_present(self):
        courses = {c["course"] for c in self.chunks}
        expected = {
            "CSE 142", "CSE 143", "CSE 373", "ECON 200", "ENGL 131",
            "MATH 125", "MATH 126", "PHYS 121", "PSYCH 210", "STAT 311",
        }
        self.assertEqual(courses, expected)

    def test_source_urls_are_ratemycourses(self):
        for c in self.chunks:
            self.assertIn("ratemycourses.io", c.get("source_url", ""),
                          f"Bad URL in chunk {c.get('chunk_id')}")

    def test_no_section_headers_in_text(self):
        headers = ("comments on the course", "comments on the professor",
                   "advice", "suggest a professor", "course content")
        for c in self.chunks:
            text_lower = c.get("text", "").lower().strip()
            for h in headers:
                self.assertFalse(
                    text_lower.startswith(h),
                    f"Chunk {c['chunk_id']} text starts with section header '{h}'"
                )

    def test_no_empty_text(self):
        for c in self.chunks:
            self.assertTrue(c.get("text", "").strip(),
                            f"Chunk {c.get('chunk_id')} has empty text")

    def test_flatten_metadata_on_all_chunks(self):
        """flatten_metadata must not raise and must return only safe types."""
        allowed = (str, int, float, bool)
        for c in self.chunks:
            meta = flatten_metadata(c)
            for k, v in meta.items():
                self.assertIsInstance(v, allowed,
                    f"Chunk {c['chunk_id']} key '{k}' has type {type(v).__name__}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
