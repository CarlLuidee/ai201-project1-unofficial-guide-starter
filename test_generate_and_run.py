"""
test_generate_and_run.py
Unit tests for the generation layer in generate_and_run.py.

All tests run without Groq, ChromaDB, or sentence-transformers installed.
The functions under test are inlined below (same logic, no imports).

Coverage:
  1. build_context_block  — numbering, format, field presence
  2. extract_answer_block — tag extraction, fallback
  3. parse_and_validate_citations — valid/invalid/duplicate citations
  4. post_process — sentinel detection, zero-citation rejection,
                    grounded vs ungrounded RAGResponse
  5. render_response — output structure, source list, ungrounded path
  6. SYSTEM_PROMPT — structural requirements
"""

import re
import textwrap
import unittest
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Inline the pure-Python logic under test
# (mirrors generate_and_run.py exactly so tests run without third-party deps)
# ─────────────────────────────────────────────────────────────────────────────

INSUFFICIENT_SENTINEL = "ONLY_CONTEXT_INSUFFICIENT"

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


@dataclass
class Source:
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
    answer_text:    str
    sources:        list
    grounded:       bool
    raw_llm_output: str = field(repr=False)


_CITATION_RE   = re.compile(r"\[SOURCE\s+(\d+)\]", re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def build_context_block(chunks: list[dict]) -> str:
    lines = ["The following sources are the ONLY information you may use:\n"]
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[SOURCE {i}]")
        lines.append(
            f"Course: {chunk.get('course', 'Unknown')} | "
            f"Professor: {chunk.get('professor', 'Unknown')} | "
            f"Term: {chunk.get('term', 'Unknown')}"
        )
        lines.append(f"Review: {chunk['text']}")
        lines.append("")
    return "\n".join(lines)


def extract_answer_block(raw: str) -> str:
    m = _ANSWER_TAG_RE.search(raw)
    return m.group(1).strip() if m else raw.strip()


def parse_and_validate_citations(
    answer_text: str,
    chunks: list[dict],
) -> tuple[str, list[Source]]:
    valid_range  = set(range(1, len(chunks) + 1))
    seen_indices: list[int] = []
    invalid_tags: set[int]  = set()

    for m in _CITATION_RE.finditer(answer_text):
        n = int(m.group(1))
        if n in valid_range:
            if n not in seen_indices:
                seen_indices.append(n)
        else:
            invalid_tags.add(n)

    def replace_tag(m: re.Match) -> str:
        n = int(m.group(1))
        return "[SOURCE ?]" if n in invalid_tags else m.group(0)

    cleaned_answer = _CITATION_RE.sub(replace_tag, answer_text)

    sources = []
    for n in seen_indices:
        chunk = chunks[n - 1]
        sources.append(Source(
            index      = n,
            course     = chunk.get("course", "Unknown"),
            professor  = chunk.get("professor", "Unknown"),
            term       = chunk.get("term", "Unknown"),
            source_url = chunk.get("source_url", ""),
        ))

    return cleaned_answer, sources


def post_process(raw_llm_output: str, chunks: list[dict]) -> RAGResponse:
    answer_text = extract_answer_block(raw_llm_output)

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

    cleaned_answer, sources = parse_and_validate_citations(answer_text, chunks)

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


def render_response(response: RAGResponse, query: str) -> str:
    sep = "─" * 60
    lines = [
        f"\nQuery: {query}",
        sep,
        "Answer",
        sep,
    ]
    if response.grounded:
        wrapped = textwrap.fill(
            response.answer_text, width=80,
            break_long_words=False, break_on_hyphens=False,
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
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_chunks(n: int = 3) -> list[dict]:
    courses = ["CSE 373", "PSYCH 210", "MATH 126"]
    profs   = ["Kasey Champion", "Nicole Mcnichols", "Andrew Loveless"]
    terms   = ["Spring 2021", "Spring 2019", "Fall 2020"]
    urls    = [
        "https://www.ratemycourses.io/uw/course/cse373",
        "https://www.ratemycourses.io/uw/course/psych210",
        "https://www.ratemycourses.io/uw/course/math126",
    ]
    return [
        {
            "course":     courses[i],
            "professor":  profs[i],
            "term":       terms[i],
            "source_url": urls[i],
            "text":       f"Sample review text for chunk {i + 1}.",
        }
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: build_context_block
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildContextBlock(unittest.TestCase):

    def test_sources_are_1_indexed(self):
        chunks = make_chunks(3)
        block  = build_context_block(chunks)
        self.assertIn("[SOURCE 1]", block)
        self.assertIn("[SOURCE 2]", block)
        self.assertIn("[SOURCE 3]", block)
        self.assertNotIn("[SOURCE 0]", block)
        self.assertNotIn("[SOURCE 4]", block)

    def test_all_course_names_present(self):
        chunks = make_chunks(3)
        block  = build_context_block(chunks)
        for c in chunks:
            self.assertIn(c["course"], block)

    def test_all_professor_names_present(self):
        chunks = make_chunks(3)
        block  = build_context_block(chunks)
        for c in chunks:
            self.assertIn(c["professor"], block)

    def test_review_text_present(self):
        chunks = make_chunks(2)
        block  = build_context_block(chunks)
        for c in chunks:
            self.assertIn(c["text"], block)

    def test_no_urls_in_context_block(self):
        """URLs are kept out of the context; the LLM should never see them."""
        chunks = make_chunks(3)
        block  = build_context_block(chunks)
        for c in chunks:
            self.assertNotIn(c["source_url"], block)

    def test_preamble_present(self):
        block = build_context_block(make_chunks(1))
        self.assertIn("ONLY information you may use", block)

    def test_single_chunk_produces_source_1_only(self):
        block = build_context_block(make_chunks(1))
        self.assertIn("[SOURCE 1]", block)
        self.assertNotIn("[SOURCE 2]", block)

    def test_empty_chunks_returns_preamble_only(self):
        block = build_context_block([])
        self.assertNotIn("[SOURCE", block)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: extract_answer_block
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractAnswerBlock(unittest.TestCase):

    def test_extracts_content_inside_tags(self):
        raw = "<answer>This is the answer.</answer>"
        self.assertEqual(extract_answer_block(raw), "This is the answer.")

    def test_strips_whitespace_inside_tags(self):
        raw = "<answer>  \n  Answer here.  \n  </answer>"
        self.assertEqual(extract_answer_block(raw), "Answer here.")

    def test_multiline_content(self):
        raw = "<answer>Line one.\nLine two.</answer>"
        result = extract_answer_block(raw)
        self.assertIn("Line one.", result)
        self.assertIn("Line two.", result)

    def test_fallback_when_no_tags(self):
        raw = "No tags here."
        self.assertEqual(extract_answer_block(raw), "No tags here.")

    def test_case_insensitive_tags(self):
        raw = "<ANSWER>Content</ANSWER>"
        self.assertEqual(extract_answer_block(raw), "Content")

    def test_ignores_text_outside_tags(self):
        raw = "Preamble. <answer>Inside.</answer> Postamble."
        self.assertEqual(extract_answer_block(raw), "Inside.")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: parse_and_validate_citations
# ─────────────────────────────────────────────────────────────────────────────

class TestParseAndValidateCitations(unittest.TestCase):

    def setUp(self):
        self.chunks = make_chunks(3)   # SOURCE 1, 2, 3 are valid

    def test_valid_single_citation(self):
        text = "CSE 373 is great for software engineers. [SOURCE 1]"
        cleaned, sources = parse_and_validate_citations(text, self.chunks)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].index, 1)
        self.assertIn("[SOURCE 1]", cleaned)

    def test_multiple_valid_citations(self):
        text = "Claim one [SOURCE 1] and claim two [SOURCE 3]."
        cleaned, sources = parse_and_validate_citations(text, self.chunks)
        self.assertEqual(len(sources), 2)
        self.assertEqual([s.index for s in sources], [1, 3])

    def test_out_of_range_citation_replaced_with_question_mark(self):
        text = "Claim [SOURCE 99]."
        cleaned, sources = parse_and_validate_citations(text, self.chunks)
        self.assertEqual(sources, [])
        self.assertIn("[SOURCE ?]", cleaned)
        self.assertNotIn("[SOURCE 99]", cleaned)

    def test_source_zero_is_invalid(self):
        text = "Claim [SOURCE 0]."
        cleaned, sources = parse_and_validate_citations(text, self.chunks)
        self.assertEqual(sources, [])
        self.assertIn("[SOURCE ?]", cleaned)

    def test_duplicate_citations_deduplicated(self):
        text = "Claim [SOURCE 2] and another [SOURCE 2]."
        cleaned, sources = parse_and_validate_citations(text, self.chunks)
        # Only one Source object for index 2
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].index, 2)

    def test_source_order_matches_first_appearance(self):
        text = "Claim A [SOURCE 3] then claim B [SOURCE 1]."
        _, sources = parse_and_validate_citations(text, self.chunks)
        self.assertEqual([s.index for s in sources], [3, 1])

    def test_source_metadata_correct(self):
        text = "Great class [SOURCE 2]."
        _, sources = parse_and_validate_citations(text, self.chunks)
        self.assertEqual(sources[0].course,     "PSYCH 210")
        self.assertEqual(sources[0].professor,  "Nicole Mcnichols")
        self.assertEqual(sources[0].source_url, "https://www.ratemycourses.io/uw/course/psych210")

    def test_mixed_valid_and_invalid(self):
        text = "Valid [SOURCE 1] and invalid [SOURCE 7]."
        cleaned, sources = parse_and_validate_citations(text, self.chunks)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].index, 1)
        self.assertIn("[SOURCE ?]", cleaned)
        self.assertIn("[SOURCE 1]", cleaned)

    def test_no_citations_returns_empty_source_list(self):
        text = "A claim with no citation at all."
        _, sources = parse_and_validate_citations(text, self.chunks)
        self.assertEqual(sources, [])

    def test_citation_regex_case_insensitive(self):
        text = "Claim [source 1]."
        _, sources = parse_and_validate_citations(text, self.chunks)
        self.assertEqual(len(sources), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: post_process
# ─────────────────────────────────────────────────────────────────────────────

class TestPostProcess(unittest.TestCase):

    def setUp(self):
        self.chunks = make_chunks(3)

    def _wrap(self, text: str) -> str:
        return f"<answer>{text}</answer>"

    # ── Grounded path ────────────────────────────────────────────────────────

    def test_grounded_response_when_valid_citations_present(self):
        raw = self._wrap("CSE 373 is great. [SOURCE 1]")
        result = post_process(raw, self.chunks)
        self.assertTrue(result.grounded)
        self.assertEqual(len(result.sources), 1)

    def test_answer_text_preserved_in_grounded_response(self):
        raw = self._wrap("Great class [SOURCE 1].")
        result = post_process(raw, self.chunks)
        self.assertIn("Great class", result.answer_text)

    def test_raw_llm_output_always_stored(self):
        raw = self._wrap("Some answer [SOURCE 1].")
        result = post_process(raw, self.chunks)
        self.assertEqual(result.raw_llm_output, raw)

    # ── Sentinel path ────────────────────────────────────────────────────────

    def test_sentinel_in_answer_block_sets_ungrounded(self):
        raw = self._wrap(INSUFFICIENT_SENTINEL)
        result = post_process(raw, self.chunks)
        self.assertFalse(result.grounded)
        self.assertEqual(result.sources, [])

    def test_sentinel_outside_answer_block_sets_ungrounded(self):
        raw = INSUFFICIENT_SENTINEL   # no <answer> tags
        result = post_process(raw, self.chunks)
        self.assertFalse(result.grounded)

    def test_sentinel_response_text_is_human_friendly(self):
        raw = self._wrap(INSUFFICIENT_SENTINEL)
        result = post_process(raw, self.chunks)
        self.assertNotIn(INSUFFICIENT_SENTINEL, result.answer_text)
        self.assertIn("not contain enough information", result.answer_text)

    # ── Zero-citation rejection ───────────────────────────────────────────────

    def test_zero_valid_citations_sets_ungrounded(self):
        """
        This is the core grounding guarantee:
        an answer without any valid [SOURCE N] tags MUST be rejected.
        """
        raw = self._wrap("Some answer with no citations at all.")
        result = post_process(raw, self.chunks)
        self.assertFalse(result.grounded)
        self.assertEqual(result.sources, [])

    def test_only_invalid_citations_sets_ungrounded(self):
        raw = self._wrap("Claim [SOURCE 99] and [SOURCE 100].")
        result = post_process(raw, self.chunks)
        self.assertFalse(result.grounded)

    def test_zero_citation_message_does_not_expose_sentinel(self):
        raw = self._wrap("No citations here.")
        result = post_process(raw, self.chunks)
        self.assertNotIn(INSUFFICIENT_SENTINEL, result.answer_text)

    # ── Out-of-range citation sanitisation ───────────────────────────────────

    def test_hallucinated_citation_replaced_not_propagated(self):
        raw = self._wrap("Real claim [SOURCE 1]. Hallucinated [SOURCE 42].")
        result = post_process(raw, self.chunks)
        self.assertTrue(result.grounded)   # SOURCE 1 is still valid
        self.assertIn("[SOURCE ?]", result.answer_text)
        self.assertNotIn("[SOURCE 42]", result.answer_text)

    def test_hallucinated_source_excluded_from_source_list(self):
        raw = self._wrap("Real [SOURCE 2]. Fake [SOURCE 42].")
        result = post_process(raw, self.chunks)
        indices = [s.index for s in result.sources]
        self.assertIn(2, indices)
        self.assertNotIn(42, indices)

    # ── Empty chunks guard ───────────────────────────────────────────────────

    def test_empty_chunks_with_no_citation_is_ungrounded(self):
        raw = self._wrap("Some answer.")
        result = post_process(raw, chunks=[])
        self.assertFalse(result.grounded)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: render_response
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderResponse(unittest.TestCase):

    def _grounded_response(self):
        return RAGResponse(
            answer_text    = "CSE 373 is great. [SOURCE 1]",
            sources        = [Source(
                index=1, course="CSE 373",
                professor="Kasey Champion", term="Spring 2021",
                source_url="https://www.ratemycourses.io/uw/course/cse373",
            )],
            grounded       = True,
            raw_llm_output = "",
        )

    def _ungrounded_response(self):
        return RAGResponse(
            answer_text    = "Could not produce a grounded answer.",
            sources        = [],
            grounded       = False,
            raw_llm_output = "",
        )

    def test_query_appears_in_output(self):
        out = render_response(self._grounded_response(), "How is CSE 373?")
        self.assertIn("How is CSE 373?", out)

    def test_answer_section_heading_present(self):
        out = render_response(self._grounded_response(), "q")
        self.assertIn("Answer", out)

    def test_sources_section_heading_present_when_grounded(self):
        out = render_response(self._grounded_response(), "q")
        self.assertIn("Sources", out)

    def test_sources_section_absent_when_ungrounded(self):
        out = render_response(self._ungrounded_response(), "q")
        self.assertNotIn("Sources", out)

    def test_source_url_appears_in_grounded_output(self):
        out = render_response(self._grounded_response(), "q")
        self.assertIn("https://www.ratemycourses.io/uw/course/cse373", out)

    def test_course_and_professor_in_source_list(self):
        out = render_response(self._grounded_response(), "q")
        self.assertIn("CSE 373", out)
        self.assertIn("Kasey Champion", out)

    def test_source_rendered_with_index(self):
        out = render_response(self._grounded_response(), "q")
        self.assertIn("[1]", out)

    def test_ungrounded_shows_fallback_message(self):
        out = render_response(self._ungrounded_response(), "q")
        self.assertIn("Could not produce a grounded answer.", out)

    def test_output_is_string(self):
        self.assertIsInstance(render_response(self._grounded_response(), "q"), str)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: SYSTEM_PROMPT structural requirements
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemPrompt(unittest.TestCase):

    def test_sentinel_in_system_prompt(self):
        self.assertIn(INSUFFICIENT_SENTINEL, SYSTEM_PROMPT)

    def test_must_language_not_suggest_language(self):
        prompt_lower = SYSTEM_PROMPT.lower()
        self.assertIn("must", prompt_lower)
        # "try to" or "prefer" would be suggestions, not requirements
        self.assertNotIn("try to", prompt_lower)
        self.assertNotIn("prefer to", prompt_lower)

    def test_output_format_tags_specified(self):
        self.assertIn("<answer>", SYSTEM_PROMPT)
        self.assertIn("</answer>", SYSTEM_PROMPT)

    def test_citation_format_specified(self):
        self.assertIn("[SOURCE N]", SYSTEM_PROMPT)

    def test_no_own_knowledge_constraint(self):
        self.assertIn("NO knowledge of your own", SYSTEM_PROMPT)

    def test_no_url_instruction_present(self):
        """LLM must not write URLs — Python does that in the source list."""
        self.assertIn("MUST NOT include URLs", SYSTEM_PROMPT)

    def test_sources_section_delegated_to_python(self):
        """The prompt must tell the LLM NOT to add a Sources section."""
        self.assertIn("do not add a sources", SYSTEM_PROMPT.lower())


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
