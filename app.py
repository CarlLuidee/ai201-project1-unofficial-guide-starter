"""
app.py
Gradio web interface for the UW Course Review RAG system.

Run:
    python app.py
Then open:  http://localhost:7860

The interface exposes:
  - A question text box
  - Optional course / professor filters
  - An Answer panel (with inline [SOURCE N] citations)
  - A Retrieved from panel (one line per cited source with URL)

All grounding logic lives in generate_and_run.py — this file is only UI.
"""

import gradio as gr
from generate_and_run import ask


# ─────────────────────────────────────────────────────────────────────────────
# Handler
# ─────────────────────────────────────────────────────────────────────────────

def handle_query(question: str, course_filter: str, professor_filter: str):
    """
    Called by Gradio on every button click or Enter keypress.
    Returns (answer_text, sources_text) to fill the two output boxes.
    """
    if not question.strip():
        return "Please enter a question.", ""

    result = ask(
        question         = question.strip(),
        filter_course    = course_filter.strip() or None,
        filter_professor = professor_filter.strip() or None,
    )

    sources_text = "\n".join(f"• {s}" for s in result["sources"]) if result["sources"] else "—"
    return result["answer"], sources_text


# ─────────────────────────────────────────────────────────────────────────────
# UI layout
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="UW Course Review Guide") as demo:

    gr.Markdown(
        """
        # 📚 The Unofficial UW Course Guide
        Ask anything about UW courses and professors based on real student reviews.
        Answers are grounded in retrieved reviews only — with source citations.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Your question",
                placeholder='e.g. "Is CSE 373 good for software engineers?" or "How hard is MATH 126?"',
                lines=2,
            )
        with gr.Column(scale=1):
            course_box = gr.Textbox(
                label="Filter by course (optional)",
                placeholder="e.g. CSE 373",
            )
            professor_box = gr.Textbox(
                label="Filter by professor (optional)",
                placeholder="e.g. Kasey Champion",
            )

    ask_btn = gr.Button("Ask", variant="primary")

    answer_box = gr.Textbox(
        label="Answer",
        lines=8,
        interactive=False,
    )
    sources_box = gr.Textbox(
        label="Retrieved from",
        lines=4,
        interactive=False,
    )

    # Wire up button click and Enter-to-submit
    ask_btn.click(
        fn      = handle_query,
        inputs  = [question_box, course_box, professor_box],
        outputs = [answer_box, sources_box],
    )
    question_box.submit(
        fn      = handle_query,
        inputs  = [question_box, course_box, professor_box],
        outputs = [answer_box, sources_box],
    )

    gr.Markdown(
        """
        ---
        **How to use:** Type a question and press **Ask** or hit Enter.
        Use the filter boxes to narrow results to a specific course or professor.
        The *Retrieved from* panel shows which review pages each answer draws from.
        """
    )


# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch()
