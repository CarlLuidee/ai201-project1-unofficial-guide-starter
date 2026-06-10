# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

Course and professor reviews

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | https://www.ratemycourses.io | Student review for PHYS 121 Mechanics course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/phys121 |
| 2 | https://www.ratemycourses.io | Student review for ECON 200 Introduction to Microeconomics course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/econ200 |
| 3 | https://www.ratemycourses.io | Student review for MATH 125 Calculus with Analytic Geometry II course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/math125 |
| 4 | https://www.ratemycourses.io | Student review for MATH 126 Calculus with Analytic Geometry III course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/math125 | https://www.ratemycourses.io/uw/course/math126 |
| 5 | https://www.ratemycourses.io | Student review for  Computer Programming II course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/math125 | https://www.ratemycourses.io/uw/course/cse143 |
| 6 | https://www.ratemycourses.io | Student review for CSE 142	Computer Programming I course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/math125 | https://www.ratemycourses.io/uw/course/cse142 |
| 7 | https://www.ratemycourses.io | Student review for ENGL 131	Composition: Exposition course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/math125 | https://www.ratemycourses.io/uw/course/engl131 |
| 8 | https://www.ratemycourses.io | Student review for PSYCH 210 The Diversity of Human Sexuality course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/math125 | https://www.ratemycourses.io/uw/course/psych210 |
| 9 | https://www.ratemycourses.io | Student review for CSE 373	Data Structures and Algorithms course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/math125 | https://www.ratemycourses.io/uw/course/cse373 |
| 10 | https://www.ratemycourses.io | Student review for STAT 311 Elements of Statistical Methods course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/math125 | https://www.ratemycourses.io/uw/course/stat311 |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:**
250 tokens
**Overlap:**
0 to 1
**Reasoning:**
The student reviews are short in length (100-300 words), organized, and tend to be more concise, therefore chunk size does not require as many tokens and minimum to no overlap is needed to get the full context.
---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**
text-embedding-3-large
**Top-k:**
5 to 6 chunks
**Production tradeoff reflection:**
If costs weren't a factor, improving accuracy would be the highest priority.
---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | Does the professor mostly read from slides, or do they engage with the students? | The professor uses slides as a guide but they mostly interact with the students. Students often mention that they feel engaged during lectures. |
| 2 | How much time do students typically need to prepare for the midterm and final exam? | It depends on the students' study habits. Most students report spending 6–10 hours preparing for each midterm and 12–20 hours for the final exam. |
| 3 | What happens if a student misses several lectures? | Students say it's possible to catch up using posted slides and notes, but makes it harder to follow the class. |
| 4 | What is the average time it takes the professor to respond to emails? | Students report receiving a response within 24 hours, but may take longer during holidays or around exam periods. |
| 5 | How do students describe their experiences during office hours? | Students describe office hours as very welcoming and helpful. The professor takes time to make sure the student understands the course lessons. |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. Returns biased results.

2. Used information and data is outdated.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

┌─────────────────────────────┐
│     Document Ingestion      │
├─────────────────────────────┤
│ Sources:                    │
│ • Course Reviews            │
│ • Professor Reviews         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│          Chunking           │
├─────────────────────────────┤
│ Split reviews into          │
│ semantic chunks             │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Embedding + Vector Store   │
├─────────────────────────────┤
│ Tools:                      │
│ • all-MiniLM-L6-v2          │
│ • ChromaDB                  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│         Retrieval           │
├─────────────────────────────┤
│ Semantic and keyword match  │
│ Tool: ChromaDB              │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│         Generation          │
├─────────────────────────────┤
│ Grounded answer with        │
│ citations to the original   │
│ review page                 │
│                             │
│ LLM: Groq                   │
│ (llama-3.3-70b-versatile)   │
└─────────────────────────────┘

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

I will use Claude to help implement each stage of the RAG pipeline. I will provide the project requirements and sample professor and course review data and ask Claude for code to clean and structure the required documents. For chunking, embeddings, retrieval, and generation, I will provide the Architecture including chosen tools: LangChain, all-MiniLM-L6-v2, ChromaDB, and Groq, and ask for code that follows my specifications. I will verify the output by testing each component, inspecting retrieved chunks, and ensuring generated answers are grounded in the retrieved reviews with proper source citations.

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
