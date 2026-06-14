# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

Course and professor reviews

---

## Documents

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | Rate My Courses | Student review for PHYS 121 Mechanics course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/phys121 |
| 2 | Rate My Courses | Student review for ECON 200 Introduction to Microeconomics course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/econ200 |
| 3 | Rate My Courses | Student review for MATH 125 Calculus with Analytic Geometry II course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/math125 |
| 4 | Rate My Courses | Student review for MATH 126 Calculus with Analytic Geometry III course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/math126 |
| 5 | Rate My Courses | Student review for CSE 143 Computer Programming II course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/cse143 |
| 6 | Rate My Courses | Student review for CSE 142 Computer Programming I course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/cse142 |
| 7 | Rate My Courses | Student review for ENGL 131 Composition: Exposition course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/engl131 |
| 8 | Rate My Courses | Student review for PSYCH 210 The Diversity of Human Sexuality course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/psych210 |
| 9 | Rate My Courses | Student review for CSE 373 Data Structures and Algorithms course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/cse373 |
| 10 | Rate My Courses | Student review for STAT 311 Elements of Statistical Methods course. It is from the University of Washington. | https://www.ratemycourses.io/uw/course/stat311 |

---

## Chunking Strategy

**Chunk size:**
250 tokens

**Overlap:**
0

**Reasoning:**
The student reviews are short in length (100-300 words), organized, and tend to be more concise, therefore chunk size does not require as many tokens and minimum to no overlap is needed to get the full context.

---

## Retrieval Approach

**Embedding model:**
all-MiniLM-L6-v2

**Top-k:**
5 to 6 chunks

**Production tradeoff reflection:**
If costs weren't a factor, improving accuracy would be the highest priority.

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | How do students describe the organization of assignments, deadlines, and course materials in the course? | Indicates whether assignments, deadlines, and course materials are clearly structured or confusing and difficult to navigate. |
| 2 | What patterns appear in student feedback about the frequency and workload of assignments, quizzes, and deadlines in the course? | Report of the course workload, whether it is light, moderate, or heavy, based on course content. |
| 3 | How do students evaluate the course instructor's ability to explain difficult concepts during lectures, discussions, or office hours? | Report of the instructor’s effectivenes at explaining the course material. |
| 4 | What do student reviews indicate about the alignment between exams and the material covered in lectures, homework, and assigned readings in the course? | Reports of whether exams closely match course materials or feel disconnected from what was taught. |
| 5 | To what extent do students feel the instructor helped offset issues with course structure, workload, textbook, or logistics in the course? | Signs of improvements in the learning experience despite challenges in the course. |

---

## Anticipated Challenges

1. Returns biased results.

2. Used information and data is outdated.

---

## Architecture

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

I will use Claude to help implement each stage of the RAG pipeline. 
I will provide Claude the project requirements and samples of professor and course review data and ask it for code to clean and structure the required documents. 
For chunking, embeddings, retrieval, and generation, I will provide the Architecture including chosen tools: LangChain, all-MiniLM-L6-v2, ChromaDB, and Groq, and ask for code that follows my specifications. 
I will verify the output by testing each component, inspecting retrieved chunks, and ensuring generated answers are grounded in the retrieved reviews with proper source citations.

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
