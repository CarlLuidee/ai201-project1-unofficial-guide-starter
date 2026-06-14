# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

Student course reviews at the University of Washington.

---

## Document Sources

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

**Why these choices fit your documents:**
The student reviews are short in length (100-300 words), organized, and tend to be more concise, therefore chunk size does not require as many tokens and minimum to no overlap is needed to get the full context.

**Final chunk count:**
59

---

## Embedding Model

**Model used:**
all-MiniLM-L6-v2

**Production tradeoff reflection:**
If costs weren't a factor, improving accuracy would be the highest priority.

---

## Grounded Generation

**System prompt grounding instruction:**
The model only takes data from the chunks based off of information I manually gathered.

**How source attribution is surfaced in the response:**
A URL leading to the original source used by the model is cited just below the response.

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | How do students describe the organization of assignments, deadlines, and course materials in PHYS 121? | Indicates whether assignments, deadlines, and course materials are clearly structured or confusing and difficult to navigate. | Some students find PHYS 121 disorganized, while others focus on its positive aspects. | Relevant | Partially accurate |
| 2 | What patterns appear in student feedback about the frequency and workload of assignments, quizzes, and deadlines in PHYS 121? | Summary of the course workload. | The course has a challenging workload, though it is manageable with classroom support. | Relevant | Accurate |
| 3 | How do students evaluate PHYS 121 instructor's ability to explain difficult concepts during lectures, discussions, or office hours? | Report of the instructor’s effectivenes at explaining the course material. | The professors themselves are excellent at their roles, but the course design hampers their ability to teach. | Relevant | Partially accurate |
| 4 | What do student reviews indicate about the alignment between exams and the material covered in lectures, homework, and assigned readings in PHYS 121? | If exams closely match course materials or feel disconnected from what was taught. | The student reviews do not provide sufficient or any information about the alignment between exams and course materials. | Off-target | Inaccurate |
| 5 | To what extent do students feel the instructor helped offset issues with course structure, workload, textbook, or logistics in PHYS 121? | Signs of improvements in the learning experience despite challenges in the course. | Some professors were accessible during office hours and very supportive of their students. | Relevant | Accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:**
What do student reviews indicate about the alignment between exams and the material covered in lectures, homework, and assigned readings in PHYS 121?

**What the system returned:**
The student reviews do not provide information about the alignment between exams and the material covered in lectures, homework, and assigned readings in PHYS 121, as the review for PHYS 121 only mentions that the practical knowledge of mechanics makes it a useful course and that the professor made the class fun, but does not discuss the exams or the material covered [SOURCE 1].

**Root cause (tied to a specific pipeline stage):**
There was insufficient student review data to address a specific question, so the model lacked sufficient information to provide a concrete answer.

**What you would change to fix it:**
Find more sources for student reviews of a specific course to yield a greater number of chunks and more accurate model responses.

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:**
The spec helped me organize and plan the project more efficiently. It also guided me on what to prompt Claude to provide, giving me the exact code I needed.

**One way your implementation diverged from the spec, and why:**
I largeley stuck to the orginal specs I wrote. The most significant deviation from my spec was limiting the review data to the University of Washington. I decided on this because courses vary wildy across different colleges. I believe it will only skew the results if I were to add review data from other colleges.

---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

**Instance 1**

- *What I gave the AI:* I provided Claude with the strategies from my planning.md and asked it to implement the chunking function.
- *What it produced:* It gave me a sentence split.
- *What I changed or overrode:* I changed the chunk size from 300 to 250 because the reviews are short.

**Instance 2**

- *What I gave the AI:* I provided Claude with requirements related to grounding and how the output should be formatted.
- *What it produced:* It gave me code that produced results exclusively from the data I supplied and outputted the appropriate source URLs.
- *What I changed or overrode:* Its output met all of my requirements, so there was no need for any edits to the code.