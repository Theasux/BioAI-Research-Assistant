# BioAI Research Assistant

A scientific literature retrieval and question-answering system for life science research, combining dense semantic retrieval, keyword retrieval, hybrid ranking, and retrieval-augmented generation (RAG).

## Overview

BioAI Research Assistant is an independent BioAI project built using publicly available PubMed literature.

The project explores how modern information retrieval and large language models can be combined to support scientific literature search and evidence-grounded question answering.

The system implements a complete pipeline from literature collection and retrieval to evidence-grounded answer generation.

### Key Components

- PubMed literature collection
- Text cleaning and chunking
- Dense semantic retrieval
- BM25 keyword retrieval
- Hybrid retrieval
- Reciprocal Rank Fusion (RRF)
- Retrieval evaluation
- Ground-truth coverage analysis
- Retrieval error analysis
- Retrieval-Augmented Generation (RAG)

## System Architecture

```text
User Query
    |
    v
+--------------------------------+
|       Hybrid Retrieval         |
|                                |
|  +--------------------------+  |
|  | Dense Retrieval          |  |
|  | Sentence Transformer     |  |
|  | FAISS                    |  |
|  +--------------------------+  |
|              +                 |
|  +--------------------------+  |
|  | Keyword Retrieval        |  |
|  | BM25                     |  |
|  +--------------------------+  |
|              |                 |
|              v                 |
|       RRF Rank Fusion          |
+--------------------------------+
    |
    v
Top-K Papers
    |
    v
Context Construction
    |
    v
DeepSeek LLM
    |
    v
Evidence-grounded Answer
    |
    v
PMID Citations
Dataset

The current development corpus contains:

555 unique PubMed papers
2,718 text chunks
384-dimensional embeddings

The literature corpus covers several life-science topics, including:

Salt stress
Alternative splicing
Ion transport
Maize regulation
Photosynthesis

All literature records used in the project are collected from publicly available PubMed data.

Retrieval
Dense Semantic Retrieval

The system uses Sentence Transformers to convert scientific text into dense semantic embeddings.

The current implementation uses:

Model: all-MiniLM-L6-v2
Embedding dimension: 384
Vector index: FAISS IndexFlatIP
Normalized embeddings

With normalized embeddings, inner product similarity corresponds to cosine similarity.

Dense retrieval is performed at the text-chunk level.

The current implementation retrieves the top 100 candidate chunks and aggregates them at the PMID level by retaining the highest-scoring chunk for each paper.

This combines fine-grained semantic matching with paper-level ranking.

BM25 Keyword Retrieval

BM25 is used as a keyword-based retrieval method to capture lexical matching.

Compared with semantic retrieval, BM25 is particularly useful when queries contain:

Gene symbols
Protein names
Abbreviations
Technical terminology
Exact scientific keywords

The current implementation performs BM25 retrieval at the paper level.

Hybrid Retrieval

Dense Retrieval and BM25 are combined using Reciprocal Rank Fusion (RRF).

The retrieval process is:

User Query
    |
    +-------------------------+
    |                         |
    v                         v
Dense Retrieval           BM25 Retrieval
    |                         |
    v                         v
Top 100 Chunks            Top 100 Papers
    |
    v
PMID-level Aggregation
    |
    +------------+------------+
                 |
                 v
             RRF Fusion
                 |
                 v
          PMID-level Ranking
                 |
                 v
              Top-K

Dense retrieval and BM25 provide complementary retrieval signals.

Dense retrieval focuses on semantic similarity, while BM25 focuses on lexical matching.

The final hybrid results are ranked at the PMID level and deduplicated before being passed to the RAG pipeline.

Reciprocal Rank Fusion

Reciprocal Rank Fusion combines ranked results from different retrieval systems without directly combining their original scores.

The RRF score is defined as:

RRF(d) = Σ 1 / (k + rank(d))

The current implementation uses:

k = 60

RRF is used because Dense Retrieval and BM25 produce scores with different scales.

Instead of directly adding incompatible similarity scores, RRF combines the relative rankings produced by the two retrieval methods.

RAG Pipeline

The project implements a retrieval-augmented generation pipeline:

Query
  |
  v
Hybrid Retrieval
  |
  v
Top-K Papers
  |
  v
Context Construction
  |
  v
Prompt Assembly
  |
  v
DeepSeek LLM
  |
  v
Evidence-grounded Answer
  |
  v
PMID Citations

The retrieval system provides scientific literature evidence as external context for the language model.

DeepSeek is used as the generation model through an API.

The generated answer is based on the retrieved literature context, with PMID identifiers retained for evidence traceability.

The core idea is:

Retrieval provides evidence
          +
LLM provides generation
          =
Evidence-grounded scientific QA
Evaluation

A small development benchmark was constructed to evaluate the retrieval system.

The benchmark contains:

8 evaluation questions
18 ground-truth relevant PMIDs
Recall@5
Precision@5
MRR

The benchmark is intended for system development and debugging rather than large-scale statistical evaluation.

Retrieval Comparison

The current development benchmark produced the following results:

Method	Recall@5	Precision@5	MRR
Dense Retrieval	0.312	0.100	0.250
BM25	0.292	0.075	0.281
Hybrid RRF	0.323	0.100	0.417

Because the benchmark contains only 8 questions, these results should be interpreted as development-level observations rather than statistically conclusive comparisons.

The evaluation code supports comparison of:

Dense Retrieval
BM25
Hybrid RRF

using standard information retrieval metrics.

Ground-truth Coverage

A ground-truth coverage audit was performed before interpreting retrieval results.

All 18 ground-truth PMIDs were found in the current literature corpus.

18 Ground-truth PMIDs
          |
          v
18 PMIDs found in corpus
          |
          v
Coverage = 100%

This analysis helps distinguish retrieval failures from corpus coverage failures.

If a relevant PMID is absent from the corpus, the retrieval system cannot retrieve it regardless of retrieval quality.

Retrieval Error Analysis

The project includes retrieval error analysis to investigate why relevant papers may not appear in the final Top-K results.

The analysis considers several potential sources of retrieval errors:

Relevant papers not retrieved by Dense Retrieval
Relevant papers not retrieved by BM25
Ranking changes after hybrid fusion
Top-K truncation
Semantic similarity without sufficient relevance
Corpus coverage limitations

The purpose of error analysis is to identify retrieval bottlenecks before modifying the retrieval architecture.

LLM-only vs. RAG

The project also includes an experimental comparison between:

LLM-only

and:

Retrieval + RAG + LLM

The purpose of this experiment is to examine the effect of providing external scientific literature as context to the language model.

This experiment is used as a development comparison rather than a large-scale benchmark.

Project Structure
BioAI-Research-Assistant/
|
├── 5.retrieval/
│   ├── 01-pubmed_retrieval.py
│   ├── 02-02-build_corpus.py
│   ├── 03-build_index.py
│   ├── 04-search.py
│   ├── 05-hybrid_search.py
│   ├── 06-build_corpus_v2.py
│   ├── 07-bm25_search.py
│   ├── 08-build_index_v2.py
│   └── 09-hybrid_search_v2.py
│
├── 6.evaluation/
│   ├── 01-retrieval_questions.json
│   ├── 02-evaluate.py
│   ├── 03-compare_llm_vs_rag.py
│   ├── 04-evaluate_retrieval_v2.py
│   ├── 05-retrieval_error_analysis.json
│   ├── 06-check_ground_truth_coverage.py.py
│   └── 07-retrieval_error_analysis.py
│
├── 7.rag/
│   ├── 01-rag.py
│   └── 01-rag02.py
│
├── README.md
├── requirements.txt
└── .gitignore
Technologies
AI / Information Retrieval
Sentence Transformers
Transformer-based Embeddings
FAISS
BM25
Reciprocal Rank Fusion (RRF)
Retrieval-Augmented Generation (RAG)
Large Language Model API
Programming / Data
Python
NumPy
REST API
XML
JSON
Git
Conda
Scientific Data
PubMed
Scientific Literature
Life Science Research
Reproducibility

The project uses Python and commonly available open-source libraries.

Install the required dependencies with:

pip install -r requirements.txt

The main project dependencies are:

numpy
requests
sentence-transformers
faiss-cpu
rank-bm25
openai

Large local datasets, FAISS indexes, generated retrieval results, and model files are excluded from Git through .gitignore.

The literature corpus and retrieval indexes are generated locally using the provided scripts.

Current Status

The current version implements the workflow from public scientific literature collection to evidence-grounded question answering.

Implemented components include:

PubMed literature collection
Corpus construction
Text cleaning and chunking
Dense semantic retrieval
BM25 keyword retrieval
Hybrid retrieval
RRF ranking
PMID-level aggregation and deduplication
Retrieval evaluation
Ground-truth coverage analysis
Retrieval error analysis
RAG-based question answering
LLM-only vs. RAG comparison
Future Development

Potential future improvements include:

Larger retrieval evaluation datasets
Improved chunking strategies
Query expansion
Reranking
Citation verification
Tool Calling
Agent-based literature search
Backend API service
Interactive frontend

These components are planned extensions and are not part of the current implemented version.

Motivation

This project was developed as an independent exploration of AI applications in life science research.

The goal is to combine scientific domain knowledge with modern AI and information retrieval techniques to support scientific data processing, literature discovery, and evidence-grounded question answering.