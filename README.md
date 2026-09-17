# BioAI Research Assistant

A literature retrieval and question-answering system for life science research, combining semantic retrieval, keyword retrieval, hybrid ranking, and retrieval-augmented generation (RAG).

## Overview

BioAI Research Assistant is an independent BioAI project built with publicly available PubMed literature.

The project explores how AI and information retrieval techniques can be applied to scientific literature search and evidence-grounded question answering.

The system implements:

- PubMed literature collection
- Text cleaning and chunking
- Dense semantic retrieval
- BM25 keyword retrieval
- Reciprocal Rank Fusion (RRF)
- Hybrid retrieval
- Retrieval evaluation
- Retrieval error analysis
- Retrieval-Augmented Generation (RAG)

## System Pipeline

```text
User Query
    |
    v
Hybrid Retrieval
    |
    v
Top-K Papers
    |
    v
RAG Context Construction
    |
    v
DeepSeek LLM
    |
    v
Evidence-grounded Answer
Dataset

The current development corpus contains:

555 unique PubMed papers
2,718 text chunks
384-dimensional sentence embeddings

The literature covers several life-science topics, including:

Salt stress
Alternative splicing
Ion transport
Maize regulation
Photosynthesis

All literature data used in this project are collected from publicly available PubMed records.

Retrieval
Dense Retrieval

Sentence Transformers are used to convert text chunks into dense semantic embeddings.

FAISS is used to build a vector index and perform efficient similarity search.

BM25 Retrieval

BM25 is implemented as a keyword-based retrieval method to capture exact terminology and lexical matching.

Hybrid Retrieval

Dense retrieval and BM25 retrieval are combined using Reciprocal Rank Fusion (RRF).

The system performs paper-level ranking and PMID-level deduplication before constructing the final retrieval context.

RAG

The project implements a complete retrieval-augmented generation pipeline:

Query
  |
  v
Retrieval
  |
  v
Top-K Papers
  |
  v
Context Construction
  |
  v
LLM
  |
  v
Answer + PMID Citations

DeepSeek is used as the generation model.

The generated answers are grounded in retrieved literature evidence, and PMID identifiers are included to support traceability.

Evaluation

A small development benchmark was constructed for retrieval evaluation:

8 evaluation questions
18 ground-truth relevant PMIDs
Recall@5
Precision@5
MRR

The project also includes:

Dense vs BM25 vs Hybrid retrieval comparison
LLM-only vs RAG comparison
Ground-truth coverage analysis
Retrieval error analysis

The benchmark is intended for development and system debugging rather than large-scale statistical evaluation.

Project Structure
BioAI-Research-Assistant/
|
├── 5.retrieval/
│   ├── 09-hybrid_search_v2.py
│   └── ...
|
├── 6.evaluation/
│   ├── 01-retrieval_questions.json
│   ├── 02-evaluate.py
│   ├── 03-compare_llm_vs_rag.py
│   ├── 04-evaluate_retrieval_v2.py
│   ├── 05-retrieval_error_analysis.json
│   └── ...
|
├── 7.rag/
│   └── ...
|
├── requirements.txt
├── .gitignore
└── README.md
Technologies
AI / Machine Learning
Python
PyTorch
scikit-learn
Sentence Transformers
Transformer
Embedding
FAISS
BM25
RRF
RAG
LLM API
Data / Engineering
NumPy
Pandas
REST API
JSON
Git
Conda
Linux / HPC
Bioinformatics
PubMed
NGS
RNA-seq
PacBio Iso-Seq
Oxford Nanopore sequencing
Alternative Splicing
lncRNA
Transcription Factors
GO / KEGG
Reproducibility

The project uses Python and common open-source libraries.

Install dependencies with:

pip install -r requirements.txt

Large local datasets, FAISS indexes, generated results, and model files are excluded from Git through .gitignore.

Current Status

The current version implements the complete workflow from literature retrieval to evidence-grounded question answering.

Future development may include:

Tool Calling
Agent-based literature search
Improved retrieval evaluation
Query expansion
Reranking
More comprehensive evaluation datasets
Motivation

This project was developed as an independent exploration of AI + life science applications.

The goal is to combine a background in bioinformatics with modern AI techniques for scientific data processing, information retrieval, and intelligent knowledge discovery.
