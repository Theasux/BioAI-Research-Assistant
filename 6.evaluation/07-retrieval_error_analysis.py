from pathlib import Path
import json
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer



# 1. Paths


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

CORPUS_PATH = PROJECT_DIR / "5.retrieval" / "pubmed_corpus_v2.json"

DENSE_INDEX_PATH = (
    Path("5.retrieval")
    / "index_v2"
    / "pubmed_v2.index"
)

CHUNKS_PATH = (
    PROJECT_DIR
    / "5.retrieval"
    / "index_v2"
    / "chunks_v2.json"
)

QUESTIONS_PATH = (
    BASE_DIR
    / "01-retrieval_questions.json"
)



# 2. Basic text tokenizer for BM25


def tokenize(text):
    """
    Simple tokenizer:
    lowercase + keep alphanumeric words.
    """
    return re.findall(r"\b\w+\b", text.lower())



# 3. Load corpus


print("=" * 70)
print("BioAI Retrieval Error Analysis")
print("=" * 70)

print("\nLoading corpus...")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)

print(f"Papers: {len(corpus)}")



# 4. Load chunks


print("\nLoading chunks...")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Chunks: {len(chunks)}")



# 5. Build paper-level BM25 corpus


print("\nBuilding BM25 index...")

paper_texts = []

for paper in corpus:
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    text = f"{title} {abstract}"
    paper_texts.append(text)

tokenized_corpus = [
    tokenize(text)
    for text in paper_texts
]

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 ready.")



# 6. Load Dense FAISS index


print("\nLoading FAISS index...")

dense_index = faiss.read_index(str(DENSE_INDEX_PATH))

print(f"Dense vectors: {dense_index.ntotal}")
print(f"Dimension: {dense_index.d}")



# 7. Load embedding model


print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model ready.")



# 8. Load evaluation questions


print("\nLoading evaluation questions...")

with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)

print(f"Questions: {len(questions)}")



# 9. Build mappings


# Corpus order:
# PMID -> paper index
pmid_to_paper_index = {}

for i, paper in enumerate(corpus):
    pmid = str(paper["pmid"])
    pmid_to_paper_index[pmid] = i


# Chunk -> PMID
chunk_pmids = [
    str(chunk["pmid"])
    for chunk in chunks
]



# 10. Analyze each question


all_results = []

for q_idx, item in enumerate(questions, start=1):

    question = item["question"]

    relevant_pmids = [
        str(pmid)
        for pmid in item["relevant_pmids"]
    ]

    print("\n")
    print("=" * 70)
    print(f"Question {q_idx}")
    print("=" * 70)

    print(f"Query:\n{question}")

    print("\nGround-truth PMIDs:")
    print(relevant_pmids)


    # --------------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------------

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    # Search ALL chunks.
    # This is important because we want to know
    # the true rank of every ground-truth paper.
    dense_scores, dense_indices = dense_index.search(
        query_embedding,
        dense_index.ntotal
    )

    dense_scores = dense_scores[0]
    dense_indices = dense_indices[0]


    # --------------------------------------------------------
    # Convert chunk-level Dense results
    # to paper-level ranking
    # --------------------------------------------------------

    dense_paper_scores = {}

    for score, chunk_idx in zip(
        dense_scores,
        dense_indices
    ):

        pmid = chunk_pmids[int(chunk_idx)]

        # Keep the best chunk score for each paper.
        if (
            pmid not in dense_paper_scores
            or score > dense_paper_scores[pmid]
        ):
            dense_paper_scores[pmid] = float(score)


    dense_ranking = sorted(
        dense_paper_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    dense_rank_dict = {
        pmid: rank
        for rank, (pmid, score)
        in enumerate(dense_ranking, start=1)
    }

    dense_score_dict = dict(dense_ranking)


    # --------------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------------

    query_tokens = tokenize(question)

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_ranking_indices = np.argsort(
        bm25_scores
    )[::-1]

    bm25_ranking = []

    for paper_idx in bm25_ranking_indices:

        pmid = str(
            corpus[int(paper_idx)]["pmid"]
        )

        score = float(
            bm25_scores[int(paper_idx)]
        )

        bm25_ranking.append(
            (pmid, score)
        )


    bm25_rank_dict = {
        pmid: rank
        for rank, (pmid, score)
        in enumerate(bm25_ranking, start=1)
    }

    bm25_score_dict = dict(bm25_ranking)



    # Print ground-truth diagnostic table


    print("\n")
    print(
        f"{'PMID':<12}"
        f"{'Dense Rank':<14}"
        f"{'Dense Score':<14}"
        f"{'BM25 Rank':<14}"
        f"{'BM25 Score':<14}"
    )

    print("-" * 68)

    for pmid in relevant_pmids:

        dense_rank = dense_rank_dict.get(
            pmid,
            None
        )

        dense_score = dense_score_dict.get(
            pmid,
            None
        )

        bm25_rank = bm25_rank_dict.get(
            pmid,
            None
        )

        bm25_score = bm25_score_dict.get(
            pmid,
            None
        )

        dense_score_text = (
            f"{dense_score:.4f}"
            if dense_score is not None
            else "NA"
        )

        bm25_score_text = (
            f"{bm25_score:.4f}"
            if bm25_score is not None
            else "NA"
        )

        print(
            f"{pmid:<12}"
            f"{str(dense_rank):<14}"
            f"{dense_score_text:<14}"
            f"{str(bm25_rank):<14}"
            f"{bm25_score_text:<14}"
        )



    # Print Top 10 for each method


    print("\nDense Top 10:")
    for rank, (pmid, score) in enumerate(
        dense_ranking[:10],
        start=1
    ):

        marker = (
            " <-- GT"
            if pmid in relevant_pmids
            else ""
        )

        print(
            f"{rank:>2}. "
            f"{pmid} "
            f"score={score:.4f}"
            f"{marker}"
        )


    print("\nBM25 Top 10:")

    for rank, (pmid, score) in enumerate(
        bm25_ranking[:10],
        start=1
    ):

        marker = (
            " <-- GT"
            if pmid in relevant_pmids
            else ""
        )

        print(
            f"{rank:>2}. "
            f"{pmid} "
            f"score={score:.4f}"
            f"{marker}"
        )



    # Save diagnostic result


    question_result = {
        "question": question,
        "relevant_pmids": relevant_pmids,
        "ground_truth": {}
    }

    for pmid in relevant_pmids:

        question_result["ground_truth"][pmid] = {
            "dense_rank": dense_rank_dict.get(pmid),
            "dense_score": dense_score_dict.get(pmid),
            "bm25_rank": bm25_rank_dict.get(pmid),
            "bm25_score": bm25_score_dict.get(pmid)
        }

    all_results.append(question_result)



# 11. Save results


OUTPUT_PATH = (
    BASE_DIR
    / "05-retrieval_error_analysis.json"
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_results,
        f,
        ensure_ascii=False,
        indent=2
    )



# 12. Finished


print("\n")
print("=" * 70)
print("Error analysis completed.")
print("=" * 70)

print(f"\nSaved:")
print(OUTPUT_PATH)