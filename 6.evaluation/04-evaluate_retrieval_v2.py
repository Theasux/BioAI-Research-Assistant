from pathlib import Path
import json
import re

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer



# Configuration


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

CORPUS_PATH = PROJECT_DIR / "5.retrieval" / "pubmed_corpus_v2.json"

DENSE_INDEX_PATH = (
    PROJECT_DIR
    / "5.retrieval"
    / "index_v2"
    / "pubmed_v2.index"
)

CHUNKS_PATH = (
    PROJECT_DIR
    / "5.retrieval"
    / "index_v2"
    / "chunks_v2.json"
)

QUESTIONS_PATH = BASE_DIR / "01-retrieval_questions.json"

RESULT_PATH = BASE_DIR / "04-retrieval_v2_evaluation.json"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5

# Number of candidates used before final Top 5
DENSE_CANDIDATE_K = 100
BM25_CANDIDATE_K = 100

# RRF constant
RRF_K = 60



# Helper functions


def tokenize(text):
    """
    Simple tokenizer for BM25.

    Convert text to lowercase and split into word-like tokens.
    """
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25_documents(corpus):
    """
    Build BM25 documents using title + abstract.
    """

    documents = []

    for paper in corpus:

        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        text = f"{title} {abstract}"

        documents.append(text)

    return documents


def get_top_bm25(corpus, bm25, query, top_k):
    """
    Retrieve top papers using BM25.
    """

    scores = bm25.get_scores(tokenize(query))

    # Sort descending by BM25 score
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, idx in enumerate(ranked_indices, start=1):

        paper = corpus[int(idx)]

        results.append({
            "pmid": paper["pmid"],
            "score": float(scores[idx]),
            "rank": rank
        })

    return results


def get_dense_paper_ranking(
    corpus,
    index,
    chunks,
    model,
    query,
    candidate_k
):
    """
    Dense retrieval.

    FAISS works at chunk level.

    We therefore:
    1. retrieve chunks
    2. aggregate by PMID
    3. keep the maximum chunk score for each paper
    4. rank papers by maximum dense similarity
    """

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        candidate_k
    )

    paper_scores = {}

    for score, chunk_idx in zip(
        scores[0],
        indices[0]
    ):

        if chunk_idx < 0:
            continue

        chunk = chunks[int(chunk_idx)]

        pmid = chunk["pmid"]

        score = float(score)

        # Keep the best matching chunk for each paper
        if (
            pmid not in paper_scores
            or score > paper_scores[pmid]
        ):
            paper_scores[pmid] = score

    # Sort papers by dense score
    ranked_pmids = sorted(
        paper_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for rank, (pmid, score) in enumerate(
        ranked_pmids,
        start=1
    ):

        results.append({
            "pmid": pmid,
            "score": float(score),
            "rank": rank
        })

    return results


def get_hybrid_ranking(
    dense_results,
    bm25_results,
    rrf_k,
    top_k
):
    """
    Reciprocal Rank Fusion (RRF).

    RRF score:

        1 / (k + rank_dense)
        +
        1 / (k + rank_bm25)

    This combines rankings rather than raw scores.
    """

    hybrid_scores = {}

    # --------------------------------------------------------
    # Dense contribution
    # --------------------------------------------------------

    for item in dense_results:

        pmid = item["pmid"]
        rank = item["rank"]

        contribution = 1.0 / (rrf_k + rank)

        hybrid_scores.setdefault(
            pmid,
            {
                "rrf_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_score": None,
                "bm25_score": None
            }
        )

        hybrid_scores[pmid]["rrf_score"] += contribution
        hybrid_scores[pmid]["dense_rank"] = rank
        hybrid_scores[pmid]["dense_score"] = item["score"]

    # --------------------------------------------------------
    # BM25 contribution
    # --------------------------------------------------------

    for item in bm25_results:

        pmid = item["pmid"]
        rank = item["rank"]

        contribution = 1.0 / (rrf_k + rank)

        hybrid_scores.setdefault(
            pmid,
            {
                "rrf_score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "dense_score": None,
                "bm25_score": None
            }
        )

        hybrid_scores[pmid]["rrf_score"] += contribution
        hybrid_scores[pmid]["bm25_rank"] = rank
        hybrid_scores[pmid]["bm25_score"] = item["score"]

    # --------------------------------------------------------
    # Sort by RRF score
    # --------------------------------------------------------

    ranked = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1]["rrf_score"],
        reverse=True
    )

    results = []

    for rank, (pmid, info) in enumerate(
        ranked[:top_k],
        start=1
    ):

        results.append({
            "pmid": pmid,
            "rrf_score": info["rrf_score"],
            "dense_rank": info["dense_rank"],
            "bm25_rank": info["bm25_rank"],
            "dense_score": info["dense_score"],
            "bm25_score": info["bm25_score"],
            "rank": rank
        })

    return results


def evaluate_method(retrieved_pmids, relevant_pmids):
    """
    Calculate:

    Recall@5
    Precision@5
    Reciprocal Rank
    """

    retrieved_pmids = retrieved_pmids[:TOP_K]

    relevant_set = set(relevant_pmids)

    retrieved_relevant = [
        pmid
        for pmid in retrieved_pmids
        if pmid in relevant_set
    ]

    # --------------------------------------------------------
    # Recall@5
    # --------------------------------------------------------

    recall = (
        len(retrieved_relevant)
        / len(relevant_set)
        if relevant_set
        else 0.0
    )

    # --------------------------------------------------------
    # Precision@5
    # --------------------------------------------------------

    precision = (
        len(retrieved_relevant)
        / TOP_K
    )

    # --------------------------------------------------------
    # Reciprocal Rank
    # --------------------------------------------------------

    reciprocal_rank = 0.0

    for rank, pmid in enumerate(
        retrieved_pmids,
        start=1
    ):

        if pmid in relevant_set:

            reciprocal_rank = 1.0 / rank
            break

    return {
        "recall_at_5": recall,
        "precision_at_5": precision,
        "reciprocal_rank": reciprocal_rank
    }



# Main


print("=" * 70)
print("BioAI Retrieval V2 Evaluation")
print("=" * 70)



# Load corpus


print("\nLoading PubMed corpus...")

with open(
    CORPUS_PATH,
    "r",
    encoding="utf-8"
) as f:

    corpus = json.load(f)

print(f"Papers loaded: {len(corpus)}")



# Load Dense index


print("\nLoading Dense Retrieval V2 index...")

# IMPORTANT:
# FAISS on Windows may have problems with absolute paths
# containing Chinese characters.
#
# Therefore use a relative path when reading the FAISS index.

relative_dense_index = Path(
    "5.retrieval"
) / "index_v2" / "pubmed_v2.index"

index = faiss.read_index(
    str(relative_dense_index)
)

print(f"FAISS vectors: {index.ntotal}")
print(f"FAISS dimension: {index.d}")



# Load chunk metadata


print("\nLoading chunk metadata...")

with open(
    CHUNKS_PATH,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)

print(f"Chunks loaded: {len(chunks)}")


if index.ntotal != len(chunks):

    raise RuntimeError(
        "FAISS index and chunk metadata are inconsistent."
    )

print("Dense index and chunk metadata are consistent.")



# Load BM25


print("\nBuilding BM25 index...")

bm25_documents = build_bm25_documents(corpus)

bm25_tokenized = [
    tokenize(doc)
    for doc in bm25_documents
]

bm25 = BM25Okapi(
    bm25_tokenized
)

print(f"BM25 documents: {len(bm25_documents)}")



# Load embedding model


print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print(f"Embedding model: {MODEL_NAME}")



# Load benchmark questions


print("\nLoading evaluation questions...")

with open(
    QUESTIONS_PATH,
    "r",
    encoding="utf-8"
) as f:

    questions = json.load(f)

print(f"Questions: {len(questions)}")



# Evaluation


all_results = []

dense_metrics = []
bm25_metrics = []
hybrid_metrics = []


for question_number, item in enumerate(
    questions,
    start=1
):

    question = item["question"]
    relevant_pmids = item["relevant_pmids"]

    print("\n")
    print("=" * 70)
    print(f"Question {question_number}")
    print("=" * 70)

    print(f"Query:")
    print(question)

    print("\nRelevant PMIDs:")
    print(relevant_pmids)

    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    dense_results = get_dense_paper_ranking(
        corpus=corpus,
        index=index,
        chunks=chunks,
        model=model,
        query=question,
        candidate_k=DENSE_CANDIDATE_K
    )

    dense_top5 = [
        item["pmid"]
        for item in dense_results[:TOP_K]
    ]

    dense_eval = evaluate_method(
        dense_top5,
        relevant_pmids
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25_results = get_top_bm25(
        corpus=corpus,
        bm25=bm25,
        query=question,
        top_k=BM25_CANDIDATE_K
    )

    bm25_top5 = [
        item["pmid"]
        for item in bm25_results[:TOP_K]
    ]

    bm25_eval = evaluate_method(
        bm25_top5,
        relevant_pmids
    )

    # --------------------------------------------------------
    # Hybrid RRF
    # --------------------------------------------------------

    hybrid_results = get_hybrid_ranking(
        dense_results=dense_results[
            :DENSE_CANDIDATE_K
        ],
        bm25_results=bm25_results[
            :BM25_CANDIDATE_K
        ],
        rrf_k=RRF_K,
        top_k=TOP_K
    )

    hybrid_top5 = [
        item["pmid"]
        for item in hybrid_results
    ]

    hybrid_eval = evaluate_method(
        hybrid_top5,
        relevant_pmids
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nDense Top 5:")
    print(dense_top5)

    print(
        "Recall@5   : "
        f"{dense_eval['recall_at_5']:.3f}"
    )

    print(
        "Precision@5: "
        f"{dense_eval['precision_at_5']:.3f}"
    )

    print(
        "RR         : "
        f"{dense_eval['reciprocal_rank']:.3f}"
    )

    print("\nBM25 Top 5:")
    print(bm25_top5)

    print(
        "Recall@5   : "
        f"{bm25_eval['recall_at_5']:.3f}"
    )

    print(
        "Precision@5: "
        f"{bm25_eval['precision_at_5']:.3f}"
    )

    print(
        "RR         : "
        f"{bm25_eval['reciprocal_rank']:.3f}"
    )

    print("\nHybrid RRF Top 5:")
    print(hybrid_top5)

    print(
        "Recall@5   : "
        f"{hybrid_eval['recall_at_5']:.3f}"
    )

    print(
        "Precision@5: "
        f"{hybrid_eval['precision_at_5']:.3f}"
    )

    print(
        "RR         : "
        f"{hybrid_eval['reciprocal_rank']:.3f}"
    )

    # --------------------------------------------------------
    # Save question result
    # --------------------------------------------------------

    all_results.append({

        "question": question,

        "relevant_pmids": relevant_pmids,

        "dense": {
            "top5": dense_top5,
            "metrics": dense_eval
        },

        "bm25": {
            "top5": bm25_top5,
            "metrics": bm25_eval
        },

        "hybrid_rrf": {
            "top5": hybrid_top5,
            "metrics": hybrid_eval
        }
    })

    dense_metrics.append(dense_eval)
    bm25_metrics.append(bm25_eval)
    hybrid_metrics.append(hybrid_eval)



# Overall metrics


def mean_metric(metrics, key):

    values = [
        item[key]
        for item in metrics
    ]

    return float(
        np.mean(values)
    )


dense_overall = {
    "recall_at_5": mean_metric(
        dense_metrics,
        "recall_at_5"
    ),
    "precision_at_5": mean_metric(
        dense_metrics,
        "precision_at_5"
    ),
    "mrr": mean_metric(
        dense_metrics,
        "reciprocal_rank"
    )
}


bm25_overall = {
    "recall_at_5": mean_metric(
        bm25_metrics,
        "recall_at_5"
    ),
    "precision_at_5": mean_metric(
        bm25_metrics,
        "precision_at_5"
    ),
    "mrr": mean_metric(
        bm25_metrics,
        "reciprocal_rank"
    )
}


hybrid_overall = {
    "recall_at_5": mean_metric(
        hybrid_metrics,
        "recall_at_5"
    ),
    "precision_at_5": mean_metric(
        hybrid_metrics,
        "precision_at_5"
    ),
    "mrr": mean_metric(
        hybrid_metrics,
        "reciprocal_rank"
    )
}



# Print overall results


print("\n")
print("=" * 70)
print("Overall Retrieval Performance")
print("=" * 70)

print(
    f"{'Method':<20}"
    f"{'Recall@5':>12}"
    f"{'Precision@5':>15}"
    f"{'MRR':>10}"
)

print("-" * 70)

print(
    f"{'Dense V2':<20}"
    f"{dense_overall['recall_at_5']:>12.3f}"
    f"{dense_overall['precision_at_5']:>15.3f}"
    f"{dense_overall['mrr']:>10.3f}"
)

print(
    f"{'BM25':<20}"
    f"{bm25_overall['recall_at_5']:>12.3f}"
    f"{bm25_overall['precision_at_5']:>15.3f}"
    f"{bm25_overall['mrr']:>10.3f}"
)

print(
    f"{'Hybrid RRF':<20}"
    f"{hybrid_overall['recall_at_5']:>12.3f}"
    f"{hybrid_overall['precision_at_5']:>15.3f}"
    f"{hybrid_overall['mrr']:>10.3f}"
)



# Save evaluation results


output = {

    "configuration": {

        "corpus_papers": len(corpus),

        "dense_chunks": len(chunks),

        "embedding_model": MODEL_NAME,

        "dense_candidate_k": DENSE_CANDIDATE_K,

        "bm25_candidate_k": BM25_CANDIDATE_K,

        "rrf_k": RRF_K,

        "top_k": TOP_K
    },

    "overall": {

        "dense_v2": dense_overall,

        "bm25": bm25_overall,

        "hybrid_rrf": hybrid_overall
    },

    "questions": all_results
}


print("\nSaving evaluation results...")

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"Saved:")
print(RESULT_PATH)


print("\n")
print("=" * 70)
print("Evaluation completed successfully.")
print("=" * 70)