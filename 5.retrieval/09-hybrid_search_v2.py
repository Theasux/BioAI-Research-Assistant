from pathlib import Path
import json
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer



# BioAI Hybrid Retrieval V2
#
# Dense Retrieval + BM25 + RRF



print("=" * 70)
print("BioAI Hybrid Retrieval V2")
print("=" * 70)



# 1. Paths


BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# PubMed V2 corpus
# ------------------------------------------------------------

CORPUS_PATH = BASE_DIR / "pubmed_corpus_v2.json"


# ------------------------------------------------------------
# Dense Retrieval V2
# ------------------------------------------------------------

DENSE_INDEX_PATH = Path(
    r".\5.retrieval\index_v2\pubmed_v2.index"
)

CHUNKS_PATH = (
    BASE_DIR
    / "index_v2"
    / "chunks_v2.json"
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

RESULTS_PATH = (
    BASE_DIR
    / "09-hybrid_search_v2_results.json"
)



# 2. Configuration


MODEL_NAME = "all-MiniLM-L6-v2"

# Dense retrieval:
# retrieve this many chunks from FAISS first
DENSE_TOP_K = 100

# BM25:
# retrieve this many papers first
BM25_TOP_K = 100

# Final Hybrid result
FINAL_TOP_K = 5

# RRF constant
#
# Standard RRF commonly uses k = 60.
#
# Larger k:
#   reduces the influence of very high ranks
#
# Smaller k:
#   gives stronger weight to top ranks
#
RRF_K = 60



# 3. Tokenizer for BM25


def tokenize(text):
    """
    Simple tokenizer.

    Convert text to lowercase and keep
    letters, numbers and + / - characters.

    Examples:

        salt stress
        HKT1;5
        Na+
        SOS1
        alternative splicing
    """

    text = text.lower()

    tokens = re.findall(
        r"[a-z0-9]+(?:[+-][a-z0-9]+)?",
        text
    )

    return tokens



# 4. Load PubMed corpus


print("\nLoading PubMed corpus...")

if not CORPUS_PATH.exists():
    raise FileNotFoundError(
        f"Corpus not found:\n{CORPUS_PATH}"
    )

with open(
    CORPUS_PATH,
    "r",
    encoding="utf-8"
) as f:

    papers = json.load(f)


print(
    f"Papers loaded: {len(papers)}"
)



# 5. Build paper-level BM25


print("\nBuilding BM25 index...")

bm25_documents = []

for paper in papers:

    title = paper.get(
        "title",
        ""
    )

    abstract = paper.get(
        "abstract",
        ""
    )

    text = (
        f"{title} {abstract}"
    )

    bm25_documents.append(
        text
    )


bm25_tokens = [
    tokenize(text)
    for text in bm25_documents
]


bm25 = BM25Okapi(
    bm25_tokens
)


print(
    f"BM25 documents: {len(bm25_documents)}"
)



# 6. Load Dense FAISS index


print("\nLoading Dense Retrieval V2 index...")

if not DENSE_INDEX_PATH.exists():

    raise FileNotFoundError(
        f"FAISS index not found:\n"
        f"{DENSE_INDEX_PATH}"
    )


index = faiss.read_index(
    str(DENSE_INDEX_PATH)
)


print(
    f"FAISS vectors: {index.ntotal}"
)

print(
    f"FAISS dimension: {index.d}"
)



# 7. Load chunk metadata


print("\nLoading chunk metadata...")

if not CHUNKS_PATH.exists():

    raise FileNotFoundError(
        f"Chunk metadata not found:\n"
        f"{CHUNKS_PATH}"
    )


with open(
    CHUNKS_PATH,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)


print(
    f"Chunks loaded: {len(chunks)}"
)


if index.ntotal != len(chunks):

    raise RuntimeError(
        "FAISS vector count does not match "
        "chunk metadata count."
    )


print(
    "Dense index and chunk metadata are consistent."
)



# 8. Load embedding model


print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print(
    f"Embedding model: {MODEL_NAME}"
)



# 9. Dense retrieval


def dense_retrieval(
    query,
    top_k=100
):
    """
    Dense retrieval operates at chunk level.

    Query
      ↓
    embedding
      ↓
    FAISS
      ↓
    top chunks
      ↓
    aggregate by PMID
      ↓
    paper-level ranking

    For each PMID, the highest-scoring chunk
    is used as the paper's dense score.
    """

    # --------------------------------------------------------
    # Encode query
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )


    # --------------------------------------------------------
    # Search FAISS
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        top_k
    )


    # --------------------------------------------------------
    # Aggregate chunk results by PMID
    # --------------------------------------------------------

    paper_results = {}

    for score, chunk_index in zip(
        scores[0],
        indices[0]
    ):

        if chunk_index < 0:
            continue

        chunk = chunks[chunk_index]

        pmid = str(
            chunk["pmid"]
        )

        score = float(score)


        # ----------------------------------------------------
        # Keep the highest scoring chunk
        # for each paper.
        # ----------------------------------------------------

        if (
            pmid not in paper_results
            or score > paper_results[pmid]["score"]
        ):

            paper_results[pmid] = {
                "pmid": pmid,
                "score": score,
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "title": chunk["title"],
                "topics": chunk.get(
                    "topics",
                    []
                ),
                "text": chunk["text"]
            }


    # --------------------------------------------------------
    # Sort papers by dense score
    # --------------------------------------------------------

    ranked = sorted(
        paper_results.values(),
        key=lambda x: x["score"],
        reverse=True
    )


    return ranked



# 10. BM25 retrieval


def bm25_retrieval(
    query,
    top_k=100
):
    """
    BM25 operates directly at paper level.

    Query
      ↓
    tokenization
      ↓
    BM25
      ↓
    paper ranking
    """

    query_tokens = tokenize(
        query
    )


    scores = bm25.get_scores(
        query_tokens
    )


    # --------------------------------------------------------
    # Get top paper indices
    # --------------------------------------------------------

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]


    ranked = []

    for paper_index in top_indices:

        paper = papers[paper_index]

        ranked.append(
            {
                "paper_index": int(
                    paper_index
                ),

                "pmid": str(
                    paper.get(
                        "pmid",
                        ""
                    )
                ),

                "score": float(
                    scores[paper_index]
                ),

                "title": paper.get(
                    "title",
                    ""
                ),

                "topics": paper.get(
                    "topics",
                    []
                )
            }
        )


    return ranked



# 11. Reciprocal Rank Fusion


def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    rrf_k=60
):
    """
    Reciprocal Rank Fusion.

    RRF score:

        1 / (k + rank)

    A paper can receive contributions from:

        Dense rank
        BM25 rank

    Example:

        Dense rank = 3
        BM25 rank  = 8

        RRF =
            1 / (60 + 3)
          + 1 / (60 + 8)

    The important point:

    RRF uses RANKS instead of raw scores.

    Therefore we do NOT need to directly compare:

        cosine similarity ~ 0.8

    with:

        BM25 score ~ 15

    """


    # --------------------------------------------------------
    # Create lookup dictionaries
    # --------------------------------------------------------

    dense_by_pmid = {
        item["pmid"]: item
        for item in dense_results
    }


    bm25_by_pmid = {
        item["pmid"]: item
        for item in bm25_results
    }


    # --------------------------------------------------------
    # Candidate union
    #
    # Any paper appearing in either retrieval system
    # becomes a Hybrid candidate.
    # --------------------------------------------------------

    candidate_pmids = (
        set(dense_by_pmid.keys())
        |
        set(bm25_by_pmid.keys())
    )


    hybrid_results = []


    for pmid in candidate_pmids:

        # ----------------------------------------------------
        # Dense rank
        # ----------------------------------------------------

        dense_rank = None

        if pmid in dense_by_pmid:

            dense_rank = (
                dense_results.index(
                    dense_by_pmid[pmid]
                ) + 1
            )


        # ----------------------------------------------------
        # BM25 rank
        # ----------------------------------------------------

        bm25_rank = None

        if pmid in bm25_by_pmid:

            bm25_rank = (
                bm25_results.index(
                    bm25_by_pmid[pmid]
                ) + 1
            )


        # ----------------------------------------------------
        # Calculate RRF score
        # ----------------------------------------------------

        rrf_score = 0.0


        if dense_rank is not None:

            rrf_score += (
                1.0
                /
                (rrf_k + dense_rank)
            )


        if bm25_rank is not None:

            rrf_score += (
                1.0
                /
                (rrf_k + bm25_rank)
            )


        # ----------------------------------------------------
        # Get metadata
        # ----------------------------------------------------

        if pmid in dense_by_pmid:

            metadata = dense_by_pmid[pmid]

        else:

            metadata = bm25_by_pmid[pmid]


        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        hybrid_results.append(
            {
                "pmid": pmid,

                "title": metadata[
                    "title"
                ],

                "topics": metadata.get(
                    "topics",
                    []
                ),

                "rrf_score": rrf_score,

                "dense_rank": dense_rank,

                "bm25_rank": bm25_rank,

                "dense_score": (
                    dense_by_pmid[pmid]["score"]
                    if pmid in dense_by_pmid
                    else None
                ),

                "bm25_score": (
                    bm25_by_pmid[pmid]["score"]
                    if pmid in bm25_by_pmid
                    else None
                ),

                "best_chunk_id": (
                    dense_by_pmid[pmid]["chunk_id"]
                    if pmid in dense_by_pmid
                    else None
                )
            }
        )


    # --------------------------------------------------------
    # Sort by RRF score
    # --------------------------------------------------------

    hybrid_results.sort(
        key=lambda x: x["rrf_score"],
        reverse=True
    )


    return hybrid_results



# 12. Run one Hybrid query


def hybrid_search(
    query,
    dense_top_k=DENSE_TOP_K,
    bm25_top_k=BM25_TOP_K,
    final_top_k=FINAL_TOP_K
):

    print("\n" + "=" * 70)

    print(
        f"Query: {query}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    dense_results = dense_retrieval(
        query,
        top_k=dense_top_k
    )


    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25_results = bm25_retrieval(
        query,
        top_k=bm25_top_k
    )


    print(
        f"\nDense papers after PMID aggregation: "
        f"{len(dense_results)}"
    )

    print(
        f"BM25 papers: "
        f"{len(bm25_results)}"
    )


    # --------------------------------------------------------
    # RRF
    # --------------------------------------------------------

    hybrid_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        rrf_k=RRF_K
    )


    final_results = hybrid_results[
        :final_top_k
    ]



    # Print Dense top 5


    print(
        "\n--- Dense Top 5 ---"
    )

    for rank, item in enumerate(
        dense_results[:5],
        start=1
    ):

        print(
            f"{rank}. "
            f"PMID={item['pmid']} "
            f"Score={item['score']:.4f}"
        )

        print(
            f"   {item['title']}"
        )



    # Print BM25 top 5


    print(
        "\n--- BM25 Top 5 ---"
    )

    for rank, item in enumerate(
        bm25_results[:5],
        start=1
    ):

        print(
            f"{rank}. "
            f"PMID={item['pmid']} "
            f"Score={item['score']:.4f}"
        )

        print(
            f"   {item['title']}"
        )



    # Print Hybrid top 5


    print(
        "\n--- Hybrid RRF Top 5 ---"
    )

    for rank, item in enumerate(
        final_results,
        start=1
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            f"PMID: {item['pmid']}"
        )

        print(
            f"RRF Score: "
            f"{item['rrf_score']:.6f}"
        )

        print(
            f"Dense Rank: "
            f"{item['dense_rank']}"
        )

        print(
            f"BM25 Rank: "
            f"{item['bm25_rank']}"
        )

        print(
            f"Dense Score: "
            f"{item['dense_score']}"
        )

        print(
            f"BM25 Score: "
            f"{item['bm25_score']}"
        )

        print(
            f"Topics: "
            f"{item['topics']}"
        )

        print(
            f"Title: "
            f"{item['title']}"
        )


    return final_results



# 13. Test queries


test_queries = [

    "What genes are involved in maize salt stress response?",

    "How does salt stress affect photosynthesis in maize?",

    "How does alternative splicing contribute to salt stress responses?",

    "Which genes regulate sodium uptake under salt stress?",

    "How do HKT and SOS1 regulate sodium transport under salt stress?"
]



# 14. Run test queries


all_results = []


for query in test_queries:

    results = hybrid_search(
        query
    )

    all_results.append(
        {
            "query": query,
            "results": results
        }
    )



# 15. Save results


print("\n" + "=" * 70)
print("Saving Hybrid results...")
print("=" * 70)


with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_results,
        f,
        ensure_ascii=False,
        indent=2
    )


print(
    f"Saved results:\n{RESULTS_PATH}"
)



# 16. Interactive search


print("\n" + "=" * 70)
print("Interactive Hybrid Search")
print("=" * 70)

print(
    "输入问题进行 Dense + BM25 + RRF Hybrid 检索。"
)

print(
    "输入 exit 退出。"
)


while True:

    query = input(
        "\nQuestion: "
    ).strip()


    if query.lower() == "exit":

        print("Bye.")

        break


    if not query:

        continue


    results = hybrid_search(
        query
    )


    print(
        "\nFinal Hybrid Results:"
    )


    for rank, item in enumerate(
        results,
        start=1
    ):

        print(
            f"\n{rank}. "
            f"PMID={item['pmid']}"
        )

        print(
            f"RRF={item['rrf_score']:.6f} "
            f"| DenseRank={item['dense_rank']} "
            f"| BM25Rank={item['bm25_rank']}"
        )

        print(
            item["title"]
        )