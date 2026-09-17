from pathlib import Path
import json
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer



# 1. 路径


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

INDEX_PATH = Path("5.retrieval") / "index" / "pubmed.index"
CHUNKS_PATH = PROJECT_DIR / "5.retrieval" / "index" / "chunks.json"



# 2. 参数


MODEL_NAME = "all-MiniLM-L6-v2"

CANDIDATE_K = 20
TOP_PAPERS = 5

# Hybrid Retrieval 权重
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3



# 3. 停用词


STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "with",
    "how",
    "what",
    "which",
    "why",
    "does",
    "do",
    "under",
    "through",
    "from",
    "by",
    "their",
    "they",
    "them",
    "this",
    "that",
    "these",
    "those",
}



# 4. 文本分词


def tokenize(text):
    """
    把文本转换成简单的英文 token。

    例如：

    "How do plants regulate sodium uptake?"

    →

    ["plants", "regulate", "sodium", "uptake"]
    """

    words = re.findall(
        r"[A-Za-z0-9]+",
        text.lower()
    )

    words = [
        word
        for word in words
        if word not in STOPWORDS
    ]

    return words



# 5. Keyword Score


def keyword_score(question, text):
    """
    计算问题关键词在文献文本中的覆盖程度。

    简单版本：

    keyword score =
    匹配到的关键词数量 / 问题关键词总数
    """

    query_words = set(tokenize(question))

    if not query_words:
        return 0.0

    text_words = set(tokenize(text))

    matched_words = query_words.intersection(text_words)

    score = len(matched_words) / len(query_words)

    return score



# 6. 加载 FAISS


print("=" * 70)
print("BioAI Hybrid Retrieval")
print("=" * 70)

print("\nLoading FAISS index...")

if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"FAISS index not found:\n{INDEX_PATH}"
    )

index = faiss.read_index(
    str(INDEX_PATH)
)

print(
    f"FAISS vectors: {index.ntotal}"
)

print(
    f"Vector dimension: {index.d}"
)



# 7. 加载 chunks


print("\nLoading chunk metadata...")

if not CHUNKS_PATH.exists():
    raise FileNotFoundError(
        f"chunks.json not found:\n{CHUNKS_PATH}"
    )

with open(
    CHUNKS_PATH,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)

print(
    f"Loaded chunks: {len(chunks)}"
)



# 8. 加载 Embedding 模型


print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print(
    "Embedding model loaded."
)



# 9. Hybrid Search


def hybrid_search(question):

    print("\n" + "=" * 70)
    print("Question")
    print("=" * 70)

    print(question)

    # --------------------------------------------------------
    # Step 1
    # Semantic Retrieval
    # --------------------------------------------------------

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        CANDIDATE_K
    )

    # --------------------------------------------------------
    # Step 2
    # Candidate chunks
    # --------------------------------------------------------

    candidates = []

    for semantic_score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        chunk = chunks[int(idx)]

        text = chunk.get(
            "text",
            ""
        )

        title = chunk.get(
            "title",
            ""
        )

        # ----------------------------------------------------
        # Keyword score
        # ----------------------------------------------------

        combined_text = (
            title + " " + text
        )

        lexical_score = keyword_score(
            question,
            combined_text
        )

        # ----------------------------------------------------
        # Hybrid score
        # ----------------------------------------------------

        hybrid_score = (
            SEMANTIC_WEIGHT * semantic_score
            +
            KEYWORD_WEIGHT * lexical_score
        )

        candidates.append(
            {
                "pmid": chunk.get("pmid"),
                "title": title,
                "text": text,
                "chunk_index": chunk.get(
                    "chunk_index"
                ),
                "semantic_score": float(
                    semantic_score
                ),
                "keyword_score": float(
                    lexical_score
                ),
                "hybrid_score": float(
                    hybrid_score
                ),
            }
        )

    # --------------------------------------------------------
    # Step 3
    # PMID Deduplication
    # --------------------------------------------------------

    best_by_pmid = {}

    for candidate in candidates:

        pmid = candidate["pmid"]

        if not pmid:
            continue

        if pmid not in best_by_pmid:

            best_by_pmid[pmid] = candidate

        elif (
            candidate["hybrid_score"]
            >
            best_by_pmid[pmid]["hybrid_score"]
        ):

            best_by_pmid[pmid] = candidate

    # --------------------------------------------------------
    # Step 4
    # Hybrid ranking
    # --------------------------------------------------------

    ranked_papers = sorted(
        best_by_pmid.values(),
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Step 5
    # Top papers
    # --------------------------------------------------------

    ranked_papers = ranked_papers[
        :TOP_PAPERS
    ]

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\nFinal Retrieved Papers:")

    for rank, paper in enumerate(
        ranked_papers,
        start=1
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            f"PMID: {paper['pmid']}"
        )

        print(
            f"Hybrid score: "
            f"{paper['hybrid_score']:.4f}"
        )

        print(
            f"Semantic score: "
            f"{paper['semantic_score']:.4f}"
        )

        print(
            f"Keyword score: "
            f"{paper['keyword_score']:.4f}"
        )

        print(
            f"Title: {paper['title']}"
        )

    return ranked_papers



# 10. 测试问题


questions = [

    "How does alternative splicing contribute to salt stress responses?",

    "What genes are involved in maize salt stress response?",

    "How does salt stress affect photosynthesis in maize?",

    "How do plants regulate sodium uptake under salt stress?",
]



# 11. 执行


all_results = []

for question in questions:

    papers = hybrid_search(
        question
    )

    all_results.append(
        {
            "question": question,
            "results": papers
        }
    )



# 12. 保存结果


OUTPUT_PATH = BASE_DIR / "05-hybrid_search_results.json"

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


print("\n")
print("=" * 70)
print("Hybrid Retrieval completed.")
print("=" * 70)

print(
    f"\nResults saved to:\n{OUTPUT_PATH}"
)