import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer



# 1. Project paths


# 当前脚本所在目录
BASE_DIR = Path(__file__).resolve().parent

# 当前项目根目录：01-python
PROJECT_DIR = BASE_DIR.parent

# ------------------------------------------------------------
# IMPORTANT:
# FAISS 在 Windows 下对包含中文字符的绝对路径可能存在兼容性问题。
# 因此 FAISS index 使用相对于项目根目录的路径。
# ------------------------------------------------------------

INDEX_PATH = Path("5.retrieval") / "index" / "pubmed.index"

# chunks.json 使用 Python 读取，可以正常处理 Unicode 路径。
CHUNKS_PATH = PROJECT_DIR / "5.retrieval" / "index" / "chunks.json"

# evaluation questions
QUESTIONS_PATH = BASE_DIR / "01-retrieval_questions.json"



# 2. Configuration


MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5



# 3. Print information


print("=" * 70)
print("Retrieval Evaluation")
print("=" * 70)

print("\nCurrent working directory:")
print(Path.cwd())

print("\nFAISS index path:")
print(INDEX_PATH)

print("\nFAISS index absolute path:")
print((Path.cwd() / INDEX_PATH).resolve())

print("\nChunk metadata:")
print(CHUNKS_PATH)

print("\nEvaluation questions:")
print(QUESTIONS_PATH)



# 4. Check files


print("\nChecking files...")

if not (Path.cwd() / INDEX_PATH).exists():
    raise FileNotFoundError(
        f"FAISS index not found:\n"
        f"{(Path.cwd() / INDEX_PATH).resolve()}"
    )

if not CHUNKS_PATH.exists():
    raise FileNotFoundError(
        f"Chunk metadata not found:\n{CHUNKS_PATH}"
    )

if not QUESTIONS_PATH.exists():
    raise FileNotFoundError(
        f"Evaluation questions not found:\n{QUESTIONS_PATH}"
    )

print("All required files exist.")



# 5. Load FAISS index


print("\nLoading FAISS index...")

# IMPORTANT:
# 这里传给 FAISS 的是相对路径，不包含中文目录。
index = faiss.read_index(str(INDEX_PATH))

print(f"FAISS vectors: {index.ntotal}")
print(f"Vector dimension: {index.d}")



# 6. Load chunk metadata


print("\nLoading chunk metadata...")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded chunks: {len(chunks)}")



# 7. Load evaluation questions


print("\nLoading evaluation questions...")

with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)

print(f"Evaluation questions: {len(questions)}")



# 8. Check consistency


if index.ntotal != len(chunks):
    raise ValueError(
        f"Index/vector number ({index.ntotal}) "
        f"does not match chunk number ({len(chunks)})"
    )

print("Index and metadata are consistent.")



# 9. Load embedding model


print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")



# 10. Evaluate one question


def evaluate_question(question_data):

    question = question_data["question"]

    relevant_pmids = set(
        str(pmid)
        for pmid in question_data["relevant_pmids"]
    )

    # --------------------------------------------------------
    # Convert question to embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
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
        TOP_K
    )

    # --------------------------------------------------------
    # Convert chunks to paper-level PMIDs
    # --------------------------------------------------------

    retrieved_pmids = []

    for idx in indices[0]:

        if idx == -1:
            continue

        pmid = str(chunks[idx]["pmid"])

        if pmid not in retrieved_pmids:
            retrieved_pmids.append(pmid)

    # --------------------------------------------------------
    # Find relevant retrieved papers
    # --------------------------------------------------------

    retrieved_relevant = (
        set(retrieved_pmids) & relevant_pmids
    )

    # --------------------------------------------------------
    # Recall@K
    # --------------------------------------------------------

    recall_at_k = (
        len(retrieved_relevant)
        / len(relevant_pmids)
    )

    # --------------------------------------------------------
    # Precision@K
    # --------------------------------------------------------

    precision_at_k = (
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

        if pmid in relevant_pmids:

            reciprocal_rank = 1.0 / rank
            break

    return {
        "question": question,
        "relevant_pmids": sorted(relevant_pmids),
        "retrieved_pmids": retrieved_pmids,
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "reciprocal_rank": reciprocal_rank
    }



# 11. Run evaluation


results = []

for i, question_data in enumerate(
    questions,
    start=1
):

    result = evaluate_question(
        question_data
    )

    results.append(result)

    print("\n" + "-" * 70)
    print(f"Question {i}")
    print("-" * 70)

    print(
        f"Question: "
        f"{result['question']}"
    )

    print(
        f"Recall@{TOP_K}: "
        f"{result['recall_at_k']:.3f}"
    )

    print(
        f"Precision@{TOP_K}: "
        f"{result['precision_at_k']:.3f}"
    )

    print(
        f"Reciprocal Rank: "
        f"{result['reciprocal_rank']:.3f}"
    )

    print(
        f"Retrieved PMIDs: "
        f"{result['retrieved_pmids']}"
    )



# 12. Calculate overall metrics


mean_recall = np.mean(
    [
        result["recall_at_k"]
        for result in results
    ]
)

mean_precision = np.mean(
    [
        result["precision_at_k"]
        for result in results
    ]
)

mrr = np.mean(
    [
        result["reciprocal_rank"]
        for result in results
    ]
)



# 13. Final results


print("\n" + "=" * 70)
print("Overall Evaluation Results")
print("=" * 70)

print(
    f"Recall@{TOP_K}: "
    f"{mean_recall:.3f}"
)

print(
    f"Precision@{TOP_K}: "
    f"{mean_precision:.3f}"
)

print(
    f"MRR: "
    f"{mrr:.3f}"
)

print("=" * 70)