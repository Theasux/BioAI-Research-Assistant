import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Configuration
# ============================================================

INDEX_PATH = "index/pubmed.index"
CHUNKS_PATH = "index/chunks.json"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5


# ============================================================
# 2. Load FAISS index
# ============================================================

print("=" * 70)
print("Loading Literature Retrieval System")
print("=" * 70)

print("\nLoading FAISS index...")

index = faiss.read_index(INDEX_PATH)

print(f"FAISS vectors: {index.ntotal}")
print(f"Vector dimension: {index.d}")


# ============================================================
# 3. Load chunk metadata
# ============================================================

print("\nLoading chunk metadata...")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded chunks: {len(chunks)}")


# ============================================================
# 4. Check index and metadata consistency
# ============================================================

if index.ntotal != len(chunks):
    raise ValueError(
        f"Index/vector number ({index.ntotal}) "
        f"does not match chunk number ({len(chunks)})"
    )

print("Index and metadata are consistent.")


# ============================================================
# 5. Load embedding model
# ============================================================

print("\nLoading Embedding Model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# ============================================================
# 6. Search function
# ============================================================

def search_literature(query, top_k=5):

    # --------------------------------------------------------
    # Convert user query into embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # FAISS expects float32
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
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        # FAISS may return -1 if there are not enough results
        if idx == -1:
            continue

        chunk = chunks[idx]

        print(f"\nRank {rank}")
        print("-" * 70)

        print(f"Score: {score:.4f}")
        print(f"PMID: {chunk['pmid']}")
        print(f"Title: {chunk['title']}")
        print(f"Chunk index: {chunk['chunk_index']}")

        print("\nText:")
        print(chunk["text"])


# ============================================================
# 7. Interactive search
# ============================================================

while True:

    query = input(
        "\nEnter your question "
        "(type 'quit' to exit): "
    ).strip()

    if query.lower() == "quit":
        print("\nExiting...")
        break

    if not query:
        print("Please enter a question.")
        continue

    search_literature(
        query,
        top_k=TOP_K
    )