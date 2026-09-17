import os
import json
import faiss
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ============================================================
# 1. Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

INDEX_PATH = Path("5.retrieval") / "index" / "pubmed.index"
CHUNKS_PATH = PROJECT_DIR / "5.retrieval" / "index" / "chunks.json"


# ============================================================
# 2. Load FAISS index
# ============================================================

print("=" * 70)
print("BioAI RAG V1")
print("=" * 70)

print("\nLoading FAISS index...")

index = faiss.read_index(str(INDEX_PATH))

print("FAISS vectors:", index.ntotal)
print("Vector dimension:", index.d)


# ============================================================
# 3. Load chunk metadata
# ============================================================

print("\nLoading chunk metadata...")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Loaded chunks:", len(chunks))


# ============================================================
# 4. Load embedding model
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 5. Initialize LLM client
# ============================================================

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. "
        "Please set it in PowerShell first."
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


# ============================================================
# 6. User question
# ============================================================

question = input(
    "\nEnter your question:\n> "
)


# ============================================================
# 7. Convert question into embedding
# ============================================================

query_embedding = embedding_model.encode(
    [question],
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")


# ============================================================
# 8. Retrieve top chunks
# ============================================================

TOP_K = 5

scores, indices = index.search(
    query_embedding,
    TOP_K
)


# ============================================================
# 9. Build context
# ============================================================

context_parts = []

seen_pmids = set()

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    chunk = chunks[int(idx)]

    pmid = str(chunk["pmid"])
    title = chunk["title"]
    text = chunk["text"]

    context_parts.append(
        f"""
[Evidence {rank}]
PMID: {pmid}
Title: {title}
Similarity score: {score:.4f}

{text}
"""
    )

    seen_pmids.add(pmid)


context = "\n".join(context_parts)


# ============================================================
# 10. Build RAG prompt
# ============================================================

system_prompt = """
You are a scientific literature assistant.

Answer the user's question using ONLY the evidence
provided in the context.

Do not introduce unsupported facts.

If the evidence is insufficient to answer the question,
clearly say that the retrieved evidence is insufficient.

When making a claim based on a paper, cite its PMID
in the form [PMID: XXXXXXXX].

Give a concise but scientifically informative answer.
"""


user_prompt = f"""
Question:
{question}

Retrieved literature evidence:

{context}

Please answer the question based only on the retrieved evidence.
"""


# ============================================================
# 11. Call LLM
# ============================================================

print("\nGenerating answer...\n")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ],
    temperature=0.1
)


answer = response.choices[0].message.content


# ============================================================
# 12. Display result
# ============================================================

print("=" * 70)
print("Answer")
print("=" * 70)

print(answer)

print("\n" + "=" * 70)
print("Retrieved Papers")
print("=" * 70)

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    chunk = chunks[int(idx)]

    print(
        f"{rank}. "
        f"PMID={chunk['pmid']} "
        f"Score={score:.4f}"
    )

    print(
        f"   {chunk['title']}"
    )