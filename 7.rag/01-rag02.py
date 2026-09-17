import os
import json
import faiss
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer
from openai import OpenAI



# 1. Project paths


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# FAISS 使用相对路径
# 避免 Windows 中文路径导致 FAISS FileIO 问题
INDEX_PATH = Path("5.retrieval") / "index" / "pubmed.index"

# chunks.json 使用绝对路径
CHUNKS_PATH = (
    PROJECT_DIR
    / "5.retrieval"
    / "index"
    / "chunks.json"
)



# 2. Retrieval parameters


# 第一步：
# FAISS 先找更多候选 chunk
CANDIDATE_K = 20

# 第二步：
# PMID 去重以后，最终只保留多少篇论文
TOP_PAPERS = 5



# 3. Basic information


print("=" * 70)
print("BioAI RAG V1.5")
print("=" * 70)

print("\nCurrent working directory:")
print(Path.cwd())

print("\nFAISS index:")
print(INDEX_PATH)

print("\nChunk metadata:")
print(CHUNKS_PATH)

print("\nRetrieval parameters:")
print("Candidate chunks:", CANDIDATE_K)
print("Final unique papers:", TOP_PAPERS)



# 4. Check files


print("\n" + "=" * 70)
print("Checking required files")
print("=" * 70)

index_absolute_path = PROJECT_DIR / INDEX_PATH

if not index_absolute_path.exists():
    raise FileNotFoundError(
        f"FAISS index not found:\n{index_absolute_path}"
    )

if not CHUNKS_PATH.exists():
    raise FileNotFoundError(
        f"Chunk metadata not found:\n{CHUNKS_PATH}"
    )

print("All required files exist.")



# 5. Load FAISS index


print("\nLoading FAISS index...")

index = faiss.read_index(
    str(INDEX_PATH)
)

print("FAISS vectors:", index.ntotal)
print("Vector dimension:", index.d)



# 6. Load chunk metadata


print("\nLoading chunk metadata...")

with open(
    CHUNKS_PATH,
    "r",
    encoding="utf-8"
) as f:
    chunks = json.load(f)

print("Loaded chunks:", len(chunks))



# 7. Check consistency


if index.ntotal != len(chunks):
    raise RuntimeError(
        "FAISS index and chunk metadata are inconsistent.\n"
        f"FAISS vectors: {index.ntotal}\n"
        f"Chunks: {len(chunks)}"
    )

print("Index and metadata are consistent.")



# 8. Load embedding model


print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")



# 9. Load DeepSeek API key


print("\nChecking DeepSeek API key...")

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set."
    )

print("DeepSeek API key detected.")



# 10. Initialize LLM client


client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)



# 11. Get user question


print("\n" + "=" * 70)
print("Question")
print("=" * 70)

question = input(
    "Enter your question:\n> "
).strip()

if not question:
    raise ValueError(
        "Question cannot be empty."
    )



# 12. Embed question


print("\nEmbedding question...")

query_embedding = embedding_model.encode(
    [question],
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")

print(
    "Query embedding shape:",
    query_embedding.shape
)



# 13. FAISS candidate retrieval


print(
    f"\nSearching FAISS for top {CANDIDATE_K} chunks..."
)

scores, indices = index.search(
    query_embedding,
    CANDIDATE_K
)



# 14. PMID-based deduplication


print("\n" + "=" * 70)
print("PMID Deduplication")
print("=" * 70)

# 使用字典：
#
# PMID
#   ↓
# 最相关的 chunk
#
# 如果同一篇论文出现多个 chunk，
# 只保留 similarity score 最高的那个 chunk。

best_chunk_by_pmid = {}

for score, idx in zip(
    scores[0],
    indices[0]
):

    idx = int(idx)

    if idx < 0:
        continue

    chunk = chunks[idx]

    pmid = str(
        chunk["pmid"]
    )

    score = float(score)

    # 第一次出现
    if pmid not in best_chunk_by_pmid:

        best_chunk_by_pmid[pmid] = {
            "score": score,
            "pmid": pmid,
            "title": chunk["title"],
            "text": chunk["text"],
            "chunk_index": chunk.get(
                "chunk_index",
                None
            )
        }

    # 如果同一个 PMID 后面出现了更高分的 chunk
    elif score > best_chunk_by_pmid[pmid]["score"]:

        best_chunk_by_pmid[pmid] = {
            "score": score,
            "pmid": pmid,
            "title": chunk["title"],
            "text": chunk["text"],
            "chunk_index": chunk.get(
                "chunk_index",
                None
            )
        }


print(
    "Candidate chunks retrieved:",
    len(indices[0])
)

print(
    "Unique papers after PMID deduplication:",
    len(best_chunk_by_pmid)
)



# 15. Sort unique papers by similarity


unique_papers = sorted(
    best_chunk_by_pmid.values(),
    key=lambda x: x["score"],
    reverse=True
)



# 16. Keep Top-K unique papers


retrieved_papers = unique_papers[
    :TOP_PAPERS
]



# 17. Display retrieval results


print("\n" + "=" * 70)
print("Final Retrieved Papers")
print("=" * 70)

for rank, paper in enumerate(
    retrieved_papers,
    start=1
):

    print(
        f"{rank}. "
        f"PMID={paper['pmid']} "
        f"Score={paper['score']:.4f}"
    )

    print(
        f"   {paper['title']}"
    )

    print(
        f"   Chunk index: "
        f"{paper['chunk_index']}"
    )

    print()



# 18. Build evidence context


context_parts = []

for rank, paper in enumerate(
    retrieved_papers,
    start=1
):

    context_parts.append(
        f"""
[Evidence {rank}]
PMID: {paper['pmid']}
Title: {paper['title']}
Similarity score: {paper['score']:.4f}

Relevant text:
{paper['text']}
"""
    )

context = "\n".join(
    context_parts
)



# 19. System prompt


system_prompt = """
You are a scientific literature assistant specializing in
life sciences and biomedical research.

请使用中文回答用户的问题。

你只能根据提供的检索证据回答问题。

不得引入检索证据中没有支持的事实。

不要凭空补充：
- 实验结果
- 基因功能
- 分子机制
- 统计结果
- 文献结论

如果提供的证据不足以回答问题，
请明确说明：

“现有检索证据不足以支持这一结论。”

不要根据自己的知识进行推测或编造答案。

回答中的科学术语应尽量使用规范的中文表达。

对于重要专业术语，可以同时保留英文。

当某个科学结论来自某篇论文时，
请在相应结论后引用 PMID。

PMID 格式：

[PMID: XXXXXXXX]

论文标题可以保留英文原文，
以方便用户进一步核对原始文献。

请将答案组织成清晰的结构。

回答应该：
1. 使用中文；
2. 基于提供的证据；
3. 引用对应 PMID；
4. 明确区分证据和推测；
5. 避免没有文献支持的过度解释。
"""



# 20. User prompt


user_prompt = f"""
用户问题：

{question}


检索到的文献证据：

{context}


请严格根据以上文献证据回答用户问题。

要求：

1. 使用中文回答。
2. 每个重要科学结论尽可能注明 PMID。
3. 不要引入证据中没有支持的信息。
4. 如果证据不足，请明确说明。
5. 不要编造具体实验结果或分子机制。
6. 可以综合多篇论文，但必须能够从提供的证据中找到支持。
"""



# 21. Call DeepSeek


print("\n" + "=" * 70)
print("Generating answer...")
print("=" * 70)

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



# 22. Get final answer


answer = response.choices[0].message.content



# 23. Display answer


print("\n" + "=" * 70)
print("Answer")
print("=" * 70)

print(answer)



# 24. Display evidence


print("\n" + "=" * 70)
print("Retrieved Evidence")
print("=" * 70)

for rank, paper in enumerate(
    retrieved_papers,
    start=1
):

    print(
        f"{rank}. "
        f"PMID={paper['pmid']} "
        f"Score={paper['score']:.4f}"
    )

    print(
        f"   {paper['title']}"
    )

    print()