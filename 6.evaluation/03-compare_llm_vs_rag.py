from pathlib import Path
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI



# 1. 路径配置


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

INDEX_PATH = Path("5.retrieval") / "index" / "pubmed.index"
CHUNKS_PATH = PROJECT_DIR / "5.retrieval" / "index" / "chunks.json"



# 2. 实验参数


CANDIDATE_K = 20
TOP_PAPERS = 5

MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "deepseek-chat"



# 3. 测试问题


QUESTIONS = [
    "How does alternative splicing contribute to salt stress responses?",
    "What genes are involved in maize salt stress response?",
    "How does salt stress affect photosynthesis in maize?",
    "How do plants regulate sodium uptake under salt stress?",
]



# 4. 加载 FAISS


print("=" * 70)
print("BioAI Evaluation: LLM-only vs RAG")
print("=" * 70)

print("\nLoading FAISS index...")

if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"FAISS index not found:\n{INDEX_PATH}"
    )

index = faiss.read_index(str(INDEX_PATH))

print(f"FAISS vectors: {index.ntotal}")
print(f"Vector dimension: {index.d}")



# 5. 加载 chunks metadata


print("\nLoading chunk metadata...")

if not CHUNKS_PATH.exists():
    raise FileNotFoundError(
        f"chunks.json not found:\n{CHUNKS_PATH}"
    )

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded chunks: {len(chunks)}")



# 6. 加载 Embedding 模型


print("\nLoading embedding model...")

embedding_model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")



# 7. 初始化 DeepSeek


api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set."
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

print("DeepSeek API key detected.")



# 8. LLM-only


def ask_llm_only(question):
    """
    直接把用户问题交给 LLM。
    不提供任何检索到的文献证据。
    """

    system_prompt = """
You are a scientific literature assistant.

请使用中文回答用户的问题。

这是一个 LLM-only baseline。

你没有获得外部检索到的文献证据，
因此不要假装自己进行了文献检索。

如果无法确定某个具体事实，请明确说明不确定性。

回答应该简洁、清晰，并尽量避免没有依据的具体结论。
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.choices[0].message.content



# 9. RAG retrieval


def retrieve_papers(question):
    """
    Question
        ↓
    embedding
        ↓
    FAISS top K chunks
        ↓
    PMID deduplication
        ↓
    Top N unique papers
    """

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        CANDIDATE_K
    )

    best_chunk_by_pmid = {}

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        chunk = chunks[int(idx)]

        pmid = chunk.get("pmid")

        if not pmid:
            continue

        candidate = {
            "pmid": pmid,
            "title": chunk.get("title", ""),
            "score": float(score),
            "text": chunk.get("text", ""),
            "chunk_index": chunk.get("chunk_index")
        }

        if pmid not in best_chunk_by_pmid:

            best_chunk_by_pmid[pmid] = candidate

        elif score > best_chunk_by_pmid[pmid]["score"]:

            best_chunk_by_pmid[pmid] = candidate

    unique_papers = sorted(
        best_chunk_by_pmid.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return unique_papers[:TOP_PAPERS]



# 10. RAG answer


def ask_rag(question, retrieved_papers):

    evidence_blocks = []

    for i, paper in enumerate(retrieved_papers, start=1):

        block = f"""
[Paper {i}]
PMID: {paper["pmid"]}
Title: {paper["title"]}
Retrieval score: {paper["score"]:.4f}

Evidence:
{paper["text"]}
"""

        evidence_blocks.append(block)

    evidence = "\n".join(evidence_blocks)

    system_prompt = """
You are a scientific literature assistant.

请使用中文回答用户的问题。

你只能根据下面提供的 Retrieved Literature Evidence 回答。

重要规则：

1. 不得把模型自身记忆中的知识当作检索证据。
2. 如果证据不足，请明确说“现有检索证据不足”。
3. 不得编造论文、PMID、基因名称或实验结果。
4. 当某个结论来自某篇论文时，在结论后标注 PMID。
5. 可以保留论文标题的英文原文。
6. 回答应该简洁、清晰、科学。

Retrieved Literature Evidence:
"""

    user_prompt = f"""
User question:

{question}

Literature evidence:

{evidence}

请基于上述证据回答问题。
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    return answer



# 11. 运行实验


results = []

for question_id, question in enumerate(QUESTIONS, start=1):

    print("\n")
    print("=" * 70)
    print(f"Question {question_id}")
    print("=" * 70)

    print(f"\n{question}")

    # --------------------------------------------------------
    # LLM-only
    # --------------------------------------------------------

    print("\n[1] Running LLM-only...")

    llm_answer = ask_llm_only(question)

    print("\nLLM-only answer:")
    print("-" * 70)
    print(llm_answer)

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    print("\n[2] Running RAG retrieval...")

    retrieved_papers = retrieve_papers(question)

    print("\nRetrieved papers:")

    for rank, paper in enumerate(retrieved_papers, start=1):

        print(
            f"{rank}. "
            f"PMID={paper['pmid']} "
            f"Score={paper['score']:.4f}"
        )

        print(
            f"   {paper['title']}"
        )

    print("\n[3] Generating RAG answer...")

    rag_answer = ask_rag(
        question,
        retrieved_papers
    )

    print("\nRAG answer:")
    print("-" * 70)
    print(rag_answer)

    # --------------------------------------------------------
    # 保存结果
    # --------------------------------------------------------

    results.append(
        {
            "question_id": question_id,
            "question": question,
            "llm_only_answer": llm_answer,
            "rag_answer": rag_answer,
            "retrieved_papers": [
                {
                    "pmid": paper["pmid"],
                    "title": paper["title"],
                    "score": paper["score"],
                    "chunk_index": paper["chunk_index"]
                }
                for paper in retrieved_papers
            ]
        }
    )



# 12. 保存 JSON


OUTPUT_PATH = BASE_DIR / "03-llm_vs_rag_results.json"

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )



# 13. 完成


print("\n")
print("=" * 70)
print("Evaluation completed.")
print("=" * 70)

print(f"\nResults saved to:")
print(OUTPUT_PATH)