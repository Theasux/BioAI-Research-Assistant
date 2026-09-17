import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


# ============================================================
# 1. 路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CORPUS_PATH = BASE_DIR / "pubmed_corpus_v2.json"


# ============================================================
# 2. 配置
# ============================================================

TOP_K = 5


# ============================================================
# 3. 文本分词
# ============================================================

def tokenize(text):
    """
    将文本转换成 BM25 使用的 token 列表。

    这里使用一个简单的正则分词器：
    - 英文字母
    - 数字
    - 带有 + / - 的简单生物学符号
    """

    text = text.lower()

    tokens = re.findall(
        r"[a-zA-Z]+(?:[+-]?\d*)?|\d+",
        text
    )

    return tokens


# ============================================================
# 4. 加载 Corpus
# ============================================================

print("=" * 70)
print("BioAI BM25 Retrieval")
print("=" * 70)

print()
print("Loading corpus...")

with open(
    CORPUS_PATH,
    "r",
    encoding="utf-8"
) as f:

    papers = json.load(f)


print(
    f"Papers loaded: {len(papers)}"
)


# ============================================================
# 5. 构建 BM25 文档
# ============================================================

documents = []

valid_papers = []

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
        title
        + " "
        + abstract
    ).strip()

    if not text:
        continue

    tokens = tokenize(text)

    if not tokens:
        continue

    documents.append(tokens)

    valid_papers.append(paper)


print(
    f"Documents indexed: {len(documents)}"
)


# ============================================================
# 6. 建立 BM25 Index
# ============================================================

print()
print("Building BM25 index...")

bm25 = BM25Okapi(
    documents
)

print("BM25 index ready.")


# ============================================================
# 7. 查询函数
# ============================================================

def search(query, top_k=5):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(
        query_tokens
    )

    # 从高到低排序
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    seen_pmids = set()

    for index in ranked_indices:

        paper = valid_papers[index]

        pmid = paper["pmid"]

        # PMID 去重
        if pmid in seen_pmids:
            continue

        seen_pmids.add(pmid)

        results.append(
            {
                "pmid": pmid,
                "score": float(
                    scores[index]
                ),
                "title": paper["title"],
                "topics": paper.get(
                    "topics",
                    []
                ),
            }
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# 8. 测试问题
# ============================================================

questions = [
    "What genes are involved in maize salt stress response?",

    "How does salt stress affect photosynthesis in maize?",

    "How does alternative splicing contribute to salt stress responses?",

    "Which genes regulate sodium uptake under salt stress?",
]


# ============================================================
# 9. 执行检索
# ============================================================

for question in questions:

    print()
    print("=" * 70)

    print(
        f"Query: {question}"
    )

    print("=" * 70)

    results = search(
        question,
        TOP_K
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print()
        print(
            f"Rank {rank}"
        )

        print(
            f"PMID: {result['pmid']}"
        )

        print(
            f"BM25 Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Topics: "
            f"{', '.join(result['topics'])}"
        )

        print(
            f"Title: "
            f"{result['title']}"
        )


# ============================================================
# 10. 交互式查询
# ============================================================

print()
print("=" * 70)
print("Interactive Search")
print("=" * 70)

print(
    "输入问题进行 BM25 检索。"
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

    results = search(
        query,
        TOP_K
    )

    print()

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. "
            f"PMID={result['pmid']} "
            f"Score={result['score']:.4f}"
        )

        print(
            f"   {result['title']}"
        )