#retrieval/pubmed_retrieval.py
import requests
import xml.etree.ElementTree as ET

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer



# 1. PubMed API：搜索论文


def search_pubmed(query, retmax=10):
    """
    根据关键词搜索 PubMed，返回 PMID 列表。
    """

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax
    }

    response = requests.get(url, params=params, timeout=30)

    response.raise_for_status()

    data = response.json()

    pmids = data["esearchresult"]["idlist"]

    return pmids



# 2. PubMed API：根据 PMID 获取论文


def fetch_pubmed_articles(pmids):
    """
    根据 PMID 列表获取论文的标题和摘要。
    """

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml"
    }

    response = requests.get(url, params=params, timeout=30)

    response.raise_for_status()

    root = ET.fromstring(response.text)

    papers = []

    for article in root.findall(".//PubmedArticle"):

        # ----------------------------------------------------
        # PMID
        # ----------------------------------------------------

        pmid_element = article.find(".//PMID")

        if pmid_element is None:
            continue

        pmid = pmid_element.text

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        title_element = article.find(".//ArticleTitle")

        if title_element is None:
            title = ""
        else:
            title = "".join(title_element.itertext())

        # ----------------------------------------------------
        # Abstract
        # ----------------------------------------------------

        abstract_parts = []

        for abstract_element in article.findall(".//AbstractText"):
            text = "".join(abstract_element.itertext())

            if text:
                label = abstract_element.attrib.get("Label")

                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)

        abstract = " ".join(abstract_parts)

        # ----------------------------------------------------
        # 如果没有摘要，就跳过
        # ----------------------------------------------------

        if not abstract:
            continue

        papers.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract
        })

    return papers



# 3. Chunking：把摘要切成多个文本片段


def chunk_text(text, chunk_size=500, overlap=100):
    """
    将文本按照字符数切分成多个 Chunk。

    chunk_size：
        每个 Chunk 的最大字符数。

    overlap：
        相邻 Chunk 之间重复的字符数。
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks



# 4. 构建 Chunk 数据集


def build_chunks(papers):
    """
    将每篇论文的 Abstract 转换成多个 Chunk。

    每个 Chunk 都保留：
        PMID
        Title
        Chunk text
    """

    all_chunks = []

    for paper in papers:

        chunks = chunk_text(
            paper["abstract"],
            chunk_size=500,
            overlap=100
        )

        for chunk in chunks:

            all_chunks.append({
                "pmid": paper["pmid"],
                "title": paper["title"],
                "text": chunk
            })

    return all_chunks



# 5. Embedding


def build_embeddings(chunks, model):
    """
    将所有 Chunk 转换为 Embedding。
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    return embeddings



# 6. 建立 FAISS Index


def build_faiss_index(embeddings):
    """
    使用 FAISS 建立向量索引。

    因为 Embedding 已经 L2 normalize，
    所以 Inner Product 就等价于 Cosine Similarity。
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index



# 7. Semantic Retrieval


def semantic_search(query, model, index, chunks, top_k=5):
    """
    根据 Query 检索最相关的 Chunk。
    """

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_id in zip(scores[0], indices[0]):

        chunk = chunks[index_id]

        results.append({
            "score": float(score),
            "pmid": chunk["pmid"],
            "title": chunk["title"],
            "text": chunk["text"]
        })

    return results



# 8. 主程序


def main():

    # --------------------------------------------------------
    # 用户输入 PubMed 搜索关键词
    # --------------------------------------------------------

    pubmed_query = input(
        "请输入 PubMed 搜索关键词："
    ).strip()

    if not pubmed_query:
        print("搜索关键词不能为空。")
        return

    # --------------------------------------------------------
    # PubMed 搜索
    # --------------------------------------------------------

    print("\n正在搜索 PubMed...")

    pmids = search_pubmed(
        pubmed_query,
        retmax=10
    )

    print(f"找到 {len(pmids)} 篇论文。")

    if not pmids:
        print("没有找到论文。")
        return

    # --------------------------------------------------------
    # 获取论文
    # --------------------------------------------------------

    print("正在获取论文内容...")

    papers = fetch_pubmed_articles(pmids)

    print(f"成功获取 {len(papers)} 篇有摘要的论文。")

    if not papers:
        print("没有获得可用论文。")
        return

    # --------------------------------------------------------
    # Chunking
    # --------------------------------------------------------

    print("正在进行 Chunking...")

    chunks = build_chunks(papers)

    print(f"生成 {len(chunks)} 个 Chunks。")

    # --------------------------------------------------------
    # 加载 Sentence Transformer
    # --------------------------------------------------------

    print("\n正在加载 Embedding Model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    print("正在生成 Embeddings...")

    embeddings = build_embeddings(
        chunks,
        model
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    print("正在建立 FAISS Index...")

    index = build_faiss_index(
        embeddings
    )

    print(
        f"FAISS 中共有 {index.ntotal} 个向量。"
    )

    # --------------------------------------------------------
    # 用户输入真正的问题
    # --------------------------------------------------------

    query = input(
        "\n请输入语义检索问题："
    ).strip()

    if not query:
        print("Query 不能为空。")
        return

    # --------------------------------------------------------
    # Semantic Retrieval
    # --------------------------------------------------------

    results = semantic_search(
        query=query,
        model=model,
        index=index,
        chunks=chunks,
        top_k=5
    )

    # --------------------------------------------------------
    # 输出结果
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("Semantic Retrieval Results")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):

        print(f"\nRank {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"PMID: {result['pmid']}")
        print(f"Title: {result['title']}")

        print("\nRelevant Chunk:")
        print(result["text"])

        print("-" * 70)


if __name__ == "__main__":
    main()