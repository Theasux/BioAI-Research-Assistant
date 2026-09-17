import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer



# 1. 文件路径


CORPUS_FILE = Path(
    "5.retrieval/pubmed_corpus.json"
)

INDEX_DIR = Path(
    "5.retrieval/index"
)

INDEX_FILE = INDEX_DIR / "pubmed.index"

CHUNKS_FILE = INDEX_DIR / "chunks.json"



# 2. Embedding Model


MODEL_NAME = "all-MiniLM-L6-v2"



# 3. Chunking 参数


CHUNK_SIZE = 500

CHUNK_OVERLAP = 100



# 4. 读取 Corpus


def load_corpus():

    with open(
        CORPUS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        papers = json.load(f)

    return papers



# 5. Chunking


def chunk_text(
    text,
    chunk_size=500,
    overlap=100
):
    """
    将一篇论文的文本切成多个有重叠的 chunks。
    """

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks



# 6. 为所有论文建立 Chunks


def build_chunks(papers):

    all_chunks = []

    for paper in papers:

        pmid = paper["pmid"]

        title = paper["title"]

        abstract = paper["abstract"]

        # ----------------------------------------------------
        # 将 Title 加入文本
        # ----------------------------------------------------

        full_text = (
            f"Title: {title}\n"
            f"Abstract: {abstract}"
        )

        # ----------------------------------------------------
        # Chunking
        # ----------------------------------------------------

        chunks = chunk_text(
            full_text,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )

        # ----------------------------------------------------
        # 保存每个 chunk 的元数据
        # ----------------------------------------------------

        for chunk_id, chunk in enumerate(chunks):

            all_chunks.append(
                {
                    "chunk_id": len(all_chunks),
                    "pmid": pmid,
                    "title": title,
                    "chunk_index": chunk_id,
                    "text": chunk,
                }
            )

    return all_chunks



# 7. 生成 Embeddings


def build_embeddings(chunks):

    print()
    print("Loading Embedding Model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Generating Embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    return embeddings



# 8. 建立 FAISS Index


def build_faiss_index(embeddings):

    print()
    print("Building FAISS Index...")

    dimension = embeddings.shape[1]

    # --------------------------------------------------------
    # IndexFlatIP:
    #
    # Flat = 精确搜索
    # IP   = Inner Product
    #
    # 因为 Embedding 已经 L2 normalize，
    # 所以 Inner Product = Cosine Similarity
    # --------------------------------------------------------

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    print(
        f"FAISS contains {index.ntotal} vectors."
    )

    return index



# 9. 保存 FAISS Index 和 Chunks


def save_index(
    index,
    chunks
):

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 保存 FAISS
    # --------------------------------------------------------

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    # --------------------------------------------------------
    # 保存 chunk 元数据
    # --------------------------------------------------------

    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("Index saved:")
    print(INDEX_FILE)

    print()
    print("Chunk metadata saved:")
    print(CHUNKS_FILE)



# 10. 主程序


def main():

    print("=" * 70)
    print("Building Literature Vector Index")
    print("=" * 70)


    # --------------------------------------------------------
    # Step 1
    # 读取论文 Corpus
    # --------------------------------------------------------

    print()
    print("Loading PubMed corpus...")

    papers = load_corpus()

    print(
        f"Loaded {len(papers)} papers."
    )


    # --------------------------------------------------------
    # Step 2
    # Chunking
    # --------------------------------------------------------

    print()
    print("Chunking papers...")

    chunks = build_chunks(
        papers
    )

    print(
        f"Generated {len(chunks)} chunks."
    )


    # --------------------------------------------------------
    # Step 3
    # Embedding
    # --------------------------------------------------------

    embeddings = build_embeddings(
        chunks
    )


    # --------------------------------------------------------
    # Step 4
    # FAISS
    # --------------------------------------------------------

    index = build_faiss_index(
        embeddings
    )


    # --------------------------------------------------------
    # Step 5
    # 保存
    # --------------------------------------------------------

    save_index(
        index,
        chunks
    )


    # --------------------------------------------------------
    # Step 6
    # 检查
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Index Construction Finished")
    print("=" * 70)

    print(
        f"Papers : {len(papers)}"
    )

    print(
        f"Chunks : {len(chunks)}"
    )

    print(
        f"Vectors: {index.ntotal}"
    )

    print(
        f"Dimension: {embeddings.shape[1]}"
    )



# 11. 程序入口


if __name__ == "__main__":
    main()