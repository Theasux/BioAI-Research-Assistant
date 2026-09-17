from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer



# BioAI Dense Retrieval V2 Index Builder


print("=" * 70)
print("BioAI Dense Retrieval V2 Index Builder")
print("=" * 70)



# 1. Paths and configuration


# 当前脚本所在目录：
# D:\...\01-python\5.retrieval
BASE_DIR = Path(__file__).resolve().parent

# V2 PubMed corpus
CORPUS_PATH = BASE_DIR / "pubmed_corpus_v2.json"

# Dense Retrieval V2 输出目录
INDEX_DIR = BASE_DIR / "index_v2"

# Python / JSON 使用的路径
CHUNKS_PATH = INDEX_DIR / "chunks_v2.json"

# ------------------------------------------------------------
# IMPORTANT:
# FAISS 在 Windows 下对包含中文字符的绝对路径可能存在问题。
# 因此 FAISS 写入时使用项目根目录下的相对路径。
# ------------------------------------------------------------
FAISS_SAVE_PATH = r".\5.retrieval\index_v2\pubmed_v2.index"



# 2. Model and chunking parameters


MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100



# 3. Check corpus


print("\nLoading corpus...")

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

print(f"Papers loaded: {len(papers)}")


if len(papers) == 0:
    raise ValueError("Corpus is empty.")



# 4. Text chunking function


def chunk_text(
    text,
    chunk_size=500,
    overlap=100
):
    """
    Split text into overlapping chunks.

    Example:

    chunk 1:
        0 ~ 500

    chunk 2:
        400 ~ 900

    chunk 3:
        800 ~ 1300

    Therefore adjacent chunks share 100 characters.
    """

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # 已经到文本末尾
        if end >= len(text):
            break

        # 下一段向前重叠 100 个字符
        start = end - overlap

    return chunks



# 5. Build chunks


print("\nBuilding chunks...")

chunks = []

for paper in papers:

    pmid = str(
        paper.get("pmid", "")
    )

    title = paper.get(
        "title",
        ""
    ).strip()

    abstract = paper.get(
        "abstract",
        ""
    ).strip()

    topics = paper.get(
        "topics",
        []
    )

    # --------------------------------------------------------
    # Combine title and abstract
    # --------------------------------------------------------

    full_text = (
        f"Title: {title}\n"
        f"Abstract: {abstract}"
    )

    # --------------------------------------------------------
    # Split into chunks
    # --------------------------------------------------------

    paper_chunks = chunk_text(
        full_text,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    # --------------------------------------------------------
    # Save metadata for each chunk
    # --------------------------------------------------------

    for chunk_index, chunk in enumerate(
        paper_chunks
    ):

        chunks.append(
            {
                "chunk_id": len(chunks),
                "pmid": pmid,
                "title": title,
                "topics": topics,
                "chunk_index": chunk_index,
                "text": chunk
            }
        )


print(f"Chunks created: {len(chunks)}")


if len(chunks) == 0:
    raise ValueError(
        "No chunks were created from the corpus."
    )



# 6. Load Sentence Transformer


print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print(
    f"Embedding model: {MODEL_NAME}"
)



# 7. Generate embeddings


print("\nGenerating embeddings...")

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = model.encode(
    texts,

    # 每次处理 32 个文本
    batch_size=32,

    # 显示进度条
    show_progress_bar=True,

    # 返回 NumPy array
    convert_to_numpy=True,

    # L2 normalize
    normalize_embeddings=True
)


# ------------------------------------------------------------
# FAISS requires float32
# ------------------------------------------------------------

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)


print(
    f"Embedding shape: {embeddings.shape}"
)



# 8. Prepare output directory


print("\nPreparing output directory...")

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(
    f"Output directory ready:\n{INDEX_DIR}"
)



# 9. Build FAISS index


print("\nBuilding FAISS index...")

# embedding dimension
dimension = embeddings.shape[1]

# ------------------------------------------------------------
# IndexFlatIP:
#
# Flat = exact search
# IP   = Inner Product
#
# Because embeddings were L2-normalized,
# inner product == cosine similarity.
# ------------------------------------------------------------

index = faiss.IndexFlatIP(
    dimension
)

# Add all chunk embeddings
index.add(
    embeddings
)


print(
    f"FAISS vectors: {index.ntotal}"
)

print(
    f"Vector dimension: {index.d}"
)



# 10. Check FAISS and chunk count


if index.ntotal != len(chunks):

    raise RuntimeError(
        "Mismatch between FAISS vectors "
        "and chunk metadata!"
    )


print(
    "FAISS vectors and chunk metadata are consistent."
)



# 11. Save FAISS index


print("\nSaving FAISS index...")

# ------------------------------------------------------------
# IMPORTANT:
#
# Do NOT use:
#
#     str(INDEX_PATH)
#
# because that becomes an absolute Windows path containing
# Chinese characters.
#
# Use the relative path below instead.
# ------------------------------------------------------------

faiss.write_index(
    index,
    FAISS_SAVE_PATH
)


print(
    "Saved FAISS index:"
)

print(
    FAISS_SAVE_PATH
)



# 12. Save chunk metadata


print("\nSaving chunk metadata...")

with open(
    CHUNKS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        chunks,
        f,
        ensure_ascii=False,
        indent=2
    )


print(
    "Saved chunk metadata:"
)

print(
    CHUNKS_PATH
)



# 13. Verify output files


print("\nChecking output files...")

# FAISS path is relative to current working directory
faiss_output = Path(
    FAISS_SAVE_PATH
)

if not faiss_output.exists():

    raise RuntimeError(
        "FAISS index was not created!"
    )


if not CHUNKS_PATH.exists():

    raise RuntimeError(
        "chunks_v2.json was not created!"
    )


print(
    "FAISS index exists:"
)

print(
    faiss_output.resolve()
)


print(
    "Chunk metadata exists:"
)

print(
    CHUNKS_PATH.resolve()
)



# 14. Final summary


print("\n" + "=" * 70)
print("Verification")
print("=" * 70)

print(
    f"Papers       : {len(papers)}"
)

print(
    f"Chunks       : {len(chunks)}"
)

print(
    f"Vectors      : {index.ntotal}"
)

print(
    f"Dimension    : {index.d}"
)

print(
    f"Chunk size   : {CHUNK_SIZE}"
)

print(
    f"Chunk overlap: {CHUNK_OVERLAP}"
)

print(
    f"Model        : {MODEL_NAME}"
)



# 15. Show first 3 chunks


print("\nFirst 3 chunks:")

for chunk in chunks[:3]:

    print("-" * 70)

    print(
        f"Chunk ID: {chunk['chunk_id']}"
    )

    print(
        f"PMID: {chunk['pmid']}"
    )

    print(
        f"Title: {chunk['title']}"
    )

    print(
        f"Topics: {chunk['topics']}"
    )

    print(
        f"Chunk index: {chunk['chunk_index']}"
    )

    print(
        f"Text: {chunk['text'][:300]}..."
    )


print("\n" + "=" * 70)
print("Dense Retrieval V2 index build completed successfully.")
print("=" * 70)