import sys
import json
from pathlib import Path

import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL_PATH

BATCH_SIZE = 16  # chunks embedded per batch; lower to 8 or 4 on small GPUs/CPU-only
CHROMA_ADD_BATCH = 500  # Chroma has a per-call size limit independent of embedding batching


def load_chunks(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.resolve()} not found. Point this script at the JSON file "
            f"chunking.py produced (pass its path as the first argument), "
            f"e.g. `python ingest.py ./all_chunks.json`."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def make_chunk_id(chunk: dict, index: int) -> str:
    """Build a stable, unique id per chunk from its metadata."""
    meta = chunk["metadata"]
    source = meta.get("source", "unknown")
    page = meta.get("page", "na")
    chunk_index = meta.get("chunk_index", index)
    return f"{source}::p{page}::{chunk_index}::{index}"


def sanitize_metadata(meta: dict) -> dict:
    """Chroma metadata values must be str, int, float, or bool (no None)."""
    return {k: (v if v is not None else "") for k, v in meta.items()}


def embed_and_store(chunks_path: Path):
    print(f"Loading model: {EMBED_MODEL_PATH} ...")
    # Qwen3 embedding models need trust_remote_code=True to load their
    # custom modeling code from HuggingFace, and are trained with an
    # instruction prefix for queries (not for the documents/passages being
    # indexed). We only embed passages here, plain; query-time embedding
    # adds prompt_name="query" (see nodes.py / retrieval_only.py).
    model = SentenceTransformer(EMBED_MODEL_PATH, trust_remote_code=True)

    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    texts = [c["text"] for c in chunks]
    ids = [make_chunk_id(c, i) for i, c in enumerate(chunks)]
    metadatas = [sanitize_metadata(c["metadata"]) for c in chunks]

    print("Embedding chunks ...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"Connecting to Chroma at {Path(CHROMA_DIR).resolve()} ...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # hnsw:space="cosine" is what makes `1 - distance` a valid similarity
    # in nodes.py / retrieval_only.py -- don't drop this if you edit ingestion.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Writing {len(chunks)} records to collection '{COLLECTION_NAME}' ...")
    for start in tqdm(range(0, len(chunks), CHROMA_ADD_BATCH)):
        end = start + CHROMA_ADD_BATCH
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} records.")


if __name__ == "__main__":
    chunks_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./all_chunks.json")
    embed_and_store(chunks_path)
