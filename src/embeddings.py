
"""Generate multilingual embeddings and create a FAISS index."""

import json
import hashlib
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# File paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
INDEX_FILE = DATA_DIR / "rules.faiss"
METADATA_FILE = DATA_DIR / "index_metadata.json"

# Embedding configuration
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 16
TOKEN_OVERLAP = 24


def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "chunks.jsonl not found. Run ingestion.py first."
        )

    chunks = []
    seen = set()

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            chunk = json.loads(line)

            if "id" not in chunk or "search_text" not in chunk:
                raise ValueError("Invalid chunk structure")

            if chunk["id"] in seen:
                raise ValueError(f"Duplicate chunk: {chunk['id']}")

            seen.add(chunk["id"])
            chunks.append(chunk)

    if not chunks:
        raise ValueError("No chunks found")

    return chunks


def prepare_passages(chunks, model):
    """Split chunks into token-sized passages for the embedding model."""

    tokenizer = model.tokenizer
    max_tokens = model.max_seq_length
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)

    passages = []
    mapping = []

    for chunk in chunks:

        # Attach rule information to every passage
        prefix = f"Rule {chunk['rule']}: {chunk['title']}\n"

        prefix_tokens = tokenizer.encode(
            prefix, add_special_tokens=False
        )

        # Reserve space for actual rule text
        if len(prefix_tokens) > max_tokens - special_tokens - 32:
            prefix = f"Rule {chunk['rule']}\n"
            prefix_tokens = tokenizer.encode(
                prefix, add_special_tokens=False
            )

        budget = max_tokens - special_tokens - len(prefix_tokens)

        if budget <= 0:
            raise ValueError("Embedding model token limit too small")

        # The first two lines of search_text contain metadata
        body = chunk["search_text"].split("\n", 2)[-1]

        tokens = tokenizer.encode(
            body, add_special_tokens=False
        )

        if not tokens:
            continue

        overlap = min(TOKEN_OVERLAP, budget // 4)
        step = budget - overlap

        window = 0

        for start in range(0, len(tokens), step):

            end = min(start + budget, len(tokens))

            passage = tokenizer.decode(
                tokens[start:end],
                skip_special_tokens=True
            )

            passages.append(prefix + passage)

            mapping.append({
                "chunk_id": chunk["id"],
                "rule": chunk["rule"],
                "window": window
            })

            window += 1

            if end == len(tokens):
                break

    return passages, mapping


def generate_embeddings():
    """Create normalized embeddings and save the FAISS index."""

    print("Loading rule chunks...")
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Preparing token windows...")
    passages, mapping = prepare_passages(chunks, model)

    if not passages:
        raise ValueError("No passages were generated")

    print(f"Prepared {len(passages)} embedding passages")

    print("Generating embeddings...")

    embeddings = model.encode(
        passages,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.ascontiguousarray(
        embeddings, dtype=np.float32
    )

    # FAISS inner-product search on normalized vectors
    # corresponds to cosine similarity.
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    if index.ntotal != len(mapping):
        raise ValueError("Embedding and metadata count mismatch")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))

    # Preserve the relationship between vector positions and chunks
    metadata = {
        "model_name": MODEL_NAME,
        "dimension": dimension,
        "source_checksum": hashlib.sha256(
            CHUNKS_FILE.read_bytes()
        ).hexdigest(),
        "total_chunks": len(chunks),
        "total_passages": len(passages),
        "mapping": mapping
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("\nEmbedding generation completed!")
    print(f"Original chunks: {len(chunks)}")
    print(f"Indexed passages: {index.ntotal}")
    print(f"Embedding dimensions: {dimension}")
    print(f"FAISS index: {INDEX_FILE}")
    print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    generate_embeddings()
