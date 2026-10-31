
import json
import re
import hashlib
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
INDEX_FILE = DATA_DIR / "rules.faiss"
METADATA_FILE = DATA_DIR / "index_metadata.json"


class RuleRetriever:

    def __init__(self):
        """Load the existing FAISS index, metadata and embedding model."""

        for path in [CHUNKS_FILE, INDEX_FILE, METADATA_FILE]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing file: {path}\n"
                    "Run ingestion.py and embeddings.py first."
                )

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(INDEX_FILE))

        with open(METADATA_FILE, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        # Check whether the index matches the current chunks
        current_checksum = hashlib.sha256(
            CHUNKS_FILE.read_bytes()
        ).hexdigest()

        if current_checksum != self.metadata["source_checksum"]:
            raise ValueError(
                "The chunks have changed since embedding generation. "
                "Run embeddings.py again."
            )

        # Load original chunks for accurate source citations
        self.chunks = {}

        with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    chunk = json.loads(line)
                    self.chunks[chunk["id"]] = chunk

        # FAISS vectors are linked to original chunks using mapping
        self.mapping = self.metadata["mapping"]

        if self.index.ntotal != len(self.mapping):
            raise ValueError("FAISS index and metadata do not match")

        if self.index.d != self.metadata["dimension"]:
            raise ValueError("Incorrect embedding dimensions")

        for item in self.mapping:
            if item["chunk_id"] not in self.chunks:
                raise ValueError(
                    f"Missing source chunk: {item['chunk_id']}"
                )

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            self.metadata["model_name"]
        )

        if self.model.get_sentence_embedding_dimension() != self.index.d:
            raise ValueError("Embedding model does not match FAISS index")

        print(f"Retriever ready: {self.index.ntotal} vectors loaded.")

    def search(self, question, top_k=5):
        """Return the most relevant original rule chunks."""

        question = question.strip()

        if not question:
            return []

        if top_k < 1:
            raise ValueError("top_k must be positive")

        # Convert the question into the same vector space
        query_vector = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_vector = np.ascontiguousarray(
            query_vector, dtype=np.float32
        )

        # Detect explicit references such as Rule 56 or नियम 56
        requested_rules = {
            match.upper()
            for match in re.findall(
                r"(?:\brule\s*|नियम\s*)(\d+(?:-[A-Za-z])?)",
                question,
                flags=re.IGNORECASE
            )
        }

        # Search more vectors than required because several vectors
        # may belong to the same original chunk.
        candidates = min(
            self.index.ntotal,
            max(100, top_k * 25)
        )

        if requested_rules:
            candidates = self.index.ntotal

        scores, indices = self.index.search(
            query_vector, candidates
        )

        best_chunks = {}

        for score, vector_id in zip(scores[0], indices[0]):

            if vector_id < 0:
                continue

            record = self.mapping[int(vector_id)]
            chunk_id = record["chunk_id"]

            # Retain the highest-scoring window per source chunk
            if (
                chunk_id not in best_chunks
                or score > best_chunks[chunk_id]["similarity"]
            ):
                best_chunks[chunk_id] = {
                    "similarity": float(score),
                    "window": record["window"]
                }

        ranked_results = []

        for chunk_id, match in best_chunks.items():

            chunk = self.chunks[chunk_id]

            # Prioritize explicitly named rules
            rule_bonus = (
                0.35
                if chunk["rule"].upper() in requested_rules
                else 0.0
            )

            ranked_results.append({
                "chunk_id": chunk_id,
                "rule": chunk["rule"],
                "title": chunk["title"],
                "part": chunk["part"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "version": chunk["version"],
                "source": chunk["source"],
                "text": chunk["text"],
                "similarity": match["similarity"],
                "rank_score": match["similarity"] + rule_bonus,
                "window": match["window"]
            })

        ranked_results.sort(
            key=lambda result: result["rank_score"],
            reverse=True
        )

        # Prevent one large rule from dominating every result
        selected = []
        rule_counts = {}

        max_per_rule = top_k if requested_rules else 2

        for result in ranked_results:

            rule = result["rule"]

            if rule_counts.get(rule, 0) >= max_per_rule:
                continue

            selected.append(result)
            rule_counts[rule] = rule_counts.get(rule, 0) + 1

            if len(selected) == top_k:
                break

        return selected


def main():
    """Interactive terminal for testing retrieval."""

    retriever = RuleRetriever()

    print("\nBhumi Vikas Rule Retriever")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("Enter your question: ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        results = retriever.search(question, top_k=5)

        if not results:
            print("No results found.\n")
            continue

        print("\nMost relevant rule sections:\n")

        for i, result in enumerate(results, start=1):

            print("=" * 65)

            print(f"Result {i}")
            print(f"Rule: {result['rule']}")
            print(f"Title: {result['title']}")
            print(
                f"PDF page range: "
                f"{result['page_start']}–{result['page_end']}"
            )
            print(
                f"Similarity: {result['similarity']:.4f}"
            )
            print(f"Source version: {result['version']}")

            print("\nSource text:")
            print(result["text"][:1000])

            if len(result["text"]) > 1000:
                print("\n[Additional source text omitted from preview]")

            print()

        print("=" * 65)
        print()


if __name__ == "__main__":
    main()
