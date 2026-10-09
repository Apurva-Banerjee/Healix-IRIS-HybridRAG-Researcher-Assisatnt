import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.chunking.chunk_schema import ChunkRecord
from src.config.settings import settings
from src.embeddings.embedder import Embedder
from src.retrieval.qdrant_store import QdrantStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for chunks and index them into Qdrant.")
    parser.add_argument("--chunks-dir", type=str, default=None, help="Directory containing chunk JSON files")
    parser.add_argument("--collection", type=str, default=settings.QDRANT_COLLECTION, help="Qdrant collection name")
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir) if args.chunks_dir else settings.PROCESSED_CHUNKS_DIR
    chunk_files = list(chunks_dir.glob("*_chunks.json"))

    if not chunk_files:
        print(f"No chunk files found in {chunks_dir}. Run `python scripts/create_chunks.py` first.")
        return

    all_chunks = []
    for cf in chunk_files:
        try:
            data = json.loads(cf.read_text(encoding="utf-8"))
            for item in data:
                all_chunks.append(ChunkRecord(**item))
        except Exception as e:
            print(f"Failed loading {cf.name}: {e}")

    print(f"Loaded {len(all_chunks)} chunks from {len(chunk_files)} files.")

    # Initialize Embedder & QdrantStore
    embedder = Embedder()
    dim = embedder.dimension
    qdrant = QdrantStore(collection_name=args.collection, dimension=dim)

    print(f"Embedding model dimension: {dim}")
    embeddings = embedder.embed_chunks(all_chunks)

    print(f"Upserting {len(embeddings)} vectors into Qdrant collection '{args.collection}'...")
    total_upserted = qdrant.upsert_chunks(all_chunks, embeddings)

    print("\n" + "=" * 60)
    print(f"INDEXING COMPLETE: {total_upserted} chunks indexed in Qdrant.")
    print(f"Total vectors in collection: {qdrant.count()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
