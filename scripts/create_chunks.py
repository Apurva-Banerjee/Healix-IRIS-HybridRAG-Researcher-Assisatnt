import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.chunking.chunker import SectionAwareChunker
from src.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Create section-aware chunks from processed JSON documents.")
    parser.add_argument("--json-dir", type=str, default=None, help="Directory containing parsed paper JSON files")
    parser.add_argument("--chunk-size", type=int, default=settings.CHUNK_SIZE_TOKENS, help="Chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=settings.CHUNK_OVERLAP_TOKENS, help="Chunk overlap in tokens")
    args = parser.parse_args()

    json_dir = Path(args.json_dir) if args.json_dir else settings.PROCESSED_JSON_DIR
    json_files = list(json_dir.glob("*.json"))

    if not json_files:
        print(f"No parsed JSON documents found in {json_dir}. Run `python scripts/process_pdfs.py` first.")
        return

    chunker = SectionAwareChunker(chunk_size_tokens=args.chunk_size, chunk_overlap_tokens=args.chunk_overlap)
    total_chunks = 0

    print(f"Processing {len(json_files)} structured documents into chunks...")
    for jf in json_files:
        try:
            doc_data = json.loads(jf.read_text(encoding="utf-8"))
            chunks = chunker.chunk_paper_document(doc_data)
            total_chunks += len(chunks)
            print(f"  [OK] {doc_data.get('paper_id')}: {len(chunks)} chunks created.")
        except Exception as e:
            print(f"  [ERROR] Chunking failed for {jf.name}: {e}")

    print("\n" + "=" * 60)
    print(f"CHUNKING COMPLETE: {total_chunks} total chunks generated across {len(json_files)} documents.")
    print("=" * 60)


if __name__ == "__main__":
    main()
