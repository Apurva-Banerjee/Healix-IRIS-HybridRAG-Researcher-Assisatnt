import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.ingestion.ingestion_pipeline import IngestionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Ingest research papers from arXiv and OpenAlex.")
    parser.add_argument("--query", type=str, default="GraphRAG", help="Search topic or keyword")
    parser.add_argument("--target", type=int, default=20, help="Target number of papers")
    parser.add_argument("--download-pdfs", action="store_true", default=True, help="Download PDFs for papers")
    parser.add_argument("--no-pdfs", dest="download_pdfs", action="store_false")
    parser.add_argument("--max-pdfs", type=int, default=5, help="Maximum number of PDFs to download")
    args = parser.parse_args()

    pipeline = IngestionPipeline()
    summary = pipeline.run(
        query=args.query,
        target_papers=args.target,
        download_pdfs=args.download_pdfs,
        max_pdf_downloads=args.max_pdfs,
    )

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Query:               {summary['query']}")
    print(f"Raw Retrieved:       {summary['raw_retrieved']}")
    print(f"Deduplicated Unique: {summary['deduplicated']}")
    print(f"Stored in SQLite:    {summary['stored_in_db']}")
    print(f"PDFs Downloaded:     {summary['pdfs_downloaded']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
