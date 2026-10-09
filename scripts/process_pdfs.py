import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config.settings import settings
from src.database.repository import PaperRepository
from src.pdf_processing.parser import PDFParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("process_pdfs")


def main():
    parser = argparse.ArgumentParser(description="Parse downloaded PDFs into clean text and structured JSON sections.")
    parser.add_argument("--pdf-dir", type=str, default=None, help="Directory containing PDFs")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else settings.RAW_PDF_DIR
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}. Run `python scripts/ingest.py` first.")
        return

    repo = PaperRepository()
    parser_engine = PDFParser()
    processed_count = 0

    print(f"Found {len(pdf_files)} PDFs to process...")
    for pdf_path in pdf_files:
        try:
            # Try to match paper in DB
            paper = None
            for p in repo.list_papers(limit=200):
                if p.local_pdf_path and Path(p.local_pdf_path).name == pdf_path.name:
                    paper = p
                    break

            paper_id = paper.paper_id if paper else pdf_path.stem
            title = paper.title if paper else pdf_path.stem

            doc_data = parser_engine.parse_pdf(str(pdf_path), paper_id=paper_id, title=title)
            processed_count += 1
            print(f"  [OK] {paper_id}: {doc_data['total_pages']} pages, {len(doc_data['sections'])} sections")
        except Exception as e:
            print(f"  [ERROR] Failed processing {pdf_path.name}: {e}")

    print("\n" + "=" * 60)
    print(f"PDF PROCESSING COMPLETE: {processed_count}/{len(pdf_files)} PDFs converted.")
    print("=" * 60)


if __name__ == "__main__":
    main()
