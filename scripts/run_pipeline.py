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
from src.chunking.chunker import SectionAwareChunker
from src.config.settings import settings
from src.database.repository import PaperRepository
from src.embeddings.embedder import Embedder
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.pdf_processing.parser import PDFParser
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.keyword_search import BM25SearchEngine
from src.retrieval.qdrant_store import QdrantStore
from src.retrieval.semantic_search import SemanticSearchEngine
from src.generation.answer_generator import ResearchAnswerGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Run complete vertical slice: Ingestion -> PDF Parsing -> Chunking -> Indexing -> Evaluation Query.")
    parser.add_argument("--query", type=str, default="GraphRAG", help="Search topic to ingest")
    parser.add_argument("--target", type=int, default=15, help="Number of papers to retrieve")
    parser.add_argument("--max-pdfs", type=int, default=3, help="Max PDFs to download and parse")
    parser.add_argument("--test-question", type=str, default="What are the key limitations and trade-offs of GraphRAG?", help="Test research question to answer after indexing")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("STEP 1: INGESTION (arXiv + OpenAlex -> Deduplication -> Metadata DB)")
    print("=" * 70)
    pipeline = IngestionPipeline()
    summary = pipeline.run(
        query=args.query,
        target_papers=args.target,
        download_pdfs=True,
        max_pdf_downloads=args.max_pdfs,
    )
    print(f"Retrieved: {summary['raw_retrieved']} | Unique: {summary['deduplicated']} | PDFs: {summary['pdfs_downloaded']}")

    print("\n" + "=" * 70)
    print("STEP 2: PDF PARSING (PyMuPDF -> Dehyphenation/Cleaning -> Section Extraction)")
    print("=" * 70)
    repo = PaperRepository()
    pdf_parser = PDFParser()
    parsed_docs = []

    pdf_files = list(settings.RAW_PDF_DIR.glob("*.pdf"))
    for pdf_path in pdf_files:
        try:
            doc_data = pdf_parser.parse_pdf(str(pdf_path))
            parsed_docs.append(doc_data)
            print(f"  Parsed: {doc_data['paper_id']} ({doc_data['total_pages']} pages, {len(doc_data['sections'])} sections)")
        except Exception as e:
            logger.error("Error parsing %s: %s", pdf_path.name, str(e))

    # Also generate document records from abstracts for papers without full PDFs so the corpus is rich
    for p in summary["papers"]:
        if not p.local_pdf_path and p.abstract:
            pseudo_doc = {
                "paper_id": p.paper_id,
                "title": p.title,
                "sections": [{"name": "Abstract", "page_start": 1, "text": p.abstract}],
            }
            parsed_docs.append(pseudo_doc)

    print("\n" + "=" * 70)
    print("STEP 3: SECTION-AWARE CHUNKING")
    print("=" * 70)
    chunker = SectionAwareChunker()
    all_chunks = []
    for doc in parsed_docs:
        chunks = chunker.chunk_paper_document(doc)
        all_chunks.extend(chunks)
    print(f"Total Chunks Generated: {len(all_chunks)}")

    print("\n" + "=" * 70)
    print("STEP 4: DENSE VECTOR EMBEDDINGS & QDRANT INDEXING")
    print("=" * 70)
    embedder = Embedder()
    dim = embedder.dimension
    qdrant = QdrantStore(dimension=dim)
    embeddings = embedder.embed_chunks(all_chunks)
    qdrant.upsert_chunks(all_chunks, embeddings)
    print(f"Total Vectors in Qdrant: {qdrant.count()}")

    print("\n" + "=" * 70)
    print("STEP 5: HYBRID RETRIEVAL & SYNTHESIS DEMO")
    print("=" * 70)
    sem_engine = SemanticSearchEngine(embedder=embedder, qdrant_store=qdrant)
    kw_engine = BM25SearchEngine(chunks=all_chunks)
    hybrid_engine = HybridSearchEngine(semantic_search=sem_engine, keyword_search=kw_engine)

    print(f"Question: {args.test_question}\n")
    retrieved_evidence = hybrid_engine.search(args.test_question, top_k=3)
    print("Top Retrieved Evidence Passages:")
    for i, ev in enumerate(retrieved_evidence, 1):
        print(f"  [{i}] {ev.get('title')} (Section: {ev.get('section')}, Score: {ev.get('score')}):")
        passage_snippet = ev.get('text', '')[:200].replace('\n', ' ')
        print(f"      \"{passage_snippet}...\"\n")

    ans_gen = ResearchAnswerGenerator()
    rag_result = ans_gen.generate_answer(query=args.test_question, retrieved_chunks=retrieved_evidence)

    print("\n" + "=" * 70)
    print("STEP 6: GENERATED GROUNDED ANSWER")
    print("=" * 70)
    print(rag_result["answer"])
    print("\nCited Sources:")
    for s in rag_result["sources"]:
        print(f"  - {s['title']} ({s['section']}, Page {s['page']})")
    print("=" * 70)


if __name__ == "__main__":
    main()
