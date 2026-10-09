import sys
import logging
from pathlib import Path

# Silence harmless Streamlit watcher warnings for lazy-loaded transformer modules
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# Add project root to sys.path so imports work seamlessly
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
from src.config.settings import settings
from src.database.repository import PaperRepository
from src.generation.answer_generator import ResearchAnswerGenerator
from src.generation.ollama_client import OllamaClient
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.keyword_search import BM25SearchEngine
from src.retrieval.qdrant_store import QdrantStore
from src.retrieval.semantic_search import SemanticSearchEngine

st.set_page_config(
    page_title="AI Research Scientist Assistant",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 AI Research Scientist Assistant")
st.caption("Evidence-grounded research intelligence, literature retrieval, and scientific synthesis")

# Initialize singletons in session state
@st.cache_resource
def get_services():
    repo = PaperRepository()
    q_store = QdrantStore()
    sem_engine = SemanticSearchEngine(qdrant_store=q_store)
    kw_engine = BM25SearchEngine()
    hyb_engine = HybridSearchEngine(semantic_search=sem_engine, keyword_search=kw_engine)
    ollama = OllamaClient()
    ans_gen = ResearchAnswerGenerator(ollama_client=ollama)
    ingest_pipe = IngestionPipeline(repository=repo)
    return repo, q_store, sem_engine, kw_engine, hyb_engine, ollama, ans_gen, ingest_pipe

repo, q_store, sem_engine, kw_engine, hyb_engine, ollama, ans_gen, ingest_pipe = get_services()

# Sidebar: System Status & Corpus Management
with st.sidebar:
    st.header("📊 Corpus & Engine Status")
    paper_count = repo.count_papers()
    vector_count = q_store.count()
    ollama_live = ollama.is_available()

    col1, col2 = st.columns(2)
    col1.metric("Indexed Papers", paper_count)
    col2.metric("Evidence Chunks", vector_count)

    st.markdown(f"**Ollama Engine:** {'🟢 Online' if ollama_live else '🟡 Offline (Structured Fallback)'}")
    st.markdown(f"**Embedding Model:** `{settings.EMBEDDING_MODEL_NAME}`")
    st.markdown(f"**Vector Store:** `Qdrant (Cosine)`")

    st.divider()
    st.subheader("📥 Ingestion & Discovery")
    ingest_query = st.text_input("Ingestion Topic / Keyword", value="GraphRAG")
    target_count = st.slider("Target Papers", min_value=5, max_value=50, value=15)
    download_pdfs = st.checkbox("Download Open-Access PDFs", value=True)
    max_pdfs = st.slider("Max PDF Downloads", min_value=1, max_value=15, value=5)

    if st.button(" Run Ingestion Pipeline", use_container_width=True):
        with st.spinner(f"Ingesting papers from arXiv and OpenAlex for '{ingest_query}'..."):
            summary = ingest_pipe.run(
                query=ingest_query,
                target_papers=target_count,
                download_pdfs=download_pdfs,
                max_pdf_downloads=max_pdfs,
            )
            st.success(f"Ingestion Finished! Retrieved {summary['raw_retrieved']} raw, saved {summary['deduplicated']} unique, downloaded {summary['pdfs_downloaded']} PDFs.")
            st.rerun()

# Main Interface Tabs
tab_qa, tab_search, tab_library = st.tabs([" Grounded Research Q&A", " Evidence Retrieval", " Paper Corpus"])

with tab_qa:
    st.subheader("Ask a Scientific Question")
    st.markdown("Query your scientific paper corpus. Answers are strictly synthesized from retrieved empirical evidence.")

    query_input = st.text_input(
        "Research Question",
        value="What are the key limitations and trade-offs of GraphRAG?",
        placeholder="e.g. Compare traditional RAG and GraphRAG architectures...",
    )

    col_m, col_k = st.columns([2, 1])
    retrieval_mode = col_m.selectbox("Retrieval Method", ["Hybrid (Dense + BM25)", "Semantic Vector (Qdrant)", "Keyword (BM25)"])
    top_k_val = col_k.slider("Evidence Chunks (Top-K)", min_value=1, max_value=10, value=5)

    if st.button(" Synthesize Answer", type="primary", use_container_width=True):
        with st.spinner("Retrieving evidence passages and synthesizing answer..."):
            if "Hybrid" in retrieval_mode:
                retrieved = hyb_engine.search(query_input, top_k=top_k_val)
            elif "Semantic" in retrieval_mode:
                retrieved = sem_engine.search(query_input, top_k=top_k_val)
            else:
                retrieved = kw_engine.search(query_input, top_k=top_k_val)

            res = ans_gen.generate_answer(query=query_input, retrieved_chunks=retrieved)

            st.markdown("###  Synthesized Research Answer")
            st.markdown(res["answer"])

            st.divider()
            st.markdown("###  Supporting Paper Citations")
            if res["sources"]:
                for s in res["sources"]:
                    st.markdown(f"- **{s['title']}** (Section: *{s['section']}*, Page {s['page']}) — *Relevance Score: {s['relevance_score']}*")
            else:
                st.info("No direct paper sources linked.")

            with st.expander(" Inspect Extracted Evidence Passages", expanded=False):
                for i, c in enumerate(res["evidence_chunks"], 1):
                    st.markdown(f"**[Evidence {i}] {c.get('title')}** | Section: *{c.get('section')}* | Page: {c.get('page')}")
                    st.code(c.get("text", ""), language="text")

with tab_search:
    st.subheader("Direct Evidence Passage Search")
    search_q = st.text_input("Search terms / topic", value="community detection Leiden algorithm")
    s_top_k = st.slider("Top Results", 1, 20, 5, key="search_slider")

    if st.button("Search Evidence"):
        with st.spinner("Searching..."):
            results = hyb_engine.search(search_q, top_k=s_top_k)
            for i, r in enumerate(results, 1):
                st.markdown(f"#### {i}. {r.get('title')}")
                st.markdown(f"**Section:** {r.get('section')} | **Page:** {r.get('page')} | **Score:** `{r.get('score')}`")
                st.write(r.get("text"))
                st.divider()

with tab_library:
    st.subheader("Cataloged Research Papers")
    papers = repo.list_papers(limit=50)
    if not papers:
        st.info("No papers currently cataloged in the database. Run ingestion from the sidebar to populate your research corpus.")
    else:
        for p in papers:
            with st.expander(f"📄 {p.title} ({p.publication_date or 'N/A'})"):
                st.markdown(f"**Authors:** {', '.join(p.authors) if p.authors else 'Unknown'}")
                st.markdown(f"**Source:** `{p.source}` | **Venue:** {p.venue} | **Citations:** {p.citation_count}")
                if p.doi:
                    st.markdown(f"**DOI:** [{p.doi}](https://doi.org/{p.doi})")
                if p.arxiv_id:
                    st.markdown(f"**arXiv:** [{p.arxiv_id}](https://arxiv.org/abs/{p.arxiv_id})")
                if p.local_pdf_path:
                    st.success(f"PDF Downloaded: {p.local_pdf_path}")
                st.markdown(f"**Abstract:**\n{p.abstract or 'No abstract available.'}")
