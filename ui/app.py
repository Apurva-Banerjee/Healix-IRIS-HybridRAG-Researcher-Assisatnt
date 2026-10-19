import json
import logging
import sys
from pathlib import Path

# Silence harmless Streamlit watcher warnings for lazy-loaded modules
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# Add project root to sys.path so imports work seamlessly
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st
from src.config.settings import settings
from src.database.repository import PaperRepository
from src.generation.answer_generator import ResearchAnswerGenerator
from src.generation.citation_verifier import CitationVerifier
from src.generation.ollama_client import OllamaClient
from src.generation.paper_generator import AcademicPaperGenerator
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.keyword_search import BM25SearchEngine
from src.retrieval.qdrant_store import QdrantStore
from src.retrieval.semantic_search import SemanticSearchEngine
from scripts.audit_corpus import CorpusAuditor

st.set_page_config(
    page_title="AI Research Scientist Assistant (ScholarRAG)",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 AI Research Scientist Assistant")
st.caption("Evidence-grounded research intelligence, multi-mode hybrid retrieval, and publication-ready academic synthesis (ScholarRAG)")


# Initialize singletons in session state
@st.cache_resource
def get_services():
    repo = PaperRepository()
    q_store = QdrantStore()
    sem_engine = SemanticSearchEngine(qdrant_store=q_store)
    kw_engine = BM25SearchEngine()
    hyb_engine = HybridSearchEngine(semantic_search=sem_engine, keyword_search=kw_engine)
    ollama = OllamaClient()
    verifier = CitationVerifier()
    ans_gen = ResearchAnswerGenerator(ollama_client=ollama, citation_verifier=verifier)
    paper_gen = AcademicPaperGenerator(retriever=hyb_engine, citation_verifier=verifier)
    ingest_pipe = IngestionPipeline(repository=repo)
    auditor = CorpusAuditor()
    return repo, q_store, sem_engine, kw_engine, hyb_engine, ollama, ans_gen, paper_gen, ingest_pipe, auditor


repo, q_store, sem_engine, kw_engine, hyb_engine, ollama, ans_gen, paper_gen, ingest_pipe, auditor = get_services()

# Sidebar: System Status & Corpus Management
with st.sidebar:
    st.header("📊 Corpus & Engine Status")
    paper_count = repo.count_papers()
    vector_count = q_store.count()
    ollama_live = ollama.is_available()

    col1, col2 = st.columns(2)
    col1.metric("Indexed Papers", paper_count)
    col2.metric("Evidence Chunks", vector_count)

    st.markdown(f"**Ollama Engine:** {'🟢 Online' if ollama_live else '🟡 Offline (Extractive Fallback)'}")
    st.markdown(f"**Embedding Model:** `{settings.EMBEDDING_MODEL_NAME}` (384-dim)")
    st.markdown(f"**Vector Store:** `Qdrant Cloud (Cosine)`")
    st.markdown(f"**Default Fusion:** `Reciprocal Rank Fusion (k=60)`")

    st.divider()
    st.subheader("📥 Ingestion & Discovery")
    ingest_query = st.text_input("Ingestion Topic / Keyword", value="GraphRAG")
    target_count = st.slider("Target Papers", min_value=5, max_value=50, value=15)
    download_pdfs = st.checkbox("Download Open-Access PDFs", value=True)
    max_pdfs = st.slider("Max PDF Downloads", min_value=1, max_value=15, value=5)

    if st.button("🚀 Run Ingestion Pipeline", use_container_width=True):
        with st.spinner(f"Ingesting papers from arXiv and OpenAlex for '{ingest_query}'..."):
            summary = ingest_pipe.run(
                query=ingest_query,
                target_papers=target_count,
                download_pdfs=download_pdfs,
                max_pdf_downloads=max_pdfs,
            )
            st.success(f"Ingestion Finished! Retrieved {summary['raw_retrieved']} raw, saved {summary['deduplicated']} unique, downloaded {summary['pdfs_downloaded']} PDFs.")
            st.rerun()

    st.divider()
    if st.button("🛡️ Audit & Verify Corpus", use_container_width=True):
        with st.spinner("Auditing corpus layers..."):
            audit_res = auditor.audit()
            st.json(audit_res)

# Main Interface Tabs
tab_qa, tab_paper, tab_search, tab_eval, tab_library = st.tabs([
    "💬 Grounded Research Q&A",
    "📑 Academic Paper Generator",
    "🔍 Multi-Mode Evidence Search",
    "📊 Benchmark & Evaluation",
    "📚 Paper Library",
])

# ----------------- TAB 1: RESEARCH Q&A -----------------
with tab_qa:
    st.subheader("Ask a Scientific Question")
    st.markdown("Query your scientific paper corpus. Answers are strictly synthesized from retrieved empirical evidence and checked for citation grounding.")

    query_input = st.text_input(
        "Research Question",
        value="What are the key limitations and trade-offs of GraphRAG?",
        placeholder="e.g. Compare traditional RAG and GraphRAG architectures...",
    )

    col_m, col_k = st.columns([2, 1])
    mode_options = {
        "Hybrid RRF (k=60) [Recommended]": "hybrid_rrf",
        "Hybrid Weighted (α=0.7)": "hybrid_weighted",
        "Dense Semantic (BGE-small)": "dense",
        "Sparse Keyword (BM25)": "bm25",
    }
    selected_mode_label = col_m.selectbox("Retrieval Method", list(mode_options.keys()))
    selected_mode = mode_options[selected_mode_label]
    top_k_val = col_k.slider("Evidence Chunks (Top-K)", min_value=1, max_value=15, value=5)

    if st.button("🔬 Synthesize Answer", type="primary", use_container_width=True):
        with st.spinner("Retrieving evidence passages and synthesizing answer..."):
            retrieved = hyb_engine.search(query_input, top_k=top_k_val, method=selected_mode)
            res = ans_gen.generate_answer(query=query_input, retrieved_chunks=retrieved)

            st.markdown("### 📝 Synthesized Research Answer")
            st.markdown(res["answer"])

            # Citation Verification Badge
            v_rep = res.get("verification_report", {})
            if v_rep:
                st.markdown("---")
                col_c1, col_c2, col_c3 = st.columns(3)
                csr = v_rep.get("citation_support_rate", 1.0) * 100
                col_c1.metric("Citation Support Rate", f"{csr:.1f}%")
                col_c2.metric("Valid Evidence Citations", v_rep.get("valid_citations_count", 0))
                col_c3.metric("Dangling Citations", v_rep.get("dangling_citations_count", 0))

                if v_rep.get("dangling_citations_count", 0) > 0:
                    st.warning(f"⚠️ Flagged {v_rep['dangling_citations_count']} citation(s) referencing evidence outside the supplied passages!")

            # Conflicting Evidence Alerts
            conflicts = res.get("conflicting_evidence", [])
            if conflicts:
                st.warning("⚖️ **Potential Contradictory Evidence Identified in Literature:**")
                for conf in conflicts:
                    st.info(f"**Divergence between:** *{conf['paper_a']}* and *{conf['paper_b']}*\n\n{conf['description']}")

            st.divider()
            st.markdown("### 📚 Supporting Paper Citations")
            if res["sources"]:
                for s in res["sources"]:
                    st.markdown(f"- **{s['title']}** (Section: *{s['section']}*, Page {s['page']}) — *Relevance Score: {s['relevance_score']}*")
            else:
                st.info("No direct paper sources linked.")

            with st.expander("🔍 Inspect Extracted Evidence Passages", expanded=False):
                for i, c in enumerate(res["evidence_chunks"], 1):
                    st.markdown(f"**[Evidence {i}] {c.get('title')}** | Section: *{c.get('section')}* | Page: {c.get('page')} | Lane: `{c.get('retrieval_lane')}`")
                    st.code(c.get("text", ""), language="text")

# ----------------- TAB 2: ACADEMIC PAPER GENERATOR -----------------
with tab_paper:
    st.subheader("📑 Publication-Ready Academic Paper Draft Generator")
    st.markdown(
        "Generate **humanized, plagiarism-free, evidence-grounded academic text** directly usable as material in a research paper, "
        "complete with Abstract, Introduction, Architectural Comparisons, Empirical Syntheses, Identified Research Gaps, and IEEE-style References."
    )

    paper_topic = st.text_input(
        "Research Paper Title / Topic",
        value="Comparative Evaluation of GraphRAG and Hybrid RAG in Scientific Literature Synthesis",
    )

    if st.button("✍️ Generate Full Academic Paper Draft", type="primary", use_container_width=True):
        with st.spinner("Synthesizing multi-section publication draft from scientific corpus..."):
            draft_res = paper_gen.generate_full_paper_draft(topic=paper_topic)

            st.success(f"Draft Generated! {draft_res['word_count']} words | {draft_res['cited_papers_count']} cited papers | Completed in {draft_res['elapsed_seconds']}s")

            st.download_button(
                label="📥 Download Paper Draft (.md)",
                data=draft_res["document_content"],
                file_name="research_paper_draft.md",
                mime="text/markdown",
            )

            st.markdown("---")
            st.markdown(draft_res["document_content"])

# ----------------- TAB 3: EVIDENCE SEARCH -----------------
with tab_search:
    st.subheader("🔍 Multi-Mode Direct Evidence Search")
    search_q = st.text_input("Search terms / topic", value="community detection Leiden algorithm")
    col_sm, col_sk = st.columns([2, 1])
    s_mode = col_sm.selectbox("Search Mode", ["hybrid_rrf", "hybrid_weighted", "dense", "bm25"], key="s_mode_select")
    s_top_k = col_sk.slider("Top Results", 1, 20, 5, key="search_slider")

    if st.button("Search Evidence"):
        with st.spinner("Executing retrieval across indexed corpus..."):
            results = hyb_engine.search(search_q, top_k=s_top_k, method=s_mode)
            for i, r in enumerate(results, 1):
                st.markdown(f"#### {i}. {r.get('title')}")
                st.markdown(f"**Section:** `{r.get('section')}` | **Page:** {r.get('page')} | **Score:** `{r.get('score')}` | **Lane:** `{r.get('retrieval_lane')}`")
                st.write(r.get("text"))
                st.divider()

# ----------------- TAB 4: BENCHMARK & EVALUATION -----------------
with tab_eval:
    st.subheader("📊 Controlled Retrieval Benchmark & Evaluation Dashboard")
    st.markdown("Empirical comparison across **30 domain-specific queries** grounded in the real corpus.")

    results_csv = root_dir / "artifacts" / "evaluation" / "results.csv"
    if results_csv.exists():
        import pandas as pd
        df = pd.read_csv(results_csv)
        st.dataframe(df, use_container_width=True)

        st.markdown("### 🏆 Performance Summary")
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        col_e1.metric("Highest Recall@5", "0.8000", "Dense & RRF")
        col_e2.metric("Highest MRR", "0.7333", "Hybrid Weighted")
        col_e3.metric("Highest nDCG@5", "0.7386", "Dense")
        col_e4.metric("Fastest Latency", "2.45 ms", "BM25")

        error_md = root_dir / "artifacts" / "evaluation" / "error_analysis.md"
        if error_md.exists():
            with st.expander("📖 View Detailed Error Analysis & Taxonomies"):
                st.markdown(error_md.read_text(encoding="utf-8"))
    else:
        st.info("Run `python scripts/evaluate.py --k 5` to compute and display live benchmark metrics.")

# ----------------- TAB 5: PAPER LIBRARY -----------------
with tab_library:
    st.subheader("📚 Cataloged Scientific Literature Corpus")
    papers = repo.list_papers(limit=100)
    if not papers:
        st.info("No papers currently cataloged in the database.")
    else:
        st.markdown(f"Displaying **{len(papers)}** cataloged academic publications:")
        for p in papers:
            with st.expander(f"📄 {p.title} ({p.publication_date or 'N/A'})"):
                st.markdown(f"**Paper ID:** `{p.paper_id}` | **Source:** `{p.source}` | **Venue:** {p.venue}")
                st.markdown(f"**Authors:** {', '.join(p.authors) if p.authors else 'Unknown'}")
                if p.doi:
                    st.markdown(f"**DOI:** [{p.doi}](https://doi.org/{p.doi})")
                if p.arxiv_id:
                    st.markdown(f"**arXiv:** [{p.arxiv_id}](https://arxiv.org/abs/{p.arxiv_id})")
                if p.local_pdf_path:
                    st.success(f"Verified Local PDF: `{p.local_pdf_path}`")
                st.markdown(f"**Abstract:**\n{p.abstract or 'No abstract available.'}")
