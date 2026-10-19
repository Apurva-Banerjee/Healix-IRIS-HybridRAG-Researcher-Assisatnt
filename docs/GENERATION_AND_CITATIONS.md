# 🖋️ Evidence-Grounded Generation & Citation Verification Protocol

**Document ID:** `docs/GENERATION_AND_CITATIONS.md`  
**System Module:** `src/generation/`  
**Components:** `ResearchAnswerGenerator`, `AcademicPaperGenerator`, `CitationVerifier`, `PromptBuilder`  

---

## 1. Principles of Academic Grounding

A fundamental vulnerability of naive LLM generation in academic domains is *hallucinated attribution*: language models synthesize plausible claims and invent fictitious authors, DOIs, or attach real paper names to claims the cited paper never made.

ScholarRAG enforces a strict five-point academic contract:
1. **Closed Evidence Context:** The generator is presented exclusively with retrieved passages identified by deterministic identifiers (`[Evidence 1]`, `[Evidence 2]`, ...).
2. **Forbidden Fabrication:** The model is prohibited from citing any source not present in the supplied evidence bundle.
3. **Explicit Insufficiency:** If the retrieved literature does not contain adequate evidence to answer the research question, the system outputs an explicit declaration of insufficiency rather than extrapolating.
4. **Empirical Conflict Highlighting:** When literature contains opposing findings (e.g. reported accuracy gains vs. contradictory benchmarks), the system surfaces the divergence rather than silently smoothing it over.
5. **Deterministic Citation Verification:** An independent verification pass parses all citations, calculates the **Citation Support Rate (CSR)**, and flags any dangling references.

---

## 2. Citation Support Rate (CSR) Formulation

$$\operatorname{CSR} = \frac{|\mathcal{C}_{\text{valid}}|}{|\mathcal{C}_{\text{total}}|}$$

Where:
- $\mathcal{C}_{\text{total}}$ is the set of all citation indices extracted from the generated answer ($i \in \{1, \dots, N\}$).
- $\mathcal{C}_{\text{valid}}$ is the subset of indices that strictly map to a supplied evidence unit ($1 \le i \le K_{\text{evidence}}$).
- If $|\mathcal{C}_{\text{total}}| = 0$ and no claims were made, $\operatorname{CSR} = 1.0$. If dangling citations ($i > K_{\text{evidence}}$) appear, $\operatorname{CSR} < 1.0$ and `is_fully_grounded = False`.

---

## 3. Humanized & Plagiarism-Free Academic Prose

The user explicitly required that:
> *"make sure the content generated is plag free and humazied so as to directly use as a material in actual research paper"*

To achieve this:
1. **Conceptual Synthesis over Verbatim Copying:**
   - Instead of copy-pasting lengthy contiguous sentences from retrieved PDFs (which triggers plagiarism detectors), the `AcademicPaperGenerator` extracts core findings, empirical numbers, and parameters, and re-expresses them in original, scholarly academic prose.
2. **Scholarly Voice & Style:**
   - Avoids repetitive conversational LLM boilerplate (*"In today's fast-paced world"*, *"It is important to remember"*, *"Delve into the tapestry"*).
   - Uses publication-grade academic syntax, clear passive and active voice balance, precise methodological verbs (*formalize*, *delineate*, *empirically corroborate*, *substantiate*), and structured LaTeX math notation.
3. **Traceable Attribution:**
   - Formal IEEE-style bibliographic entries linking each paper to its canonical DOI or arXiv identifier.

---

## 4. Extractive Scientific Synthesis Fallback

When the local Ollama inference daemon is offline or experiencing heavy GPU/CPU latency, ScholarRAG does not crash or emit an uninformative error. Instead, it activates the `_generate_extractive_synthesis` pipeline:
- Extracts empirical parameters, algorithms, and key sentences directly from the top-ranked section-aware chunks.
- Formats them into structured sections: *Core Findings*, *Technical Methodology*, *Limitations*, and *Grounded Sources*.
- Attributes each bullet point to its exact `[Evidence X]` chunk and section.
