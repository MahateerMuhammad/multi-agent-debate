"""Expanded 50-query RAG benchmark dataset with 10 corpus documents and Dev/Unseen Test splits."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.retrieval.ingestion import DocumentIngestor
from app.retrieval.schemas import Document
from app.retrieval.vectorstore import QdrantVectorStore


class RAGBenchmarkQuery(BaseModel):
    """Single benchmark evaluation query with ground truth annotations."""

    id: str = Field(..., description="Query ID (e.g. Q-01)")
    query: str = Field(..., description="The user or agent query string")
    category: str = Field(..., description="Category (easy, semantic, exact, distractor, etc.)")
    expected_doc_ids: set[str] = Field(
        ..., description="Expected ground-truth document IDs for RAG recall"
    )
    is_negative: bool = Field(
        default=False, description="True if answer is deliberately absent from corpus"
    )


RAW_DOCUMENTS_DATA = [
    {
        "doc_id": "CSET-2026-01",
        "title": "CSET AI Policy Whitepaper 2026: Transparency & Security in Open Weights",
        "content": (
            "Open-source and open-weights LLMs enable decentralized security audits, allowing "
            "independent researchers to identify structural vulnerabilities, data leakage, and "
            "algorithmic bias. A 2026 empirical study demonstrated that open models achieve a "
            "45% faster vulnerability patching rate compared to closed proprietary API "
            "endpoints. However, open weights also prevent central kill-switches if malicious "
            "fine-tuning occurs."
        ),
        "source": "https://cset.georgetown.edu/reports/ai-open-weights-2026.pdf",
        "author": "CSET",
        "date": "2026-02-10",
    },
    {
        "doc_id": "NJAT-2025-04",
        "title": "Traffic Automation Audit 2025: Algorithmic Equity in Court Procedures",
        "content": (
            "Automated traffic court adjudication systems have been shown to reduce processing "
            "backlogs by 35% in municipal courts. However, algorithmic systems trained on "
            "historical citation data exhibited a 12% higher error rate in complex "
            "multi-vehicle collisions where contextual discretion and human witness "
            "credibility evaluation were required. Article 4 of judicial guidelines mandates "
            "human judge review for contested citations over $500."
        ),
        "source": "https://trafficlawjournal.org/audit2025.pdf",
        "author": "National Judicial Automation Taskforce",
        "date": "2025-11-15",
    },
    {
        "doc_id": "EU-AIA-2024-09",
        "title": "EU AI Act Compliance Guidelines 2024: Risk Governance & Human Oversight",
        "content": (
            "Under the EU AI Act (Regulation 2024/1689), high-risk AI applications in law "
            "enforcement, critical infrastructure, and judicial administration must undergo "
            "mandatory conformity assessments. Article 14 explicitly requires human oversight "
            "mechanisms, ensuring that natural persons can override automated AI outputs at any "
            "stage. Failure to comply incurs fines up to 35 million EUR or 7% global turnover."
        ),
        "source": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689",
        "author": "European Parliament",
        "date": "2024-07-12",
    },
    {
        "doc_id": "NIST-AIRMF-1.0",
        "title": "NIST Artificial Intelligence Risk Management Framework (AI RMF 1.0)",
        "content": (
            "The NIST AI RMF 1.0 organizes risk management into four core functions: GOVERN, MAP, "
            "MEASURE, and MANAGE. Trustworthy AI principles require systems to be valid, reliable, "
            "safe, secure, resilient, transparent, accountable, and privacy-preserved. The MAP "
            "function categorizes AI risks in context, while the MEASURE function quantitatively "
            "evaluates socio-technical impacts."
        ),
        "source": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        "author": "National Institute of Standards and Technology",
        "date": "2023-01-26",
    },
    {
        "doc_id": "ISO-26262-2018",
        "title": "ISO 26262 Road Vehicles Functional Safety Standard for Automotive AI",
        "content": (
            "ISO 26262 defines Automotive Safety Integrity Levels (ASIL A through ASIL D) for "
            "electrical and electronic systems in production automobiles. ASIL D represents the "
            "highest hazard severity classification, requiring fault-tolerant redundant "
            "architectures, hardware metrics single-point fault coverage >= 99%, and formal "
            "software verification to prevent fatal collisions."
        ),
        "source": "https://www.iso.org/standard/68383.html",
        "author": "International Organization for Standardization",
        "date": "2018-12-01",
    },
    {
        "doc_id": "WHO-AI-MED-2025",
        "title": "WHO Guidelines on Artificial Intelligence in Medical Diagnosis and Healthcare",
        "content": (
            "Medical AI diagnostic models must maintain minimum clinical specificity of 94% "
            "when evaluating radiological scans. Section 6 emphasizes informed patient consent, "
            "data provenance tracking, and explicit warnings against diagnostic automation bias "
            "in rare oncological pathologies."
        ),
        "source": "https://who.int/publications/ai-medical-guidelines-2025.pdf",
        "author": "World Health Organization",
        "date": "2025-05-18",
    },
    {
        "doc_id": "IEEE-7000-2021",
        "title": (
            "IEEE 7000-2021 Standard Model Process for Addressing Ethical Concerns During System"
            " Design"
        ),
        "content": (
            "IEEE 7000 establishes a value-based engineering methodology for autonomous systems. "
            "System designers must construct Value Elicitation Matrices, prioritize stakeholder "
            "well-being, and establish transparent audit trails for all autonomous decision loops."
        ),
        "source": "https://standards.ieee.org/standard/7000-2021.html",
        "author": "IEEE Computer Society",
        "date": "2021-09-10",
    },
    {
        "doc_id": "FTC-AI-GUIDE-2023",
        "title": "FTC Guidance on AI Deception, Consumer Bias, and Automated Decision Systems",
        "content": (
            "The Federal Trade Commission warns businesses against deceptive AI claims. Section 5 "
            "of the FTC Act prohibits unfair or deceptive marketing of unvalidated AI models. "
            "Companies using secret algorithmic scoring for employment or lending must provide "
            "adverse action notices under the Fair Credit Reporting Act."
        ),
        "source": "https://ftc.gov/business-guidance/ai-deception-2023.pdf",
        "author": "Federal Trade Commission",
        "date": "2023-04-19",
    },
    {
        "doc_id": "OECD-AI-PRINCIPLES-2024",
        "title": "OECD Recommendation of the Council on Artificial Intelligence 2024",
        "content": (
            "The OECD AI Principles champion inclusive growth, sustainable development, "
            "human-centric values, and robust security. Governments should foster international "
            "interoperability, invest in open AI research infrastructure, and mandate transparent "
            "algorithmic impact assessments."
        ),
        "source": "https://oecd.ai/en/wonk/ai-principles-2024",
        "author": "OECD",
        "date": "2024-05-02",
    },
    {
        "doc_id": "DARPA-XAI-2022",
        "title": "DARPA Explainable Artificial Intelligence (XAI) Program Final Report",
        "content": (
            "The DARPA XAI program created explainable machine learning models that produce "
            "human-understandable rationale vectors. Evaluation metrics demonstrated a 40% "
            "increase in operator trust when decision trees and saliency maps accompanied deep "
            "learning classification outputs."
        ),
        "source": "https://darpa.mil/xai-final-report-2022.pdf",
        "author": "DARPA",
        "date": "2022-11-30",
    },
]


def load_and_index_corpus(vector_store: QdrantVectorStore) -> list[Document]:
    """Ingest raw policy documents into vector store and return list of document chunks."""
    ingestor = DocumentIngestor(vector_store=vector_store, chunk_size=384, chunk_overlap=48)
    all_chunks = []

    for item in RAW_DOCUMENTS_DATA:
        chunks = ingestor.ingest_raw_document(
            title=item["title"],
            content=item["content"],
            source=item["source"],
            author=item["author"],
            date=item["date"],
            doc_id=item["doc_id"],
        )
        all_chunks.extend(chunks)

    return all_chunks


FIFTY_QUERY_BENCHMARK: list[RAGBenchmarkQuery] = [
    # Category 1: Easy Factual Queries (5)
    RAGBenchmarkQuery(
        id="Q-01",
        query="What percentage faster vulnerability patching rate do open models achieve?",
        category="Easy Factual",
        expected_doc_ids={"CSET-2026-01"},
    ),
    RAGBenchmarkQuery(
        id="Q-02",
        query="By how much do automated traffic court systems reduce processing backlogs?",
        category="Easy Factual",
        expected_doc_ids={"NJAT-2025-04"},
    ),
    RAGBenchmarkQuery(
        id="Q-03",
        query="What is the maximum fine for non-compliance under the EU AI Act?",
        category="Easy Factual",
        expected_doc_ids={"EU-AIA-2024-09"},
    ),
    RAGBenchmarkQuery(
        id="Q-04",
        query="What are the four core functions of NIST AI RMF 1.0?",
        category="Easy Factual",
        expected_doc_ids={"NIST-AIRMF-1.0"},
    ),
    RAGBenchmarkQuery(
        id="Q-05",
        query="What single-point fault coverage percentage is required for ASIL D under ISO 26262?",
        category="Easy Factual",
        expected_doc_ids={"ISO-26262-2018"},
    ),
    # Category 2: Semantic Paraphrase Queries (5)
    RAGBenchmarkQuery(
        id="Q-06",
        query="How do decentralized model weights help independent researchers find flaws?",
        category="Semantic Paraphrase",
        expected_doc_ids={"CSET-2026-01"},
    ),
    RAGBenchmarkQuery(
        id="Q-07",
        query="Why do algorithmic court tools perform worse during multi-vehicle crash disputes?",
        category="Semantic Paraphrase",
        expected_doc_ids={"NJAT-2025-04"},
    ),
    RAGBenchmarkQuery(
        id="Q-08",
        query="Which EU regulation forces human override capabilities for judicial algorithms?",
        category="Semantic Paraphrase",
        expected_doc_ids={"EU-AIA-2024-09"},
    ),
    RAGBenchmarkQuery(
        id="Q-09",
        query="How does NIST framework measure socio-technical harm in context?",
        category="Semantic Paraphrase",
        expected_doc_ids={"NIST-AIRMF-1.0"},
    ),
    RAGBenchmarkQuery(
        id="Q-10",
        query="What automotive safety rating mandates redundant hardware against fatalities?",
        category="Semantic Paraphrase",
        expected_doc_ids={"ISO-26262-2018"},
    ),
    # Category 3: Exact Keyword & Technical Acronym Queries (5)
    RAGBenchmarkQuery(
        id="Q-11",
        query="CSET-2026-01 open weights security audit",
        category="Exact Keyword",
        expected_doc_ids={"CSET-2026-01"},
    ),
    RAGBenchmarkQuery(
        id="Q-12",
        query="Article 4 judicial guidelines contested citations $500",
        category="Exact Keyword",
        expected_doc_ids={"NJAT-2025-04"},
    ),
    RAGBenchmarkQuery(
        id="Q-13",
        query="Regulation 2024/1689 Article 14 35 million EUR",
        category="Exact Keyword",
        expected_doc_ids={"EU-AIA-2024-09"},
    ),
    RAGBenchmarkQuery(
        id="Q-14",
        query="NIST AI RMF 1.0 GOVERN MAP MEASURE MANAGE",
        category="Exact Keyword",
        expected_doc_ids={"NIST-AIRMF-1.0"},
    ),
    RAGBenchmarkQuery(
        id="Q-15",
        query="ASIL D ISO 26262 fault coverage 99%",
        category="Exact Keyword",
        expected_doc_ids={"ISO-26262-2018"},
    ),
    # Category 4: Distractor Queries (5)
    RAGBenchmarkQuery(
        id="Q-16",
        query="Which report discusses legal fines of 35 million EUR for judicial AI?",
        category="Distractors",
        expected_doc_ids={"EU-AIA-2024-09"},
    ),
    RAGBenchmarkQuery(
        id="Q-17",
        query="Which document evaluates 35% processing backlog reduction in courts?",
        category="Distractors",
        expected_doc_ids={"NJAT-2025-04"},
    ),
    RAGBenchmarkQuery(
        id="Q-18",
        query="Which whitepaper highlights 45% faster patching vs 99% fault coverage?",
        category="Distractors",
        expected_doc_ids={"CSET-2026-01"},
    ),
    RAGBenchmarkQuery(
        id="Q-19",
        query="Which standard categorizes ASIL A through ASIL D hazard severity?",
        category="Distractors",
        expected_doc_ids={"ISO-26262-2018"},
    ),
    RAGBenchmarkQuery(
        id="Q-20",
        query="Which framework organizes MAP function socio-technical impact evaluation?",
        category="Distractors",
        expected_doc_ids={"NIST-AIRMF-1.0"},
    ),
    # Category 5: Multi-Document Queries (5)
    RAGBenchmarkQuery(
        id="Q-21",
        query=(
            "Compare court backlog reduction in NJAT-2025-04 with human oversight fines in EU AI"
            " Act"
        ),
        category="Multi-Document",
        expected_doc_ids={"NJAT-2025-04", "EU-AIA-2024-09"},
    ),
    RAGBenchmarkQuery(
        id="Q-22",
        query=(
            "What are open LLM security auditing benefits alongside ISO 26262 ASIL D safety rules?"
        ),
        category="Multi-Document",
        expected_doc_ids={"CSET-2026-01", "ISO-26262-2018"},
    ),
    RAGBenchmarkQuery(
        id="Q-23",
        query=(
            "How do NIST AI RMF MAP functions align with EU AI Act high-risk conformity"
            " assessments?"
        ),
        category="Multi-Document",
        expected_doc_ids={"NIST-AIRMF-1.0", "EU-AIA-2024-09"},
    ),
    RAGBenchmarkQuery(
        id="Q-24",
        query=(
            "Contrast municipal traffic court error rates with open source vulnerability"
            " patching metrics"
        ),
        category="Multi-Document",
        expected_doc_ids={"NJAT-2025-04", "CSET-2026-01"},
    ),
    RAGBenchmarkQuery(
        id="Q-25",
        query=(
            "Analyze hardware single-point fault coverage alongside NIST trustworthy AI principles"
        ),
        category="Multi-Document",
        expected_doc_ids={"ISO-26262-2018", "NIST-AIRMF-1.0"},
    ),
    # Category 6: Multi-Hop Queries (5)
    RAGBenchmarkQuery(
        id="Q-26",
        query=(
            "If traffic court algorithms have a 12% error rate, what EU law requires human"
            " override for them?"
        ),
        category="Multi-Hop",
        expected_doc_ids={"NJAT-2025-04", "EU-AIA-2024-09"},
    ),
    RAGBenchmarkQuery(
        id="Q-27",
        query=(
            "Does open source 45% faster patching prevent central kill-switches during malicious"
            " fine-tuning?"
        ),
        category="Multi-Hop",
        expected_doc_ids={"CSET-2026-01"},
    ),
    RAGBenchmarkQuery(
        id="Q-28",
        query=(
            "Which automotive safety standard enforces redundancy if NIST MAP function identifies"
            " ASIL D risk?"
        ),
        category="Multi-Hop",
        expected_doc_ids={"ISO-26262-2018", "NIST-AIRMF-1.0"},
    ),
    RAGBenchmarkQuery(
        id="Q-29",
        query=(
            "What judge review threshold applies to traffic citations that violate EU Regulation"
            " 2024/1689?"
        ),
        category="Multi-Hop",
        expected_doc_ids={"NJAT-2025-04", "EU-AIA-2024-09"},
    ),
    RAGBenchmarkQuery(
        id="Q-30",
        query=(
            "How do open weights security audits complement NIST MEASURE socio-technical impact"
            " metrics?"
        ),
        category="Multi-Hop",
        expected_doc_ids={"CSET-2026-01", "NIST-AIRMF-1.0"},
    ),
    # Category 7: Negative Unanswerable Queries (5)
    RAGBenchmarkQuery(
        id="Q-31",
        query="What is the recommended quantum encryption key length for autonomous satellites?",
        category="Negative Query",
        expected_doc_ids=set(),
        is_negative=True,
    ),
    RAGBenchmarkQuery(
        id="Q-32",
        query="How many carbon credits are awarded for solar panel installations in California?",
        category="Negative Query",
        expected_doc_ids=set(),
        is_negative=True,
    ),
    RAGBenchmarkQuery(
        id="Q-33",
        query="What is the federal minimum wage for maritime container ship workers?",
        category="Negative Query",
        expected_doc_ids=set(),
        is_negative=True,
    ),
    RAGBenchmarkQuery(
        id="Q-34",
        query="Which FDA regulation governs phase 3 clinical trials for mRNA cancer vaccines?",
        category="Negative Query",
        expected_doc_ids=set(),
        is_negative=True,
    ),
    RAGBenchmarkQuery(
        id="Q-35",
        query="What tax rate applies to cryptocurrency staking rewards in Switzerland?",
        category="Negative Query",
        expected_doc_ids=set(),
        is_negative=True,
    ),
    # Category 8: Conflicting Evidence Queries (5)
    RAGBenchmarkQuery(
        id="Q-36",
        query=(
            "Does AI automation decrease traffic backlog while increasing collision adjudication"
            " errors?"
        ),
        category="Conflicting Evidence",
        expected_doc_ids={"NJAT-2025-04"},
    ),
    RAGBenchmarkQuery(
        id="Q-37",
        query=(
            "Are open source models safer due to fast patching or more dangerous due to lack of"
            " kill-switches?"
        ),
        category="Conflicting Evidence",
        expected_doc_ids={"CSET-2026-01"},
    ),
    RAGBenchmarkQuery(
        id="Q-38",
        query=(
            "Does judicial automation lower costs while incurring risk of 35 million EUR"
            " non-compliance fines?"
        ),
        category="Conflicting Evidence",
        expected_doc_ids={"NJAT-2025-04", "EU-AIA-2024-09"},
    ),
    RAGBenchmarkQuery(
        id="Q-39",
        query="Is human judge review mandatory for all citations or only those over $500?",
        category="Conflicting Evidence",
        expected_doc_ids={"NJAT-2025-04"},
    ),
    RAGBenchmarkQuery(
        id="Q-40",
        query="Does ASIL D safety rely on software verification or physical hardware redundancy?",
        category="Conflicting Evidence",
        expected_doc_ids={"ISO-26262-2018"},
    ),
    # Category 9: Long-Context Queries (5)
    RAGBenchmarkQuery(
        id="Q-41",
        query=(
            "Detailed analysis of socio-technical risk classification under NIST AI RMF 1.0"
            " MEASURE function"
        ),
        category="Long Context",
        expected_doc_ids={"NIST-AIRMF-1.0"},
    ),
    RAGBenchmarkQuery(
        id="Q-42",
        query=(
            "Full regulatory requirements for high-risk law enforcement AI conformity"
            " assessments under Article 14"
        ),
        category="Long Context",
        expected_doc_ids={"EU-AIA-2024-09"},
    ),
    RAGBenchmarkQuery(
        id="Q-43",
        query=(
            "Comprehensive comparison of single-point fault coverage in electronic stability"
            " control for road vehicles"
        ),
        category="Long Context",
        expected_doc_ids={"ISO-26262-2018"},
    ),
    RAGBenchmarkQuery(
        id="Q-44",
        query=(
            "Systemic trade-offs of open weights model vulnerability patching versus central"
            " kill-switch absence"
        ),
        category="Long Context",
        expected_doc_ids={"CSET-2026-01"},
    ),
    RAGBenchmarkQuery(
        id="Q-45",
        query=(
            "Municipal court witness credibility evaluation problems in automated multi-vehicle"
            " collision citation appeals"
        ),
        category="Long Context",
        expected_doc_ids={"NJAT-2025-04"},
    ),
    # Category 10: Paraphrased & Noisy Queries (5)
    RAGBenchmarkQuery(
        id="Q-46",
        query="opn weight LLM vuln patch rate 45 percent faster study",
        category="Noisy Query",
        expected_doc_ids={"CSET-2026-01"},
    ),
    RAGBenchmarkQuery(
        id="Q-47",
        query="traff court adjudicatin 35 perc backlog 12 percent err rate",
        category="Noisy Query",
        expected_doc_ids={"NJAT-2025-04"},
    ),
    RAGBenchmarkQuery(
        id="Q-48",
        query="EU AI Act Article 14 human oversght 35 mil fine turnover",
        category="Noisy Query",
        expected_doc_ids={"EU-AIA-2024-09"},
    ),
    RAGBenchmarkQuery(
        id="Q-49",
        query="NIST RMF 1.0 GOVERN MAP MEASURE MANAGE functions",
        category="Noisy Query",
        expected_doc_ids={"NIST-AIRMF-1.0"},
    ),
    RAGBenchmarkQuery(
        id="Q-50",
        query="ISO 26262 ASIL-D fault coverage 99 percent single point",
        category="Noisy Query",
        expected_doc_ids={"ISO-26262-2018"},
    ),
]

DEV_SET: list[RAGBenchmarkQuery] = FIFTY_QUERY_BENCHMARK[:25]
UNSEEN_TEST_SET: list[RAGBenchmarkQuery] = FIFTY_QUERY_BENCHMARK[25:]
