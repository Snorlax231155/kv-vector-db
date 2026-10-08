"""One-click demo scenarios shown in the UI. Each states what the audience should see."""

from __future__ import annotations

from medmemory.contracts.schemas import Scenario

SCENARIOS: list[Scenario] = [
    Scenario(
        id="exact-lookup",
        title="Exact lookup (KV)",
        query="What is the latest eGFR for P0001?",
        patient_scope="P0001",
        description="A known patient and a known lab: a prefix scan on patient:P0001:lab:egfr:.",
        expect="Route KV, one KV record cited, retrieval in well under a millisecond.",
    ),
    Scenario(
        id="code-lookup",
        title="Code lookup (KV)",
        query="What does ICD-10-CM code N18.30 mean?",
        description="Exact ID lookup in the ICD-10-CM table. No embeddings involved.",
        expect="Route KV, the official ICD-10-CM description.",
    ),
    Scenario(
        id="semantic",
        title="Semantic search (vector)",
        query="What does the metformin label say about patients with reduced kidney function?",
        description="No IDs, a meaning-based question: search the drug_labels namespace, filtered to drug=metformin.",
        expect="Route VECTOR, label chunks with similarity scores, every sentence cited.",
    ),
    Scenario(
        id="hybrid",
        title="Hybrid (KV ∥ vector)",
        patient_scope="P0001",
        query="Given her latest eGFR, what does the metformin label say about kidney function?",
        description="Needs the patient's exact eGFR (KV) and the label's renal guidance (vector), retrieved in parallel.",
        expect="Route HYBRID, KV and vector stages overlapping in the waterfall, citations from both.",
    ),
    Scenario(
        id="cache",
        title="Cache hit vs miss",
        query="What is the latest HbA1c for P0001?",
        patient_scope="P0001",
        run_twice=True,
        description="Ask the same question twice: the second answer comes from the exact cache.",
        expect="First run cache=miss, second run cache=exact with a far shorter waterfall.",
    ),
    Scenario(
        id="semantic-cache",
        title="Semantic cache (scoped)",
        query="what's the latest HbA1c for P0001",
        patient_scope="P0001",
        description="A near-paraphrase of a cached question with the same patient and entities. Run after 'Cache hit vs miss'.",
        expect="cache=semantic with the similarity shown. The same question for another patient never matches.",
    ),
    Scenario(
        id="red-flag",
        title="Emergency red flag",
        query="I have crushing chest pain spreading to my left arm and I'm sweating",
        description="Safety pre-check short-circuits: no retrieval, no LLM.",
        expect="Status red_flag with emergency guidance; the waterfall shows only the safety stage.",
    ),
    Scenario(
        id="abstain-conflict",
        title="Abstention: conflicting sources",
        patient_scope="P0004",
        query="Is her levothyroxine dose in the notes consistent with her medication record?",
        description="The structured record and the latest note's medication list disagree (a planted test case).",
        expect="Status abstained, with both conflicting sources named instead of a guess.",
    ),
    Scenario(
        id="abstain-unknown",
        title="Abstention: not in knowledge base",
        query="What is the recommended dose of semaglutide?",
        description="A dosing question for a drug with no FDA label in the knowledge base. Doses never come from model memory.",
        expect="Status abstained: 'I don't have enough information'.",
    ),
    Scenario(
        id="scope-violation",
        title="Patient scope enforcement",
        query="What medications is P0002 taking?",
        patient_scope="P0001",
        description="The active patient is P0001 but the question names P0002.",
        expect="Status scope_violation before any retrieval runs.",
    ),
]
