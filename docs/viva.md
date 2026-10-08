# Viva pack

## 60-second script

> "Clinical questions come in two shapes. 'What was P0001's last HbA1c?' is an **exact lookup**:
> a key-value store answers it in microseconds and can't be wrong about it. 'What does the
> metformin label say about kidney function?' is a **meaning** question that needs semantic
> search. Real questions often need both: 'Given her eGFR, is metformin still appropriate?'
>
> MedMemory is the orchestration layer that combines them. A router classifies each query
> as KV, vector or hybrid and extracts the entities. It is rules-first, with a LoRA-tuned
> classifier and calibrated fallback. A two-level cache sits in front, an exact LRU plus a
> semantic cache, and the semantic cache is deliberately locked to the same patient,
> entities and data version. Retrieval runs KV and Pinecone in parallel. Then a grounded
> generator answers only from cited sources, and a deterministic checker deletes any
> sentence whose numbers don't appear in its source.
>
> Safety is built in rather than bolted on. Emergencies short-circuit before retrieval,
> another patient's data is blocked at four layers, and 'I don't have enough information'
> is a tested output. We integrated SQLite, RocksDB, Pinecone, FAISS and Claude rather than
> building them. What we built is the routing, scoping, caching, grounding and the
> evaluation that measures all of it. Let me show you."

**Demo order (≈3 min):** exact lookup → hybrid, pointing at the overlapping bars in the
waterfall → cache miss then hit → red flag → conflicting-sources abstention → scope
violation → dashboard confusion matrix → cluster ring, adding a node.

## Likely reviewer questions

**Why a GPU for a database?**
We don't use one for the database. KV gets take ~1–4 µs on CPU and there is nothing to
parallelise. Pinecone is managed, so its hardware isn't ours. Our GPU does three jobs:
batch embedding (bge-small runs ~12× faster on our RTX 4060 than on CPU, see the
benchmarks), LoRA training of the router, and a local exact-search baseline. Our own
benchmark also shows the GPU does **not** help vector search at our real corpus size
(~2k chunks). It only pulls ahead at 100k+ vectors or in large batches.

**Isn't the KV-vs-vector latency comparison obvious?**
Yes. A hash lookup beats a nearest-neighbour search, and LLM generation (seconds)
dwarfs both. That's why the dashboard separates *retrieval* latency from end-to-end
latency. The interesting results are routing accuracy, grounding, abstention correctness
and leakage, not "KV is fast".

**What exactly did you build vs integrate?**
See the table in the README. In short: engines, models and Pinecone are integrated. The
router, scoping, caches, sufficiency and citation checks, chunking, instrumentation,
eval harness, UI and sharding layer are ours.

**Your semantic cache could return the wrong patient's answer.**
It can't by construction. Lookups only compare entries in the same bucket, and the bucket
key is (patient, filters, entity signature, data version). There is a test that stores
P0001's answer and asks the identical question for P0006: the similarity is 1.0, and it
is still blocked. The adversarial near-miss tests (negation, CKD 3 vs 4, LDL vs A1c) pass
even at a 0.5 threshold.

**How do you know answers are grounded?**
Every sentence must carry a `[S#]` marker that points at provided evidence, and every
number in the sentence must appear in the cited source. Sentences that fail are removed.
If less than 60% of the draft survives, we abstain. The limitation is that a wrong claim
with no numbers can pass. We measure that with a human spot-check and say so.

**Why should we believe the router numbers? You wrote the rules and the test set.**
The gold set was written by a separate author that had no access to the router code or
the training templates. It was frozen before training, and it has a validator that
checks every expected source exists. Training data is template-generated and filtered
by trigram overlap against the gold set. Dev-split accuracy is reported separately and
labelled as in-distribution.

**Your LoRA router is worse than your rules. Why keep it?**
On our gold set, rules score 0.86 and LoRA 0.81. The adapter scores 0.85 on held-out *templates*, but it is
over-confident on human phrasings that look like no template. That is a real finding about template-generated
training data, and we changed the policy to rules-first because of it (ADR-0004, revised). LoRA now only breaks
ties when the rules are unsure. The path to making it useful is more varied training data, such as
LLM-paraphrased queries or real clinician queries, not a bigger model.

**Didn't you tune on your test set?**
We split the gold set by hash into an analysis half and a held-out half, and inspected failures only on the
analysis half. On the held-out half, status accuracy went from 0.65 to 0.88 and abstention accuracy from 0.62 to
0.88. One remaining safety miss was fixed afterwards, and we label that fix as post hoc.

**Why LoRA on a small encoder? Full fine-tuning would work.**
It would. LoRA trains ~1% of the weights, gives a few-MB adapter, trains in about a
minute on a laptop GPU, and was the technique we set out to demonstrate. We report it
honestly against the rules router, including where rules win.

**What happens if Pinecone or the LLM is down?**
Errors are typed (`VectorStoreError`, `GenerationError`) and return a 502 with a
trace ID. Mock mode runs the whole system offline. The design fallback for a vector
outage is to answer KV-only questions and abstain on the rest.

**Is this HIPAA-compliant?**
No, and it doesn't claim to be. It uses synthetic data only. The auth layer is a
labelled stub. The audit log hashes patient IDs, but a real deployment would need real
authentication, encryption at rest, BAAs with vendors and a security review.

**What would you do with another month?**
Real auth (OIDC), an LLM-judge plus clinician review for faithfulness, Pinecone hybrid
evaluated at scale, conflict detection beyond medication doses, and moving the semantic
cache onto an ANN index.
