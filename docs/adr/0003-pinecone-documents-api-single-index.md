# ADR-0003: Pinecone v10 Documents API, one index, three namespaces, client-side hybrid

- Status: accepted
- Date: 2026-10-07

## Context
SDK `pinecone==10.0.0` deprecates `dimension=`/`metric=` index creation in favour of a
field `schema`. Schema-based indexes are read and written through `index.documents`. A
`dense_vector` scoring clause must appear alone in a search, so dense and BM25 text
scoring cannot be combined in a single call. The Starter plan allows 5 indexes,
100 namespaces per index, and 2 GB of storage, in us-east-1 only.

## Decision
- Create one index with the schema
  `{embedding: dense_vector(384, cosine), text: string(full_text_search)}`. The text field
  must be declared at creation, because sparse/text fields can't be added later.
- Use the namespaces `patient_notes`, `drug_labels` and `guidelines`, each with an optional
  per-developer prefix (`MEDMEMORY_PINECONE_NAMESPACE_PREFIX`).
- Patient scoping uses a metadata filter on `patient_id`, plus a post-filter in our code.
  We chose this over one namespace per patient because the per-patient approach would hit
  the 100-namespace cap at around 100 patients.
- Hybrid search is two searches, dense and BM25 `TextQuery`, run concurrently and fused
  with Reciprocal Rank Fusion (k=60) on the client. It sits behind `options.hybrid_sparse`
  and is reported as an ablation.
- Embeddings are computed locally (on the GPU when available), not with Pinecone's hosted
  inference, so that Pinecone and FAISS index identical vectors.

## Consequences
- The adapter is written against the installed SDK's source (`pinecone/client/documents.py`).
  It has not yet been run against a live index in CI. `tests/vector/test_pinecone_live.py`
  runs only when `PINECONE_API_KEY` is set.
- Upserts become visible eventually. The ingest command polls until the namespace count
  matches before reporting success.
