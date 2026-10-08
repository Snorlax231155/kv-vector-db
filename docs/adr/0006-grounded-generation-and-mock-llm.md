# ADR-0006: Grounded generation with citation markers and a deterministic post-check

- Status: accepted
- Date: 2026-10-07

## Context
Every factual claim must map to retrieved evidence. Doses must come from labels, never from
the model's memory. CI and teammates without keys must still be able to run the full
pipeline.

## Decision
- Evidence is numbered `S1..Sn`, and the prompt requires a `[S#]` marker at the end of every
  sentence. The model may reply `INSUFFICIENT_EVIDENCE` instead.
- The **post-check is deterministic, not an LLM judge.** A sentence is kept only if it
  carries at least one valid marker *and* every number in it (doses, lab values, dates)
  appears in the text of a cited source. Unsupported sentences are removed. If less than
  60% of the answer survives, the response becomes an abstention.
- The chain is the same in every mode: LangChain `ChatPromptTemplate`, then a chat model,
  then the parser. In mock mode the chat model is `ExtractiveChatModel`, a `BaseChatModel`
  that answers by quoting the most relevant evidence sentences. Mock mode therefore
  exercises the real prompt, the streaming and the post-check path.
- Text streams to the UI over SSE, and the final event carries the checked answer. If
  sentences were removed, the UI replaces the streamed draft with the checked version.

## Consequences
- Faithfulness, measured as the citation-grounding rate, is computed the same way in tests,
  in eval and in production.
- A numbers-only check misses paraphrased claims that are wrong but number-free. That gap
  is documented as a limitation, and the human spot-check in the eval protocol measures it.
