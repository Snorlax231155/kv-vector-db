# ADR-0004: Rules-first router, with a LoRA classifier for the route only

- Status: accepted
- Date: 2026-10-07

## Context
KV, VECTOR and HYBRID are largely separable by surface cues: an ID or code in the query
points to KV. No labelled data exists. If a model is trained on the same templates the
evaluation set was written from, any "learned router beats rules" result is an artifact.

## Decision
- **Entities** always come from deterministic extractors: regex for IDs and codes, plus
  gazetteers for drugs, labs and conditions, after abbreviation normalisation.
- **Route** comes from `RulesRouter`, or from `LoraRouter`, which is a PEFT LoRA adapter on
  `distilbert-base-uncased` with temperature-scaled confidence.
- `AutoRouter` uses LoRA when its calibrated confidence is at least τ. Otherwise it falls
  back to rules. If rules confidence is also below τ, it routes HYBRID, which is the safe
  default because it retrieves from both sides.
- Training data is template-generated plus paraphrased. The gold set
  (`eval/gold/queries.jsonl`) is written independently, frozen, and never used for
  training or threshold tuning. A leakage check rejects any training query whose word
  trigrams overlap a gold query too much.

## Consequences
- If the rules router wins, we report that. The useful finding is where the two routers
  disagree.
- Without torch installed (CI, mock mode) the system degrades to rules with no code changes.

## Revision (2026-10-07, after the first gold-set evaluation)
- **Evidence.** The first policy was LoRA-first. It was evaluated on the independently written
  gold set, split by `sha256(id)` parity into an **analysis half** (failures inspected) and a
  **held-out half** (aggregate metrics only). On the analysis half: rules 0.922, rules-first
  0.902, LoRA-first 0.882, LoRA alone 0.843. The LoRA classifier reaches 0.85 accuracy on
  held-out *templates*, but it is over-confident (p≈0.95 when wrong) on human phrasings
  unlike any template.
- **Decision.** `AutoRouter` is now **rules → LoRA → HYBRID**. LoRA is consulted only when
  rules confidence < 0.6, and it is trusted only at ≥ 0.8.
- **Calibration fix.** The dev split is now template-disjoint. The earlier random split put
  paraphrases of training templates in dev, which gave accuracy 1.0 and a sharpening
  temperature (T=0.18). With held-out templates, T=2.0 and ECE falls from 0.10 to 0.05.
- Results before and after these fixes are both reported in `eval/results/`, per half. The
  held-out half is the number to quote.
