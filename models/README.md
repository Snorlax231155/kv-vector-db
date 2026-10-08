# models/

This directory is reserved for trained model artifacts and weights.

In the current **Phase 1 Presentation Milestone**, MedMemory uses deterministic clinical lexicons, regex-based entity extractors, and rules-based routing (`RulesRouter`), keeping the runtime lightweight, offline, and reproducible without external model weight files.

Future milestones (Phase 2 & Phase 3) will utilize this folder for fine-tuned LoRA router adapter weights and cross-encoder reranker models.
