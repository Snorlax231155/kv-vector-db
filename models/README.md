# models/

This directory is reserved for trained model artifacts and weights.

By default, MedMemory operates with deterministic clinical lexicons, regex-based entity extractors, and rules-based routing (`RulesRouter`), keeping the runtime lightweight, offline, and reproducible without requiring multi-gigabyte weight downloads.

Future extensions can utilize this directory for fine-tuned LoRA router adapter weights and cross-encoder reranker models.
