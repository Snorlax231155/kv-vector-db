"""Embedders.

`HashingEmbedder` is the mock: deterministic feature hashing (no model, no GPU, no network)
over word unigrams, bigrams and character trigrams, with a small clinical synonym map so
'HbA1c' and 'hemoglobin a1c' land near each other. It is a *lexical* embedder: good enough
to exercise every code path and give sensible retrieval on the seed set, but it has no real
semantics. Scores are not comparable with neural embedders; thresholds are per embedder.

`SentenceTransformerEmbedder` wraps sentence-transformers (default BAAI/bge-small-en-v1.5,
384-d). Batches on CUDA when available. bge v1.5 recommends a query instruction prefix for
retrieval; documents are embedded without it.
"""

from __future__ import annotations

import itertools
import math
import re
import threading
from collections import Counter
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import xxhash

from medmemory.contracts.protocols import FloatArray

_TOKEN = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
        "does",
        "did",
        "do",
        "patient",
        "patients",
        "please",
        "tell",
        "show",
        "give",
        "about",
        "s",
        "t",
    ]
)
# Tiny shared vocabulary so the mock embedder isn't hopeless on abbreviations. The real
# normalizer lives in router/normalize.py; this is deliberately minimal and independent.
_SYN = {
    "hba1c": "hemoglobin a1c",
    "a1c": "hemoglobin a1c",
    "htn": "hypertension",
    "bp": "blood pressure",
    "mi": "myocardial infarction",
    "ckd": "chronic kidney disease",
    "t2dm": "type 2 diabetes",
    "dm2": "type 2 diabetes",
    "afib": "atrial fibrillation",
    "chf": "heart failure",
    "copd": "chronic obstructive pulmonary disease",
    "gerd": "reflux",
    "sob": "shortness of breath",
    "egfr": "egfr kidney function",
    "ldl": "ldl cholesterol",
    "nsaid": "nsaids ibuprofen",
    "nsaids": "nsaids ibuprofen",
    "diagnosis": "condition diagnosis",
    "diagnoses": "condition diagnoses",
    "diagnosed": "condition diagnosed",
    "diganosis": "condition diagnosis",
    "diganoses": "condition diagnoses",
    "problem": "condition problem",
    "problems": "condition problems",
    "illness": "condition illness",
    "illnesses": "condition illnesses",
    "visit": "encounter visit",
    "visits": "encounter visits",
    "appointment": "encounter appointment",
    "appointments": "encounter appointments",
    "meds": "medication meds",
    "medicine": "medication medicine",
    "medicines": "medication medicines",
    "prescription": "medication prescription",
    "prescriptions": "medication prescriptions",
    "drug": "medication drug",
    "drugs": "medication drugs",
    "regimen": "medication regimen",
    "present": "present active",
}


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    words = []
    for tok in _TOKEN.findall(text.lower()):
        for w in _SYN.get(tok, tok).split():
            if w not in _STOP:
                words.append(_stem(w))
    return words


class HashingEmbedder:
    def __init__(self, dim: int = 384, seed: int = 13) -> None:
        self.dim = dim
        self.seed = seed
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> Counter[str]:
        words = tokenize(text)
        feats: Counter[str] = Counter()
        for w in words:
            feats["w:" + w] += 1.0  # type: ignore[assignment]
            if len(w) >= 5:
                padded = f"#{w}#"
                for i in range(len(padded) - 2):
                    feats["c:" + padded[i : i + 3]] += 0.25  # type: ignore[assignment]
        for a, b in itertools.pairwise(words):
            feats[f"b:{a}_{b}"] += 0.7  # type: ignore[assignment]
        return feats

    def _embed(self, text: str) -> FloatArray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat, tf in self._features(text).items():
            h = xxhash.xxh64_intdigest(feat.encode("utf-8"), seed=self.seed)
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(tf)) if tf >= 1 else sign * tf
        n = float(np.linalg.norm(vec))
        return vec / n if n > 0 else vec

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._embed(t) for t in texts])

    def embed_query(self, text: str) -> FloatArray:
        return self._embed(text)


class SentenceTransformerEmbedder:
    QUERY_PREFIX: ClassVar[dict[str, str]] = {
        "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
        "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    }

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)
        self.batch_size = batch_size
        self.name = f"{model_name}@{device}"
        self._prefix = self.QUERY_PREFIX.get(model_name, "")
        self._lock = threading.Lock()  # the model isn't guaranteed thread-safe on GPU

    def embed_documents(self, texts: Sequence[str]) -> FloatArray:
        with self._lock:
            arr = self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 2000,
            )
        return np.asarray(arr, dtype=np.float32)

    def embed_query(self, text: str) -> FloatArray:
        with self._lock:
            arr = self._model.encode(
                [self._prefix + text], normalize_embeddings=True, convert_to_numpy=True
            )
        return np.asarray(arr[0], dtype=np.float32)
