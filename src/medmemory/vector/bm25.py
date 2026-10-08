"""Minimal Okapi BM25 used by the local stores' `text_query` (Pinecone does BM25
server-side for schema fields declared with full_text_search)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from medmemory.vector.embedders import tokenize


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(self.tfs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for tf, dl in zip(self.tfs, self.lens, strict=True):
            s = 0.0
            for t in terms:
                f = tf.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out
