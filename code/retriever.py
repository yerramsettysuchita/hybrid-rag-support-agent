"""
retriever.py — BM25-based offline retriever with stemmed vocabulary and reranking.

Tokenization goes through normalizer.tokenize (stemmed + de-duped) so the
vocabulary is consistent between index time and query time.  A reranking step
adds additive bonuses when query terms appear in the document title or
breadcrumbs, rewarding articles whose headings directly name the topic.

RetrieverProtocol defines the search() interface; BM25Retriever satisfies it.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from rank_bm25 import BM25Okapi

from corpus import Document
from normalizer import tokenize

logger = logging.getLogger(__name__)

# (document, combined score after reranking)
SearchResult = tuple[Document, float]


# ---------------------------------------------------------------------------
# Protocol — thin interface; swap BM25Retriever for any compliant class
# ---------------------------------------------------------------------------

@runtime_checkable
class RetrieverProtocol(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 5,
        company: str | None = None,
    ) -> list[SearchResult]: ...


# ---------------------------------------------------------------------------
# Reranker — additive bonuses on top of BM25 scores
# ---------------------------------------------------------------------------

def _rerank(
    results: list[SearchResult],
    query_tokens: set[str],
    title_bonus: float = 4.0,
    breadcrumb_bonus: float = 2.0,
) -> list[SearchResult]:
    """
    Add per-result bonuses for query-term overlap in title and breadcrumbs.

    Additive scoring (not multiplicative) keeps the adjustments easy to reason
    about during debugging: final_score = bm25_score + title_bonus * overlap
    + breadcrumb_bonus * overlap.
    """
    reranked: list[SearchResult] = []
    for doc, score in results:
        bonus = 0.0

        title_tokens = set(tokenize(doc.title))
        overlap_title = len(query_tokens & title_tokens)
        if overlap_title:
            bonus += title_bonus * overlap_title

        bc_text = " ".join(doc.breadcrumbs)
        bc_tokens = set(tokenize(bc_text))
        overlap_bc = len(query_tokens & bc_tokens)
        if overlap_bc:
            bonus += breadcrumb_bonus * overlap_bc

        reranked.append((doc, score + bonus))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Offline BM25 retriever.

    Index is built once at construction; all searches are deterministic.
    Retrieval order:
      1. BM25 scoring over stemmed vocabulary.
      2. Title / breadcrumb reranking (additive bonus).
      3. Company-scoped results listed first; cross-domain fallback fills gaps.
    """

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

        logger.debug("Tokenising %d documents for BM25 index ...", len(docs))
        tokenized = [tokenize(d.raw_text) for d in docs]
        self._bm25 = BM25Okapi(tokenized)

        # company_key (lowercase) → list of doc positions
        self._company_idx: dict[str, list[int]] = {}
        for i, doc in enumerate(docs):
            key = doc.company.lower()
            self._company_idx.setdefault(key, []).append(i)

        logger.debug(
            "Index built. Company buckets: %s",
            {k: len(v) for k, v in self._company_idx.items()},
        )

    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        company: str | None = None,
    ) -> list[SearchResult]:
        """
        Return up to top_k (Document, score) pairs, highest score first.

        Parameters
        ----------
        query   : free-text ticket description or question
        top_k   : maximum results returned
        company : optional filter ("HackerRank", "Claude", "Visa").
                  Company-scoped hits come first; cross-domain fallback
                  fills remaining slots when company docs are scarce.
        """
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        query_token_set = set(tokens)

        # Fetch a wider pool before reranking (3× top_k)
        pool_k = top_k * 3

        if company:
            pool = self._scoped_search(scores, company.lower(), pool_k)
        else:
            pool = self._global_search(scores, pool_k)

        reranked = _rerank(pool, query_token_set)
        return reranked[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scoped_search(
        self,
        scores: list[float],
        company_key: str,
        pool_k: int,
    ) -> list[SearchResult]:
        primary_idx = self._company_idx.get(company_key, [])
        primary = sorted(primary_idx, key=lambda i: scores[i], reverse=True)
        results: list[SearchResult] = [
            (self._docs[i], float(scores[i]))
            for i in primary
            if scores[i] > 0
        ]

        # Fill remaining slots from the full corpus (cross-domain fallback)
        if len(results) < pool_k:
            seen = set(primary_idx)
            fallback = [
                (self._docs[i], float(scores[i]))
                for i in range(len(self._docs))
                if i not in seen and scores[i] > 0
            ]
            fallback.sort(key=lambda x: x[1], reverse=True)
            results += fallback

        return results[:pool_k]

    def _global_search(
        self,
        scores: list[float],
        pool_k: int,
    ) -> list[SearchResult]:
        ranked = [
            (self._docs[i], float(scores[i]))
            for i in range(len(self._docs))
            if scores[i] > 0
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:pool_k]
