"""
normalizer.py — deterministic text normalization for BM25 retrieval.

Provides a suffix-based stemmer (no external dependencies) plus a tokenizer
that returns stemmed tokens. Applying the same normalizer at both index time
and query time is what makes vocabulary match across morphological variants
(e.g. "conversations" and "conversation" collapse to the same stem).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Suffix rules — longest first so shorter rules don't shadow longer ones.
# Tuple: (suffix, replacement, min_stem_length)
# min_stem_length guards against over-stripping short words like "sting"→"".
# ---------------------------------------------------------------------------
_SUFFIX_RULES: list[tuple[str, str, int]] = [
    # 7-char suffixes
    ("ational", "ate", 4),
    # 6-char suffixes
    ("ations",  "ate", 4),
    ("nesses",  "",    4),
    ("izable",  "ize", 4),
    # 5-char suffixes
    ("ation",   "ate", 4),
    ("alize",   "al",  4),
    ("ments",   "",    4),
    ("izing",   "ize", 4),
    ("ising",   "ise", 4),
    ("iness",   "y",   4),
    # 4-char suffixes
    ("ment",    "",    4),
    ("ness",    "",    4),
    ("ying",    "y",   3),
    ("ting",    "",    4),
    ("sing",    "",    4),
    ("ring",    "",    4),
    ("ning",    "",    4),
    ("ding",    "",    4),
    ("king",    "",    4),
    ("ling",    "",    4),
    ("ving",    "",    4),
    ("able",    "",    4),
    ("ible",    "",    4),
    # 3-char suffixes
    ("ing",     "",    4),
    ("ies",     "y",   3),
    ("ied",     "y",   3),
    ("ion",     "",    4),
    ("ers",     "",    4),
    # 2-char suffixes
    ("ed",      "",    4),
    ("er",      "",    4),
    ("es",      "",    4),
    ("ly",      "",    4),
    # 1-char suffix — only strip plural-s from longer words
    ("s",       "",    5),
]

# Common stop-words to drop from BM25 tokens (they inflate IDF noise)
_STOP_WORDS: frozenset[str] = frozenset([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "i", "my", "we", "our",
    "you", "your", "he", "she", "they", "this", "that", "these", "those",
    "be", "are", "was", "were", "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "may", "might", "not", "no", "so",
    "as", "if", "up", "out", "into", "about", "what", "which", "who",
    "how", "when", "where", "why", "all", "any", "some", "more", "also",
    "been", "than", "then", "very", "just", "me", "him", "her", "us",
    "them", "their", "there", "here", "get", "got",
])


def stem(token: str) -> str:
    """
    Apply the first matching suffix rule and return the resulting stem.
    Idempotent and deterministic: same input always produces same output.
    """
    for suffix, replacement, min_len in _SUFFIX_RULES:
        if token.endswith(suffix):
            stem_candidate = token[: -len(suffix)] + replacement
            if len(stem_candidate) >= min_len:
                return stem_candidate
    return token


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Lowercase → extract alphanumeric tokens → stem → optionally drop stop-words.

    De-duplicates: if the original and stemmed form are distinct, both are
    kept so the index contains the exact surface form as well as its stem.
    This maximises recall without sacrificing precision on exact matches.
    """
    raw_tokens: list[str] = re.findall(r"[a-z0-9]+", text.lower())

    seen: set[str] = set()
    result: list[str] = []

    for t in raw_tokens:
        if remove_stopwords and t in _STOP_WORDS:
            continue
        s = stem(t)
        for variant in (t, s):
            if variant and variant not in seen:
                result.append(variant)
                seen.add(variant)

    return result
