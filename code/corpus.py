"""
corpus.py — load and parse the local support article corpus.

Each .md file under data/ is parsed into a Document. All three corpora
(HackerRank, Claude, Visa) use YAML frontmatter; field names differ slightly
across domains but all are handled here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Map top-level data/ subdirectory name → canonical company label
_COMPANY_MAP: dict[str, str] = {
    "hackerrank": "HackerRank",
    "claude": "Claude",
    "visa": "Visa",
}

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Document:
    doc_id: str          # filename stem (unique enough within a corpus)
    title: str           # article title from frontmatter or first heading
    source_url: str      # canonical URL from frontmatter
    breadcrumbs: list[str]  # navigation path, e.g. ["Screen", "Best Practice Guides"]
    body: str            # article body with frontmatter stripped
    raw_text: str        # title + breadcrumbs + body — used for BM25 indexing
    company: str         # "HackerRank" | "Claude" | "Visa"
    category: str        # leaf breadcrumb or folder name
    file_path: str       # absolute path as string
    description: str = field(default="")  # present in Visa docs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (metadata_dict, body_text). Gracefully handles missing frontmatter."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        meta: dict = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, text[m.end():]


def _company_from_path(file_path: Path, data_dir: Path) -> str:
    """Derive company label from the file's position inside data/."""
    try:
        rel = file_path.relative_to(data_dir)
        key = rel.parts[0].lower()
        return _COMPANY_MAP.get(key, key.title())
    except ValueError:
        return "Unknown"


def _first_heading(text: str) -> str:
    """Return the text of the first Markdown heading found in body."""
    m = re.search(r"^#{1,3}\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _category_from_path(file_path: Path, data_dir: Path) -> str:
    """
    Derive a category string from the file path when breadcrumbs are absent
    (primarily Visa docs). Uses the immediate parent folder name.
    """
    try:
        rel = file_path.relative_to(data_dir)
        # rel.parts: ("visa", "support", "consumer", "file.md")
        # parent name is parts[-2] when there are at least 2 parts
        parts = rel.parts
        return parts[-2] if len(parts) >= 2 else parts[0]
    except ValueError:
        return "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_corpus(data_dir: Path) -> list[Document]:
    """
    Recursively load all .md files under data_dir and return a list of Documents.

    BM25 index text is built by combining title (doubled for weight),
    breadcrumbs, description, and the full body — all lowercase is handled
    at query time so the raw_text keeps original casing for readability.
    """
    docs: list[Document] = []

    for md_file in sorted(data_dir.rglob("*.md")):
        raw = md_file.read_text(encoding="utf-8", errors="ignore")
        meta, body = _parse_frontmatter(raw)

        company = _company_from_path(md_file, data_dir)

        title = str(meta.get("title") or _first_heading(body) or md_file.stem)
        source_url = str(meta.get("source_url") or "")
        breadcrumbs: list[str] = list(meta.get("breadcrumbs") or [])
        description = str(meta.get("description") or "")

        if breadcrumbs:
            category = breadcrumbs[-1]
        else:
            category = _category_from_path(md_file, data_dir)

        # Index text: title is repeated to boost its weight in BM25 scoring.
        # breadcrumbs help match product-area terminology; description adds
        # Visa-specific context that isn't always in the body.
        raw_text = " ".join([
            title, title,
            " ".join(breadcrumbs),
            description,
            body,
        ])

        docs.append(Document(
            doc_id=md_file.stem,
            title=title,
            source_url=source_url,
            breadcrumbs=breadcrumbs,
            body=body,
            raw_text=raw_text,
            company=company,
            category=category,
            file_path=str(md_file),
            description=description,
        ))

    return docs
