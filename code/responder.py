"""
responder.py — grounded response generation for the support triage agent.

Hybrid architecture:
  1. If an API key is configured, Claude Haiku synthesises a concise, accurate
     response from the top retrieved corpus documents (RAG, temperature=0).
     The provider is chosen by which key is set: OpenRouter if
     OPENROUTER_API_KEY is present, otherwise Anthropic via ANTHROPIC_API_KEY.
     There is no failover between the two providers.
  2. Falls back to deterministic extractive passage retrieval when no key is
     present, so the system works fully offline with zero API dependencies.

Both paths ground every word in retrieved corpus documents — the LLM is
explicitly instructed to use ONLY the provided articles, and the extractive
path generates no text at all.
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path

from corpus import Document
from normalizer import tokenize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixed response strings
# ---------------------------------------------------------------------------

_OUT_OF_SCOPE = (
    "Thank you for reaching out. This request is outside the scope of what "
    "our support team can assist with. If you have a question about "
    "HackerRank, Claude, or Visa, please describe your issue and we will "
    "be happy to help."
)

_ESCALATION_ACK = (
    "Thank you for contacting support. Your request has been escalated to "
    "our team and a human agent will follow up with you as soon as possible."
)


# ---------------------------------------------------------------------------
# Markdown cleaning
# ---------------------------------------------------------------------------

_IMAGE_RE          = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE           = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HEADING_RE        = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_BLANK_RE          = re.compile(r"\n{3,}")
_LAST_UPDATED_RE   = re.compile(r"_Last updated:.*?_\s*\n?", re.I | re.DOTALL)
_RELATED_RE        = re.compile(r"\n\s*\*?\s*Related Articles?\s*\*?\s*\n.*", re.DOTALL | re.I)
_TRAILING_BS_RE    = re.compile(r"^\\\s*$", re.MULTILINE)
_EMOJI_LINK_RE     = re.compile(r"[\U0001F4C4\U0001F4C5\U0001F4CB]\s*\S+")  # 📄📅📋 links
_HR_LINE_RE        = re.compile(r"^-{3,}$", re.MULTILINE)
_BOLD_RE           = re.compile(r"\*\*([^*\n]+)\*\*")   # **bold** → bold
_ITALIC_RE         = re.compile(r"\*([^*\n]+)\*")         # *italic* → italic


def _clean(text: str) -> str:
    """Strip markdown noise, timestamps, Related Articles sections, and artifact lines."""
    text = _IMAGE_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _HEADING_RE.sub("", text)
    text = _LAST_UPDATED_RE.sub("", text)   # strip "Last updated: ..." timestamps
    text = _RELATED_RE.sub("", text)         # strip Related Articles sections
    text = _TRAILING_BS_RE.sub("", text)     # strip lone backslash artifact lines
    text = _EMOJI_LINK_RE.sub("", text)      # strip 📄 emoji article links
    text = _HR_LINE_RE.sub("", text)         # strip --- dividers
    text = _BOLD_RE.sub(r"\1", text)         # **bold** → bold (clean for CSV)
    text = _ITALIC_RE.sub(r"\1", text)       # *italic* → italic
    text = _BLANK_RE.sub("\n\n", text)
    return text.strip()


# Boilerplate phrases the LLM occasionally produces despite explicit prohibition.
# These are safe to strip because they add no information to the response.
_BOILERPLATE_RES: list[re.Pattern] = [
    re.compile(
        r"based on the (?:"
        r"information provided|"
        r"provided[,.]|"                                                   # "based on the provided,"
        r"information (?:provided )?in the (?:reference |knowledge base |)?articles?|"
        r"(?:provided |reference |knowledge base |)?(?:articles?|information)"
        r")[,.]?\s*",
        re.I,
    ),
    re.compile(r"according to the (?:provided |reference |)?(?:articles?|information)[,.]?\s*", re.I),
    re.compile(
        r"(?:the |these )?(?:provided |reference )?articles? (?:above |provided |)?"
        r"(?:indicate|state|show|say|mention|note|suggest)[,.]?\s*",
        re.I,
    ),
    re.compile(r"using (?:only )?the (?:provided |reference )?(?:articles?|information)[,.]?\s*", re.I),
    re.compile(r"here (?:is|are) (?:a |the )?(?:helpful |professional )?(?:response|answer|information)[:.]\s*", re.I),
]

# Consecutive word-repetition pattern (hallucination / garbled output signal)
_STUTTER_RE = re.compile(r"\b(\w{3,})\s+\1\b", re.I)


def _is_garbled(text: str) -> bool:
    """Return True if the response shows signs of LLM hallucination / garbling."""
    stutters = _STUTTER_RE.findall(text)
    return len(stutters) >= 3


def _strip_boilerplate(text: str) -> str:
    """
    Remove LLM meta-commentary from the response opening.

    Only operates on the first ~150 chars of the text to avoid accidentally
    corrupting legitimate in-sentence references further into the response.
    """
    cut = 150
    prefix = text[:cut]
    suffix = text[cut:]

    for pat in _BOILERPLATE_RES:
        m = pat.search(prefix)
        if m:
            prefix = prefix[:m.start()] + prefix[m.end():]
            break  # one pass is enough; avoid cascading replacements

    result = (prefix + suffix).strip()
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    return result


# ---------------------------------------------------------------------------
# LLM-backed generation — OpenRouter (primary) or Anthropic (fallback)
# ---------------------------------------------------------------------------

_OPENROUTER_BASE    = "https://openrouter.ai/api/v1"
_OPENROUTER_MODEL   = "anthropic/claude-haiku-4.5"
_ANTHROPIC_MODEL    = "claude-haiku-4-5-20251001"
_MAX_CHARS_PER_DOC  = 1500

_SYSTEM_PROMPT = (
    "You are a professional customer support agent. Use ONLY the reference "
    "articles provided to answer the customer's question.\n\n"
    "MANDATORY RULES — violations will be rejected:\n"
    "1. DO NOT use these phrases anywhere in your response (not even in the "
    "middle of a sentence):\n"
    "   'Based on the knowledge base', 'Based on the information provided',\n"
    "   'According to the provided', 'According to the information',\n"
    "   'knowledge base articles', 'provided articles', 'provided information',\n"
    "   'the articles above', 'the information in the articles'.\n"
    "2. Start directly with the answer. No preamble, no meta-commentary.\n"
    "3. Do not invent facts. Do not use outside knowledge.\n"
    "4. If the articles do not fully cover the question, give what is most "
    "relevant and acknowledge the gap in one short sentence.\n"
    "5. Be concise (under 200 words), professional, empathetic, and direct.\n"
    "6. Do not mention article titles, document names, or that you consulted "
    "any reference material.\n"
    "7. Use plain prose only — no markdown asterisks for bold, no underscores "
    "for italics, no code fences. Numbered lists are acceptable for steps."
)

# Self-evaluation prompt — scores response relevance 1–5.
_EVAL_SYSTEM = (
    "You are a support-response quality evaluator. "
    "Rate how well the RESPONSE answers the QUESTION. "
    "Reply with a single digit only:\n"
    "1=completely wrong or irrelevant, 2=mostly wrong, "
    "3=partially correct, 4=mostly correct, 5=perfectly answers. "
    "Nothing else — just the single digit."
)


def _load_credentials() -> tuple[str, str] | tuple[None, None]:
    """
    Return (api_key, provider) where provider is 'openrouter' or 'anthropic'.
    Checks environment variables first, then .env file in the repo root.
    Returns (None, None) if no key is found.
    """
    sources = [
        ("OPENROUTER_API_KEY", "openrouter"),
        ("ANTHROPIC_API_KEY",  "anthropic"),
    ]

    # 1. Check live environment
    for var, provider in sources:
        val = os.environ.get(var, "").strip()
        if val:
            return val, provider

    # 2. Parse .env file
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return None, None
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for var, provider in sources:
                if line.startswith(f"{var}="):
                    val = line.split("=", 1)[1].strip().strip("\"'")
                    if val:
                        return val, provider
    except OSError:
        pass

    return None, None


def _format_docs_for_llm(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs[:3], 1):
        body = _clean(doc.body)[:_MAX_CHARS_PER_DOC]
        parts.append(f"[Article {i}: {doc.title}]\n{body}")
    return "\n\n---\n\n".join(parts)


def _call_openrouter(api_key: str, system: str, user: str) -> str | None:
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=_OPENROUTER_BASE,
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://hackerrank.com",
                "X-Title": "HackerRank Support Triage Agent",
            },
        )
        resp = client.chat.completions.create(
            model=_OPENROUTER_MODEL,
            temperature=0,
            max_tokens=450,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("OpenRouter call failed: %s", exc)
        return None


def _call_anthropic(api_key: str, system: str, user: str) -> str | None:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=450,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text.strip()
    except Exception as exc:
        logger.warning("Anthropic call failed: %s", exc)
        return None


def _quality_score(issue: str, response: str, api_key: str, provider: str) -> int:
    """
    Self-evaluate the LLM response against the original question.

    Returns a relevance score 1–5. Falls back to 3 (neutral) on any failure
    so that self-evaluation never blocks a usable response.
    Only called when retrieval confidence is weak (< 0.45) — avoids doubling
    API calls for strong retrievals that need no verification.
    """
    prompt = f"QUESTION: {issue[:400]}\n\nRESPONSE: {response[:800]}"
    try:
        if provider == "openrouter":
            raw = _call_openrouter(api_key, _EVAL_SYSTEM, prompt)
        else:
            raw = _call_anthropic(api_key, _EVAL_SYSTEM, prompt)
        if raw:
            m = re.search(r"[1-5]", raw.strip()[:20])
            if m:
                return int(m.group())
    except Exception:
        pass
    return 3  # neutral — don't block the response


def _generate_llm_response(
    issue: str,
    subject: str,
    company: str | None,
    top_docs: list[Document],
    confidence: float = 1.0,
) -> str | None:
    """
    Synthesise a grounded response via Claude Haiku (temperature=0).
    Uses OpenRouter if OPENROUTER_API_KEY is set, otherwise Anthropic direct
    (no failover between them). Returns None on failure so the caller can
    fall back to extractive retrieval.
    """
    api_key, provider = _load_credentials()
    if not api_key:
        logger.debug("No API key found — using extractive fallback")
        return None

    articles    = _format_docs_for_llm(top_docs)
    company_str = company or "our platform"
    ticket_text = f"Subject: {subject}\n{issue}" if subject.strip() else issue

    user_message = (
        f"REFERENCE ARTICLES:\n{articles}\n\n"
        f"CUSTOMER TICKET ({company_str} support):\n{ticket_text}\n\n"
        f"Write a professional, direct support response using only the "
        f"reference articles. Do not mention the articles or the knowledge "
        f"base. Under 200 words."
    )

    if provider == "openrouter":
        text = _call_openrouter(api_key, _SYSTEM_PROMPT, user_message)
    else:
        text = _call_anthropic(api_key, _SYSTEM_PROMPT, user_message)

    if text and len(text) >= 20:
        if _is_garbled(text):
            logger.warning("LLM response appears garbled — using extractive fallback")
            return None
        text = _strip_boilerplate(text)
        if len(text) < 20:
            return None
        # Self-evaluation: only for shaky retrievals — avoids doubling API calls
        # on strong matches that don't need verification.
        if confidence < 0.45:
            score = _quality_score(issue, text, api_key, provider)
            logger.debug("Self-eval quality: %d/5 (conf=%.2f)", score, confidence)
            if score <= 1:
                logger.warning("Self-eval: response irrelevant (score=%d) — extractive fallback", score)
                return None
        logger.debug("LLM response: %d chars via %s/%s", len(text), provider,
                     _OPENROUTER_MODEL if provider == "openrouter" else _ANTHROPIC_MODEL)
        return text

    logger.warning("LLM returned empty/short response — using extractive fallback")
    return None


# ---------------------------------------------------------------------------
# Extractive passage retrieval (deterministic offline fallback)
# ---------------------------------------------------------------------------

_MIN_SUBSTANTIVE_CHARS = 80


def _score_paragraph(para: str, query_token_set: set[str]) -> float:
    tokens  = tokenize(para)
    overlap = sum(1 for t in tokens if t in query_token_set)
    if not overlap:
        return 0.0
    word_count = len(para.split())
    score = overlap / math.log(word_count + 2)
    if len(para) < _MIN_SUBSTANTIVE_CHARS:
        score *= 0.25
    return score


def _is_metadata_line(para: str) -> bool:
    """Return True for non-informative metadata or navigation paragraphs."""
    p = para.strip()
    if re.match(r"_Last updated:", p, re.I):
        return True
    if re.match(r"\*?\s*Related Articles?", p, re.I):
        return True
    if re.match(r"^[\\—-]+$", p):
        return True
    # Short navigation labels: "Get support", "FAQ", "Merchants", "Visa Concierge"
    # Keep lines with '?' (FAQ questions) or long substantive lines.
    words = p.split()
    if len(words) <= 4 and "?" not in p and not any(c in p for c in ".!,;:("):
        return True
    return False


def extract_passage(
    doc: Document,
    query: str,
    max_chars: int = 1000,
) -> str:
    """
    Return the most query-relevant passage from doc.body.

    Algorithm
    ---------
    1. Clean markdown from the body (strips timestamps, bold, Related Articles).
    2. Strip leading title line if the body opens with the article title.
    3. If the body is short enough, return it in full.
    4. Otherwise score each paragraph by query-token overlap and select
       the top paragraphs that fit within max_chars, re-ordered by original
       position so the response reads coherently.
    """
    body = _clean(doc.body)

    # Strip the article title if it appears verbatim as the first line of the body
    # (common when markdown H1 is the first element).
    title_stripped = doc.title.strip()
    if body.startswith(title_stripped):
        body = body[len(title_stripped):].lstrip("\n ")

    if len(body) <= max_chars:
        logger.debug("Returning full body (%d chars) from '%s'", len(body), doc.title[:50])
        return body

    query_tokens = set(tokenize(query))
    paragraphs   = [p.strip() for p in body.split("\n\n") if p.strip() and not _is_metadata_line(p)]

    if not paragraphs:
        return body[:max_chars]

    scored = sorted(
        enumerate(paragraphs),
        key=lambda x: (_score_paragraph(x[1], query_tokens), -x[0]),
        reverse=True,
    )

    selected: list[tuple[int, str]] = []
    total_chars = 0
    for original_idx, para in scored:
        needed = len(para) + (2 if selected else 0)
        if total_chars + needed > max_chars:
            continue
        selected.append((original_idx, para))
        total_chars += needed

    if not selected:
        return paragraphs[0][:max_chars]

    selected.sort(key=lambda x: x[0])
    result = "\n\n".join(p for _, p in selected)
    logger.debug(
        "Extracted %d paragraph(s) / %d chars from '%s'",
        len(selected), len(result), doc.title[:50],
    )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_response(
    issue: str,
    status: str,
    request_type: str,
    top_docs: list[Document],
    confidence: float,
    company: str | None = None,
    subject: str = "",
) -> str:
    """
    Produce the final user-facing response string.

    Decision table
    ──────────────
    status=escalated                   → escalation acknowledgement
    status=replied, type=invalid       → out-of-scope message
    status=replied, API key present    → LLM-synthesised grounded response
    status=replied, no API key / fail  → extractive passage from top doc
    """
    if status == "escalated":
        return _ESCALATION_ACK

    if request_type == "invalid":
        return _OUT_OF_SCOPE

    if not top_docs or confidence < 0.12:
        logger.warning(
            "Falling back to escalation ACK (no docs or very low confidence %.3f)",
            confidence,
        )
        return _ESCALATION_ACK

    # Try LLM synthesis first (best accuracy, still grounded in corpus)
    llm_response = _generate_llm_response(issue, subject, company, top_docs, confidence)
    if llm_response:
        return llm_response

    # Deterministic extractive fallback
    passage = extract_passage(top_docs[0], issue)
    return passage or _OUT_OF_SCOPE
