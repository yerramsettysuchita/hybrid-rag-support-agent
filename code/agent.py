"""
agent.py — pipeline orchestrator for the HackerRank Orchestrate support agent.

Wires together:
  corpus.load_corpus  ->  BM25Retriever.search  ->  triage  ->  build_response

Each call to SupportAgent.process() emits structured DEBUG log lines covering:
  - raw ticket fields
  - top-k retrieved documents with scores
  - triage decision (status, type, area, confidence)
  - escalation reason (when applicable)

Logging is controlled from the caller (main.py) via logging.basicConfig.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from corpus import load_corpus
from responder import _load_credentials, _call_openrouter, _call_anthropic, build_response
from retriever import BM25Retriever
from triage import TriageResult, _classify_request_type, compute_confidence, triage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic query seeding — pre-retrieval intent injection
# ---------------------------------------------------------------------------

# Intent signals and the BM25-friendly terms they inject.
# Runs before LLM expansion so the raw query already reflects the user's intent.
_SEED_RULES: list[tuple[list[str], str]] = [
    (["refund", "money back", "reimburs", "charged wrongly", "charge me"],
     "billing payment subscription refund"),
    (["cancel", "cancellation", "unsubscribe"],
     "subscription cancel billing plan"),
    (["pause", "pause subscription", "pausing"],
     "subscription pause billing plan"),
    (["invoice", "receipt", "billing statement"],
     "billing invoice subscription payment"),
    (["forgot password", "reset password", "password reset"],
     "account login password reset"),
    (["delete account", "close my account", "remove my account"],
     "account delete close subscription"),
]


def _seed_query(base_query: str, issue: str, company: str | None) -> str:
    """
    Inject intent-specific vocabulary into the raw BM25 query without an LLM call.

    The returned string appends corpus-vocabulary terms for signals (like 'refund')
    that the user expresses in natural language but that may not match the exact
    words used in help-article titles.  The original query is never replaced —
    seeds are appended so existing strong matches are not disrupted.
    """
    issue_lower = issue.lower()
    appended: list[str] = []
    for signals, terms in _SEED_RULES:
        if any(sig in issue_lower for sig in signals):
            appended.append(terms)
    if appended:
        seeded = base_query + " " + " ".join(appended)
        logger.debug("SEED    query   : %s", seeded[:120])
        return seeded
    return base_query


# ---------------------------------------------------------------------------
# Query expansion — pre-retrieval LLM step
# ---------------------------------------------------------------------------

_EXPANSION_SYSTEM = (
    "You are a search query optimizer for a customer support retrieval system. "
    "Your job is to expand a support ticket into a richer search query that will "
    "surface the most relevant help articles. Output ONLY the expanded query — "
    "no explanation, no punctuation, no labels."
)


def _expand_query(
    issue: str,
    subject: str,
    company: str | None,
) -> str:
    """
    Use Claude Haiku to expand the raw ticket into a retrieval-optimised query.

    Adds synonyms, related product terms, and clarifying context so BM25
    can surface articles the literal ticket text would miss.
    Falls back to the original combined text if no API key is available or
    the call fails — retrieval continues normally in either case.
    """
    api_key, provider = _load_credentials()
    if not api_key:
        return f"{subject} {issue}".strip() if subject else issue

    company_str = company or "unknown"
    raw = f"{subject}\n{issue}".strip() if subject else issue

    user_msg = (
        f"Company: {company_str}\n"
        f"Ticket:\n{raw}\n\n"
        "Expand this into a 1–2 sentence search query that includes the specific "
        "product feature, relevant synonyms, and the user's core need. "
        "Do not include personal details or issue numbers."
    )

    # Don't spend an API call expanding tickets that are already invalid
    # (pattern-matched injection / trivia / greeting) — retrieval order doesn't
    # affect the response, but it does affect the area via _area_for_invalid().
    # Keeping the raw query preserves deterministic area assignment.
    if _classify_request_type(issue) == "invalid":
        logger.debug("Skipping query expansion for invalid ticket")
        return raw

    try:
        if provider == "openrouter":
            result = _call_openrouter(api_key, _EXPANSION_SYSTEM, user_msg)
        else:
            result = _call_anthropic(api_key, _EXPANSION_SYSTEM, user_msg)

        if result and len(result) >= 10:
            logger.debug("Query expanded: %d -> %d chars", len(raw), len(result))
            return result
    except Exception as exc:
        logger.warning("Query expansion failed: %s", exc)

    return raw


# ---------------------------------------------------------------------------
# Output schema — mirrors the output.csv column order
# ---------------------------------------------------------------------------

@dataclass
class AgentOutput:
    issue: str
    subject: str
    company: str
    status: str          # "replied" | "escalated"
    product_area: str
    response: str
    request_type: str    # "product_issue" | "feature_request" | "bug" | "invalid"
    justification: str
    # Decision trace fields
    confidence: float    # 0.0 - 1.0 retrieval confidence
    fingerprint: str     # 8-char SHA256 of the decision path
    rationale: str       # one-line human-readable decision explanation
    top_doc_title: str   # title of top retrieved document (empty if none)
    top_doc_url: str     # URL of top retrieved document (empty if none)
    summary: str         # compact one-liner for display and interview use


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SupportAgent:
    """
    Loads the corpus once, then processes tickets on demand.

    Parameters
    ----------
    data_dir : path to the repo-root data/ directory
    top_k    : number of documents to retrieve per ticket
    """

    def __init__(self, data_dir: Path, top_k: int = 5) -> None:
        self._top_k = top_k

        logger.info("Loading corpus from %s", data_dir)
        docs = load_corpus(data_dir)
        by_company = {
            c: sum(1 for d in docs if d.company == c)
            for c in ("HackerRank", "Claude", "Visa")
        }
        logger.info(
            "Corpus: %d documents  %s",
            len(docs),
            "  ".join(f"{c}={n}" for c, n in by_company.items()),
        )

        logger.info("Building BM25 index ...")
        self._retriever = BM25Retriever(docs)
        logger.info("Index ready.")

    # ------------------------------------------------------------------

    def process(
        self,
        issue: str,
        subject: str = "",
        company: str | None = None,
    ) -> AgentOutput:
        """
        Process a single support ticket end-to-end.

        Parameters
        ----------
        issue   : ticket body (required)
        subject : ticket subject line (optional; prepended to query)
        company : "HackerRank" | "Claude" | "Visa" | None
        """
        logger.debug("-" * 60)
        logger.debug("TICKET  issue   : %s", issue[:120])
        logger.debug("        subject : %s", subject[:80] if subject else "(none)")
        logger.debug("        company : %s", company or "None")

        # -- 1. Build raw query, seeding intent terms for common patterns --------
        # Deterministic keyword seeding runs before BM25 and before any LLM call.
        # It ensures that billing/refund intents are reflected in the query even
        # when the ticket's wording doesn't contain the exact corpus vocabulary.
        base = f"{subject} {issue}".strip() if subject else issue
        raw_query = _seed_query(base, issue, company)
        results   = self._retriever.search(raw_query, top_k=self._top_k, company=company)
        query     = raw_query

        # -- 2. Expand query only when raw retrieval is still weak -------------
        # Threshold 0.45: strong retrievals (>= 0.45) are left untouched so that
        # expansion doesn't dilute a well-matching query.  Weak retrievals are the
        # cases where synonym expansion actually helps (paraphrased intent, rare
        # terminology, or queries that BM25 simply can't anchor to a single doc).
        raw_conf = compute_confidence(results, company)
        if raw_conf < 0.45 and _classify_request_type(issue) != "invalid":
            expanded = _expand_query(issue, subject, company)
            if expanded != raw_query:
                exp_results = self._retriever.search(expanded, top_k=self._top_k, company=company)
                exp_conf    = compute_confidence(exp_results, company)
                if exp_conf > raw_conf:
                    results = exp_results
                    query   = expanded
                    logger.debug("EXPAND  improved conf %.3f -> %.3f  query: %s",
                                 raw_conf, exp_conf, expanded[:100])

        logger.debug("RETRIEVE  %d result(s):", len(results))
        for rank, (doc, score) in enumerate(results, 1):
            logger.debug(
                "  %d. [%-12s] %-60s score=%.2f",
                rank, doc.company, doc.title[:60], score,
            )

        # -- 2. Triage ---------------------------------------------------------
        tr: TriageResult = triage(issue, company, results)

        # -- 3. Build response -------------------------------------------------
        response = build_response(
            issue=issue,
            status=tr.status,
            request_type=tr.request_type,
            top_docs=tr.top_docs,
            confidence=tr.confidence,
            company=company,
            subject=subject,
        )

        # -- 4. Compose justification and trace --------------------------------
        justification = _make_justification(tr)
        rationale     = _make_rationale(tr)
        fingerprint   = _make_fingerprint(
            issue, company, tr.status, tr.request_type, tr.product_area
        )

        top_doc_title = tr.top_docs[0].title      if tr.top_docs else ""
        top_doc_url   = tr.top_docs[0].source_url  if tr.top_docs else ""
        summary       = _make_summary(tr, fingerprint, rationale)

        logger.debug(
            "DECISION  status=%-10s type=%-16s area=%s  conf=%.2f  fp=%s",
            tr.status, tr.request_type, tr.product_area, tr.confidence, fingerprint,
        )
        logger.debug("SUMMARY   %s", summary)

        return AgentOutput(
            issue=issue,
            subject=subject,
            company=company or "None",
            status=tr.status,
            product_area=tr.product_area,
            response=response,
            request_type=tr.request_type,
            justification=justification,
            confidence=tr.confidence,
            fingerprint=fingerprint,
            rationale=rationale,
            top_doc_title=top_doc_title,
            top_doc_url=top_doc_url,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# Justification builder
# ---------------------------------------------------------------------------

def _escalation_category(reason: str) -> str:
    r = reason.lower()
    if "outage" in r or "site-wide" in r or "down" in r:
        return "active_outage"
    if "security" in r or "safety" in r or "identity" in r or "breach" in r or "bounty" in r:
        return "security_concern"
    if "cannot be fulfilled" in r or "structurally" in r:
        return "outside_scope"
    if "insufficient corpus" in r or "confidence=" in r:
        return "low_confidence"
    return "requires_review"


def _make_justification(tr: TriageResult) -> str:
    if tr.status == "escalated":
        reason = tr.escalation_reason or "requires human review"
        category = _escalation_category(reason)
        return (
            f"Escalated [{category}]: {reason}. "
            f"Retrieval confidence={tr.confidence:.2f}."
        )

    if tr.request_type == "invalid":
        return "Ticket is out of scope for this support domain; replied with out-of-scope message."

    if tr.top_docs:
        src_title = tr.top_docs[0].title
        src_url   = tr.top_docs[0].source_url
        citation  = f"'{src_title}'" + (f" ({src_url})" if src_url else "")
        return (
            f"Answered using corpus document {citation}. "
            f"Confidence={tr.confidence:.2f}. "
            f"Type: {tr.request_type}."
        )

    return f"Replied with best available answer. Confidence={tr.confidence:.2f}."


# ---------------------------------------------------------------------------
# Decision trace helpers
# ---------------------------------------------------------------------------

def _make_summary(tr: TriageResult, fingerprint: str, rationale: str) -> str:
    """Compact one-liner: bracket prefix with routing facts, then the rationale."""
    prefix = (
        f"[{tr.status} | {tr.request_type} | {tr.product_area}"
        f" | conf={tr.confidence:.2f} | fp={fingerprint}]"
    )
    return f"{prefix} {rationale}"


def explain(out: AgentOutput) -> str:
    """
    Return a multi-line interview-ready explanation for a single AgentOutput.

    Designed to be called interactively (e.g. print(explain(out))) so a
    developer can quickly narrate a decision to a judge.
    """
    lines = [
        f"Fingerprint : {out.fingerprint}",
        f"Decision    : {out.status} ({out.request_type}) -> {out.product_area}",
        f"Confidence  : {out.confidence:.2f}",
    ]
    if out.top_doc_title:
        doc_ref = f"'{out.top_doc_title}'"
        if out.top_doc_url:
            doc_ref += f"  <{out.top_doc_url}>"
        lines.append(f"Top document: {doc_ref}")
    lines.append(f"Rationale   : {out.rationale}")
    return "\n".join(lines)


def _make_rationale(tr: TriageResult) -> str:
    if tr.request_type == "invalid":
        return "Invalid request -- out-of-scope, adversarial, or greeting."
    if tr.status == "escalated":
        reason = tr.escalation_reason or "requires human review"
        return f"Escalated: {reason}."
    if tr.top_docs:
        doc = tr.top_docs[0]
        return (
            f"Replied using '{doc.title[:60]}' "
            f"(conf={tr.confidence:.2f}, area={tr.product_area})."
        )
    return f"Replied with best-effort response (conf={tr.confidence:.2f})."


def _make_fingerprint(
    issue: str,
    company: str | None,
    status: str,
    request_type: str,
    product_area: str,
) -> str:
    key = "|".join([
        issue.lower()[:100],
        (company or "").lower(),
        status,
        request_type,
        product_area,
    ])
    return hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:8]
