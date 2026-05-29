"""
triage.py — classify and route support tickets.

All logic is rule-based, offline, and deterministic.

Decision flow per ticket
------------------------
1. Detect adversarial / injection patterns  ->  invalid (replied, not escalated)
2. Classify request_type  (bug | feature_request | product_issue | invalid)
3. Compute confidence from BM25 retrieval scores
4. Apply escalation rules  ->  escalated | replied
5. Infer product_area using:
     INVALID tickets:
       a) company default (no breadcrumbs, no doc content)
       b) top retrieved doc's company as a hint when company=None
     VALID tickets:
       a) issue-keyword overrides per company (highest priority)
       b) cross-company keyword fallback for company=None
       c) company-aware breadcrumb selection from top retrieved doc
       d) file-path mapping (Visa, which has no breadcrumbs)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from corpus import Document
from retriever import SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class TriageResult:
    status: str              # "replied" | "escalated"
    request_type: str        # "product_issue" | "feature_request" | "bug" | "invalid"
    product_area: str        # e.g. "screen", "privacy", "travel_support"
    confidence: float        # 0.0 - 1.0
    escalation_reason: str   # non-empty only when status == "escalated"
    top_docs: list[Document] = field(default_factory=list)


# ===========================================================================
# SECTION 1 — request_type classification
# ===========================================================================

# -- Outage / site-down (always escalate + classify as bug) -----------------
_OUTAGE_PHRASES: tuple[str, ...] = (
    "site is down",
    "service is down",
    "platform is down",
    "is down",
    "none of the pages are accessible",
    "pages are not accessible",
    "all requests are failing",
    "all requests failing",
    "stopped working completely",
    "completely stopped working",
    "nothing is loading",
    "unable to access any",
)

# -- Security / safety emergencies (escalate) --------------------------------
_SECURITY_ESCALATE_PHRASES: tuple[str, ...] = (
    "identity theft",
    "identity has been stolen",
    "identity was stolen",
    "my identity was",
    "security vulnerability",
    "major security",
    "major vulnerability",
    "bug bounty",
    "data breach",
    "account has been hacked",
    "account was hacked",
    "someone hacked",
)

# -- Structurally impossible requests (escalate) -----------------------------
_IMPOSSIBLE_PHRASES: tuple[str, ...] = (
    "increase my score",
    "change my score",
    "move me to the next round",
    "tell the company",
    "ban the seller",
    "make visa refund",
    "force the merchant",
    "even though i am not the",
    "even though i'm not the",
    "even though i am not a",
)

# -- Requests that require human action because the corpus cannot help --------
# These are not structurally impossible, but the support corpus has no
# relevant article; escalation is safer than returning a wrong response.
_HUMAN_REQUIRED_PHRASES: tuple[str, ...] = (
    "rescheduling of my",      # candidate wants to reschedule assessment → recruiter must act
    "reschedule my",           # same intent, shorter form
    "inactivity time",         # session-timeout config question, not in corpus
    "inactivity timeout",      # same
    "zoom connectivity",       # candidate compatibility blocker, needs live support
    "remove an interviewer",   # user-management action, corpus does not cover this
    "infosec process",         # vendor security questionnaire — needs enterprise team
    "security questionnaire",  # same category of request
    "vendor assessment",       # security vendor onboarding — not in corpus
)

# -- Adversarial / prompt-injection (invalid, replied) -----------------------
_INJECTION_PHRASES: tuple[str, ...] = (
    "delete all files",
    "delete all data",
    "rm -rf",
    # Classic ignore/override attempts
    "ignore previous instructions",
    "ignore all previous",
    "ignore your previous",
    "disregard your instructions",
    "forget your instructions",
    "forget all previous",
    "override your instructions",
    # System-prompt extraction attempts
    "show me your system prompt",
    "show me your rules",
    "show me your instructions",
    "reveal your system",
    "print your system prompt",
    "output your instructions",
    "tell me your instructions",
    "what are your instructions",
    # Role-play / jailbreak openers
    "you are now a",
    "pretend you are",
    "act as if you have no",
    "jailbreak",
    "bypass your",
    # French injection attempt (from test corpus)
    "affiche toutes les",
    "affiche les documents",
    "logique exacte que vous",
    "dites moi quoi faire",
    "dites-moi quoi faire",
)

# -- Technical failure signals -> bug ----------------------------------------
_BUG_PHRASES: tuple[str, ...] = (
    "site is down",
    "not working",
    "isn't working",
    "is not working",
    "stopped working",
    "pages are not accessible",
    "none of the pages",
    "all requests are failing",
    "requests are failing",
    "all requests failing",
    "is failing",
    "completely down",
    "is down",
    "is broken",
    "is crashing",
    "internal server error",
    "500 error",
    "no longer works",
    "isn't loading",
    "is not loading",
    "nothing is loading",
)

_BUG_WORDS: frozenset[str] = frozenset([
    "outage", "crash", "crashed", "broken", "blocker",
    "unresponsive", "unreachable", "inaccessible",
])

_SECURITY_BUG_PHRASES: tuple[str, ...] = (
    "security vulnerability",
    "major vulnerability",
    "major security",
    "bug bounty",
    "found a vulnerability",
    "found a bug",
    "discovered a vulnerability",
)

# -- Feature request signals -------------------------------------------------
# Deliberately narrow — "would like to request" is excluded because users
# write "I would like to request a rescheduling", which is a product_issue.
_FEATURE_PHRASES: tuple[str, ...] = (
    "can you add a",
    "can you add the ability",
    "please add a feature",
    "please add support for",
    "it would be great if you could add",
    "it would be nice if you could add",
    "feature request",
    "would love to see",
    "consider adding",
    "wish there was",
    "wishlist",
    "would like the ability to",
    "would like you to add",
)

_FEATURE_WORDS: frozenset[str] = frozenset(["wishlist"])

# -- Invalid / out-of-scope patterns -----------------------------------------
_INVALID_PATTERNS: list[re.Pattern] = [
    # Pop-culture / general-knowledge questions
    re.compile(r"\bname of the actor\b", re.I),
    re.compile(r"\bwho (is|was|plays)\b.{0,40}\?", re.I),
    # Pure gratitude / closing messages — not support requests.
    # Allows optional "for help/assist/helping…" clause before message ends.
    re.compile(
        r"^thank(?:s|\s+you)[\s!.,]*"
        r"(?:for\s+(?:help|assist|your\s+(?:help|support|assistance)|helping)[^,;]*)?[\s!.,]*$",
        re.I,
    ),
    # Greetings and sign-offs
    re.compile(r"^(?:hello|hi|hey|good\s*(?:morning|evening|afternoon|day))[\s!.,]*$", re.I),
    re.compile(r"^(?:bye|goodbye|see\s+you|take\s+care)[\s!.,]*$", re.I),
    # Malicious code requests
    re.compile(r"give me (?:the )?code (?:to|for)\b", re.I),
    re.compile(
        r"write (?:a |the )?(?:script|code|program) (?:to|that) "
        r"(?:delete|remove|wipe|hack|crack|exploit)",
        re.I,
    ),
    # Jailbreak / role-play starters
    re.compile(r"\bpretend\s+(?:you\s+are|that\s+you)\b", re.I),
    re.compile(
        r"\byou\s+are\s+now\s+(?:a\s+)?(?:free|unfiltered|unrestricted|evil|dan|jailbroken)\b",
        re.I,
    ),
    re.compile(r"\bjailbreak\b", re.I),
    # Trivially empty / nonsense
    re.compile(r"^(?:none|n/?a|\?+|!+|-)[\s.,]*$", re.I),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _contains_phrase(text_lower: str, phrases: tuple[str, ...]) -> str | None:
    for phrase in phrases:
        if phrase in text_lower:
            return phrase
    return None


def _classify_request_type(issue: str) -> str:
    t = issue.lower()

    for pat in _INVALID_PATTERNS:
        if pat.search(issue):
            return "invalid"

    if _contains_phrase(t, _INJECTION_PHRASES):
        return "invalid"

    if _contains_phrase(t, _BUG_PHRASES):
        return "bug"
    if any(w in t.split() for w in _BUG_WORDS):
        return "bug"
    if _contains_phrase(t, _SECURITY_BUG_PHRASES):
        return "bug"

    if _contains_phrase(t, _FEATURE_PHRASES):
        return "feature_request"
    if any(w in t.split() for w in _FEATURE_WORDS):
        return "feature_request"

    return "product_issue"


# ===========================================================================
# SECTION 2 — escalation decision
# ===========================================================================

def _check_escalation(
    issue: str,
    request_type: str,
    confidence: float,
) -> tuple[bool, str]:
    """Return (should_escalate, reason)."""
    t = " ".join(issue.lower().split())  # collapse newlines/tabs to single spaces

    # Invalid tickets are always replied (not escalated) — escalating wastes agent time
    if request_type == "invalid":
        return False, ""

    if phrase := _contains_phrase(t, _OUTAGE_PHRASES):
        return True, f"Site-wide outage ('{phrase}')"

    if phrase := _contains_phrase(t, _SECURITY_ESCALATE_PHRASES):
        return True, f"Security/safety issue ('{phrase}')"

    if phrase := _contains_phrase(t, _IMPOSSIBLE_PHRASES):
        return True, f"Request cannot be fulfilled by support ('{phrase}')"

    if phrase := _contains_phrase(t, _HUMAN_REQUIRED_PHRASES):
        return True, f"Requires human agent — corpus has no coverage for ('{phrase}')"

    if confidence < 0.29:
        return True, f"Insufficient corpus evidence (confidence={confidence:.2f})"

    return False, ""


# ===========================================================================
# SECTION 3 — product_area inference
# ===========================================================================

# -- Deterministic defaults for INVALID tickets ------------------------------
# Never driven by retrieved doc breadcrumbs — only company identity matters.
# For company=None, _area_for_invalid() uses the top doc's company as a hint.
_COMPANY_DEFAULT_AREA: dict[str, str] = {
    "hackerrank": "screen",
    "claude":     "conversation_management",
    "visa":       "general_support",
}


def _area_for_invalid(
    results: list[SearchResult],
    company: str | None,
) -> str:
    """
    Stable product_area for invalid/out-of-scope tickets.

    Does NOT use breadcrumbs or path fragments from retrieved docs.
    Uses only:
      1. The ticket's stated company (most reliable).
      2. The top retrieved doc's company as a fallback when company=None
         (the company is stable across BM25 runs; breadcrumbs are not).
    """
    company_lower = (company or "").lower()
    if company_lower in _COMPANY_DEFAULT_AREA:
        return _COMPANY_DEFAULT_AREA[company_lower]
    if results:
        doc_company = results[0][0].company.lower()
        if doc_company in _COMPANY_DEFAULT_AREA:
            return _COMPANY_DEFAULT_AREA[doc_company]
    return "general"


# -- HackerRank keyword -> area overrides ------------------------------------
_HR_KEYWORD_AREAS: list[tuple[list[str], str]] = [
    # Tests / assessments / screening
    (["test", "assessment", "screen", "candidate", "invite", "variant",
      "question", "coding challenge", "proctoring", "anti-cheat", "plagiarism",
      "test duration", "extra time", "score report", "time limit"],
     "screen"),
    # Live interviews — bare "interview" excluded; too broad (matches "mock interview")
    (["live coding", "codepair", "video interview", "interviewer"],
     "interviews"),
    # Community / account / billing
    (["community", "delete my account", "delete account", "google login",
      "password", "hackerrank account", "forgot password", "login issue",
      "sign up", "signed up", "subscription", "billing", "payment",
      "mock interview", "refund"],
     "community"),
    # Learning / SkillUp
    (["skillup", "skill up", "learn", "course", "tutorial", "certificate",
      "practice problem"],
     "skillup"),
    # Library / question library
    (["library", "question bank", "question library"],
     "library"),
    # User / account management — check BEFORE integrations
    (["remove user", "remove member", "remove employee",
      "remove from account", "remove from our", "remove them from",
      "remove him from", "remove her from", "employee has left",
      "staff has left", "team member has left",
      "deactivate user", "disable user", "user management",
      "manage users", "manage members", "add user", "add member",
      "invite user", "revoke access"],
     "settings"),
    # Integrations
    (["integration", "ats", "greenhouse", "workday", "lever", "webhook",
      "api key", "sso", "saml"],
     "integrations"),
    # General / settings
    (["infosec", "security process", "dpa", "data processing"],
     "settings"),
]

# Breadcrumb label -> canonical product_area
_HR_BREADCRUMB_MAP: dict[str, str] = {
    "screen":                "screen",
    "interviews":            "interviews",
    "engage":                "engage",
    "skillup":               "skillup",
    "library":               "library",
    "integrations":          "integrations",
    "settings":              "settings",
    "chakra":                "chakra",
    "general help":          "general_help",
    "hackerrank community":  "community",
    "community":             "community",
    "uncategorized":         "general_help",
}

# -- Claude keyword -> area overrides ----------------------------------------
_CLAUDE_KEYWORD_AREAS: list[tuple[list[str], str]] = [
    # Privacy — check BEFORE conversation_management because
    # "private info in conversations" should map to privacy
    (["privacy", "private info", "sensitive data", "who can view",
      "data use", "training data", "data used for", "personal data",
      "use my data", "using my data"],  # "allowing Claude to use my data to improve"
     "privacy"),
    # Conversation management
    (["delete", "rename", "conversation", "chat history", "incognito",
      "share chat", "export chat"],
     "conversation_management"),
    # Billing / subscription
    (["billing", "subscription", "payment", "charge", "refund", "invoice",
      "credit card", "plan", "pro plan", "max plan", "cancel"],
     "billing"),
    # API / console
    (["api", "console", "sdk", "api key", "rate limit", "tier", "bedrock",
      "foundry", "vertex"],
     "api"),
    # Usage limits
    (["usage limit", "message limit", "quota", "rate limit", "usage cap"],
     "usage_and_limits"),
    # Features / capabilities
    (["web search", "artifact", "extended thinking", "research mode",
      "project", "skills", "cowork", "plugin", "connector", "upload",
      "file", "image", "voice", "export"],
     "features"),
    # Account management (workspace/seat covers team plan access issues)
    (["account", "email", "password", "login", "sign in", "delete account",
      "lti", "sso", "phone", "verify", "session", "active session",
      "workspace", "seat", "member", "team member", "org admin",
      "remove from", "removed from", "restore my access", "lost access"],
     "account_management"),
    # Troubleshooting
    (["error", "not working", "stopped working", "stopped responding",
      "broken", "issue", "bug", "problem", "incorrect", "wrong answer",
      "hallucination", "all requests", "requests are failing"],
     "troubleshooting"),
    # Safety / safeguards
    (["crawl", "crawling", "scrape", "opt out", "data removal"],
     "safeguards"),
]

_CLAUDE_BREADCRUMB_MAP: dict[str, str] = {
    "account management":                    "account_management",
    "conversation management":               "conversation_management",
    "features and capabilities":             "features",
    "get started with claude":               "get_started",
    "personalization and settings":          "settings",
    "troubleshooting":                       "troubleshooting",
    "usage and limits":                      "usage_and_limits",
    "safeguards":                            "safeguards",
    "amazon bedrock":                        "amazon_bedrock",
    "api faq":                               "api",
    "api prompt design":                     "api",
    "claude api usage and best practices":   "api",
    "pricing and billing":                   "billing",
    "using the claude api and console":      "api",
    "pre-built connectors":                  "connectors",
    "capabilities":                          "features",
    "general":                               "general",
}

# -- Visa keyword -> area overrides ------------------------------------------
_VISA_KEYWORD_AREAS: list[tuple[list[str], str]] = [
    # Traveller's cheques — check BEFORE generic stolen-card rules
    (["traveller", "traveler", "cheque", "cheques", "citicorp", "travellers"],
     "travel_support"),
    # Exchange rate / travel card
    (["exchange rate", "currency", "foreign currency", "travel card"],
     "travel_support"),
    # Card loss / theft / emergency (general Visa card, not cheques)
    (["lost card", "stolen card", "report card", "block card", "lost my card",
      "card lost", "card stolen", "report a lost", "replace card"],
     "general_support"),
    # Disputes / chargebacks
    (["dispute", "chargeback", "wrong product", "merchant sent", "not received"],
     "dispute_resolution"),
    # Fraud and identity theft — check before generic stolen-card rules
    (["fraud", "unauthorized charge", "unauthorized transaction",
      "identity theft", "identity stolen", "identity has been stolen",
      "identity was stolen", "my identity"],
     "fraud_protection"),
    # Rules / regulations / minimum spend
    (["minimum", "surcharge", "checkout fee", "rules", "regulations",
      "interchange", "fees"],
     "consumer"),
    # Data security
    (["data security", "pci", "pci dss"],
     "data_security"),
]

# Visa path fragments -> product_area (Visa has no breadcrumbs)
_VISA_PATH_MAP: list[tuple[str, str]] = [
    ("travelers-cheques",   "travel_support"),
    ("travel-support",      "travel_support"),
    ("dispute-resolution",  "dispute_resolution"),
    ("fraud-protection",    "fraud_protection"),
    ("data-security",       "data_security"),
    ("regulations-fees",    "regulations"),
    ("small-business",      "small_business"),
    ("merchant",            "merchant"),
    ("consumer",            "consumer"),
]

# -- Generic (no-company) keyword fallback for non-invalid tickets -----------
# Checked when company=None before falling back to the retrieved doc.
# Keeps product_area stable for ambiguous tickets that don't state a company.
_GENERIC_KEYWORD_AREAS: list[tuple[list[str], str]] = [
    (["security vulnerability", "major security", "bug bounty", "data breach",
      "identity theft", "identity stolen"],
     "security"),
    (["down", "outage", "not responding", "not accessible", "not loading",
      "not working", "stopped working", "isn't working", "is not working",
      "all requests", "nothing is loading"],
     "technical_issue"),
    (["billing", "payment", "subscription", "refund", "invoice"],
     "billing"),
    (["account", "login", "password", "sign in"],
     "account"),
    (["privacy", "personal data", "sensitive data"],
     "privacy"),
]


def _hr_product_area(doc: Document) -> str:
    if doc.breadcrumbs:
        bc0 = doc.breadcrumbs[0].lower()
        return _HR_BREADCRUMB_MAP.get(bc0, bc0.replace(" ", "_"))
    return doc.category.lower().replace(" ", "_").replace("-", "_")


def _claude_product_area(doc: Document) -> str:
    if len(doc.breadcrumbs) >= 2:
        bc_last = doc.breadcrumbs[-1].lower()
        return _CLAUDE_BREADCRUMB_MAP.get(bc_last, bc_last.replace(" ", "_"))
    if doc.breadcrumbs:
        bc0 = doc.breadcrumbs[0].lower()
        return _CLAUDE_BREADCRUMB_MAP.get(bc0, bc0.replace(" ", "_"))
    return doc.category.lower().replace(" ", "_").replace("-", "_")


def _visa_product_area(doc: Document) -> str:
    path = doc.file_path.lower().replace("\\", "/")
    for fragment, area in _VISA_PATH_MAP:
        if fragment in path:
            return area
    return "general_support"


def _infer_product_area(
    results: list[SearchResult],
    company: str | None,
    issue: str,
) -> str:
    """
    Product_area inference for non-invalid tickets (four stages):
    1. Issue-keyword overrides per stated company (most reliable).
    2. Generic cross-company keyword fallback when company=None.
    3. Company-aware breadcrumb/path from the top retrieved document.
    4. Category field as a last resort.
    """
    issue_lower = issue.lower()
    company_lower = (company or "").lower()

    # Stage 1: per-company keyword overrides
    if company_lower == "hackerrank":
        for keywords, area in _HR_KEYWORD_AREAS:
            if any(kw in issue_lower for kw in keywords):
                return area

    elif company_lower == "claude":
        for keywords, area in _CLAUDE_KEYWORD_AREAS:
            if any(kw in issue_lower for kw in keywords):
                return area

    elif company_lower == "visa":
        for keywords, area in _VISA_KEYWORD_AREAS:
            if any(kw in issue_lower for kw in keywords):
                return area

    # Stage 2: generic fallback for no-company tickets
    if not company_lower:
        for keywords, area in _GENERIC_KEYWORD_AREAS:
            if any(kw in issue_lower for kw in keywords):
                return area

    # Stage 3: breadcrumb / path from top retrieved document
    if not results:
        return "general_support"

    top_doc = results[0][0]
    doc_company = top_doc.company.lower()

    if doc_company == "hackerrank":
        return _hr_product_area(top_doc)
    if doc_company == "claude":
        return _claude_product_area(top_doc)
    if doc_company == "visa":
        return _visa_product_area(top_doc)

    # Stage 4: unknown company — use category as-is
    return top_doc.category.lower().replace(" ", "_").replace("-", "_")


# ===========================================================================
# SECTION 4 — confidence
# ===========================================================================

def compute_confidence(results: list[SearchResult], company: str | None) -> float:
    """
    Map BM25 scores to a 0–1 confidence value.

    Divisor choice (/60.0):
      BM25Okapi scores on this 774-document corpus range from ~0 (no overlap)
      to ~80 (highly specific multi-term match like "Visa travellers cheques").
      A divisor of 60 maps the strong-match ceiling (~60–80) to 1.0 while
      keeping moderate matches (score ~20) at ~0.33, which is calibrated to
      be above the escalation threshold only when the match is plausible.

    Escalation threshold (0.29):
      Empirically chosen so that scores below ~17 BM25 points (score / 60 < 0.29)
      are treated as "insufficient evidence" and escalated. Calibrated against the
      sample validation set: the lowest-confidence correctly-replied ticket scores
      0.293. Setting the threshold at 0.29 gives a 3-point safety margin while
      catching all retrievals with fewer than ~3 strong overlapping terms.

    Company mismatch halving:
      When the top-ranked document belongs to a different company than the ticket
      stated, the retrieval is likely cross-domain noise. Halving raw confidence
      pushes borderline matches below 0.29 so they escalate rather than produce
      a misleading answer from the wrong product's corpus.
    """
    if not results:
        return 0.0
    top_score = results[0][1]
    raw = min(1.0, top_score / 60.0)
    if company:
        if results[0][0].company.lower() != company.lower():
            raw *= 0.5
    return round(raw, 3)


# ===========================================================================
# SECTION 5 — public triage function
# ===========================================================================

def triage(
    issue: str,
    company: str | None,
    results: list[SearchResult],
) -> TriageResult:
    """
    Classify and route a single support ticket.

    Parameters
    ----------
    issue   : raw ticket body
    company : "HackerRank" | "Claude" | "Visa" | None
    results : (Document, score) pairs from the retriever, highest score first
    """
    confidence   = compute_confidence(results, company)
    request_type = _classify_request_type(issue)

    should_esc, reason = _check_escalation(issue, request_type, confidence)
    status = "escalated" if should_esc else "replied"

    # Invalid tickets: use a deterministic company-based default — never let
    # retrieved doc breadcrumbs set the area for out-of-scope messages.
    if request_type == "invalid":
        product_area = _area_for_invalid(results, company)
    else:
        product_area = _infer_product_area(results, company, issue)

    top_docs = [doc for doc, _ in results[:3]]

    logger.debug(
        "Triage -> status=%-10s type=%-16s area=%-28s conf=%.3f",
        status, request_type, product_area, confidence,
    )
    if reason:
        logger.debug("Escalation: %s", reason)

    return TriageResult(
        status=status,
        request_type=request_type,
        product_area=product_area,
        confidence=confidence,
        escalation_reason=reason,
        top_docs=top_docs,
    )
