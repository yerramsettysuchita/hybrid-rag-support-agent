"""
runner.py — batch processor for support tickets.

Reads support_tickets/support_tickets.csv, runs the full triage pipeline on
each row, and writes support_tickets/output.csv.

Column order in output.csv mirrors AgentOutput:
  Issue, Subject, Company, Response, Product Area, Status, Request Type, Justification

A sidecar output_trace.jsonl is also written with one JSON object per ticket,
containing the full decision trace (confidence, fingerprint, rationale, top doc).

Also used by main.py --validate to score against sample_support_tickets.csv.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
logger = logging.getLogger(__name__)


def _parse_company(raw: str) -> str | None:
    """Normalise CSV company values; return None for 'none'/blank."""
    v = raw.strip()
    if not v or v.lower() == "none":
        return None
    return v


def _normalise_status(raw: str) -> str:
    return raw.strip().lower()


def _normalise_type(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")


def _normalise_area(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Core batch runner
# ---------------------------------------------------------------------------

def run_batch(
    agent,
    tickets_csv: Path,
    output_csv: Path,
) -> list[dict]:
    """
    Process every ticket in tickets_csv through agent and write output_csv.

    Also writes a sidecar output_trace.jsonl with the full decision trace
    (confidence, fingerprint, rationale, top_doc_title, top_doc_url).

    Returns the list of result dicts (Issue, Subject, Company, Response,
    Product Area, Status, Request Type, Justification) for downstream use.
    """
    rows: list[dict] = []
    traces: list[dict] = []

    with open(tickets_csv, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader, 1):
            issue   = row.get("Issue", "").strip()
            subject = row.get("Subject", "").strip()
            company = _parse_company(row.get("Company", ""))

            if not issue:
                logger.warning("Row %d: empty issue -- skipping", i)
                continue

            logger.info("Row %d: %s | company=%s", i, issue[:60], company or "None")
            out = agent.process(issue=issue, subject=subject, company=company)

            rows.append({
                "Issue":         out.issue,
                "Subject":       out.subject,
                "Company":       out.company,
                "Response":      out.response,
                "Product Area":  out.product_area,
                "Status":        out.status.capitalize(),
                "Request Type":  out.request_type,
                "Justification": out.justification,
            })

            traces.append({
                "row":           i,
                "issue":         out.issue[:100],
                "subject":       out.subject,
                "company":       out.company,
                "status":        out.status,
                "request_type":  out.request_type,
                "product_area":  out.product_area,
                "confidence":    round(out.confidence, 4),
                "fingerprint":   out.fingerprint,
                "rationale":     out.rationale,
                "top_doc_title": out.top_doc_title,
                "top_doc_url":   out.top_doc_url,
                "summary":       out.summary,
            })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Issue", "Subject", "Company", "Response",
                  "Product Area", "Status", "Request Type", "Justification"]

    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    trace_path = output_csv.parent / "output_trace.jsonl"
    with open(trace_path, "w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    logger.info("Wrote %d rows to %s", len(rows), output_csv)
    logger.info("Wrote decision trace to %s", trace_path)
    return rows


# ---------------------------------------------------------------------------
# Validation against sample tickets
# ---------------------------------------------------------------------------

def validate_against_sample(
    agent,
    sample_csv: Path,
) -> dict:
    """
    Run the agent on sample_csv (which has expected Status/Request Type/Product Area
    columns) and return a dict with accuracy stats and per-row details.
    """
    results = []
    with open(sample_csv, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader, 1):
            issue   = row.get("Issue", "").strip()
            subject = row.get("Subject", "").strip()
            company = _parse_company(row.get("Company", ""))

            exp_status = _normalise_status(row.get("Status", ""))
            exp_type   = _normalise_type(row.get("Request Type", ""))
            exp_area   = _normalise_area(row.get("Product Area", ""))

            if not issue:
                continue

            out = agent.process(issue=issue, subject=subject, company=company)

            got_status = out.status.lower()
            got_type   = out.request_type.lower()
            got_area   = out.product_area.lower()

            status_ok = got_status == exp_status
            type_ok   = got_type == exp_type
            area_ok   = got_area == exp_area or not exp_area  # blank expected → skip

            results.append({
                "row":         i,
                "issue":       issue[:60],
                "got_status":  got_status,
                "exp_status":  exp_status,
                "got_type":    got_type,
                "exp_type":    exp_type,
                "got_area":    got_area,
                "exp_area":    exp_area,
                "status_ok":   status_ok,
                "type_ok":     type_ok,
                "area_ok":     area_ok,
                "all_ok":      status_ok and type_ok and area_ok,
            })

    n = len(results)
    if n == 0:
        return {"n": 0, "rows": []}

    status_acc = sum(r["status_ok"] for r in results) / n
    type_acc   = sum(r["type_ok"]   for r in results) / n
    area_acc   = sum(r["area_ok"]   for r in results) / n
    overall    = sum(r["all_ok"]    for r in results) / n

    return {
        "n":          n,
        "status_acc": status_acc,
        "type_acc":   type_acc,
        "area_acc":   area_acc,
        "overall":    overall,
        "rows":       results,
    }


# ---------------------------------------------------------------------------
# Standalone entry point (python code/runner.py)
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    from pathlib import Path

    # Ensure UTF-8 output on Windows CP1252 consoles before any logging starts
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    _HERE = Path(__file__).parent
    DATA_DIR    = (_HERE.parent / "data").resolve()
    TICKETS_CSV = (_HERE.parent / "support_tickets" / "support_tickets.csv").resolve()
    OUTPUT_CSV  = (_HERE.parent / "support_tickets" / "output.csv").resolve()

    p = argparse.ArgumentParser(description="Batch runner for support tickets")
    p.add_argument("--tickets", default=str(TICKETS_CSV),
                   help="Input CSV (default: support_tickets/support_tickets.csv)")
    p.add_argument("--output",  default=str(OUTPUT_CSV),
                   help="Output CSV (default: support_tickets/output.csv)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("rank_bm25").setLevel(logging.WARNING)

    sys.path.insert(0, str(_HERE))
    from agent import SupportAgent

    agent = SupportAgent(DATA_DIR)
    rows  = run_batch(agent, Path(args.tickets), Path(args.output))
    print(f"\nDone. {len(rows)} tickets written to {args.output}")


if __name__ == "__main__":
    main()
