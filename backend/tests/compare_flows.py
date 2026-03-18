#!/usr/bin/env python3
"""
compare_flows.py — Side-by-side quality comparison of Normal RAG vs LangChain Agent.

Usage (from backend/ dir, with venv activated):
  python tests/compare_flows.py --email you@example.com --password yourpass

Or set env vars:
  export TEST_EMAIL=you@example.com
  export TEST_PASSWORD=yourpass
  python tests/compare_flows.py

The script:
  1. Logs in and gets a Bearer token.
  2. Sends each test question to both SSE endpoints concurrently.
  3. Parses SSE events, accumulates answers, extracts sources.
  4. Scores each answer on: grounding, Part citations, § refs, completeness.
  5. Prints a detailed side-by-side report and an overall winner tally.
  6. Writes a JSON report to tests/compare_results.json for iteration tracking.
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import requests

# ──────────────────────────────────────────────────────────────────────────────
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

TEST_QUESTIONS: List[Dict] = [
    {
        "id": "capital_reqs",
        "question": "What are the Federal Home Loan Bank capital requirements under 12 CFR Part 1277?",
        "expect_part": "1277",
        "expect_sections": ["1277.4", "1277.6", "1277.29", "1277.20"],
        "expect_keywords": ["capital", "stock", "retained earnings", "adequately capitalized"],
    },
    {
        "id": "capital_plan",
        "question": "What must a Federal Home Loan Bank capital plan include?",
        "expect_part": "1277",
        "expect_sections": ["1277.22", "1277.20", "1277.21"],
        "expect_keywords": ["capital plan", "board of directors", "membership stock"],
    },
    {
        "id": "membership_stock",
        "question": "What are the membership stock requirements for Federal Home Loan Banks?",
        "expect_part": "1277",
        "expect_sections": ["1277.4", "1277.6", "1277.10"],
        "expect_keywords": ["membership stock", "member", "capital"],
    },
    {
        "id": "retained_earnings",
        "question": "How must Federal Home Loan Banks manage retained earnings under FHFA regulations?",
        "expect_part": "1277",
        "expect_sections": ["1277.4", "1277.6"],
        "expect_keywords": ["retained earnings", "capital", "FHFA"],
    },
    {
        "id": "part_1206_check",
        "question": "What definitions apply to Federal Home Loan Bank regulations under 12 CFR Part 1206?",
        "expect_part": "1206",
        "expect_sections": ["1206.2"],
        "expect_keywords": ["definitions", "bank", "member"],
    },
]

LLM_PROVIDER = "openai"
TOP_K = 20
MIN_SIMILARITY = 0.2

# ──────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FlowResult:
    flow: str                     # "normal" | "agent"
    question_id: str
    answer: str = ""
    is_grounded: bool = False
    retrieved_count: int = 0
    prompt_context_count: int = 0
    source_parts: List[str] = field(default_factory=list)  # "1277", "1206", …
    source_sections: List[str] = field(default_factory=list)
    timing_ms: Dict[str, float] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)
    error: Optional[str] = None
    # Scores (0-100)
    score_grounding: int = 0
    score_part_accuracy: int = 0
    score_section_citations: int = 0
    score_keyword_coverage: int = 0
    score_total: int = 0


SECTION_RE = re.compile(r"§\s*(\d+\.\d+[\w\.]*)")
PART_RE = re.compile(r"[Pp]art\s+(\d+)|12\s+CFR\s+Part\s+(\d+)|1277|1206", re.IGNORECASE)


def score_result(result: FlowResult, spec: Dict) -> FlowResult:
    """Compute quality scores based on expected values in the test spec."""
    answer_lower = result.answer.lower()
    expect_part = spec.get("expect_part", "")
    expect_sections = [s.lower() for s in spec.get("expect_sections", [])]
    expect_keywords = [kw.lower() for kw in spec.get("expect_keywords", [])]

    # 1. Grounding score (0 or 40)
    result.score_grounding = 40 if result.is_grounded else 0

    # 2. Part accuracy (0 or 30) — did the answer mention the expected part?
    cited_parts = set(re.findall(r"\b1\d{3}\b", result.answer))
    # also from source metadata
    cited_parts |= set(result.source_parts)
    result.score_part_accuracy = 30 if expect_part in cited_parts else 0

    # 3. Section citations (0-20) — § refs in answer
    cited_sections = [m.lower() for m in SECTION_RE.findall(result.answer)]
    matched_sections = [s for s in expect_sections if any(s in c for c in cited_sections)]
    if expect_sections:
        result.score_section_citations = round(20 * len(matched_sections) / len(expect_sections))
    else:
        result.score_section_citations = 20

    # 4. Keyword coverage (0-10)
    matched_kw = [kw for kw in expect_keywords if kw in answer_lower]
    if expect_keywords:
        result.score_keyword_coverage = round(10 * len(matched_kw) / len(expect_keywords))
    else:
        result.score_keyword_coverage = 10

    result.score_total = (
        result.score_grounding
        + result.score_part_accuracy
        + result.score_section_citations
        + result.score_keyword_coverage
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# SSE stream parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_sse_stream(
    url: str,
    body: Dict,
    headers: Dict,
    flow_label: str,
    question_id: str,
    timeout: int = 120,
) -> FlowResult:
    result = FlowResult(flow=flow_label, question_id=question_id)
    token_buf: List[str] = []

    try:
        with requests.post(
            url,
            json=body,
            headers=headers,
            stream=True,
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                return result

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                json_str = raw_line[len("data:"):].strip()
                try:
                    envelope = json.loads(json_str)
                except json.JSONDecodeError:
                    continue

                etype = envelope.get("type", "")
                data = envelope.get("data", {})
                result.events.append({"type": etype, "data": data})

                if etype == "token":
                    token_buf.append(data.get("text", ""))

                elif etype == "source":
                    meta = data.get("metadata", {})
                    chunk_meta = meta.get("chunk_metadata", {})
                    doc_meta = meta.get("doc_metadata", {})
                    part = str(chunk_meta.get("part") or doc_meta.get("part") or "")
                    section = str(chunk_meta.get("section") or "")
                    if part:
                        result.source_parts.append(part)
                    if section:
                        result.source_sections.append(section)

                elif etype == "final":
                    result.answer = data.get("answer", "").strip()
                    result.is_grounded = data.get("is_grounded", False)
                    result.retrieved_count = data.get("retrieved_count", 0)
                    result.prompt_context_count = data.get("prompt_context_count", 0)
                    result.timing_ms = data.get("timings_ms", {})

    except requests.exceptions.Timeout:
        result.error = "Request timed out"
    except Exception as exc:
        result.error = str(exc)

    # Fallback: build answer from token buffer if final event had no answer
    if not result.answer and token_buf:
        result.answer = "".join(token_buf).strip()
        result.is_grounded = bool(result.source_parts)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────────────────────────────────────

def login(email: str, password: str) -> str:
    resp = requests.post(
        f"{BASE_URL}/auth/login",
        json={"email": email, "password": password},
        timeout=15,
    )
    if resp.status_code != 200:
        print(f"❌  Login failed: {resp.status_code} {resp.text[:200]}")
        sys.exit(1)
    token = resp.json().get("access_token")
    if not token:
        print(f"❌  No access_token in login response: {resp.text[:200]}")
        sys.exit(1)
    print(f"✅  Logged in as {email}\n")
    return token


# ──────────────────────────────────────────────────────────────────────────────
# Printer
# ──────────────────────────────────────────────────────────────────────────────

def _bar(score: int, max_score: int = 100, width: int = 20) -> str:
    filled = round(width * score / max_score) if max_score else 0
    return "█" * filled + "░" * (width - filled)


def print_comparison(normal: FlowResult, agent: FlowResult, spec: Dict):
    q_id = spec["id"]
    print("=" * 80)
    print(f"  Q: {spec['question']}")
    print(f"  Expect part={spec.get('expect_part')}  sections={spec.get('expect_sections')}")
    print("=" * 80)

    def row(label: str, nval, aval, highlight: bool = False):
        marker = "◀" if highlight and nval != aval else " "
        print(f"  {label:<26} │ NORMAL: {str(nval):<30} │ AGENT: {str(aval)}{marker}")

    row("Grounded", normal.is_grounded, agent.is_grounded)
    row("Retrieved", normal.retrieved_count, agent.retrieved_count)
    row("Prompt context", normal.prompt_context_count, agent.prompt_context_count)
    row("Source parts", sorted(set(normal.source_parts)), sorted(set(agent.source_parts)))
    row("Timing total_ms", normal.timing_ms.get("total"), agent.timing_ms.get("total"))
    row("Error", normal.error or "—", agent.error or "—")

    print()
    print(f"  {'SCORES':<26} │ {'NORMAL':<30} │ AGENT")
    print(f"  {'─'*26}─┼─{'─'*30}─┼─{'─'*30}")

    def score_row(label: str, n_s: int, a_s: int, max_s: int):
        n_bar = _bar(n_s, max_s)
        a_bar = _bar(a_s, max_s)
        winner = "◀ Normal" if n_s > a_s else ("◀ Agent" if a_s > n_s else "  Tie")
        print(f"  {label:<26} │ {n_s:>3}/{max_s} {n_bar} │ {a_s:>3}/{max_s} {a_bar}  {winner}")

    score_row("Grounding     (40pt)", normal.score_grounding, agent.score_grounding, 40)
    score_row("Part accuracy (30pt)", normal.score_part_accuracy, agent.score_part_accuracy, 30)
    score_row("§ Citations   (20pt)", normal.score_section_citations, agent.score_section_citations, 20)
    score_row("Keywords      (10pt)", normal.score_keyword_coverage, agent.score_keyword_coverage, 10)
    score_row("TOTAL        (100pt)", normal.score_total, agent.score_total, 100)

    print()
    # Show answer previews
    for label, result in [("NORMAL", normal), ("AGENT", agent)]:
        preview = (result.answer or "(no answer)").replace("\n", " ")[:300]
        print(f"  [{label}] {preview}{'…' if len(result.answer) > 300 else ''}")
    print()


def print_summary(pairs: List[Tuple[FlowResult, FlowResult]]):
    normal_wins = agent_wins = ties = 0
    for n, a in pairs:
        if n.score_total > a.score_total:
            normal_wins += 1
        elif a.score_total > n.score_total:
            agent_wins += 1
        else:
            ties += 1

    total_n = sum(n.score_total for n, _ in pairs)
    total_a = sum(a.score_total for _, a in pairs)
    q = len(pairs)

    print("\n" + "═" * 80)
    print("  OVERALL SUMMARY")
    print("═" * 80)
    print(f"  Questions tested : {q}")
    print(f"  Normal wins      : {normal_wins}")
    print(f"  Agent  wins      : {agent_wins}")
    print(f"  Ties             : {ties}")
    print(f"  Avg Normal score : {total_n / q:.1f}/100")
    print(f"  Avg Agent  score : {total_a / q:.1f}/100")
    delta = total_a / q - total_n / q
    sign = "+" if delta >= 0 else ""
    print(f"  Agent delta      : {sign}{delta:.1f}  {'✅ Agent ahead' if delta > 0 else ('🔴 Normal ahead' if delta < 0 else '— Equal')}")
    print("═" * 80)

    # Diagnose gaps
    print("\n  📋 DIAGNOSIS")
    for n, a in pairs:
        if a.score_total < n.score_total:
            gaps = []
            if a.score_grounding < n.score_grounding:
                gaps.append("not grounded")
            if a.score_part_accuracy < n.score_part_accuracy:
                gaps.append(f"wrong part (got {sorted(set(a.source_parts))} want {sorted(set(n.source_parts))})")
            if a.score_section_citations < n.score_section_citations:
                gaps.append("missing § citations in answer")
            if a.score_keyword_coverage < n.score_keyword_coverage:
                gaps.append("missing expected keywords")
            print(f"  ⚠  [{a.question_id}] Agent lags: {', '.join(gaps) or 'unclear'}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare Normal RAG vs LangChain Agent")
    parser.add_argument("--email", default=os.getenv("TEST_EMAIL", ""))
    parser.add_argument("--password", default=os.getenv("TEST_PASSWORD", ""))
    parser.add_argument("--questions", nargs="*", help="Subset of question IDs to run")
    parser.add_argument("--json-out", default="tests/compare_results.json")
    args = parser.parse_args()

    if not args.email or not args.password:
        print("❌  Provide --email and --password (or set TEST_EMAIL / TEST_PASSWORD).")
        sys.exit(1)

    token = login(args.email, args.password)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    questions = TEST_QUESTIONS
    if args.questions:
        questions = [q for q in TEST_QUESTIONS if q["id"] in args.questions]

    pairs: List[Tuple[FlowResult, FlowResult]] = []
    all_results = []

    for spec in questions:
        q = spec["question"]
        qid = spec["id"]
        body = {
            "question": q,
            "llm_provider": LLM_PROVIDER,
            "classification_filter": ["public"],
            "top_k": TOP_K,
            "min_similarity": MIN_SIMILARITY,
            "retrieval_mode": "hybrid",
            "temperature": 0.3,
        }

        print(f"\n🔍  [{qid}] {q[:70]}…")
        print("    Calling both endpoints (this may take 10–30 s each)…")

        t0 = time.perf_counter()
        normal_result = parse_sse_stream(
            f"{BASE_URL}/rag/query/stream/events",
            body, headers, "normal", qid,
        )
        t_normal = round((time.perf_counter() - t0) * 1000)
        print(f"    ✓ Normal  done in {t_normal} ms  (score will be computed)")

        t1 = time.perf_counter()
        agent_result = parse_sse_stream(
            f"{BASE_URL}/rag/agent/stream/events",
            body, headers, "agent", qid,
        )
        t_agent = round((time.perf_counter() - t1) * 1000)
        print(f"    ✓ Agent   done in {t_agent} ms")

        # Score
        normal_result = score_result(normal_result, spec)
        agent_result = score_result(agent_result, spec)

        pairs.append((normal_result, agent_result))
        all_results.append({"normal": asdict(normal_result), "agent": asdict(agent_result)})

    # Print reports
    for (n, a), spec in zip(pairs, questions):
        print_comparison(n, a, spec)

    print_summary(pairs)

    # Write JSON report
    out_path = args.json_out
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "results": all_results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n📄  Full results written to {out_path}\n")


if __name__ == "__main__":
    main()
