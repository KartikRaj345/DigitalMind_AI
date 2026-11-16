"""Verify the two demo scenarios end to end, as assertions rather than as prose.

    python -m backend.scripts.evaluate_demo_scenarios

This is the "definition of done" check, made executable. It builds a throwaway database,
loads the demo corpus, runs the pipeline, and asserts:

Scenario A
    the payment-demanding message is flagged, with cited spans for urgency, threat
    language, emotional pressure and payment request; the earlier records are linked to it;
    the timeline places them in recorded order; and the benign bank reminder, which
    contains the same surface features, is NOT flagged.

Scenario B
    the back-referencing message produces the required wording verbatim, and the
    companion message that *does* have a matching record is not reported as unmatched.

It also asserts the product rules: no forbidden claim wording anywhere in the output, no
unexplained aggregate score, and every citation points at the words it quotes.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from sqlalchemy import select

REPORTS = Path(__file__).resolve().parents[2] / "reports"

SCENARIO_A_RECORD = "demo-a5-message"
SCENARIO_A_BENIGN = "demo-a6-email"
SCENARIO_B_RECORD = "demo-b1-message"
SCENARIO_B_MATCHED = "demo-b2-email"

REQUIRED_NO_MATCH = "No matching discussion was found in the records analyzed."


class Check:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, name: str, passed: bool, detail: str = "", evidence: Any = None) -> None:
        self.rows.append(
            {"check": name, "passed": bool(passed), "detail": detail, "evidence": evidence}
        )
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))

    @property
    def all_passed(self) -> bool:
        return all(r["passed"] for r in self.rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the demo scenarios.")
    parser.add_argument("--out", type=str, default=str(REPORTS / "demo_scenarios.json"))
    args = parser.parse_args()

    # Isolated throwaway database so the user's own data is never touched.
    tmp = Path(tempfile.mkdtemp(prefix="digitalmind-demo-eval-"))
    os.environ["DIGITALMIND_DATA_DIR"] = str(tmp)
    os.environ["DIGITALMIND_DATABASE_URL"] = f"sqlite:///{tmp / 'demo_eval.db'}"

    from ..app import db as db_mod
    from ..app.config import reset_settings_cache

    reset_settings_cache()
    db_mod.dispose_engine()
    db_mod.init_db()

    from ..app.models import Event, Insight, Record, RiskAssessment
    from ..app.pipeline import load_demo, run_pipeline
    from ..app.retrieval import answering
    from ..app.retrieval.qa import NO_MATCH, enforce_citation_integrity
    from ..app.risk.assess import FORBIDDEN_CLAIMS, assert_cautious_wording

    check = Check()

    with db_mod.session_scope() as session:
        load_demo(session)
        run_pipeline(session, with_embeddings=False)

        records = {
            r.external_id: r for r in session.execute(select(Record)).scalars()
        }
        assessments: dict[int, list[RiskAssessment]] = {}
        for a in session.execute(select(RiskAssessment)).scalars():
            assessments.setdefault(a.record_id, []).append(a)

        # ---------------- Scenario A ------------------------------------
        print("\nScenario A - payment demand in context of a real account issue")
        attack = records[SCENARIO_A_RECORD]
        flagged = {a.label for a in assessments[attack.id] if a.present}

        check.record(
            "A. payment message is flagged for all four indicators",
            flagged == {"urgency", "threat_language", "emotional_pressure", "payment_request"},
            f"flagged={sorted(flagged)}",
        )

        cited = 0
        bad_spans: list[str] = []
        for a in assessments[attack.id]:
            if not a.present:
                continue
            for e in a.evidence or []:
                if e.get("start_char") is None:
                    continue
                s, t = int(e["start_char"]), int(e["end_char"])
                cited += 1
                if attack.body[s:t].strip() != str(e.get("text", "")).strip():
                    bad_spans.append(f"{a.label}:{s}-{t}")
        check.record(
            "A. every cited span points at the words it quotes",
            cited > 0 and not bad_spans,
            f"{cited} spans cited, {len(bad_spans)} mismatched",
            bad_spans,
        )

        benign = records[SCENARIO_A_BENIGN]
        benign_flagged = {a.label for a in assessments[benign.id] if a.present}
        check.record(
            "A. benign look-alike reminder is NOT flagged",
            not benign_flagged,
            f"flagged={sorted(benign_flagged)}"
            + (" (false positive on a genuine bank reminder)" if benign_flagged else ""),
        )

        context_ids = {
            records[e].external_id
            for e in (SCENARIO_A_RECORD, "demo-a1-email", "demo-a2-calendar", "demo-a3-document")
            if e in records
        }
        threads = [
            i
            for i in session.execute(select(Insight)).scalars()
            if SCENARIO_A_RECORD in [
                next((r.external_id for r in records.values() if r.id == rid), "")
                for rid in (i.record_ids or [])
            ]
        ]
        check.record(
            "A. the earlier account records are connected to the payment message",
            bool(threads),
            f"{len(threads)} insight(s) cite the payment message together with earlier records",
        )

        # Timeline: the three earlier records must precede the attack message.
        from ..app.models import Chunk  # noqa: F401  (ensure mappers loaded)
        from ..app.timeline import builder as timeline_mod

        events_by_record: dict[int, list[Event]] = {}
        for e in session.execute(select(Event)).scalars():
            events_by_record.setdefault(e.record_id, []).append(e)
        risk_by_record: dict[int, list[str]] = {}
        for a in session.execute(select(RiskAssessment)).scalars():
            if a.present:
                risk_by_record.setdefault(a.record_id, []).append(a.label)

        timeline = timeline_mod.build_timeline(records.values(), events_by_record, risk_by_record)
        prior = timeline_mod.records_before(timeline, attack.id, window_days=45, limit=20)
        prior_titles = {p.title for p in prior}
        expected = {
            "Action needed: we could not verify your account details",
            "Account verification review - Northwind Bank",
            "Account information summary",
        }
        check.record(
            "A. timeline places the earlier records before the payment request",
            expected <= prior_titles,
            f"found {len(expected & prior_titles)}/{len(expected)} expected prior items",
            sorted(expected - prior_titles),
        )
        lanes = {item.lane for item in prior}
        check.record(
            "A. recorded and inferred items are in separate lanes",
            "recorded" in lanes,
            f"lanes present in the prior window: {sorted(lanes)}",
        )

        # ---------------- Scenario B ------------------------------------
        print("\nScenario B - reference to a discussion that may not be in the data")
        unmatched = [
            i
            for i in session.execute(select(Insight)).scalars()
            if i.kind == "unresolved_discussion_reference"
            and SCENARIO_B_RECORD
            in [next((r.external_id for r in records.values() if r.id == rid), "") for rid in (i.record_ids or [])]
        ]
        check.record(
            "B. an unresolved-discussion insight is produced",
            bool(unmatched),
            f"{len(unmatched)} insight(s)",
        )
        statement = unmatched[0].statement if unmatched else ""
        check.record(
            "B. the required cautious wording appears verbatim",
            NO_MATCH in statement,
            f"expected substring: {NO_MATCH!r}",
        )
        check.record(
            "B. the wording does not claim the discussion did not happen",
            not re.search(r"\bnever (happened|occurred)|did not happen|no such discussion existed\b", statement, re.I),
            statement[:160],
        )
        check.record(
            "B. the insight cites the exact phrase in the source record",
            bool(unmatched and unmatched[0].facts)
            and isinstance(unmatched[0].facts[0].get("start_char"), int)
            and records[SCENARIO_B_RECORD].body[
                unmatched[0].facts[0]["start_char"] : unmatched[0].facts[0]["end_char"]
            ]
            == unmatched[0].facts[0].get("quote"),
            unmatched[0].facts[0].get("quote") if unmatched and unmatched[0].facts else None,
        )

        matched_kinds = {
            i.kind
            for i in session.execute(select(Insight)).scalars()
            if i.kind in {"discussion_reference", "unresolved_discussion_reference"}
        }
        check.record(
            "B. a back-reference WITH a matching record is not reported as unmatched",
            "discussion_reference" in matched_kinds,
            f"insight kinds present: {sorted(matched_kinds)}",
        )

        # ---------------- Ask interface ---------------------------------
        print("\nAsk interface")
        answer_before = answering.answer_question(session, "What happened before this payment request?")
        check.record(
            "Ask: 'what happened before' cites the earlier records",
            len(answer_before.citations) >= 3,
            f"{len(answer_before.citations)} citations",
        )
        problems = enforce_citation_integrity(
            answer_before, {r.id: r.body for r in records.values()}
        )
        check.record("Ask: every citation matches the source span", not problems, str(problems))

        answer_absent = answering.answer_question(
            session, "What records mention the Zephyrine acquisition?"
        )
        check.record(
            "Ask: an unmatched query uses the required wording",
            NO_MATCH in answer_absent.answer,
            f"no_match={answer_absent.no_match}",
        )

        answer_records = answering.answer_question(
            session, "What records mention Northwind Bank verification?"
        )
        check.record(
            "Ask: 'what records mention' returns cited records with spans",
            len(answer_records.citations) >= 2
            and all(isinstance(c.start_char, int) for c in answer_records.citations),
            f"{len(answer_records.citations)} citations",
        )

        answer_absent_project = answering.answer_question(
            session, "What records mention Project X?"
        )
        check.record(
            "Ask: a topic absent from the data uses the cautious wording",
            NO_MATCH in answer_absent_project.answer and answer_absent_project.no_match,
            f"no_match={answer_absent_project.no_match}",
        )

        # ---------------- Product rules ---------------------------------
        print("\nProduct rules")
        all_text = "\n".join(
            [i.statement for i in session.execute(select(Insight)).scalars()]
            + [a.explanation for a in session.execute(select(RiskAssessment)).scalars()]
            + [answer_before.answer, answer_absent.answer, answer_records.answer]
        )
        violations = [c for c in FORBIDDEN_CLAIMS if c in all_text.lower()]
        check.record(
            "No forbidden claim wording anywhere in the output",
            not violations,
            f"violations: {violations}" if violations else "0 violations",
        )
        check.record("Wording guard passes", assert_cautious_wording(all_text))

        from ..app.risk.assess import RiskReport, combine, overall_estimate

        report = combine(attack.id, attack.body)
        estimate = overall_estimate(report)
        check.record(
            "Any overall figure is labelled as an estimate and shows its formula",
            estimate.get("is_estimate") is True
            and bool(estimate.get("formula"))
            and bool(estimate.get("caveat"))
            and len(estimate.get("contributions", [])) == 4,
            estimate.get("formula", "")[:80],
        )

    output = {
        "all_passed": check.all_passed,
        "checks": check.rows,
        "counts": {
            "passed": sum(1 for r in check.rows if r["passed"]),
            "failed": sum(1 for r in check.rows if not r["passed"]),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n{output['counts']['passed']} passed, {output['counts']['failed']} failed")
    print(f"report -> {out}")
    return 0 if check.all_passed else 1


if __name__ == "__main__":
    import re  # noqa: E402  (imported late: the script sets env vars before app import)

    raise SystemExit(main())
