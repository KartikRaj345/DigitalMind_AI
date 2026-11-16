"""API contract tests.

These assert the product promises at the HTTP boundary: the app runs with demo data, the
scenarios behave, and no response leaks a claim the project does not support.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def demo_client(client):
    """A client with the demo corpus loaded and analysed."""
    response = client.post("/api/import/demo")
    assert response.status_code == 200, response.text
    assert response.json()["inserted"] > 0
    return client


# ---------------------------------------------------------------------------
# Health and privacy posture
# ---------------------------------------------------------------------------


def test_health(client):
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["llm_enabled"] is False
    assert payload["external_transmission"] is False


def test_privacy_notice_states_local_first(client):
    payload = client.get("/api/privacy-notice").json()
    assert payload["local_first"] is True
    assert payload["external_transmission"] is False
    assert "None third-party" in payload["integrations"] or "no third-party" in payload["integrations"].lower()


def test_settings_reports_models_honestly(demo_client):
    payload = demo_client.get("/api/settings").json()
    assert payload["risk_models"]["rules"]["available"] is True
    assert "available" in payload["risk_models"]["tfidf_lr"]
    assert "available" in payload["risk_models"]["distilbert"]
    assert payload["privacy"]["external_transmission"] is False


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def test_dashboard_counts(demo_client):
    payload = demo_client.get("/api/dashboard").json()
    assert payload["records"]["total"] > 0
    assert set(payload["risk"]["indicator_counts"]) == {
        "urgency", "threat_language", "emotional_pressure", "payment_request",
    }
    assert "not verdicts" in payload["risk"]["disclaimer"]
    assert payload["graph"]["entities"] > 0


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def test_records_list(demo_client):
    payload = demo_client.get("/api/records", params={"limit": 100}).json()
    assert payload["total"] > 0
    assert all(item["id"] > 0 for item in payload["items"])
    assert any(item["risk_labels"] for item in payload["items"])


def test_records_filter_by_risk(demo_client):
    flagged = demo_client.get("/api/records", params={"risk": True, "limit": 50}).json()
    assert all(item["risk_labels"] for item in flagged["items"])


def test_record_detail_spans_are_exact(demo_client):
    listing = demo_client.get("/api/records", params={"limit": 100}).json()
    for item in listing["items"]:
        detail = demo_client.get(f"/api/records/{item['id']}").json()
        body = detail["body"]
        for span in detail["spans"]:
            assert 0 <= span["start_char"] <= span["end_char"] <= len(body)
            assert body[span["start_char"] : span["end_char"]] == span["text"]
        for indicator in detail["indicators"]:
            for evidence in indicator["evidence"]:
                if evidence["start_char"] is None:
                    continue
                start, end = evidence["start_char"], evidence["end_char"]
                assert 0 <= start <= end <= len(body)


def test_record_detail_estimate_is_labelled(demo_client):
    listing = demo_client.get("/api/records", params={"limit": 100}).json()
    flagged = next(i for i in listing["items"] if i["risk_labels"])
    detail = demo_client.get(f"/api/records/{flagged['id']}").json()
    estimate = detail["overall_estimate"]
    assert estimate["is_estimate"] is True
    assert estimate["formula"]
    assert estimate["caveat"]
    assert len(estimate["contributions"]) == 4


def test_record_not_found_is_404(client):
    assert client.get("/api/records/424242").status_code == 404


# ---------------------------------------------------------------------------
# Scenario A
# ---------------------------------------------------------------------------


def test_scenario_a_flags_payment_message_with_citations(demo_client):
    records = demo_client.get("/api/records", params={"risk": True, "limit": 50}).json()
    attack = next(
        item for item in records["items"] if "URGENT FINAL NOTICE" in item["title"]
    )
    assert set(attack["risk_labels"]) == {
        "urgency", "threat_language", "emotional_pressure", "payment_request",
    }

    detail = demo_client.get(f"/api/records/{attack['id']}").json()
    assert detail["overall_estimate"]["is_estimate"] is True
    quoted = [
        evidence["text"]
        for indicator in detail["indicators"]
        for evidence in indicator["evidence"]
        if evidence["text"]
    ]
    assert any("4,250" in q for q in quoted)
    assert any("10 minutes" in q for q in quoted)


def test_scenario_a_leaves_the_benign_reminder_alone(demo_client):
    records = demo_client.get("/api/records", params={"limit": 100}).json()
    benign = next(
        item for item in records["items"] if "Reminder: 10 days left" in item["title"]
    )
    assert benign["risk_labels"] == []
    assert benign["severity"] == "none"


def test_timeline_separates_lanes(demo_client):
    payload = demo_client.get("/api/timeline").json()
    lanes = {item["lane"] for item in payload["items"]}
    assert "recorded" in lanes
    assert set(payload["summary"]["explanations"]) == {"recorded", "explicit", "inferred"}
    for item in payload["items"]:
        assert item["explanation"]


# ---------------------------------------------------------------------------
# Scenario B
# ---------------------------------------------------------------------------


def test_scenario_b_uses_required_wording(demo_client):
    insights = demo_client.get("/api/insights", params={"status": "all"}).json()
    unresolved = [i for i in insights if i["kind"] == "unresolved_discussion_reference"]
    assert unresolved, "the demo corpus must produce an unmatched back-reference"
    for insight in unresolved:
        assert "No matching discussion was found in the records analyzed." in insight["statement"]
        assert "not evidence that no such discussion" in insight["statement"]
        assert insight["facts"]


# ---------------------------------------------------------------------------
# Retrieval and Ask
# ---------------------------------------------------------------------------


def test_search_returns_offsets_and_explanation(demo_client):
    payload = demo_client.get("/api/search", params={"q": "Northwind Bank verification"}).json()
    assert payload["hits"]
    for hit in payload["hits"]:
        assert 0 <= hit["start_char"] <= hit["end_char"]
        assert hit["agreement"] in {"both", "lexical_only", "semantic_only"}
    assert payload["explanation"]


def test_search_modes(demo_client):
    for mode in ("lexical", "semantic", "hybrid"):
        payload = demo_client.get(
            "/api/search", params={"q": "account", "mode": mode}
        ).json()
        assert payload["mode"] == mode


def test_ask_what_records_mention(demo_client):
    payload = demo_client.post(
        "/api/ask", json={"question": "What records mention Northwind Bank?"}
    ).json()
    assert payload["citations"]
    for citation in payload["citations"]:
        assert citation["start_char"] >= 0
        assert citation["quote"]
    assert payload["coverage"]["records_in_database"] > 0


def test_ask_what_happened_before(demo_client):
    payload = demo_client.post(
        "/api/ask", json={"question": "What happened before this payment request?"}
    ).json()
    assert payload["intent"] == "what_happened_before"
    assert payload["citations"]
    assert "recorded order" in payload["answer"] or "Before" in payload["answer"]


def test_ask_absent_topic_uses_cautious_wording(demo_client):
    payload = demo_client.post(
        "/api/ask", json={"question": "What records mention the Zephyrine acquisition?"}
    ).json()
    assert payload["no_match"] is True
    assert "No matching discussion was found in the records analyzed." in payload["answer"]
    assert "did not happen" not in payload["answer"].lower()
    assert payload["caveat"]


def test_ask_risk_reports_indicators_separately(demo_client):
    payload = demo_client.post("/api/ask", json={"question": "Which records are risky?"}).json()
    assert "separately" in payload["answer"] or "each" in payload["answer"].lower()
    assert "not a judgement" in payload["answer"] or "not verdicts" in payload["answer"]
    for fact in payload["facts"]:
        assert fact.get("indicator")


def test_ask_never_emits_a_forbidden_claim(demo_client):
    from backend.app.risk.assess import FORBIDDEN_CLAIMS

    for question in (
        "What records mention Northwind Bank?",
        "What happened before this payment request?",
        "Which records are risky?",
        "Who was involved?",
    ):
        answer = demo_client.post("/api/ask", json={"question": question}).json()["answer"]
        for claim in FORBIDDEN_CLAIMS:
            assert claim not in answer.lower()


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def test_graph_nodes_and_edges(demo_client):
    payload = demo_client.get("/api/graph", params={"max_nodes": 60}).json()
    assert payload["nodes"]
    node_ids = {node["id"] for node in payload["nodes"]}
    for edge in payload["edges"]:
        assert edge["source"] in node_ids
        assert edge["target"] in node_ids
        assert edge["confidence"] >= 0.0
    assert "evidenced by a record" in payload["explanation"]


def test_entity_detail_lists_merges(demo_client):
    graph = demo_client.get("/api/graph", params={"max_nodes": 60}).json()
    entity = next(node for node in graph["nodes"] if node["kind"] == "entity")
    payload = demo_client.get(f"/api/entities/{entity['record_id'] or ''}")
    if payload.status_code == 404:
        entity_id = int(entity["id"].split(":")[1])
        payload = demo_client.get(f"/api/entities/{entity_id}")
    assert payload.status_code == 200
    detail = payload.json()
    assert detail["name"]
    assert "caveat" in detail


# ---------------------------------------------------------------------------
# Insights lifecycle
# ---------------------------------------------------------------------------


def test_insight_can_be_dismissed_restored_and_corrected(demo_client):
    insights = demo_client.get("/api/insights").json()
    assert insights
    insight_id = insights[0]["id"]

    updated = demo_client.patch(
        f"/api/insights/{insight_id}", json={"status": "dismissed"}
    ).json()
    assert updated["status"] == "dismissed"

    corrected = demo_client.patch(
        f"/api/insights/{insight_id}",
        json={"status": "corrected", "user_note": "This is a legitimate reminder."},
    ).json()
    assert corrected["status"] == "corrected"
    assert corrected["user_note"] == "This is a legitimate reminder."

    active = demo_client.get("/api/insights").json()
    assert all(i["id"] != insight_id for i in active)
    all_insights = demo_client.get("/api/insights", params={"status": "all"}).json()
    assert any(i["id"] == insight_id for i in all_insights)


def test_insight_not_found_is_404(demo_client):
    assert demo_client.patch("/api/insights/9999", json={"status": "dismissed"}).status_code == 404


# ---------------------------------------------------------------------------
# Import and integrations
# ---------------------------------------------------------------------------


def test_import_status_shows_integrations_as_not_configured(demo_client):
    payload = demo_client.get("/api/import/status").json()
    assert payload["integrations"]
    for integration in payload["integrations"]:
        assert integration["configured"] is False
        assert integration["status"] == "not configured"
        assert integration["note"]
    assert set(payload["supported_files"]) == {".txt", ".md", ".eml", ".csv", ".json", ".ics"}


def test_import_rejects_unsupported_file(demo_client):
    response = demo_client.post(
        "/api/import/files",
        files=[("files", ("virus.exe", b"MZ...", "application/octet-stream"))],
        params={"reanalyze": "false"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["inserted"] == 0
    assert payload["errors"]
    assert "unsupported file type" in payload["errors"][0].lower()


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


def test_delete_record_cascades(demo_client):
    listing = demo_client.get("/api/records", params={"limit": 100}).json()
    record_id = listing["items"][0]["id"]

    payload = demo_client.delete(f"/api/records/{record_id}").json()
    assert payload["deleted"]["verified"] is True
    assert payload["deleted"]["problems"] == []
    assert "everything derived from it" in payload["message"]

    assert demo_client.get(f"/api/records/{record_id}").status_code == 404
    detail = demo_client.get("/api/graph", params={"max_nodes": 100}).json()
    assert all(node.get("record_id") != record_id for node in detail["nodes"])


def test_delete_all_data(demo_client):
    payload = demo_client.post("/api/settings/delete-all").json()
    assert payload["deleted"]["records"] > 0
    assert demo_client.get("/api/records").json()["total"] == 0
    dashboard = demo_client.get("/api/dashboard").json()
    assert dashboard["records"]["total"] == 0
    assert dashboard["graph"]["entities"] == 0


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_roundtrip(demo_client):
    updated = demo_client.patch(
        "/api/settings",
        json={"retrieval_mode": "lexical", "min_evidence_confidence": 0.7},
    ).json()
    assert updated["retrieval_mode"] == "lexical"
    assert updated["min_evidence_confidence"] == 0.7

    assert demo_client.get("/api/settings").json()["retrieval_mode"] == "lexical"


def test_cannot_enable_external_llm_without_an_endpoint(demo_client):
    response = demo_client.patch("/api/settings", json={"allow_external_llm": True})
    assert response.status_code == 400
    assert "No LLM endpoint is configured" in response.json()["detail"]


def test_unknown_api_route_is_404(client):
    assert client.get("/api/definitely-not-a-route").status_code == 404
