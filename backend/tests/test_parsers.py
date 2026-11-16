"""File-parser tests for the upload path."""

from __future__ import annotations

import datetime as dt

import pytest

from backend.app.ingest import parsers

UTC = dt.timezone.utc

EML = """From: Priya Raman <priya.raman@studio.example>
To: Marcus Delgado <marcus.delgado@studio.example>
Date: Tue, 10 Mar 2026 08:12:44 +0000
Subject: Account verification
Content-Type: text/plain; charset="utf-8"

Hi Marcus,

The account verification for Northwind Bank is still open.

Priya
"""

CSV_TEXT = """title,body,date,participants,source_type
Verification note,Please confirm the Northwind Bank details.,2026-03-11 14:00,Priya Raman;Marcus Delgado,note
Second note,Follow up on the same review.,2026-03-12,"Marcus Delgado, Priya Raman",message
"""

JSON_TEXT = """[
  {"title": "Supplier note", "text": "The supplier bank details are unchanged.", "date": "2026-03-11", "type": "document"},
  {"title": "Task", "text": "Call the branch to verify.", "date": "2026-03-12T10:00:00Z", "type": "task"}
]"""

ICS_TEXT = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:evt-1@example
DTSTAMP:20260301T090000Z
DTSTART:20260311T140000Z
DTEND:20260311T150000Z
SUMMARY:Account verification review
LOCATION:Room 4B
ORGANIZER;CN=Priya Raman:mailto:priya.raman@studio.example
ATTENDEE;CN=Marcus Delgado:mailto:marcus.delgado@studio.example
DESCRIPTION:Walk through the verification notice\\, then decide who calls the branch.
END:VEVENT
BEGIN:VEVENT
UID:evt-2@example
DTSTAMP:20260301T090000Z
DTSTART:20260313T090000Z
SUMMARY:Follow-up call
DESCRIPTION:Confirm the branch accepted the electronic confirmation.
END:VEVENT
END:VCALENDAR
"""

MD_TEXT = """# Week of 9 March

Open items are listed in the note below.

---

# Northwind Bank verification

Awaiting branch confirmation.

- Priya Raman to call
- Marcus Delgado to prepare documents
"""


def test_parses_eml():
    records = parsers.parse("mail.eml", EML)
    assert len(records) == 1
    rec = records[0]
    assert rec.source_type == "email"
    assert rec.title == "Account verification"
    assert rec.occurred_at is not None
    assert rec.occurred_at.year == 2026 and rec.occurred_at.month == 3
    assert "Subject: Account verification" in rec.body
    assert "The account verification for Northwind Bank is still open." in rec.body
    assert any("Priya Raman" in p for p in rec.participants)


def test_parses_csv_with_date_and_participants():
    records = parsers.parse("rows.csv", CSV_TEXT)
    assert len(records) == 2
    first, second = records
    assert first.occurred_at is not None
    assert first.occurred_at.day == 11
    assert set(first.participants) == {"Priya Raman", "Marcus Delgado"}
    assert second.source_type == "message"


def test_csv_without_date_is_flagged_not_invented():
    text = "title,body\nNo date here,Body without a date.\n"
    records = parsers.parse("nodate.csv", text)
    assert records[0].extra["has_source_timestamp"] is False


def test_csv_missing_required_columns_raises_clear_error():
    with pytest.raises(parsers.ParseError) as exc:
        parsers.parse("bad.csv", "colour,size\nred,large\n")
    assert "body" in str(exc.value)


def test_parses_json_list():
    records = parsers.parse("data.json", JSON_TEXT)
    assert len(records) == 2
    assert records[0].source_type == "document"
    assert records[1].source_type == "task"
    assert records[1].occurred_at.hour == 10


def test_json_object_with_records_key():
    records = parsers.parse("wrapped.json", '{"records": [{"text": "hello"}]}')
    assert len(records) == 1


def test_invalid_json_raises():
    with pytest.raises(parsers.ParseError) as exc:
        parsers.parse("broken.json", "{not json")
    assert "Invalid JSON" in str(exc.value)


def test_parses_ics_events():
    records = parsers.parse("cal.ics", ICS_TEXT)
    assert len(records) == 2
    first = records[0]
    assert first.source_type == "calendar"
    assert first.title == "Account verification review"
    assert first.occurred_at == dt.datetime(2026, 3, 11, 14, 0, tzinfo=UTC)
    assert first.extra["location"] == "Room 4B"
    assert any("Priya Raman" in p for p in first.participants)
    assert any("Marcus Delgado" in p for p in first.participants)
    # The escaped comma must be unescaped in the description.
    assert "notice, then decide" in first.body


def test_ics_without_events_raises():
    with pytest.raises(parsers.ParseError):
        parsers.parse("empty.ics", "BEGIN:VCALENDAR\nVERSION:2.0\nEND:VCALENDAR\n")


def test_parses_markdown_sections():
    records = parsers.parse("notes.md", MD_TEXT)
    assert len(records) >= 2
    titles = [r.title for r in records]
    assert any("Northwind Bank verification" in t for t in titles)


def test_parses_plain_text_sections():
    text = "First note body.\n\n---\n\nSecond note body.\n"
    records = parsers.parse("notes.txt", text)
    assert len(records) == 2
    assert records[0].body == "First note body."
    assert records[1].body == "Second note body."


def test_unsupported_extension_is_refused():
    with pytest.raises(parsers.ParseError) as exc:
        parsers.parse("evil.exe", "binary")
    assert "Unsupported file type" in str(exc.value)


def test_supported_extensions_cover_the_documented_set():
    assert parsers.SUPPORTED_EXTENSIONS == {".txt", ".md", ".eml", ".csv", ".json", ".ics"}


def test_empty_text_produces_no_records():
    assert parsers.parse("empty.md", "   \n\n  ") == []


def test_parsed_records_survive_the_pipeline(client, tmp_path):
    """An uploaded record must be analysable, with valid spans."""
    response = client.post(
        "/api/import/files",
        files=[("files", ("mail.eml", EML, "message/rfc822"))],
        params={"reanalyze": "true"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["inserted"] == 1
    record_id = payload["record_ids"][0]

    detail = client.get(f"/api/records/{record_id}").json()
    for span in detail["spans"]:
        assert 0 <= span["start_char"] < span["end_char"] <= len(detail["body"])
        assert detail["body"][span["start_char"] : span["end_char"]] == span["text"]
