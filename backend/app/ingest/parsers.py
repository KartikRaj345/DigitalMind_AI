"""File parsers for uploads.

Supports ``.txt``, ``.md``, ``.eml``, ``.csv``, ``.json``, ``.ics``. Everything is parsed
with the standard library - no external converter, nothing leaves the machine.

Each parser returns :class:`~app.ingest.demo.DemoRecord` objects with the raw text
preserved as ``body``. Preserving the body exactly matters: every character offset in the
UI refers to it, so normalising whitespace here would silently break every citation.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import re
from email import message_from_string
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from .demo import DemoRecord

UTC = dt.timezone.utc
SUPPORTED_EXTENSIONS = {".txt", ".md", ".eml", ".csv", ".json", ".ics"}

MAX_BYTES = 8 * 1024 * 1024


class ParseError(ValueError):
    """Raised when a file cannot be parsed into records."""


def parse(filename: str, text: str) -> list[DemoRecord]:
    """Dispatch on file extension."""
    path = Path(filename)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ParseError(
            f"Unsupported file type '{ext or filename}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if len(text.encode("utf-8", errors="replace")) > MAX_BYTES:
        raise ParseError("File is larger than the 8 MB limit.")

    handler = {
        ".txt": _parse_plain,
        ".md": _parse_markdown,
        ".eml": _parse_eml,
        ".csv": _parse_csv,
        ".json": _parse_json,
        ".ics": _parse_ics,
    }[ext]
    return handler(path, text)


# ---------------------------------------------------------------------------
# .txt / .md
# ---------------------------------------------------------------------------


def _split_sections(text: str, marker: str = "---") -> list[tuple[str, str]]:
    """Split a file into ``(heading, body)`` sections on a horizontal rule.

    A single-section file returns one record, which is the common case. A section's first
    line becomes the heading only when it looks like one (a markdown ``#`` heading, or a
    short line with content beneath it) - otherwise a one-line plain-text section would
    lose its entire body to the title.
    """
    pattern = re.compile(rf"^\s*{re.escape(marker)}\s*$", re.MULTILINE)
    parts = pattern.split(text)
    if len(parts) == 1:
        return [("", text)]
    out: list[tuple[str, str]] = []
    preamble = parts[0].strip()
    if preamble:
        out.append(("", preamble))
    for part in parts[1:]:
        stripped = part.lstrip("\n")
        lines = stripped.split("\n")
        remainder = "\n".join(lines[1:]).strip()
        first = lines[0].strip()
        looks_like_heading = bool(first) and bool(remainder) and (
            first.startswith("#") or len(first) <= 80
        )
        if looks_like_heading:
            out.append((first.lstrip("#").strip(), remainder))
        else:
            out.append(("", stripped.strip()))
    return out


def _parse_plain(path: Path, text: str) -> list[DemoRecord]:
    if not text.strip():
        return []
    records: list[DemoRecord] = []
    for i, (heading, body) in enumerate(_split_sections(text)):
        content = body.strip()
        if not content:
            continue
        title = heading or _first_line(content) or path.stem
        records.append(
            DemoRecord(
                source_type="note",
                title=title[:512],
                body=content,
                occurred_at=_file_mtime_placeholder(),
                participants=[],
                external_id=f"upload:{path.name}:{i}",
            )
        )
    return records


def _parse_markdown(path: Path, text: str) -> list[DemoRecord]:
    if not text.strip():
        return []
    records: list[DemoRecord] = []
    sections = _split_sections(text)
    if len(sections) == 1:
        sections = _split_on_headings(text)
    for i, (heading, body) in enumerate(sections):
        content = body.strip()
        if not content:
            continue
        title = heading or _first_line(content) or path.stem
        records.append(
            DemoRecord(
                source_type="document" if len(sections) > 1 else "note",
                title=title[:512],
                body=content,
                occurred_at=_file_mtime_placeholder(),
                participants=_extract_participants(content),
                external_id=f"upload:{path.name}:{i}",
            )
        )
    return records


def _split_on_headings(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"^#{1,3}\s+(.+)$", text, re.MULTILINE))
    if not matches:
        return [("", text)]
    out: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        out.append(("", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group(1).strip(), text[m.start() : end]))
    return out


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _extract_participants(body: str) -> list[str]:
    m = re.search(
        r"^\s*(?:participants?|attendees?|author|from)\s*:\s*(.+)$",
        body,
        re.IGNORECASE | re.MULTILINE,
    )
    if not m:
        return []
    raw = m.group(1)
    parts = re.split(r"[,;]| and ", raw)
    return [p.strip() for p in parts if p.strip()][:20]


def _file_mtime_placeholder() -> dt.datetime:
    """Imported files have no reliable source timestamp.

    We use the process start time and mark the record so the timeline keeps it out of the
    "recorded" lane. See timeline/builder.py: a record with no source timestamp is not
    presented as a fact.
    """
    return dt.datetime.now(UTC)


# ---------------------------------------------------------------------------
# .eml
# ---------------------------------------------------------------------------


def _parse_eml(path: Path, text: str) -> list[DemoRecord]:
    msg = message_from_string(text)
    subject = _header(msg, "Subject") or path.stem
    sender = _parse_address(_header(msg, "From"))
    recipients = [
        _parse_address(a)
        for a in re.split(r"[;,]", _header(msg, "To") or "")
        if a.strip()
    ]
    cc = [_parse_address(a) for a in re.split(r"[;,]", _header(msg, "Cc") or "") if a.strip()]
    date_hdr = _header(msg, "Date")
    occurred = _parse_email_date(date_hdr)

    body = _eml_body(msg)
    headers = "\n".join(
        f"{k}: {v}" for k, v in [("From", _header(msg, "From")), ("To", _header(msg, "To")),
                                 ("Date", date_hdr), ("Subject", subject)] if v
    )
    content = f"{headers}\n\n{body}" if headers else body

    participants = [p for p in [sender, *recipients, *cc] if p]
    return [
        DemoRecord(
            source_type="email",
            title=subject[:512],
            body=content.strip(),
            occurred_at=occurred,
            participants=participants,
            external_id=f"upload:{path.name}",
            extra={"has_source_timestamp": occurred is not None},
        )
    ]


def _header(msg, name: str) -> str | None:
    value = msg.get(name)
    return str(value).strip() if value else None


def _parse_address(raw: str | None) -> str | None:
    if not raw:
        return None
    name, addr = parseaddress(raw)
    return name or addr or None


def parseaddress(raw: str) -> tuple[str, str]:
    from email.utils import parseaddr as _pa

    try:
        return _pa(raw)
    except Exception:
        return "", raw


def _parse_email_date(raw: str | None) -> dt.datetime | None:
    if not raw:
        return None
    try:
        dtv = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dtv is None:
        return None
    if dtv.tzinfo is None:
        dtv = dtv.replace(tzinfo=UTC)
    return dtv.astimezone(UTC)


def _eml_body(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload is None:
                    return str(part.get_payload())
                charset = part.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")
        return ""
    payload = msg.get_payload(decode=True)
    if payload is None:
        return str(msg.get_payload())
    charset = msg.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


# ---------------------------------------------------------------------------
# .csv
# ---------------------------------------------------------------------------

_COLUMN_ALIASES = {
    "title": {"title", "subject", "name", "summary", "headline"},
    "body": {"body", "text", "content", "message", "description", "snippet"},
    "date": {"date", "occurred_at", "timestamp", "datetime", "sent_at", "created_at", "time"},
    "source_type": {"source_type", "type", "kind", "channel"},
    "participants": {"participants", "from", "sender", "attendees", "author", "to"},
    "id": {"id", "external_id", "uid", "message_id"},
}

VALID_SOURCE_TYPES = {"email", "message", "document", "note", "calendar", "task"}


def _parse_csv(path: Path, text: str) -> list[DemoRecord]:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise ParseError("CSV has no header row.")

    mapping = _map_columns(reader.fieldnames)
    if "body" not in mapping and "title" not in mapping:
        raise ParseError(
            "CSV needs at least a 'body'/'text'/'content' or 'subject'/'title' column. "
            f"Found: {', '.join(reader.fieldnames)}"
        )

    records: list[DemoRecord] = []
    for i, row in enumerate(reader):
        body = (row.get(mapping["body"]) if "body" in mapping else "") or ""
        title = (row.get(mapping["title"]) if "title" in mapping else "") or ""
        if not body.strip() and not title.strip():
            continue
        content = body.strip() or title.strip()
        stype = (row.get(mapping["source_type"]) if "source_type" in mapping else "") or "note"
        stype = stype.strip().lower()
        if stype not in VALID_SOURCE_TYPES:
            stype = "message" if re.search(r"@", content) else "note"

        participants = []
        if "participants" in mapping:
            raw = row.get(mapping["participants"]) or ""
            participants = [p.strip() for p in re.split(r"[;,]| and ", raw) if p.strip()][:20]

        occurred = None
        if "date" in mapping:
            occurred = _parse_loose_datetime(row.get(mapping["date"]))

        ext_id = (row.get(mapping["id"]) if "id" in mapping else "") or f"{path.name}:{i}"
        records.append(
            DemoRecord(
                source_type=stype,
                title=(title.strip() or _first_line(content))[:512],
                body=content,
                occurred_at=occurred or _file_mtime_placeholder(),
                participants=participants,
                external_id=f"upload:{path.stem}:{ext_id}"[:250],
                extra={"has_source_timestamp": occurred is not None},
            )
        )
    if not records:
        raise ParseError("CSV contained no non-empty rows.")
    return records


def _map_columns(fieldnames: list[str]) -> dict[str, str]:
    lowered = {f.strip().lower(): f for f in fieldnames if f}
    out: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                out[canonical] = lowered[alias]
                break
    return out


def _parse_loose_datetime(raw: str | None) -> dt.datetime | None:
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    from dateutil import parser as dateparser

    try:
        dtv = dateparser.parse(raw, fuzzy=False, default=None)
    except (ValueError, OverflowError, TypeError):
        return None
    if dtv is None:
        return None
    if dtv.tzinfo is None:
        dtv = dtv.replace(tzinfo=UTC)
    return dtv.astimezone(UTC)


# ---------------------------------------------------------------------------
# .json
# ---------------------------------------------------------------------------


def _parse_json(path: Path, text: str) -> list[DemoRecord]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"Invalid JSON: {exc.msg} at line {exc.lineno}") from exc

    if isinstance(data, dict):
        for key in ("records", "messages", "items", "data", "emails"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise ParseError("JSON must be an object, a list, or an object with a 'records' list.")

    records: list[DemoRecord] = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            records.append(
                DemoRecord(
                    source_type="note",
                    title=_first_line(item)[:512] or path.stem,
                    body=item.strip(),
                    occurred_at=_file_mtime_placeholder(),
                    participants=[],
                    external_id=f"upload:{path.name}:{i}",
                )
            )
            continue
        if not isinstance(item, dict):
            continue
        mapping = _map_columns([str(k) for k in item.keys()])
        body = str(item.get(mapping["body"], "") if "body" in mapping else "")
        title = str(item.get(mapping["title"], "") if "title" in mapping else "")
        if not body.strip() and not title.strip():
            continue
        content = body.strip() or title.strip()
        stype = str(item.get(mapping["source_type"], "") if "source_type" in mapping else "note").lower()
        if stype not in VALID_SOURCE_TYPES:
            stype = "message" if re.search(r"@", content) else "note"
        occurred = (
            _parse_loose_datetime(str(item.get(mapping["date"])))
            if "date" in mapping
            else None
        )
        participants: list[str] = []
        if "participants" in mapping:
            raw_participants = item.get(mapping["participants"])
            if isinstance(raw_participants, list):
                participants = [str(p).strip() for p in raw_participants if str(p).strip()][:20]
            elif raw_participants:
                participants = [
                    p.strip() for p in re.split(r"[;,]| and ", str(raw_participants)) if p.strip()
                ][:20]
        ext = str(item.get(mapping["id"], "")) if "id" in mapping else str(i)
        records.append(
            DemoRecord(
                source_type=stype,
                title=(title.strip() or _first_line(content))[:512],
                body=content,
                occurred_at=occurred or _file_mtime_placeholder(),
                participants=participants,
                external_id=f"upload:{path.stem}:{ext}"[:250],
                extra={"has_source_timestamp": occurred is not None},
            )
        )
    if not records:
        raise ParseError("JSON contained no usable records.")
    return records


# ---------------------------------------------------------------------------
# .ics
# ---------------------------------------------------------------------------


def _parse_ics(path: Path, text: str) -> list[DemoRecord]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    unfolded: list[str] = []
    for line in lines:
        if line[:1] in (" ", "\t") and unfolded:
            unfolded[-1] += line[1:]
        else:
            unfolded.append(line)

    events: list[list[str]] = []
    current: list[str] | None = None
    in_event = False
    for line in unfolded:
        stripped = line.strip()
        if stripped.upper() == "BEGIN:VEVENT":
            current = []
            in_event = True
            continue
        if stripped.upper() == "END:VEVENT":
            if current:
                events.append(current)
            current = None
            in_event = False
            continue
        if in_event and current is not None:
            current.append(stripped)

    records: list[DemoRecord] = []
    for i, block in enumerate(events):
        props: dict[str, list[str]] = {}
        params: dict[str, list[str]] = {}
        for line in block:
            if ":" not in line:
                continue
            key_part, value = line.split(":", 1)
            segments = key_part.split(";")
            key = segments[0].upper()
            props.setdefault(key, []).append(value)
            # ORGANIZER;CN=Priya Raman:mailto:... - the display name lives in the CN
            # parameter, not in the value, so it has to be captured separately.
            for segment in segments[1:]:
                if "=" in segment:
                    name, _, param_value = segment.partition("=")
                    params.setdefault(name.upper(), []).append(param_value)
            for key_name in ("ATTENDEE", "ORGANIZER"):
                if key == key_name:
                    for segment in segments[1:]:
                        if segment.upper().startswith("CN="):
                            params.setdefault(f"{key_name}.CN", []).append(
                                segment[3:].strip('"')
                            )

        summary = props.get("SUMMARY", ["(untitled event)"])[0]
        description = props.get("DESCRIPTION", [""])[0]
        location = props.get("LOCATION", [""])[0]
        organizer = props.get("ORGANIZER", [""])[0]
        attendees = props.get("ATTENDEE", [])
        start_raw = (props.get("DTSTART") or [""])[0]
        end_raw = (props.get("DTEND") or [""])[0]

        start = _parse_ics_datetime(start_raw)
        end = _parse_ics_datetime(end_raw)

        participants: list[str] = []
        organizer_name = (params.get("ORGANIZER.CN") or [""])[0]
        if organizer_name:
            participants.append(organizer_name)
        elif organizer:
            participants.append(_parse_address(organizer) or organizer)
        for j, attendee in enumerate(attendees):
            cn_values = params.get("ATTENDEE.CN") or []
            display = cn_values[j] if j < len(cn_values) else ""
            parsed = display or _parse_address(attendee)
            if parsed:
                participants.append(parsed)

        header_lines = [
            f"Summary: {summary}",
            f"Start: {start_raw}",
            *([f"End: {end_raw}"] if end_raw else []),
            *([f"Location: {location}"] if location else []),
            *([f"Organiser: {organizer}"] if organizer else []),
            *([f"Attendees: {', '.join(participants)}"] if participants else []),
        ]
        content = "\n".join(header_lines) + "\n\n" + _unescape_ics(description)
        uid = props.get("UID", [f"{i}"])[0]

        records.append(
            DemoRecord(
                source_type="calendar",
                title=summary[:512],
                body=content.strip(),
                occurred_at=start or _file_mtime_placeholder(),
                participants=participants,
                external_id=f"upload:{path.stem}:{uid}"[:250],
                extra={
                    "has_source_timestamp": start is not None,
                    "location": location,
                    "ends_at": end.isoformat() if end else None,
                },
            )
        )
    if not records:
        raise ParseError("No VEVENT blocks found in the .ics file.")
    return records


def _parse_ics_datetime(raw: str) -> dt.datetime | None:
    if not raw:
        return None
    value = raw.strip()
    # 20260311T140000Z / 20260311T140000 / 20260311
    m = re.match(r"^(\d{4})(\d{2})(\d{2})(?:T(\d{2})(\d{2})(\d{2})(Z)?)?$", value)
    if not m:
        return _parse_loose_datetime(value)
    year, month, day, hour, minute, second, zulu = m.groups()
    try:
        dtv = dt.datetime(
            int(year), int(month), int(day),
            int(hour or 0), int(minute or 0), int(second or 0),
            tzinfo=UTC if zulu else None,
        )
    except ValueError:
        return None
    return dtv if dtv.tzinfo else dtv.replace(tzinfo=UTC)


def _unescape_ics(value: str) -> str:
    return (
        value.replace("\\n", "\n")
        .replace("\\N", "\n")
        .replace("\\,", ",")
        .replace("\\;", ";")
        .replace("\\\\", "\\")
    )


def _parse_address(raw: str) -> str | None:  # noqa: F811 - ICS values may carry mailto:
    if not raw:
        return None
    value = raw.split(":", 1)[-1] if raw.lower().startswith("mailto:") else raw
    name, addr = parseaddress(value)
    return name or addr or None


def suggested_origin(filename: str) -> str:
    return f"upload:{Path(filename).name}"


def describe_formats() -> list[dict[str, Any]]:
    return [
        {"ext": ".txt", "what": "Plain text, one record per `---` section"},
        {"ext": ".md", "what": "Markdown, split on headings or `---`"},
        {"ext": ".eml", "what": "Email message with headers and body"},
        {"ext": ".csv", "what": "Needs a body/text column, optional date and participants"},
        {"ext": ".json", "what": "List of objects, or an object with a 'records' list"},
        {"ext": ".ics", "what": "Calendar events (VEVENT blocks)"},
    ]
