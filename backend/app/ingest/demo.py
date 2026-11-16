"""Hand-authored demo corpus: Scenario A, Scenario B, and benign look-alikes.

These records are written by hand rather than generated so the demo tells a
*coherent story*: one account problem, a meeting about it, a document with the account
details, and then a message that demands money against that same context.

Timestamps are fixed so the demo is reproducible. Everything is fictional.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

UTC = dt.timezone.utc

# A fixed "now" for the demo, so relative expressions resolve predictably.
DEMO_NOW = dt.datetime(2026, 3, 12, 16, 30, tzinfo=UTC)


@dataclass(frozen=True)
class DemoRecord:
    source_type: str
    title: str
    body: str
    occurred_at: dt.datetime
    participants: list[str]
    external_id: str
    extra: dict | None = None


# ---------------------------------------------------------------------------
# Scenario A - the escalation narrative
# ---------------------------------------------------------------------------

A1_EMAIL = """From: Northwind Bank <no-reply@northwind.example>
To: Priya Raman <priya.raman@studio.example>
Date: Tue, 10 Mar 2026 08:12:44 +0000
Subject: Action needed: we could not verify your account details

Dear Priya Raman,

We could not complete our routine verification of the account held with Northwind Bank.

Our records show that the account number and the registered address do not match what we
have on file. This sometimes happens after a move, a name change, or a recent account
update.

To keep your account open, please review your details in the secure form linked below and
confirm your current address. There is no charge for this verification.

If we do not hear from you within 14 days we will pause the account and you will need to
visit a branch in person with photo identification.

Regards,
Northwind Bank Customer Verification
"""

A2_CALENDAR = """Subject: Account verification review - Northwind Bank

When: 2026-03-11T14:00:00Z
Where: Room 4B / video link
Organiser: Priya Raman
Attendees: Priya Raman, Marcus Delgado

Agenda
------
1. Walk through the Northwind Bank verification notice Priya received on 10 March 2026.
2. Compare the account details on file with the paper statement.
3. Agree who contacts the bank branch, and by when.
4. Review the supplier bank details separately - these are still unchanged.

Action items from the last meeting are captured in the linked notes.
"""

A3_DOCUMENT = """Account information summary
Account holder: Priya Raman
Institution: Northwind Bank
Account reference: ending 4471
Registered address: 14 Ashcombe Row, Leeds LS2 4QQ

Verification status
-------------------
A verification request was issued on 10 March 2026 after the account details did not
match. The request asked the account holder to confirm the registered address.

Current position
----------------
The account remains open. No restriction has been applied. The account holder has an
appointment to review the details with the bank, scheduled for 11 March 2026.

Supplier payment details on file
--------------------------------
Northwind Bank, sort 20-45-71, account 88314472. These details are the ones on record and
should be verified by telephone before any change is made.

Prepared by Marcus Delgado, Operations.
"""

A4_NOTE = """Verification review - notes

Priya Raman and Marcus Delgado met on 11 March 2026 to go through the Northwind Bank
verification request.

Points agreed:
- The address on file predates Priya's move; the paper statement confirms the older
  address, which is why the check failed.
- Priya will call the branch rather than replying to the email, because the email asks
  for a link she does not trust.
- The supplier bank details in the account summary are unchanged and were not discussed
  beyond confirming they still read the same.
- Nothing about this review required a payment. Marcus noted explicitly that the bank
  representative on the phone confirmed there is no fee for verification.

Open question: whether the branch will accept the electronic confirmation instead of an
in-person visit.
"""

# The attack record. It borrows the exact context of A1-A4 and then asks for money.
A5_MESSAGE = """From: Northwind Bank <support@northwind-secure.example>
To: Priya Raman
Date: Thu, 12 Mar 2026 16:24:03 +0000
Subject: URGENT FINAL NOTICE - account closure in 10 minutes

Dear Priya Raman,

I am writing with genuine concern about your Northwind Bank account. We are extremely
worried about what this delay may mean for you and your family.

Your account will be closed today unless you act immediately. Our system can hold the
balance for only a short window.

To release the funds and keep your account open, please transfer $4,250.00 to the
verification account within 10 minutes. Once we confirm the payment, the closure is
lifted automatically and you will lose nothing.

If payment is not received we will have no choice but to close your account permanently
and report the balance to the collections team.

I am terrified that you will lose everything if this is not resolved right now. This is
devastating for our support team. Please reply only to this message.

Account Services
Northwind Bank
"""

# A1 continued: a later, genuine reminder from the real domain. It contains urgency and
# the word "closed", but nothing else. It must NOT be flagged as a payment request.
A6_EMAIL = """From: Northwind Bank <no-reply@northwind.example>
To: Priya Raman <priya.raman@studio.example>
Date: Thu, 12 Mar 2026 09:02:10 +0000
Subject: Reminder: 10 days left to verify your account details

Dear Priya Raman,

This is a reminder about the account verification request sent on 10 March 2026.

You have 10 days remaining to confirm your registered address in the secure form. If we
do not receive your confirmation, the account will be paused until you visit a branch.

There is no charge for verification, and we will never ask you to move money to unlock an
account. If you receive a request like that, please contact us using the number on your
card.

Regards,
Northwind Bank Customer Verification
"""

# ---------------------------------------------------------------------------
# Scenario B - the reference with no matching record
# ---------------------------------------------------------------------------

B1_MESSAGE = """From: Colin Whitfield <colin@quillpublishing.example>
To: Priya Raman
Date: Wed, 11 Mar 2026 11:40:52 +0000
Subject: Q2 supplier reconciliation

Hi Priya,

As we discussed last week, we agreed to push the Q2 supplier reconciliation to the end of
the month, and I have updated the tracker to reflect that.

Could you confirm the split between the Harbourline Credit Union account and the general
ledger before I circulate the summary? The amounts do not balance on my side.

Thanks,
Colin
"""

# A separate conversation that refers back to a discussion DigitalMind *does* have, so the
# search is demonstrably working in both directions. Its subject matter is deliberately
# unrelated to B1, so B1's reference stays unmatched.
B2_MATCH_RECORD = """From: Marcus Delgado <marcus.delgado@studio.example>
To: Priya Raman <priya.raman@studio.example>
Date: Tue, 3 Mar 2026 15:18:09 +0000
Subject: Atlas Freight booking window

Priya,

As we agreed on 2 March, Atlas Freight have moved our booking window by two days, so the
collection slot is now the 19th rather than the 17th.

I have updated the tracker and told the warehouse. Nothing needs your approval, but please
use the new date if you speak to them.

Thanks,
Marcus
"""

B2_SUPPORTING_RECORD = """From: Marcus Delgado <marcus.delgado@studio.example>
To: Priya Raman <priya.raman@studio.example>
Date: Mon, 2 Mar 2026 10:04:31 +0000
Subject: Booking slot

Priya,

Atlas Freight have offered us a collection slot on the 17th. Let me know if that still
works for the warehouse, otherwise I will take the 19th.

Marcus
"""


DEMO_RECORDS: list[DemoRecord] = [
    DemoRecord(
        source_type="email",
        title="Action needed: we could not verify your account details",
        body=A1_EMAIL,
        occurred_at=dt.datetime(2026, 3, 10, 8, 12, 44, tzinfo=UTC),
        participants=["Priya Raman", "Northwind Bank"],
        external_id="demo-a1-email",
    ),
    DemoRecord(
        source_type="calendar",
        title="Account verification review - Northwind Bank",
        body=A2_CALENDAR,
        occurred_at=dt.datetime(2026, 3, 11, 14, 0, 0, tzinfo=UTC),
        participants=["Priya Raman", "Marcus Delgado", "Northwind Bank"],
        external_id="demo-a2-calendar",
    ),
    DemoRecord(
        source_type="document",
        title="Account information summary",
        body=A3_DOCUMENT,
        occurred_at=dt.datetime(2026, 3, 11, 17, 30, 0, tzinfo=UTC),
        participants=["Priya Raman", "Marcus Delgado", "Northwind Bank"],
        external_id="demo-a3-document",
    ),
    DemoRecord(
        source_type="note",
        title="Verification review - notes",
        body=A4_NOTE,
        occurred_at=dt.datetime(2026, 3, 11, 15, 45, 0, tzinfo=UTC),
        participants=["Priya Raman", "Marcus Delgado"],
        external_id="demo-a4-note",
    ),
    DemoRecord(
        source_type="message",
        title="URGENT FINAL NOTICE - account closure in 10 minutes",
        body=A5_MESSAGE,
        occurred_at=dt.datetime(2026, 3, 12, 16, 24, 3, tzinfo=UTC),
        participants=["Priya Raman", "Northwind Bank"],
        external_id="demo-a5-message",
        extra={"scenario": "A"},
    ),
    DemoRecord(
        source_type="email",
        title="Reminder: 10 days left to verify your account details",
        body=A6_EMAIL,
        occurred_at=dt.datetime(2026, 3, 12, 9, 2, 10, tzinfo=UTC),
        participants=["Priya Raman", "Northwind Bank"],
        external_id="demo-a6-email",
        extra={"scenario": "A", "benign_lookalike": True},
    ),
    DemoRecord(
        source_type="message",
        title="Q2 supplier reconciliation",
        body=B1_MESSAGE,
        occurred_at=dt.datetime(2026, 3, 11, 11, 40, 52, tzinfo=UTC),
        participants=["Priya Raman", "Colin Whitfield"],
        external_id="demo-b1-message",
        extra={"scenario": "B"},
    ),
    DemoRecord(
        source_type="email",
        title="Atlas Freight booking window",
        body=B2_MATCH_RECORD,
        occurred_at=dt.datetime(2026, 3, 3, 15, 18, 9, tzinfo=UTC),
        participants=["Priya Raman", "Marcus Delgado", "Atlas Freight"],
        external_id="demo-b2-email",
        extra={"scenario": "B"},
    ),
    DemoRecord(
        source_type="email",
        title="Booking slot",
        body=B2_SUPPORTING_RECORD,
        occurred_at=dt.datetime(2026, 3, 2, 10, 4, 31, tzinfo=UTC),
        participants=["Priya Raman", "Marcus Delgado", "Atlas Freight"],
        external_id="demo-b3-email",
        extra={"scenario": "B"},
    ),
    DemoRecord(
        source_type="task",
        title="Confirm supplier bank details by phone",
        body=(
            "Confirm supplier bank details by phone before any payment is released.\n"
            "Owner: Marcus Delgado.\n"
            "Due: 13 March 2026.\n"
            "Note: use the number published on the supplier's own website, not one from an\n"
            "email or message."
        ),
        occurred_at=dt.datetime(2026, 3, 12, 10, 5, 0, tzinfo=UTC),
        participants=["Marcus Delgado"],
        external_id="demo-c1-task",
    ),
    DemoRecord(
        source_type="note",
        title="Week of 9 March - open items",
        body=(
            "Open items for the week of 9 March 2026:\n"
            "- Northwind Bank verification: awaiting branch confirmation.\n"
            "- Q2 supplier reconciliation: Colin Whitfield to send the ledger split.\n"
            "- Shipping window for the Atlas Freight order moved by two days.\n"
            "- Operations handover document needs a second reviewer."
        ),
        occurred_at=dt.datetime(2026, 3, 9, 9, 0, 0, tzinfo=UTC),
        participants=["Priya Raman", "Marcus Delgado", "Colin Whitfield"],
        external_id="demo-c2-note",
    ),
]
