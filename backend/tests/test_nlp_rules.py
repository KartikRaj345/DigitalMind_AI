from backend.app.nlp import rules


def test_money_symbol():
    spans = rules.extract_money("Please transfer $4,250.00 to the verification account.")
    assert len(spans) == 1
    assert spans[0].label == rules.MONEY
    assert spans[0].normalized == "4250.00 USD"


def test_money_words():
    spans = rules.extract_money("They asked for 850 euros immediately.")
    assert spans[0].normalized == "850.00 EUR"


def test_money_no_overlap():
    text = "Pay £99.50 or 1200 dollars."
    spans = rules.extract_money(text)
    assert len(spans) == 2


def test_emails():
    spans = rules.extract_emails("Write to priya.raman@northwind.example for details.")
    assert spans[0].label == rules.PERSON
    assert "Priya" in spans[0].normalized


def test_orgs_suffix():
    spans = rules.extract_orgs("Northwind Bank and Acme Holdings Ltd confirmed.")
    names = {s.text for s in spans}
    assert "Northwind Bank" in names
    assert "Acme Holdings Ltd" in names


def test_bank_of():
    spans = rules.extract_orgs("Contact Bank of Raval about the transfer.")
    assert any("Bank of Raval" in s.text for s in spans)


def test_person_title():
    spans = rules.extract_person_names("Dear Dr Alvarez, please review.")
    assert any(s.text == "Dr Alvarez" or s.text == "Alvarez" for s in spans)


def test_person_rejects_sentence_starters():
    spans = rules.extract_person_names("Meeting Account Review tomorrow.")
    assert all(s.text not in {"Meeting Account", "Account Review"} for s in spans)


def test_events_needs_qualifier():
    spans = rules.extract_events("The meeting is scheduled for Tuesday.")
    assert any("scheduled" in s.text for s in spans)
    bare = rules.extract_events("We had a meeting.")
    assert all("meeting" != s.text.strip().lower() for s in bare)


def test_extract_all_overlaps_resolved():
    text = "Northwind Bank emailed Priya Raman on 4 March 2026 about $500.00."
    spans = rules.extract_all(text)
    for i, a in enumerate(spans):
        for b in spans[i + 1 :]:
            assert a.end <= b.start or b.end <= a.start


def test_extract_all_spans_are_exact_substrings():
    text = "Northwind Bank emailed Priya Raman on 4 March 2026 about $500.00."
    for s in rules.extract_all(text):
        assert text[s.start : s.end] == s.text
