"""spaCy NER wrapper.

Loads the model lazily and degrades cleanly: if ``en_core_web_sm`` is not installed the
pipeline still works, using only :mod:`app.nlp.rules`, and the API reports
``spacy_available: false`` so the UI can be honest about what produced a result.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from functools import lru_cache

from ..logging_config import get_logger

logger = get_logger(__name__)

# spaCy labels -> DigitalMind labels. ``None`` means "do not surface".
LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "PRODUCT": "PRODUCT",
    "GPE": None,        # places are not organizations here; see _gpe_is_org
    "LOC": None,
    "EVENT": "EVENT",
    "DATE": "DATE",
    "TIME": "DATE",
    "MONEY": "MONEY",
    "CARDINAL": None,
    "ORDINAL": None,
    "PERCENT": None,
    "QUANTITY": None,
    "LANGUAGE": None,
    "LAW": None,
    "NORP": None,
}

# A GPE/LOC is treated as an organization only when it carries an institutional
# descriptor. Without this, spaCy tags "Leeds" and "Dear Priya Raman" style spans as
# places, and the graph fills up with locations pretending to be counterparties.
_ORG_DESCRIPTOR = re.compile(
    r"\b(?:bank|trust|union|savings|lending|mutual|capital|financial|insurance|"
    r"university|college|school|hospital|clinic|foundation|institute|"
    r"department|ministry|agency|authority|commission|council|"
    r"inc|ltd|llc|plc|gmbh|group|holdings|partners|"
    r"verification|support|service|services|team|office|billing|payments|"
    r"security|reminder|notice|desk|centre|center|division)\b",
    re.IGNORECASE,
)

# "16:24:03", "14:30", "09:02:10"
_TIME_LIKE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")

# "Dear Priya Raman", "Hi Marcus", "Regards Priya"
_SALUTATION_PREFIX = re.compile(
    r"^(?:dear|hi|hello|hail|greetings|regards|kind regards|best regards|thanks|"
    r"good (?:morning|afternoon|evening))\b",
    re.IGNORECASE,
)

# "Leeds LS2", "LS1 4QQ" - a postcode is not a person.
_POSTCODE_LIKE = re.compile(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b")

_lock = threading.Lock()
_nlp = None
_load_failed = False


@dataclass
class NerSpan:
    text: str
    label: str
    start: int
    end: int
    confidence: float
    extractor: str = "spacy"


def load_nlp(model_name: str = "en_core_web_sm"):
    """Load and cache the spaCy pipeline. Returns ``None`` if unavailable."""
    global _nlp, _load_failed
    if _nlp is not None or _load_failed:
        return _nlp
    with _lock:
        if _nlp is not None or _load_failed:
            return _nlp
        try:
            import spacy

            _nlp = spacy.load(model_name, disable=["lemmatizer", "textcat"])
            logger.info("loaded spacy model=%s pipelines=%s", model_name, _nlp.pipe_names)
        except Exception as exc:  # pragma: no cover - depends on local install
            _load_failed = True
            logger.warning("spaCy unavailable (%s); using rules extractor only", type(exc).__name__)
    return _nlp


def spacy_available(model_name: str = "en_core_web_sm") -> bool:
    return load_nlp(model_name) is not None


def _is_noise_span(text: str, label: str) -> bool:
    """Reject spans that are obviously not the labelled kind of thing.

    spaCy is a general model and will happily call "16:24:03" a PERSON and
    "Dear Priya Raman" an ORG. Those become nodes in the knowledge graph, so they have to
    be filtered here rather than shown to the user and defended as model output.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if _TIME_LIKE.match(stripped):
        return True
    if label in {"PERSON", "ORG"} and _SALUTATION_PREFIX.match(stripped):
        return True
    if label in {"PERSON", "ORG"} and _ORG_DESCRIPTOR.fullmatch(stripped):
        # A bare descriptor like "Customer Verification" is a role, not a name.
        return True
    if _POSTCODE_LIKE.search(stripped):
        return True
    if label in {"PERSON", "ORG"} and ("\n" in stripped or len(stripped) > 60):
        # A name or organization does not span a line break in these records. spaCy
        # sometimes runs an entity from a signature into the next header line; left alone
        # that becomes a graph node covering two unrelated things.
        return True
    return False


def extract(text: str, model_name: str = "en_core_web_sm") -> list[NerSpan]:
    """Run spaCy NER and map its labels onto DigitalMind's taxonomy."""
    nlp = load_nlp(model_name)
    if nlp is None or not text.strip():
        return []
    try:
        doc = nlp(text)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("spaCy inference failed (%s)", type(exc).__name__)
        return []

    out: list[NerSpan] = []
    for ent in doc.ents:
        label = LABEL_MAP.get(ent.label_, None)
        if ent.label_ in {"GPE", "LOC"}:
            label = "ORG" if _ORG_DESCRIPTOR.search(ent.text) else None
        if label is None:
            continue
        if _is_noise_span(ent.text, label):
            continue
        # Trim leading/trailing whitespace so offsets stay tight.
        raw = ent.text
        lead = len(raw) - len(raw.lstrip())
        trimmed = raw.strip()
        if not trimmed:
            continue
        start = ent.start_char + lead
        end = start + len(trimmed)
        # spaCy has no per-entity probability; we assign a stable proxy so the UI can
        # still sort and filter. Documented as a limitation in docs/DECISIONS.md.
        confidence = 0.85 if ent.label_ in {"PERSON", "ORG", "GPE", "MONEY"} else 0.8
        out.append(
            NerSpan(
                text=trimmed,
                label=label,
                start=start,
                end=end,
                confidence=confidence,
            )
        )
    return out


def sentences(text: str, model_name: str = "en_core_web_sm") -> list[tuple[int, int, str]]:
    """Sentence segmentation with character offsets. Falls back to line splitting."""
    nlp = load_nlp(model_name)
    if nlp is None:
        return _fallback_sentences(text)
    try:
        doc = nlp(text)
    except Exception:  # pragma: no cover - defensive
        return _fallback_sentences(text)
    if not doc.has_annotation("SENT_START") and not doc.sents:
        return _fallback_sentences(text)
    return [(s.start_char, s.end_char, s.text) for s in doc.sents]


def _fallback_sentences(text: str) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    pos = 0
    for block in text.splitlines(keepends=True):
        stripped = block.strip()
        if stripped:
            offset = block.index(stripped)
            out.append((pos + offset, pos + offset + len(stripped), stripped))
        pos += len(block)
    return out


@lru_cache(maxsize=4)
def _cached_model(name: str):  # pragma: no cover - trivial
    return load_nlp(name)


def reset() -> None:
    """Test helper: forget the cached pipeline."""
    global _nlp, _load_failed
    with _lock:
        _nlp = None
        _load_failed = False
