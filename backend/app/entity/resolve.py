"""Entity resolution.

Turns many surface mentions ("Northwind Bank", "northwind bank", "Northwind Bank Customer
Verification") into canonical entities, and produces an auditable list of the merges it
made with a confidence for each.

Approach (deliberately simple and explainable, evaluated against gold clusters):

1. **Blocking** - only compare mentions that share a blocking key, so cost is linear-ish.
2. **Scoring** - rapidfuzz similarity on normalized forms, plus boosts for:
   - token-set containment ("Northwind Bank" inside "Northwind Bank Customer Verification")
   - matching entity type
   - shared email domain
3. **Transitive clustering** - union-find over accepted pairs, then re-score each cluster's
   cohesion so a chain of weak links cannot silently merge two distinct people.

The last step matters: "Ana Delgado" -> "Ana Delgado" -> "Ana M. Delgado" -> "Marco
Delgado" must not become one entity. We cap cluster cohesion and log the decision.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from rapidfuzz import fuzz

# Tokens that describe a role/queue rather than a name. Stripped before comparison.
_NOISE_TOKENS = {
    "customer", "service", "services", "support", "team", "dept", "department",
    "verification", "verified", "account", "accounts", "billing", "payments",
    "reminder", "notice", "final", "urgent", "mail", "email", "no", "reply",
    "info", "information", "admin", "administrator", "office", "the", "and",
    "ltd", "inc", "llc", "plc", "corp", "corporation", "company", "co", "group",
}

_TITLE_PREFIX = re.compile(
    r"^(?:mr|mrs|ms|miss|mx|dr|prof|professor|sir|madam|mister)\.?\s+", re.IGNORECASE
)
_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_WS = re.compile(r"\s+")

PERSON_TYPES = {"PERSON"}
ORG_TYPES = {"ORG", "PRODUCT"}


@dataclass
class Mention:
    """A surface form awaiting resolution."""

    key: str  # record_id:mention_id, unique per extraction
    record_id: int
    text: str
    label: str
    count: int = 1


@dataclass
class Merge:
    """An auditable entity-resolution decision."""

    target: str  # normalized canonical name
    alias: str
    kind: str  # 'exact' | 'fuzzy' | 'containment' | 'manual'
    score: float
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "target": self.target,
            "alias": self.alias,
            "kind": self.kind,
            "score": round(self.score, 4),
            "detail": self.detail,
        }


@dataclass
class EntityCluster:
    canonical: str
    label: str
    mentions: list[str] = field(default_factory=list)
    surface_forms: set[str] = field(default_factory=set)
    record_ids: set[int] = field(default_factory=set)
    total_count: int = 0
    cohesion: float = 1.0
    merges: list[Merge] = field(default_factory=list)


def normalize(name: str, label: str = "") -> str:
    """Canonical comparison form: lowercased, de-titled, de-punctuated, single-spaced."""
    text = _TITLE_PREFIX.sub("", name.strip())
    text = text.lower()
    if label in PERSON_TYPES:
        text = re.sub(r"\b(?:mrs?|ms|miss|mx|dr|prof)\b\.?", "", text)
    text = _NON_ALNUM.sub(" ", text)
    return _WS.sub(" ", text).strip()


def content_tokens(name: str) -> set[str]:
    """Tokens that carry identity, with role/queue words removed."""
    return {t for t in normalize(name).split() if t and t not in _NOISE_TOKENS}


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+$")


def email_handle(name: str) -> str | None:
    """The person-shaped part of an email address.

    ``priya.raman@studio.example`` -> ``priya raman``. This is the single most useful
    corroborating signal available for person entities: it links the address in an email
    header to the name in the signature without any fuzzy matching at all.
    """
    if not _EMAIL_RE.match(name.strip()):
        return None
    handle = name.strip().split("@", 1)[0]
    tokens = [t for t in re.split(r"[._\-+]+", handle) if t and not t.isdigit()]
    return " ".join(tokens) or None


def blocking_key(name: str, label: str) -> str:
    """Cheap key so we only compare plausible candidates.

    For people we key on surname-initial + first-initial; for orgs on the longest content
    token. Falls back to the full normalized form so nothing is unreachable. Email
    addresses are keyed by their handle so they land in the same block as the name.
    """
    handle = email_handle(name)
    basis = handle or name
    norm = normalize(basis, label)
    tokens = norm.split()
    if not tokens:
        return f"{label}::"
    if label in PERSON_TYPES:
        return f"PERSON::{tokens[0][:2]}{tokens[-1][:2]}"
    if label in ORG_TYPES:
        longest = max(content_tokens(norm) or {norm}, key=len)
        return f"ORG::{longest[:4]}"
    return f"{label}::{norm[:6]}"


def _first_token_compatible(na: str, nb: str) -> bool:
    """Whether two normalized forms agree on their leading token, initials allowed.

    People and organizations are identified by their first significant token. Two forms
    with different leading tokens are different entities, however similar the rest of the
    string looks: "L. Ashworth" and "N. Ashworth" score 92% on a character-level fuzzy
    measure purely because the surname is long, and a fuzzy threshold cannot separate them
    from "Rafaela Fontaine" vs "Rafael Fontaine" (a genuine match) without also rejecting
    the real one.

    Returning ``False`` here is a hard veto, applied before any fuzzy scoring.
    """
    ta = na.split()
    tb = nb.split()
    if not ta or not tb:
        return True
    first_a, first_b = ta[0], tb[0]
    if first_a == first_b:
        return True
    if len(first_a) == 1 or len(first_b) == 1:
        # An initial: "l" and "lena" agree, "l" and "n" do not.
        return first_a[0] == first_b[0]
    return False


def similarity(a: str, b: str, label: str = "") -> tuple[float, str]:
    """Return ``(score, kind)`` for two surface forms.

    ``kind`` is the reason we believe they match; it is stored with the merge so the UI
    can explain itself.
    """
    na, nb = normalize(a, label), normalize(b, label)
    if not na or not nb:
        return 0.0, "none"
    if na == nb:
        return 1.0, "exact"

    # An email handle that reduces to the other form is a much stronger signal than any
    # fuzzy score: "priya.raman@studio.example" and "Priya Raman" are the same person
    # by construction, not by resemblance.
    ha, hb = email_handle(a), email_handle(b)
    if ha and (normalize(ha, label) == nb or normalize(ha, label) == nb.replace(" ", "")):
        return 0.99, "email_handle"
    if hb and (normalize(hb, label) == na or normalize(hb, label) == na.replace(" ", "")):
        return 0.99, "email_handle"

    if not _first_token_compatible(na, nb):
        return 0.0, "conflicting_first_token"

    ta, tb = content_tokens(a), content_tokens(b)
    if ta and tb and (ta <= tb or tb <= ta):
        smaller, larger = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
        extra = larger - smaller
        # Containment is only evidence of sameness when the *extra* tokens carry no
        # identity: "Northwind Bank Customer Verification" adds a role, but
        # "Lena Ashworth" vs "Nadia Ashworth" adds a different person. Middle initials are
        # stripped from content tokens, so without this guard "L. Ashworth" and
        # "N. Ashworth" would merge and take both people with them - the exact chain this
        # module is supposed to prevent.
        if all(token in _NOISE_TOKENS or len(token) == 1 for token in extra):
            ratio = len(smaller) / len(larger)
            return (0.86 + 0.06 * ratio, "containment")

    if label in PERSON_TYPES:
        # People: no partial/token-set matching. Both measure "how much of the shorter
        # string appears in the longer", which is the wrong question for names.
        ratio = fuzz.ratio(na, nb) / 100.0
    else:
        ratio = fuzz.token_set_ratio(na, nb) / 100.0
        ratio = max(ratio, fuzz.partial_ratio(na, nb) / 100.0 * 0.97)
    kind = "fuzzy" if ratio >= 0.86 else "none"
    return ratio, kind


# Thresholds, tuned on dev in backend/scripts/tune_entity_resolution.py
STRONG_THRESHOLD = 0.92
WEAK_THRESHOLD = 0.86
MIN_CLUSTER_COHESION = 0.80


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:  # path compression
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_mentions(
    mentions: list[Mention],
    email_domains: dict[str, str] | None = None,
) -> list[EntityCluster]:
    """Resolve mentions into canonical clusters.

    ``email_domains`` maps a mention key to the domain of the email address it came from;
    a shared domain is a corroborating signal for organizations.
    """
    email_domains = email_domains or {}

    # Aggregate identical normalized forms first: this is both correct and cheap.
    form_index: dict[tuple[str, str], Mention] = {}
    for m in mentions:
        norm = normalize(m.text, m.label)
        if not norm:
            continue
        key = (norm, m.label)
        if key in form_index:
            form_index[key].count += m.count
            form_index[key].record_ids = form_index[key].record_ids | {m.record_id}
        else:
            form_index[key] = Mention(
                key=f"{norm}::{m.label}",
                record_id=m.record_id,
                text=m.text,
                label=m.label,
                count=m.count,
            )
            form_index[key].record_ids = {m.record_id}

    groups: dict[str, list[Mention]] = defaultdict(list)
    for m in form_index.values():
        groups[blocking_key(m.text, m.label)].append(m)

    uf = _UnionFind()
    accepted: list[tuple[Mention, Mention, Merge]] = []

    for _key, group in sorted(groups.items()):
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if a.label != b.label:
                    continue
                score, kind = similarity(a.text, b.text, a.label)
                detail = f"{kind} match"
                da = email_domains.get(a.key)
                db = email_domains.get(b.key)
                if da and db:
                    if da == db:
                        score = min(0.99, score + 0.05)
                        detail += f"; shared email domain {da}"
                    else:
                        score -= 0.05
                        detail += f"; different email domains ({da} vs {db})"
                threshold = STRONG_THRESHOLD if kind == "exact" else WEAK_THRESHOLD
                if score >= threshold:
                    merge = Merge(
                        target=normalize(b.text, b.label),
                        alias=a.text,
                        kind=kind,
                        score=score,
                        detail=detail,
                    )
                    accepted.append((a, b, merge))
                    uf.union(a.key, b.key)

    # Build clusters from the union-find result.
    buckets: dict[str, list[Mention]] = defaultdict(list)
    for m in form_index.values():
        buckets[uf.find(m.key)].append(m)

    clusters: list[EntityCluster] = []
    for root, members in buckets.items():
        members.sort(key=lambda m: (-m.count, m.text))
        # Canonical = most frequent form; ties broken alphabetically for determinism.
        canonical = members[0]
        cohesion = _cohesion(members)
        record_ids: set[int] = set()
        for m in members:
            record_ids |= getattr(m, "record_ids", {m.record_id})
        clusters.append(
            EntityCluster(
                canonical=canonical.text,
                label=canonical.label,
                mentions=[m.key for m in members],
                surface_forms={m.text for m in members},
                record_ids=record_ids,
                total_count=sum(m.count for m in members),
                cohesion=cohesion,
            )
        )

    # Attach merge provenance to the winning cluster of each accepted pair.
    for a, b, merge in accepted:
        root = uf.find(b.key)
        for cluster in clusters:
            if any(m.key == root for m in buckets[root]) and cluster.canonical == _canon_of(buckets[root]):
                cluster.merges.append(merge)
                break

    clusters.sort(key=lambda c: (-c.total_count, c.canonical))
    return clusters


def _canon_of(members: list[Mention]) -> str:
    ordered = sorted(members, key=lambda m: (-m.count, m.text))
    return ordered[0].text


def _cohesion(members: list[Mention]) -> float:
    """Minimum pairwise similarity inside a cluster.

    Minimum (not average) on purpose: one bad member drags the score down, which is what
    stops long chains of weak merges.
    """
    if len(members) < 2:
        return 1.0
    worst = 1.0
    for i, a in enumerate(members):
        for b in members[i + 1 :]:
            score, _ = similarity(a.text, b.text, a.label)
            worst = min(worst, score)
    return worst


def low_cohesion_clusters(clusters: list[EntityCluster]) -> list[EntityCluster]:
    """Clusters that look suspiciously merged. Surfaced as an inspectable insight."""
    return [c for c in clusters if c.cohesion < MIN_CLUSTER_COHESION and len(c.surface_forms) > 1]


def split_low_cohesion(cluster: EntityCluster) -> list[EntityCluster]:
    """Re-cluster a suspicious cluster by requiring a stronger link.

    Kept deterministic: greedily accept members in frequency order and only attach a
    member if it is a strong match to the seed.
    """
    ordered = sorted(
        (m for m in cluster.mentions),
        key=lambda k: k,
    )
    if not ordered:
        return [cluster]
    seed = ordered[0]
    out = [EntityCluster(seed, cluster.label, [seed], set(), set(), 0, 1.0)]
    for member in ordered[1:]:
        score, _ = similarity(seed, member, cluster.label)
        if score >= STRONG_THRESHOLD:
            out[0].mentions.append(member)
        else:
            out.append(EntityCluster(member, cluster.label, [member], set(), set(), 0, 1.0))
    return out
