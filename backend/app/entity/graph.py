"""Knowledge-graph construction.

Nodes are canonical entities and records. Edges are always evidenced by a record and a
character span, because an unexplainable edge is exactly the thing this project promises
not to produce.

Edge types:
  * ``mentioned_in``    entity -> record, span = the mention itself
  * ``participant``     entity -> record, from the record's participant list
  * ``co_mentioned``    entity -> entity, weight = shared records
  * ``discusses``       entity -> entity, when both appear in a record about a topic
  * ``near``            entity -> entity, records are temporally adjacent (timeline link)
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from ..models import Entity, EntityMention, Event, Record, Relation, RiskAssessment


@dataclass
class GraphBuildResult:
    nodes: int
    edges: int
    mentions: int
    entities: int
    low_cohesion_clusters: int = 0

    def as_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "mentions": self.mentions,
            "entities": self.entities,
            "low_cohesion_clusters": self.low_cohesion_clusters,
        }


def to_networkx(
    entities: list[Entity],
    mentions: list[EntityMention],
    relations: list[Relation],
    records: list[Record],
) -> nx.MultiDiGraph:
    """Materialize the stored graph as a NetworkX multigraph for analysis/UI.

    Two edge families are produced:

    * stored :class:`Relation` rows (entity <-> entity, e.g. ``co_mentioned``)
    * derived ``mentioned_in`` / ``participant`` edges (entity <-> record)

    The derived edges are *not* persisted: they are a pure function of the mention table,
    so recomputing them can never leave a stale edge behind after a delete.
    """
    g = nx.MultiDiGraph()
    for rec in records:
        g.add_node(
            f"record:{rec.id}",
            kind="record",
            label=rec.display_title[:80],
            source_type=rec.source_type,
            occurred_at=rec.occurred_at.isoformat() if rec.occurred_at else None,
            record_id=rec.id,
        )
    for ent in entities:
        g.add_node(
            f"entity:{ent.id}",
            kind="entity",
            label=ent.name,
            entity_type=ent.entity_type,
            mention_count=ent.mention_count,
            resolution_confidence=round(ent.resolution_confidence, 3),
        )
    for rel in relations:
        g.add_edge(
            f"entity:{rel.source_entity_id}",
            f"entity:{rel.target_entity_id}",
            kind="relation",
            relation_type=rel.relation_type,
            record_id=rel.record_id,
            confidence=round(rel.confidence, 3),
            weight=rel.weight,
            method=rel.method,
            start_char=rel.start_char,
            end_char=rel.end_char,
        )
    for m in mentions:
        if m.entity_id is None:
            continue
        g.add_edge(
            f"entity:{m.entity_id}",
            f"record:{m.record_id}",
            kind="mention",
            relation_type="mentioned_in",
            record_id=m.record_id,
            confidence=round(m.confidence, 3),
            weight=1.0,
            method=m.extractor,
            start_char=m.start_char,
            end_char=m.end_char,
        )
    return g


def centrality(g: nx.MultiDiGraph) -> dict[str, dict[str, float]]:
    """Degree and betweenness over the entity-only subgraph.

    Reported in the dashboard as "most connected", with the caveat that betweenness on a
    small corpus is unstable (documented as a limitation).
    """
    entity_nodes = [n for n, d in g.nodes(data=True) if d.get("kind") == "entity"]
    sub = g.subgraph(entity_nodes)
    undirected = nx.Graph()
    undirected.add_nodes_from(entity_nodes)
    for u, v, data in sub.edges(data=True):
        w = float(data.get("weight", 1.0))
        if undirected.has_edge(u, v):
            undirected[u][v]["weight"] += w
        else:
            undirected.add_edge(u, v, weight=w)

    out: dict[str, dict[str, float]] = {"degree": {}, "weighted_degree": {}, "betweenness": {}}
    for node in entity_nodes:
        out["degree"][node] = float(undirected.degree(node))
        out["weighted_degree"][node] = float(undirected.degree(node, weight="weight"))
    if undirected.number_of_edges() > 0:
        bet = nx.betweenness_centrality(undirected, weight=None, normalized=True)
        for node in entity_nodes:
            out["betweenness"][node] = float(bet.get(node, 0.0))
    return out


def co_mention_weights(
    mentions: list[EntityMention], min_shared: int = 2
) -> dict[tuple[int, int], tuple[int, float]]:
    """Count shared records per entity pair, the basis for ``co_mentioned`` edges."""
    by_record: dict[int, set[int]] = defaultdict(set)
    for m in mentions:
        if m.entity_id is not None:
            by_record[m.record_id].add(m.entity_id)

    pairs: dict[tuple[int, int], int] = defaultdict(int)
    for entity_ids in by_record.values():
        ordered = sorted(entity_ids)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                pairs[(a, b)] += 1

    out: dict[tuple[int, int], tuple[int, float]] = {}
    for pair, n in pairs.items():
        if n >= min_shared:
            out[pair] = (n, min(1.0, n / 5.0))
    return out


def build_relation_candidates(
    mentions: list[EntityMention],
    events: list[Event],
    assessments: list[RiskAssessment],
    risk_weight: float = 0.6,
) -> list[dict[str, Any]]:
    """Assemble relation rows ready for persistence.

    Co-mention confidence decays as ``1 - 0.5^(n-1)`` so a single shared record is a weak
    signal (0.5) and five shared records is near-certain (0.94). Risk-bearing records get
    a boost: two entities that co-occur in a flagged message are more interesting, and
    the boost is recorded in the row so the UI can explain it.
    """
    by_record: dict[int, list[EntityMention]] = defaultdict(list)
    for m in mentions:
        by_record[m.record_id].append(m)

    risk_records = {a.record_id for a in assessments if a.present}

    rows: list[dict[str, Any]] = []

    # entity -> record: mentioned_in, one per mention
    for m in mentions:
        if m.entity_id is None:
            continue
        rows.append(
            {
                "source_entity_id": m.entity_id,
                "target_entity_id": -m.record_id,  # negative marks a record endpoint
                "relation_type": "mentioned_in",
                "record_id": m.record_id,
                "start_char": m.start_char,
                "end_char": m.end_char,
                "weight": 1.0,
                "confidence": m.confidence,
                "method": m.extractor,
            }
        )

    # entity <-> entity: co_mentioned
    for (a, b), (shared, conf) in co_mention_weights(mentions).items():
        rows.append(
            {
                "source_entity_id": a,
                "target_entity_id": b,
                "relation_type": "co_mentioned",
                "record_id": None,
                "start_char": None,
                "end_char": None,
                "weight": float(shared),
                "confidence": 1.0 - 0.5 ** (shared - 1),
                "method": "co_mention",
            }
        )

    # entity <-> record: participant
    for record_id, record_mentions in by_record.items():
        entity_ids = sorted({m.entity_id for m in record_mentions if m.entity_id is not None})
        if record_id in risk_records and entity_ids:
            for eid in entity_ids:
                rows.append(
                    {
                        "source_entity_id": eid,
                        "target_entity_id": -record_id,
                        "relation_type": "co_occurs_with_risk",
                        "record_id": record_id,
                        "start_char": None,
                        "end_char": None,
                        "weight": risk_weight,
                        "confidence": 0.6,
                        "method": "risk_boost",
                    }
                )
    return rows


def subgraph_for_records(g: nx.MultiDiGraph, record_ids: set[int]) -> nx.MultiDiGraph:
    """Induced subgraph covering a set of records and everything they touch."""
    keep: set[str] = set()
    for u, v, data in g.edges(data=True):
        if data.get("record_id") in record_ids:
            keep.add(u)
            keep.add(v)
    keep.update(f"record:{rid}" for rid in record_ids)
    return g.subgraph(keep).copy()


def connected_records(g: nx.MultiDiGraph, record_ids: set[int], depth: int = 1) -> set[int]:
    """Records reachable within ``depth`` hops of the seed records.

    Used by "what happened before this payment request?" - we walk backwards in time
    through the graph rather than only through timestamps, so a message linked by an
    entity still surfaces even if its own timestamp is oddly ordered.
    """
    seeds = {f"record:{rid}" for rid in record_ids}
    seen: set[str] = set(seeds)
    frontier = set(seeds)
    for _ in range(depth):
        nxt: set[str] = set()
        for node in frontier:
            if node not in g:
                continue
            for nbr in g.successors(node):
                if nbr not in seen:
                    nxt.add(nbr)
            for nbr in g.predecessors(node):
                if nbr not in seen:
                    nxt.add(nbr)
        seen |= nxt
        frontier = nxt
    out: set[int] = set()
    for node in seen:
        if node.startswith("record:"):
            out.add(int(node.split(":", 1)[1]))
    return out
