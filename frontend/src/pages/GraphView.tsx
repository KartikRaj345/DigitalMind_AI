import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { getEntity, getGraph, type EntityDetail, type GraphEdge, type GraphNode } from "../lib/api";
import { formatDateTime } from "../lib/format";
import { recordLink } from "../lib/router";
import {
  Card,
  EmptyState,
  ErrorBlock,
  LoadingBlock,
  NeutralChip,
  PageHeader,
  SourceBadge,
  cx,
} from "../components/ui";

interface SimNode {
  id: string;
  kind: string;
  label: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  degree: number;
  data: GraphNode;
}

const REPULSION = 5200;
const SPRING = 0.012;
const SPRING_LENGTH = 96;
const DAMPING = 0.86;
const CENTER_PULL = 0.0045;

function radiusFor(node: GraphNode, degree: number): number {
  if (node.kind === "record") return 7 + Math.min(9, Math.sqrt(degree) * 2.2);
  return 6 + Math.min(11, Math.sqrt(node.mention_count ?? degree) * 2.6);
}

export function GraphView({ focusParam }: { focusParam: string | null }) {
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [explanation, setExplanation] = useState("");
  const [centrality, setCentrality] = useState<Record<string, Record<string, number>>>({});
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [showEntities, setShowEntities] = useState(true);
  const [showRecords, setShowRecords] = useState(true);
  const [entityDetail, setEntityDetail] = useState<EntityDetail | null>(null);
  const [entityError, setEntityError] = useState<unknown>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const simRef = useRef<SimNode[]>([]);
  const [tick, setTick] = useState(0);
  const [hovered, setHovered] = useState<string | null>(null);
  const transform = useRef({ x: 0, y: 0, k: 1 });
  const dragState = useRef<{
    mode: "none" | "pan" | "node";
    nodeId: string | null;
    lastX: number;
    lastY: number;
    moved: boolean;
  }>({ mode: "none", nodeId: null, lastX: 0, lastY: 0, moved: false });

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    getGraph({ max_nodes: 160 })
      .then((result) => {
        setNodes(result.nodes);
        setEdges(result.edges);
        setExplanation(result.explanation);
        setCentrality(result.centrality);
        setLoading(false);
      })
      .catch((err) => {
        setError(err);
        setLoading(false);
      });
  }, []);

  useEffect(load, [load]);

  const visible = useMemo(
    () => nodes.filter((node) => (node.kind === "record" ? showRecords : showEntities)),
    [nodes, showRecords, showEntities],
  );
  const visibleIds = useMemo(() => new Set(visible.map((node) => node.id)), [visible]);
  const visibleEdges = useMemo(
    () => edges.filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target)),
    [edges, visibleIds],
  );

  // Seed the simulation once per dataset, deterministically so the layout is stable.
  useEffect(() => {
    const degree = new Map<string, number>();
    for (const edge of edges) {
      degree.set(edge.source, (degree.get(edge.source) ?? 0) + 1);
      degree.set(edge.target, (degree.get(edge.target) ?? 0) + 1);
    }
    simRef.current = nodes.map((node, index) => {
      const deg = degree.get(node.id) ?? 0;
      // Deterministic ring seeding: no randomness, so re-renders do not reshuffle.
      const angle = (index / Math.max(1, nodes.length)) * Math.PI * 2;
      const ring = 120 + (index % 7) * 26;
      return {
        id: node.id,
        kind: node.kind,
        label: node.label,
        x: Math.cos(angle) * ring,
        y: Math.sin(angle) * ring,
        vx: 0,
        vy: 0,
        degree: deg,
        radius: radiusFor(node, deg),
        data: node,
      };
    });
    setTick((value) => value + 1);
  }, [nodes, edges]);

  // One relaxation burst per dataset change; the result is then static until reset.
  useEffect(() => {
    if (simRef.current.length === 0) return;
    let cancelled = false;
    const step = () => {
      if (cancelled) return;
      relax(simRef.current, visibleEdges);
      setTick((value) => value + 1);
      if (iterationsRef.current < 320) {
        iterationsRef.current += 1;
        window.requestAnimationFrame(step);
      }
    };
    iterationsRef.current = 0;
    const frame = window.requestAnimationFrame(step);
    return () => {
      cancelled = true;
      window.cancelAnimationFrame(frame);
    };
  }, [visibleEdges]);

  const iterationsRef = useRef(0);

  const resetLayout = () => {
    const degree = new Map<string, number>();
    for (const edge of visibleEdges) {
      degree.set(edge.source, (degree.get(edge.source) ?? 0) + 1);
      degree.set(edge.target, (degree.get(edge.target) ?? 0) + 1);
    }
    const seeded: SimNode[] = visible.map((node, index) => {
      const deg = degree.get(node.id) ?? 0;
      const angle = (index / Math.max(1, visible.length)) * Math.PI * 2;
      return {
        id: node.id,
        kind: node.kind,
        label: node.label,
        x: Math.cos(angle) * 150,
        y: Math.sin(angle) * 150,
        vx: 0,
        vy: 0,
        degree: deg,
        radius: radiusFor(node, deg),
        data: node,
      };
    });
    simRef.current = seeded;
    transform.current = { x: 0, y: 0, k: 1 };
    setTick((value) => value + 1);
  };

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const dpr = window.devicePixelRatio || 1;
    const width = wrap.clientWidth;
    const height = wrap.clientHeight;
    if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const { x: tx, y: ty, k } = transform.current;
    const sim = simRef.current;
    const byId = new Map(sim.map((node) => [node.id, node]));
    const positions = visible.map((node) => node.id);

    ctx.save();
    ctx.translate(width / 2 + tx, height / 2 + ty);
    ctx.scale(k, k);

    // Edges first. `mentioned_in` is solid; `co_mentioned` is dashed and fainter,
    // because it is a heuristic rather than something the text states.
    for (const edge of visibleEdges) {
      const a = byId.get(edge.source);
      const b = byId.get(edge.target);
      if (!a || !b) continue;
      const highlighted =
        hovered !== null && (edge.source === hovered || edge.target === hovered);
      const isMention = edge.relation_type !== "co_mentioned";
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      if (isMention) {
        ctx.strokeStyle = highlighted ? "rgba(13,124,120,0.95)" : "rgba(13,124,120,0.55)";
        ctx.lineWidth = (highlighted ? 2.4 : 1.4) / k;
        ctx.setLineDash([]);
      } else {
        ctx.strokeStyle = highlighted ? "rgba(100,116,139,0.8)" : "rgba(100,116,139,0.32)";
        ctx.lineWidth = (highlighted ? 1.8 : 1.1) / k;
        ctx.setLineDash([5 / k, 4 / k]);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    for (const id of positions) {
      const node = byId.get(id);
      if (!node) continue;
      const isRecord = node.kind === "record";
      const isHovered = hovered === node.id;
      const isSelected = selected?.id === node.id;

      ctx.beginPath();
      if (isRecord) {
        const size = node.radius;
        // `roundRect` is unavailable on older engines; fall back to a plain square.
        if (typeof ctx.roundRect === "function") {
          ctx.roundRect(node.x - size, node.y - size, size * 2, size * 2, 4);
        } else {
          ctx.rect(node.x - size, node.y - size, size * 2, size * 2);
        }
        ctx.fillStyle = isSelected ? "#0f766e" : isHovered ? "#5eead4" : "#0d9488";
        ctx.fill();
        ctx.lineWidth = (isSelected ? 3 : 1.5) / k;
        ctx.strokeStyle = "#134e4a";
        ctx.stroke();
      } else {
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = isSelected ? "#1d4ed8" : isHovered ? "#93c5fd" : "#3b82f6";
        ctx.fill();
        ctx.lineWidth = (isSelected ? 3 : 1.5) / k;
        ctx.strokeStyle = "#1e3a8a";
        ctx.stroke();
      }

      if (k > 0.75 || isHovered || isSelected) {
        const label = node.label.length > 28 ? `${node.label.slice(0, 27)}…` : node.label;
        ctx.font = `${11 / k}px ui-sans-serif, system-ui, sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = "rgba(255,255,255,0.85)";
        const w = ctx.measureText(label).width + 6 / k;
        ctx.fillRect(node.x - w / 2, node.y + node.radius + 2 / k, w, 14 / k);
        ctx.fillStyle = "#0f172a";
        ctx.fillText(label, node.x, node.y + node.radius + 4 / k);
      }
    }

    ctx.restore();
  }, [visible, visibleEdges, hovered, selected]);

  useEffect(redraw, [redraw, tick]);

  useEffect(() => {
    const onResize = () => redraw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [redraw]);

  // Focus support: `#/graph?focus=entity:12` opens that node's panel.
  useEffect(() => {
    if (!focusParam || nodes.length === 0) return;
    const found = nodes.find((node) => node.id === focusParam);
    if (found) setSelected(found);
  }, [focusParam, nodes]);

  useEffect(() => {
    if (!selected || selected.kind !== "entity") {
      setEntityDetail(null);
      return;
    }
    const id = Number.parseInt(selected.id.replace("entity:", ""), 10);
    if (!Number.isFinite(id)) return;
    let cancelled = false;
    setEntityError(null);
    getEntity(id)
      .then((detail) => {
        if (!cancelled) setEntityDetail(detail);
      })
      .catch((err) => {
        if (!cancelled) setEntityError(err);
      });
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const canvasPoint = (event: React.PointerEvent | React.MouseEvent | React.WheelEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const { x, y, k } = transform.current;
    return {
      x: (event.clientX - rect.left - rect.width / 2 - x) / k,
      y: (event.clientY - rect.top - rect.height / 2 - y) / k,
    };
  };

  const nodeAt = (px: number, py: number): SimNode | null => {
    let best: SimNode | null = null;
    let bestDistance = Infinity;
    for (const node of simRef.current) {
      if (!visibleIds.has(node.id)) continue;
      const dx = node.x - px;
      const dy = node.y - py;
      const distance = Math.hypot(dx, dy);
      if (distance <= node.radius + 6 / transform.current.k && distance < bestDistance) {
        best = node;
        bestDistance = distance;
      }
    }
    return best;
  };

  return (
    <>
      <PageHeader
        eyebrow="Knowledge graph"
        title="Graph"
        description="Entities and records, linked only where a record evidences the link. Written by hand on a canvas: no graph library."
        actions={
          <>
            <button type="button" className="dm-btn" onClick={resetLayout}>
              Reset layout
            </button>
            <button type="button" className="dm-btn" onClick={() => setSelected(null)}>
              Clear selection
            </button>
          </>
        }
      />

      <div className="grid gap-5 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <Card
          title="Node and edge legend"
          subtitle="Shape and line style carry meaning. Read the legend before drawing a conclusion from a line."
        >
          <div className="flex flex-wrap gap-4">
            <LegendSwatch shape="circle" label="Entity (person, organisation, …)" />
            <LegendSwatch shape="square" label="Record (an imported document)" />
            <LegendSwatch shape="line" variant="solid" label="mentioned_in — stated by a record" />
            <LegendSwatch
              shape="line"
              variant="dashed"
              label="co_mentioned — heuristic, lower opacity"
            />
          </div>

          <div className="mt-4 flex flex-wrap items-end gap-4 border-t border-ink-100 pt-3">
            <div className="flex items-center gap-2">
              <input
                id="show-entities"
                type="checkbox"
                className="h-4 w-4 rounded border-ink-400"
                checked={showEntities}
                onChange={(event) => setShowEntities(event.target.checked)}
              />
              <label htmlFor="show-entities" className="text-sm text-ink-800">
                Entities
              </label>
            </div>
            <div className="flex items-center gap-2">
              <input
                id="show-records"
                type="checkbox"
                className="h-4 w-4 rounded border-ink-400"
                checked={showRecords}
                onChange={(event) => setShowRecords(event.target.checked)}
              />
              <label htmlFor="show-records" className="text-sm text-ink-800">
                Records
              </label>
            </div>
            <p className="text-xs text-ink-600">
              Drag to pan or move a node · scroll to zoom · click a node for details.
            </p>
          </div>

          <div
            ref={wrapRef}
            className="mt-3 h-[520px] w-full overflow-hidden rounded-md border border-ink-200 bg-white"
          >
            <canvas
              ref={canvasRef}
              role="img"
              aria-label={`Knowledge graph with ${visible.length} node(s) and ${visibleEdges.length} edge(s). A text summary of the selected node appears in the details panel.`}
              className="h-full w-full cursor-grab touch-none active:cursor-grabbing"
              onPointerDown={(event) => {
                (event.target as HTMLCanvasElement).setPointerCapture(event.pointerId);
                const point = canvasPoint(event);
                const node = nodeAt(point.x, point.y);
                dragState.current = {
                  mode: node ? "node" : "pan",
                  nodeId: node?.id ?? null,
                  lastX: event.clientX,
                  lastY: event.clientY,
                  moved: false,
                };
              }}
              onPointerMove={(event) => {
                const state = dragState.current;
                if (state.mode === "pan") {
                  transform.current.x += event.clientX - state.lastX;
                  transform.current.y += event.clientY - state.lastY;
                  state.lastX = event.clientX;
                  state.lastY = event.clientY;
                  state.moved = true;
                  redraw();
                } else if (state.mode === "node" && state.nodeId) {
                  const point = canvasPoint(event);
                  const node = simRef.current.find((item) => item.id === state.nodeId);
                  if (node) {
                    node.x = point.x;
                    node.y = point.y;
                    node.vx = 0;
                    node.vy = 0;
                  }
                  state.moved = true;
                  redraw();
                } else {
                  const point = canvasPoint(event);
                  const node = nodeAt(point.x, point.y);
                  const next = node?.id ?? null;
                  if (next !== hovered) {
                    setHovered(next);
                  }
                }
              }}
              onPointerUp={(event) => {
                const state = dragState.current;
                if (state.mode === "node" && !state.moved && state.nodeId) {
                  const node = simRef.current.find((item) => item.id === state.nodeId);
                  if (node) setSelected(node.data);
                }
                dragState.current = { mode: "none", nodeId: null, lastX: 0, lastY: 0, moved: false };
                (event.target as HTMLCanvasElement).releasePointerCapture?.(event.pointerId);
              }}
              onPointerLeave={() => setHovered(null)}
              onWheel={(event) => {
                const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
                const next = Math.min(3, Math.max(0.35, transform.current.k * factor));
                transform.current.k = next;
                redraw();
              }}
              onKeyDown={(event) => {
                const step = 40;
                if (event.key === "ArrowLeft") transform.current.x += step;
                else if (event.key === "ArrowRight") transform.current.x -= step;
                else if (event.key === "ArrowUp") transform.current.y += step;
                else if (event.key === "ArrowDown") transform.current.y -= step;
                else if (event.key === "+" || event.key === "=") transform.current.k = Math.min(3, transform.current.k * 1.12);
                else if (event.key === "-") transform.current.k = Math.max(0.35, transform.current.k / 1.12);
                else return;
                event.preventDefault();
                redraw();
              }}
              tabIndex={0}
            />
          </div>
          <p className="mt-2 text-xs text-ink-600">{explanation}</p>

          {loading ? <LoadingBlock label="Loading graph" /> : null}
          {error ? <ErrorBlock error={error} onRetry={load} /> : null}
          {!loading && !error && nodes.length === 0 ? (
            <EmptyState title="No graph to draw yet">
              The graph is built from entity mentions inside records. Import records and the nodes
              and evidenced edges appear here.
            </EmptyState>
          ) : null}
        </Card>

        <div className="space-y-5">
          <Card
            title="Selected node"
            subtitle={selected ? selected.id : "Click a node in the canvas"}
          >
            {!selected ? (
              <EmptyState title="Nothing selected">
                Selecting a node shows what it is, how it was extracted and which records mention it.
              </EmptyState>
            ) : (
              <div>
                <p className="text-sm font-semibold text-ink-900">{selected.label}</p>
                <div className="mt-2 flex flex-wrap items-center gap-1.5">
                  <NeutralChip>{selected.kind}</NeutralChip>
                  {selected.entity_type ? <NeutralChip>{selected.entity_type}</NeutralChip> : null}
                  {selected.mention_count !== null ? (
                    <NeutralChip>{selected.mention_count} mention(s)</NeutralChip>
                  ) : null}
                  {selected.source_type ? <SourceBadge sourceType={selected.source_type} /> : null}
                </div>
                {selected.record_id !== null ? (
                  <p className="mt-2 text-xs text-ink-700">
                    <a
                      href={`#${recordLink(selected.record_id)}`}
                      className="font-medium text-teal-800 underline decoration-teal-400 underline-offset-2 hover:bg-teal-50"
                    >
                      Open the record
                    </a>{" "}
                    · {formatDateTime(selected.occurred_at)}
                  </p>
                ) : null}
                {selected.resolution_confidence !== null ? (
                  <p className="mt-2 text-xs text-ink-700">
                    Entity resolution confidence {selected.resolution_confidence.toFixed(2)}.
                    Resolution merges similar names and can be wrong.
                  </p>
                ) : null}

                {selected.kind === "entity" ? (
                  <div className="mt-4 border-t border-ink-100 pt-3">
                    {entityError ? <ErrorBlock error={entityError} /> : null}
                    {!entityError && !entityDetail ? (
                      <LoadingBlock label="Loading entity detail" />
                    ) : null}
                    {entityDetail ? (
                      <>
                        <p className="dm-section-title">Mentions ({entityDetail.mentions.length})</p>
                        <ul className="mt-2 space-y-2">
                          {entityDetail.mentions.slice(0, 8).map((mention, index) => (
                            <li key={`${mention.record_id}-${mention.start_char}-${index}`}>
                              <a
                                href={`#${recordLink(mention.record_id, mention.start_char, mention.end_char)}`}
                                className="block rounded-md border border-ink-200 bg-ink-50 px-3 py-2 text-sm hover:border-teal-600 hover:bg-teal-50"
                              >
                                <span className="italic text-ink-900">“{mention.text}”</span>
                                <span className="mt-1 block text-xs text-ink-600">
                                  Record {mention.record_id} · {mention.extractor} · confidence{" "}
                                  {mention.confidence.toFixed(2)}
                                </span>
                              </a>
                            </li>
                          ))}
                        </ul>
                        {entityDetail.aliases.length > 0 ? (
                          <>
                            <p className="dm-section-title mt-4">Aliases merged into this entity</p>
                            <ul className="mt-2 flex flex-wrap gap-1.5">
                              {entityDetail.aliases.map((alias) => (
                                <li key={alias}>
                                  <span className="dm-chip border-ink-300 bg-white text-ink-800">{alias}</span>
                                </li>
                              ))}
                            </ul>
                          </>
                        ) : null}
                        <p className="mt-3 text-xs text-ink-600">{entityDetail.caveat}</p>
                      </>
                    ) : null}
                  </div>
                ) : null}
              </div>
            )}
          </Card>

          <Card title="Most connected nodes" subtitle="Degree from the edges currently shown">
            {Object.keys(centrality).length === 0 ? (
              <EmptyState title="No centrality available">
                Centrality appears once the graph has edges to measure.
              </EmptyState>
            ) : (
              <ul className="space-y-2">
                {topCentralNodes(nodes, centrality, visibleIds, 8).map(({ node, degree }) => (
                  <li key={node.id} className="flex items-center justify-between gap-2">
                    <button
                      type="button"
                      className="text-left text-sm text-teal-800 underline decoration-teal-400 underline-offset-2 hover:bg-teal-50"
                      onClick={() => setSelected(node)}
                    >
                      {node.label}
                    </button>
                    <span className="text-xs tabular-nums text-ink-600">{degree} edge(s)</span>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card title="Edge breakdown" subtitle="Only evidenced links are drawn">
            <ul className="space-y-1.5 text-sm text-ink-800">
              {edgeTypeCounts(visibleEdges).map(([type, count]) => (
                <li key={type} className="flex items-center justify-between gap-2">
                  <span className={cx(type === "co_mentioned" ? "text-ink-600" : "text-ink-900")}>
                    {type}
                    {type === "co_mentioned" ? " (dashed, heuristic)" : " (solid, evidenced)"}
                  </span>
                  <span className="tabular-nums text-ink-700">{count}</span>
                </li>
              ))}
              {visibleEdges.length === 0 ? (
                <li className="text-sm text-ink-600">No edges in the current filter.</li>
              ) : null}
            </ul>
          </Card>
        </div>
      </div>
    </>
  );
}

function relax(nodes: SimNode[], edges: GraphEdge[]): void {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  for (let i = 0; i < nodes.length; i += 1) {
    const a = nodes[i];
    for (let j = i + 1; j < nodes.length; j += 1) {
      const b = nodes[j];
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      let distanceSq = dx * dx + dy * dy;
      if (distanceSq < 1) {
        dx = (i % 2 === 0 ? 1 : -1) * 0.5;
        dy = 0.5;
        distanceSq = 0.5;
      }
      const distance = Math.sqrt(distanceSq);
      const force = REPULSION / distanceSq;
      const fx = (dx / distance) * force;
      const fy = (dy / distance) * force;
      a.vx -= fx;
      a.vy -= fy;
      b.vx += fx;
      b.vy += fy;
    }
  }
  for (const edge of edges) {
    const a = byId.get(edge.source);
    const b = byId.get(edge.target);
    if (!a || !b) continue;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const distance = Math.max(1, Math.hypot(dx, dy));
    const target = edge.relation_type === "co_mentioned" ? SPRING_LENGTH * 1.4 : SPRING_LENGTH;
    const force = (distance - target) * SPRING;
    const fx = (dx / distance) * force;
    const fy = (dy / distance) * force;
    a.vx += fx;
    a.vy += fy;
    b.vx -= fx;
    b.vy -= fy;
  }
  for (const node of nodes) {
    node.vx -= node.x * CENTER_PULL;
    node.vy -= node.y * CENTER_PULL;
    node.vx *= DAMPING;
    node.vy *= DAMPING;
    const speed = Math.hypot(node.vx, node.vy);
    if (speed > 24) {
      node.vx = (node.vx / speed) * 24;
      node.vy = (node.vy / speed) * 24;
    }
    node.x += node.vx;
    node.y += node.vy;
  }
}

function LegendSwatch({
  shape,
  label,
  variant,
}: {
  shape: "circle" | "square" | "line";
  label: string;
  variant?: "solid" | "dashed";
}) {
  return (
    <span className="inline-flex items-center gap-2 text-xs text-ink-800">
      {shape === "line" ? (
        <svg width="34" height="10" aria-hidden="true" className="shrink-0">
          <line
            x1="1"
            y1="5"
            x2="33"
            y2="5"
            stroke={variant === "dashed" ? "#64748b" : "#0d9488"}
            strokeWidth="2"
            strokeDasharray={variant === "dashed" ? "5 4" : undefined}
            opacity={variant === "dashed" ? 0.6 : 1}
          />
        </svg>
      ) : (
        <span
          aria-hidden="true"
          className={cx(
            "inline-block h-3.5 w-3.5 shrink-0",
            shape === "circle" ? "rounded-full bg-blue-600" : "rounded-[3px] bg-teal-600",
          )}
        />
      )}
      {label}
    </span>
  );
}

function edgeTypeCounts(edges: GraphEdge[]): [string, number][] {
  const counts = new Map<string, number>();
  for (const edge of edges) {
    counts.set(edge.relation_type, (counts.get(edge.relation_type) ?? 0) + 1);
  }
  return Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
}

function topCentralNodes(
  nodes: GraphNode[],
  centrality: Record<string, Record<string, number>>,
  visibleIds: Set<string>,
  limit: number,
): { node: GraphNode; degree: number }[] {
  const metrics = centrality.degree ?? {};
  return nodes
    .filter((node) => visibleIds.has(node.id))
    .map((node) => ({ node, degree: metrics[node.id] ?? 0 }))
    .sort((a, b) => b.degree - a.degree)
    .slice(0, limit);
}
