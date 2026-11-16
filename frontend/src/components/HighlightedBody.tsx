import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

/**
 * `HighlightedBody` renders a record's raw body and overlays one absolutely
 * positioned box per active mark.
 *
 * Why an overlay instead of wrapping the text in `<mark>` elements: the API gives
 * character offsets into the *raw* body, and every downstream feature (citations,
 * evidence quotes, event spans) refers to those offsets. Wrapping the text would
 * change the DOM the offsets point into and would silently break as soon as two
 * marks overlap. Instead we keep the body as a single untouched text node and use
 * `Range.getClientRects()` to find where each character range landed on screen.
 * The text stays byte-for-byte what the record contains; only decoration is added.
 */

export type MarkKind = "entity" | "risk" | "event";

export interface BodyMark {
  id: string;
  kind: MarkKind;
  start: number;
  end: number;
  label: string;
  /** Accessible description, e.g. "Urgency evidence: a deadline is referenced". */
  description?: string;
  entityId?: number | null;
  eventId?: number | null;
}

interface Box {
  left: number;
  top: number;
  width: number;
  height: number;
}

const KIND_CLASS: Record<MarkKind, string> = {
  entity: "bg-teal-200/70 ring-1 ring-teal-600/70",
  risk: "bg-risk-300/70 ring-1 ring-risk-700",
  event: "bg-blue-200/70 ring-1 ring-blue-700/70",
};

const KIND_BUTTON: Record<MarkKind, string> = {
  entity: "hover:bg-teal-300/80",
  risk: "hover:bg-risk-400/80",
  event: "hover:bg-blue-300/80",
};

const KIND_RING: Record<MarkKind, string> = {
  entity: "ring-2 ring-teal-900",
  risk: "ring-2 ring-risk-900",
  event: "ring-2 ring-blue-900",
};

export function HighlightedBody({
  body,
  marks,
  activeMarkId,
  onSelect,
}: {
  body: string;
  marks: BodyMark[];
  activeMarkId?: string | null;
  onSelect?: (mark: BodyMark) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const textRef = useRef<HTMLDivElement | null>(null);
  const [boxes, setBoxes] = useState<Record<string, Box[]>>({});
  const boxRefs = useRef<Record<string, HTMLButtonElement | null>>({});

  const clampedMarks = useMemo(() => {
    return marks
      .map((mark) => ({
        ...mark,
        start: Math.max(0, Math.min(mark.start, body.length)),
        end: Math.max(0, Math.min(mark.end, body.length)),
      }))
      .filter((mark) => mark.end > mark.start);
  }, [marks, body.length]);

  const measure = useCallback(() => {
    const container = containerRef.current;
    const textNode = textRef.current;
    if (!container || !textNode) return;
    if (textNode.firstChild === null) {
      textNode.textContent = body;
    }
    const raw = textNode.firstChild;
    if (!raw || raw.nodeType !== Node.TEXT_NODE) return;
    const containerBox = container.getBoundingClientRect();
    const next: Record<string, Box[]> = {};
    for (const mark of clampedMarks) {
      try {
        const range = document.createRange();
        range.setStart(raw, mark.start);
        range.setEnd(raw, mark.end);
        const rects = Array.from(range.getClientRects());
        next[mark.id] = rects
          .filter((rect) => rect.width > 0 && rect.height > 0)
          .map((rect) => ({
            left: rect.left - containerBox.left,
            top: rect.top - containerBox.top,
            width: rect.width,
            height: rect.height,
          }));
      } catch {
        next[mark.id] = [];
      }
    }
    setBoxes(next);
  }, [body, clampedMarks]);

  useLayoutEffect(() => {
    measure();
  }, [measure]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", measure);
      return () => window.removeEventListener("resize", measure);
    }
    const observer = new ResizeObserver(() => measure());
    observer.observe(container);
    return () => observer.disconnect();
  }, [measure]);

  // Bring the focused citation's characters into view when the route changes.
  useEffect(() => {
    if (!activeMarkId) return;
    const element = boxRefs.current[activeMarkId];
    if (element) element.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [activeMarkId, boxes]);

  const ordered = useMemo(() => {
    // Draw order: entities at the bottom, risk evidence on top, so the most
    // consequential decoration stays visible when marks overlap.
    const rank: Record<MarkKind, number> = { entity: 0, event: 1, risk: 2 };
    return [...clampedMarks].sort((a, b) => rank[a.kind] - rank[b.kind]);
  }, [clampedMarks]);

  const overlapping = useMemo(() => {
    let max = 0;
    for (let i = 0; i < clampedMarks.length; i += 1) {
      for (let j = i + 1; j < clampedMarks.length; j += 1) {
        const a = clampedMarks[i];
        const b = clampedMarks[j];
        if (a.start < b.end && b.start < a.end) max += 1;
      }
    }
    return max;
  }, [clampedMarks]);

  return (
    <div className="dm-body-layer" ref={containerRef}>
      <div
        ref={textRef}
        className="dm-body-text text-[0.95rem] leading-7 text-ink-900"
        data-testid="record-body-text"
      >
        {body}
      </div>
      <div className="dm-body-overlay">
        {ordered.map((mark) =>
          (boxes[mark.id] ?? []).map((box, index) => (
            <button
              key={`${mark.id}-${index}`}
              type="button"
              ref={(node) => {
                if (index === 0) boxRefs.current[mark.id] = node;
              }}
              aria-label={`${mark.label}${mark.description ? `: ${mark.description}` : ""}`}
              title={mark.description ? `${mark.label} — ${mark.description}` : mark.label}
              onClick={() => onSelect?.(mark)}
              className={[
                "dm-mark-box cursor-pointer border-0 p-0 transition-colors",
                KIND_CLASS[mark.kind],
                KIND_BUTTON[mark.kind],
                activeMarkId === mark.id ? KIND_RING[mark.kind] : "",
              ].join(" ")}
              style={{
                left: `${box.left}px`,
                top: `${box.top}px`,
                width: `${box.width}px`,
                height: `${box.height}px`,
              }}
            />
          )),
        )}
      </div>
      {overlapping > 0 ? (
        <p className="mt-3 text-xs text-ink-600">
          {overlapping} highlighted range(s) overlap. Each colour is drawn from the same character
          offsets, so a phrase that is both an entity mention and risk evidence shows both.
        </p>
      ) : null}
    </div>
  );
}
