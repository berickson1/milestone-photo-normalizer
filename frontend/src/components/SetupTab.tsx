import { useEffect, useRef, useState } from "react";
import { api, fromAnalysis, svgPoint, toAnalysis, urls, type Pt, type Wrinkles } from "../api";
import type { Ctx } from "./ProjectPage";

type Box = { x0: number; y0: number; x1: number; y1: number };
type Drag =
  | { kind: "vertex"; i: number }
  | { kind: "quad"; i: number }
  | { kind: "corner"; cx: 0 | 1; cy: 0 | 1 }
  | { kind: "move"; start: Pt; orig: Box };

const QUAD_LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"];

/** Four outermost corners (TL, TR, BR, BL) of an outline; mirrors the backend's quad_from_polygon. */
function quadFromPolygon(poly: Pt[]): Pt[] {
  const pick = (score: (p: Pt) => number) => poly.reduce((best, p) => (score(p) > score(best) ? p : best));
  return [pick((p) => -p[0] - p[1]), pick((p) => p[0] - p[1]), pick((p) => p[0] + p[1]), pick((p) => p[1] - p[0])];
}

export default function SetupTab({ project, run }: Ctx) {
  const a = project.analysis;
  const ref = project.photos.find((p) => p.id === project.reference_id);
  const svg = useRef<SVGSVGElement>(null);
  const [poly, setPoly] = useState<Pt[]>([]);
  const [box, setBox] = useState<Box | null>(null);
  const [quad, setQuad] = useState<Pt[] | null>(null);
  const [perspective, setPerspective] = useState(true);
  const [aspect, setAspect] = useState("");
  const [width, setWidth] = useState("");
  const [dirty, setDirty] = useState(false);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [bg, setBg] = useState<"reference" | "combined">("reference");
  const [bgMissing, setBgMissing] = useState(false);

  useEffect(() => {
    if (!a || dirty) return;
    setPoly((project.blanket_polygon ?? []).map((p) => toAnalysis(a, p)));
    if (project.crop) {
      const [x0, y0] = toAnalysis(a, [project.crop.x, project.crop.y]);
      const [x1, y1] = toAnalysis(a, [project.crop.x + project.crop.w, project.crop.y + project.crop.h]);
      setBox({ x0, y0, x1, y1 });
    } else setBox(null);
    setQuad(project.blanket_quad ? project.blanket_quad.map((p) => toAnalysis(a, p)) : null);
    setPerspective(project.perspective);
    setAspect(project.blanket_aspect ? String(project.blanket_aspect) : "");
    setWidth(project.output_width ? String(project.output_width) : "");
    setBgMissing(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.revision, a?.width, dirty]);

  if (!a || !ref) return <p className="muted">Upload photos and choose a reference first.</p>;
  if (!ref.homography) return <p className="muted">Waiting for the reference photo to be processed…</p>;

  const r = a.width / 140;
  const norm = (b: Box): Box => ({
    x0: Math.min(b.x0, b.x1),
    y0: Math.min(b.y0, b.y1),
    x1: Math.max(b.x0, b.x1),
    y1: Math.max(b.y0, b.y1),
  });

  const onMove = (e: React.PointerEvent) => {
    if (!drag || !svg.current) return;
    const [x, y] = svgPoint(svg.current, e.clientX, e.clientY);
    const px = Math.max(0, Math.min(a.width, x));
    const py = Math.max(0, Math.min(a.height, y));
    if (drag.kind === "vertex") setPoly((ps) => ps.map((p, i) => (i === drag.i ? [px, py] : p)));
    else if (drag.kind === "quad") setQuad((qs) => qs && qs.map((p, i) => (i === drag.i ? [px, py] : p)));
    else if (drag.kind === "corner" && box)
      setBox({ ...box, [drag.cx ? "x1" : "x0"]: px, [drag.cy ? "y1" : "y0"]: py });
    else if (drag.kind === "move") {
      const dx = x - drag.start[0];
      const dy = y - drag.start[1];
      setBox({ x0: drag.orig.x0 + dx, y0: drag.orig.y0 + dy, x1: drag.orig.x1 + dx, y1: drag.orig.y1 + dy });
    }
    setDirty(true);
  };

  const start = (d: Drag) => (e: React.PointerEvent) => {
    e.stopPropagation();
    setDrag(d);
    svg.current?.setPointerCapture(e.pointerId);
  };

  const end = () => {
    if (drag?.kind === "corner" && box) setBox(norm(box));
    setDrag(null);
  };

  const insertAt = (i: number) => (e: React.PointerEvent) => {
    const a0 = poly[i];
    const a1 = poly[(i + 1) % poly.length];
    setPoly([...poly.slice(0, i + 1), [(a0[0] + a1[0]) / 2, (a0[1] + a1[1]) / 2], ...poly.slice(i + 1)]);
    setDirty(true);
    start({ kind: "vertex", i: i + 1 })(e);
  };

  const removeVertex = (i: number) => {
    if (poly.length <= 3) return;
    setPoly(poly.filter((_, j) => j !== i));
    setDirty(true);
  };

  const cropToBlanket = () => {
    if (poly.length < 3) return;
    const xs = poly.map((p) => p[0]);
    const ys = poly.map((p) => p[1]);
    const m = a.width * 0.01;
    setBox({ x0: Math.min(...xs) - m, y0: Math.min(...ys) - m, x1: Math.max(...xs) + m, y1: Math.max(...ys) + m });
    setDirty(true);
  };

  const fitQuad = () => {
    if (poly.length < 3) return;
    setQuad(quadFromPolygon(poly));
    setDirty(true);
  };

  const setWrinkles = (w: Wrinkles) => run(() => api.updateProject(project.id, { wrinkles: w }));

  const save = async () => {
    const body: Parameters<typeof api.updateProject>[1] = {};
    if (poly.length >= 3) body.blanket_polygon = poly.map((p) => fromAnalysis(a, p));
    body.perspective = perspective;
    if (quad && quad.length === 4) body.blanket_quad = quad.map((p) => fromAnalysis(a, p));
    const asp = parseFloat(aspect);
    body.blanket_aspect = Number.isFinite(asp) && asp > 0 ? asp : null;
    if (box) {
      const b = norm(box);
      const [x, y] = fromAnalysis(a, [b.x0, b.y0]);
      const [x1, y1] = fromAnalysis(a, [b.x1, b.y1]);
      body.crop = { x, y, w: x1 - x, h: y1 - y };
    }
    const w = parseInt(width, 10);
    body.output_width = Number.isFinite(w) ? w : null;
    const res = await run(() => api.updateProject(project.id, body));
    if (res) setDirty(false);
  };

  const bgUrl =
    bg === "combined"
      ? urls.background(project)
      : urls.render(project, ref.id, { view: "analysis", blackout: false, reconstruct: false, color: false });

  const useQuad = perspective && !!quad && quad.length === 4;
  // The crop box only frames the output when the perspective corners are not in use.
  const nb = !useQuad && box ? norm(box) : null;
  return (
    <section className="setup">
      <div className="toolbar">
        <button onClick={() => confirm("Replace the blanket outline and crop with an automatic guess?") && run(() => api.autoBlanket(project.id)).then(() => setDirty(false))}>
          Auto-detect blanket
        </button>
        {useQuad ? (
          <button onClick={fitQuad} disabled={poly.length < 3}>
            Fit corners to outline
          </button>
        ) : (
          <button onClick={cropToBlanket} disabled={poly.length < 3}>
            Crop to blanket
          </button>
        )}
        <label>
          Background{" "}
          <select value={bg} onChange={(e) => setBg(e.target.value as typeof bg)}>
            <option value="reference">Reference photo</option>
            <option value="combined">Combined (baby removed)</option>
          </select>
        </label>
        <label>
          Output width{" "}
          <input
            type="number"
            min={16}
            max={8000}
            value={width}
            placeholder={project.default_output_width ? `auto (${project.default_output_width})` : "auto"}
            onChange={(e) => {
              setWidth(e.target.value);
              setDirty(true);
            }}
            style={{ width: "9em" }}
          />{" "}
          px
        </label>
        <span className="spacer" />
        {dirty && (
          <button onClick={() => setDirty(false)}>
            Discard changes
          </button>
        )}
        <button className="primary" disabled={!dirty} onClick={save}>
          Save
        </button>
      </div>
      <div className="toolbar">
        <label title="Map the blanket's four corners to a true rectangle, removing the camera's perspective skew">
          <input
            type="checkbox"
            checked={perspective}
            onChange={(e) => {
              setPerspective(e.target.checked);
              if (e.target.checked && !quad && poly.length >= 3) setQuad(quadFromPolygon(poly));
              setDirty(true);
            }}
          />{" "}
          Straighten perspective
        </label>
        {perspective && (
          <label title="The real blanket's width divided by its height">
            Blanket shape{" "}
            <input
              type="number"
              min={0.2}
              max={5}
              step={0.01}
              value={aspect}
              placeholder={project.estimated_aspect ? `auto (${project.estimated_aspect.toFixed(2)})` : "auto"}
              onChange={(e) => {
                setAspect(e.target.value);
                setDirty(true);
              }}
              style={{ width: "7em" }}
            />{" "}
            width ÷ height
            {project.estimated_aspect && aspect && Math.abs(parseFloat(aspect) - project.estimated_aspect) > 0.005 && (
              <>
                {" "}
                <button
                  className="link"
                  onClick={() => {
                    setAspect(String(project.estimated_aspect));
                    setDirty(true);
                  }}
                >
                  use estimate {project.estimated_aspect.toFixed(2)}
                </button>
              </>
            )}
          </label>
        )}
        <label title="Applies immediately and re-processes the photos">
          Wrinkles{" "}
          <select value={project.wrinkles} onChange={(e) => setWrinkles(e.target.value as Wrinkles)}>
            <option value="off">Leave as photographed</option>
            <option value="straighten">Straighten the pattern</option>
            <option value="clean">Clean composite (remove creases)</option>
          </select>
        </label>
      </div>
      <p className="muted small">
        Drag the <span className="swatch blanket" /> blanket corners to trace its edge; drag a small midpoint dot to add a
        corner and double-click a corner to remove it.{" "}
        {useQuad
          ? "Drag the four square frame handles onto the blanket's real corners; that shape is stretched into a rectangle."
          : "Drag the crop or its corners to frame the output."}{" "}
        Everything outside the blanket becomes black. Changing the outline resets approvals.
      </p>
      {project.wrinkles !== "off" && (
        <p className="muted small">
          {project.wrinkles === "straighten"
            ? "Wrinkles: each photo is gently warped so the printed pattern lines up with the average shape."
            : "Wrinkles: the blanket is rebuilt from a crease-free blend of all photos; only the baby and props come from each photo."}
        </p>
      )}
      {bg === "combined" && bgMissing && <p className="banner warn">The combined background appears after processing.</p>}
      <div className="stage">
        <svg
          ref={svg}
          viewBox={`0 0 ${a.width} ${a.height}`}
          onPointerMove={onMove}
          onPointerUp={end}
          onPointerCancel={end}
          className={drag ? "dragging" : ""}
        >
          <rect width={a.width} height={a.height} fill="#111" />
          <image href={bgUrl} width={a.width} height={a.height} preserveAspectRatio="none" onError={() => setBgMissing(true)} />
          {nb && (
            <>
              <path
                d={`M0 0H${a.width}V${a.height}H0Z M${nb.x0} ${nb.y0}V${nb.y1}H${nb.x1}V${nb.y0}Z`}
                fill="rgba(0,0,0,.55)"
                fillRule="evenodd"
                pointerEvents="none"
              />
              <rect
                x={nb.x0}
                y={nb.y0}
                width={nb.x1 - nb.x0}
                height={nb.y1 - nb.y0}
                className="crop-rect"
                strokeWidth={r / 3}
                onPointerDown={(e) => svg.current && start({ kind: "move", start: svgPoint(svg.current, e.clientX, e.clientY), orig: nb })(e)}
              />
            </>
          )}
          {poly.length >= 3 && (
            <polygon points={poly.map((p) => p.join(",")).join(" ")} className="blanket-poly" strokeWidth={r / 3} />
          )}
          {nb &&
            ([0, 1] as const).flatMap((cx) =>
              ([0, 1] as const).map((cy) => (
                <rect
                  key={`c${cx}${cy}`}
                  x={(cx ? nb.x1 : nb.x0) - r}
                  y={(cy ? nb.y1 : nb.y0) - r}
                  width={2 * r}
                  height={2 * r}
                  className="handle crop-handle"
                  onPointerDown={start({ kind: "corner", cx, cy })}
                />
              )),
            )}
          {poly.map((p, i) => {
            const q = poly[(i + 1) % poly.length];
            return (
              <circle
                key={`m${i}`}
                cx={(p[0] + q[0]) / 2}
                cy={(p[1] + q[1]) / 2}
                r={r * 0.55}
                className="handle mid-handle"
                onPointerDown={insertAt(i)}
              >
                <title>Drag to add a corner</title>
              </circle>
            );
          })}
          {poly.map((p, i) => (
            <circle
              key={`v${i}`}
              cx={p[0]}
              cy={p[1]}
              r={r}
              className="handle vertex-handle"
              onPointerDown={start({ kind: "vertex", i })}
              onDoubleClick={() => removeVertex(i)}
            />
          ))}
          {useQuad && quad && (
            <>
              <polygon points={quad.map((p) => p.join(",")).join(" ")} className="crop-rect" strokeWidth={r / 3} pointerEvents="none" />
              {quad.map((p, i) => (
                <rect
                  key={`q${i}`}
                  x={p[0] - r}
                  y={p[1] - r}
                  width={2 * r}
                  height={2 * r}
                  className="handle crop-handle"
                  onPointerDown={start({ kind: "quad", i })}
                >
                  <title>{`Blanket ${QUAD_LABELS[i]} corner`}</title>
                </rect>
              ))}
            </>
          )}
        </svg>
      </div>
    </section>
  );
}
