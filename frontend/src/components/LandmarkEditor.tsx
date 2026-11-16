import { useState } from "react";
import { api, svgPoint, urls, workSize, type Landmark, type Photo, type Pt } from "../api";
import type { Ctx } from "./ProjectPage";

export default function LandmarkEditor({ project, photo, run }: Ctx & { photo: Photo }) {
  const ref = project.photos.find((p) => p.id === project.reference_id)!;
  const [pairs, setPairs] = useState<Landmark[]>(photo.landmarks);
  const [pending, setPending] = useState<Pt | null>(null);

  const clickRef = (e: React.MouseEvent<SVGSVGElement>) => {
    const [x, y] = svgPoint(e.currentTarget, e.clientX, e.clientY);
    setPending([x / ref.work_scale, y / ref.work_scale]);
  };
  const clickSrc = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!pending) return;
    const [x, y] = svgPoint(e.currentTarget, e.clientX, e.clientY);
    setPairs([...pairs, { ref: pending, src: [x / photo.work_scale, y / photo.work_scale] }]);
    setPending(null);
  };

  const apply = () => run(() => api.setLandmarks(project.id, photo.id, pairs));
  const clear = () => {
    setPairs([]);
    setPending(null);
    run(() => api.setLandmarks(project.id, photo.id, []));
  };
  const changed = JSON.stringify(pairs) !== JSON.stringify(photo.landmarks);

  return (
    <div>
      <div className="toolbar">
        <span>
          {pending ? "Now click the same spot in this photo (right)." : "Click a distinctive blanket spot in the reference (left)."}
        </span>
        <span className="spacer" />
        <span className="muted small">{pairs.length} pair{pairs.length === 1 ? "" : "s"} (need 4+)</span>
        <button disabled={!pairs.length && !pending} onClick={() => (pending ? setPending(null) : setPairs(pairs.slice(0, -1)))}>
          Undo
        </button>
        {photo.landmarks.length > 0 && <button onClick={clear}>Back to automatic</button>}
        <button className="primary" disabled={pairs.length < 4 || !changed} onClick={apply}>
          Apply landmarks
        </button>
      </div>
      <p className="muted small">
        Pick corners of letters, number markings or pattern features spread across the whole blanket, not just one area.
      </p>
      <div className="pair-view">
        <Pane title="Reference" photo={ref} url={urls.source(project, ref.id)} pts={pairs.map((p) => p.ref)} pending={pending} onClick={clickRef} />
        <Pane title="This photo" photo={photo} url={urls.source(project, photo.id)} pts={pairs.map((p) => p.src)} onClick={clickSrc} waiting={!!pending} />
      </div>
    </div>
  );
}

function Pane(props: {
  title: string;
  photo: Photo;
  url: string;
  pts: Pt[];
  pending?: Pt | null;
  waiting?: boolean;
  onClick: (e: React.MouseEvent<SVGSVGElement>) => void;
}) {
  const [w, h] = workSize(props.photo);
  const s = props.photo.work_scale;
  const r = w / 90;
  const mark = (p: Pt, label: string, cls: string) => (
    <g key={label} className={cls}>
      <circle cx={p[0] * s} cy={p[1] * s} r={r} />
      <text x={p[0] * s + r * 1.2} y={p[1] * s - r * 1.2} fontSize={r * 2.2}>
        {label}
      </text>
    </g>
  );
  return (
    <figure className={`pane ${props.waiting ? "waiting" : ""}`}>
      <figcaption>{props.title}</figcaption>
      <svg viewBox={`0 0 ${w} ${h}`} onClick={props.onClick}>
        <image href={props.url} width={w} height={h} />
        {props.pts.map((p, i) => mark(p, String(i + 1), "landmark"))}
        {props.pending && mark(props.pending, String(props.pts.length + 1), "landmark pending")}
      </svg>
    </figure>
  );
}
