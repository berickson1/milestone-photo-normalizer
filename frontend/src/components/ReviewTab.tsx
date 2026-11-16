import { useEffect, useRef, useState } from "react";
import { api, photoLabel, urls, type Photo, type Readiness, type Stats } from "../api";
import type { Ctx } from "./ProjectPage";
import { qualityBadge } from "./PhotosTab";
import MaskEditor from "./MaskEditor";
import LandmarkEditor from "./LandmarkEditor";

type Mode = "result" | "compare" | "mask" | "landmarks";
const MODES: [Mode, string][] = [
  ["result", "Result"],
  ["compare", "Compare to reference"],
  ["mask", "Baby mask"],
  ["landmarks", "Fix alignment"],
];

export default function ReviewTab(ctx: Ctx) {
  const { project, run } = ctx;
  const [selected, setSelected] = useState<string | null>(project.photos[0]?.id ?? null);
  const [mode, setMode] = useState<Mode>("result");
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const idle = project.job.state === "idle" && !project.processing;

  useEffect(() => {
    if (!idle) return;
    api.readiness(project.id).then(setReadiness, () => {});
  }, [project.id, project.revision, idle]);

  const photo = project.photos.find((p) => p.id === selected) ?? project.photos[0];
  if (!photo || !project.analysis) return <p className="muted">Upload photos and choose a reference first.</p>;

  const idx = project.photos.indexOf(photo);
  const issuesFor = (id: string) => readiness?.photos.find((p) => p.photo_id === id)?.issues ?? [];
  const go = (d: number) => setSelected(project.photos[(idx + d + project.photos.length) % project.photos.length].id);
  const isRef = photo.id === project.reference_id;

  return (
    <section className="review">
      <aside className="sidebar">
        {project.photos.map((ph) => {
          const issues = issuesFor(ph.id).filter((i) => i.blocking);
          return (
            <button key={ph.id} className={`side-item ${ph.id === photo.id ? "active" : ""}`} onClick={() => setSelected(ph.id)}>
              <img src={urls.thumb(project, ph.id)} alt="" />
              <span>
                <span className="name">{photoLabel(ph)}</span>
                <span className="row">
                  {qualityBadge(ph, ph.id === project.reference_id)}
                  {issues.length > 0 && <span className="badge warn">{issues.length} to fix</span>}
                </span>
              </span>
            </button>
          );
        })}
      </aside>
      <div className="review-main">
        <div className="toolbar">
          <button onClick={() => go(-1)} aria-label="Previous photo">
            ←
          </button>
          <strong className="name">{photoLabel(photo)}</strong>
          <button onClick={() => go(1)} aria-label="Next photo">
            →
          </button>
          <span className="spacer" />
          <div className="segmented">
            {MODES.map(([m, label]) => (
              <button
                key={m}
                className={m === mode ? "active" : ""}
                onClick={() => setMode(m)}
                disabled={m === "landmarks" && isRef}
                title={m === "landmarks" && isRef ? "The reference defines the alignment" : undefined}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        {issuesFor(photo.id).map((i) => (
          <div key={i.code} className={`banner ${i.blocking ? "warn" : "info"}`}>
            {i.message}
          </div>
        ))}
        {photo.status === "error" && photo.error && <div className="banner error">{photo.error}</div>}
        {!photo.homography && mode !== "landmarks" && mode !== "mask" ? (
          <p className="muted">
            {photo.status === "aligned" || photo.status === "error"
              ? "Automatic alignment failed. Use “Fix alignment” to place matching points by hand."
              : "Waiting for alignment…"}
          </p>
        ) : mode === "result" ? (
          <ResultView {...ctx} photo={photo} />
        ) : mode === "compare" ? (
          <CompareView {...ctx} photo={photo} />
        ) : mode === "mask" ? (
          <MaskEditor key={photo.id} {...ctx} photo={photo} />
        ) : (
          !isRef && <LandmarkEditor key={photo.id} {...ctx} photo={photo} />
        )}
        <PhotoControls {...ctx} photo={photo} isRef={isRef} run={run} unresolved={issuesFor(photo.id).some((i) => i.code.startsWith("unresolved"))} />
      </div>
    </section>
  );
}

function ResultView({ project, photo }: Ctx & { photo: Photo }) {
  const [overlay, setOverlay] = useState(false);
  const [stats, setStats] = useState<Stats | null>(null);
  const width = 1200;
  useEffect(() => {
    if (!project.crop) return;
    api.stats(project.id, photo.id).then(setStats, () => setStats(null));
  }, [project.id, photo.id, project.revision, project.crop]);
  if (!project.crop) return <p className="muted">Set the blanket outline and crop on the “Blanket &amp; crop” tab first.</p>;
  return (
    <div>
      <div className="row">
        <label>
          <input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} /> Highlight rebuilt
          areas
        </label>
        {stats && (
          <span className="muted small">
            {(stats.reconstructed_fraction * 100).toFixed(1)}% rebuilt from other photos ·{" "}
            {(stats.unresolved_fraction * 100).toFixed(1)}% missing
          </span>
        )}
      </div>
      <div className="stage result">
        <img src={urls.render(project, photo.id, { width })} alt="Normalized result" />
        {overlay && <img className="overlay" src={urls.provenance(project, photo.id, width)} alt="" />}
      </div>
      {overlay && stats && (
        <ul className="legend">
          {stats.donors.map((d) => (
            <li key={d.photo_id}>
              <span className="swatch" style={{ background: d.color }} /> from {d.name}
            </li>
          ))}
          {stats.unresolved_px > 0 && (
            <li>
              <span className="swatch hatch" /> no source available (will be black)
            </li>
          )}
          {stats.donors.length === 0 && stats.unresolved_px === 0 && <li className="muted">Nothing was rebuilt.</li>}
        </ul>
      )}
    </div>
  );
}

function CompareView({ project, photo }: Ctx & { photo: Photo }) {
  const [opacity, setOpacity] = useState(0.5);
  const [blink, setBlink] = useState(false);
  const [on, setOn] = useState(true);
  useEffect(() => {
    if (!blink) return;
    const t = setInterval(() => setOn((v) => !v), 600);
    return () => clearInterval(t);
  }, [blink]);
  const a = project.analysis!;
  const poly = project.blanket_polygon?.map(([x, y]) => `${x * a.scale + a.offset_x},${y * a.scale + a.offset_y}`).join(" ");
  const raw = { view: "analysis" as const, blackout: false, reconstruct: false };
  return (
    <div>
      <div className="row">
        <label>
          Photo opacity{" "}
          <input type="range" min={0} max={1} step={0.05} value={opacity} disabled={blink} onChange={(e) => setOpacity(+e.target.value)} />
        </label>
        <label>
          <input type="checkbox" checked={blink} onChange={(e) => setBlink(e.target.checked)} /> Blink
        </label>
        <span className="muted small">Blanket features should stay still. If they jump, use “Fix alignment”.</span>
      </div>
      <div className="stage">
        <svg viewBox={`0 0 ${a.width} ${a.height}`}>
          <image href={urls.render(project, project.reference_id!, { ...raw, color: false })} width={a.width} height={a.height} />
          <image
            href={urls.render(project, photo.id, raw)}
            width={a.width}
            height={a.height}
            opacity={blink ? (on ? 1 : 0) : opacity}
          />
          {poly && <polygon points={poly} className="blanket-poly" fill="none" strokeWidth={a.width / 500} pointerEvents="none" />}
        </svg>
      </div>
    </div>
  );
}

function PhotoControls({
  project,
  photo,
  isRef,
  run,
  unresolved,
}: Ctx & { photo: Photo; isRef: boolean; unresolved: boolean }) {
  const [strength, setStrength] = useState(photo.color.strength);
  const timer = useRef<number>(undefined);
  useEffect(() => setStrength(photo.color.strength), [photo.id, photo.color.strength]);
  const update = (body: Parameters<typeof api.updatePhoto>[2]) => run(() => api.updatePhoto(project.id, photo.id, body));
  const al = photo.alignment;
  const g = photo.color.gains;
  return (
    <div className="controls">
      <fieldset>
        <legend>Alignment</legend>
        {al ? (
          <div className="small">
            {isRef ? "Reference photo" : `${al.method === "manual" ? "Manual landmarks" : "Automatic"} · ${al.quality}`}
            {!isRef && (
              <div className="muted">
                {al.inliers} matches · {(al.coverage * 100).toFixed(0)}% blanket coverage · {al.rms_error.toFixed(1)} px error
              </div>
            )}
            {al.reasons.length > 0 && <div className="muted">{al.reasons.join("; ")}</div>}
          </div>
        ) : (
          <span className="muted small">Not aligned yet</span>
        )}
      </fieldset>
      <fieldset disabled={isRef}>
        <legend>Colour</legend>
        <label>
          <input type="checkbox" checked={photo.color.enabled} onChange={(e) => update({ color_enabled: e.target.checked })} /> Match
          the reference’s colour
        </label>
        <label>
          Strength{" "}
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={strength}
            disabled={!photo.color.enabled}
            onChange={(e) => {
              const v = +e.target.value;
              setStrength(v);
              clearTimeout(timer.current);
              timer.current = window.setTimeout(() => update({ color_strength: v }), 400);
            }}
          />{" "}
          {Math.round(strength * 100)}%
        </label>
        {isRef ? (
          <div className="muted small">Other photos are matched to this one.</div>
        ) : g ? (
          <div className="muted small">Gains R {g[0].toFixed(2)} · G {g[1].toFixed(2)} · B {g[2].toFixed(2)}</div>
        ) : (
          <div className="muted small">Not enough shared blanket to estimate a correction.</div>
        )}
      </fieldset>
      <fieldset>
        <legend>Sign-off</legend>
        <label>
          <input
            type="checkbox"
            checked={photo.approved}
            disabled={!photo.homography}
            onChange={(e) => update({ approved: e.target.checked })}
          />{" "}
          I’ve reviewed this photo (alignment and rebuilt areas look right)
        </label>
        {(unresolved || photo.accept_unresolved) && (
          <label>
            <input type="checkbox" checked={photo.accept_unresolved} onChange={(e) => update({ accept_unresolved: e.target.checked })} />{" "}
            Leave missing blanket areas black
          </label>
        )}
      </fieldset>
    </div>
  );
}
