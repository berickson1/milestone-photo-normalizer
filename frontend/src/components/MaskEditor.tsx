import { useEffect, useRef, useState } from "react";
import { api, urls, type Photo } from "../api";
import type { Ctx } from "./ProjectPage";

const PAINT = "rgb(255, 40, 120)";

export default function MaskEditor({ project, photo, run }: Ctx & { photo: Photo }) {
  const a = project.analysis!;
  const canvas = useRef<HTMLCanvasElement>(null);
  const last = useRef<[number, number] | null>(null);
  const [tool, setTool] = useState<"add" | "erase">("add");
  const [brush, setBrush] = useState(Math.round(a.width / 60));
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (dirty) return;
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      const c = canvas.current;
      if (cancelled || !c) return;
      const ctx = c.getContext("2d")!;
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.drawImage(img, 0, 0, c.width, c.height);
      const data = ctx.getImageData(0, 0, c.width, c.height);
      const d = data.data;
      for (let i = 0; i < d.length; i += 4) {
        const on = d[i] > 127;
        d[i] = 255;
        d[i + 1] = 40;
        d[i + 2] = 120;
        d[i + 3] = on ? 255 : 0;
      }
      ctx.putImageData(data, 0, 0);
    };
    img.src = urls.mask(project, photo.id);
    return () => {
      cancelled = true;
    };
  }, [project.revision, photo.id, dirty]);

  const pos = (e: React.PointerEvent): [number, number] => {
    const c = canvas.current!;
    const r = c.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * c.width, ((e.clientY - r.top) / r.height) * c.height];
  };

  const stroke = (from: [number, number], to: [number, number]) => {
    const ctx = canvas.current!.getContext("2d")!;
    ctx.globalCompositeOperation = tool === "add" ? "source-over" : "destination-out";
    ctx.strokeStyle = PAINT;
    ctx.fillStyle = PAINT;
    ctx.lineWidth = brush;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(...from);
    ctx.lineTo(...to);
    ctx.stroke();
    ctx.globalCompositeOperation = "source-over";
  };

  const save = async () => {
    const c = canvas.current!;
    const src = c.getContext("2d")!.getImageData(0, 0, c.width, c.height);
    const out = document.createElement("canvas");
    out.width = c.width;
    out.height = c.height;
    const octx = out.getContext("2d")!;
    const img = octx.createImageData(c.width, c.height);
    for (let i = 0; i < src.data.length; i += 4) {
      const v = src.data[i + 3] > 127 ? 255 : 0;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
      img.data[i + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    const blob = await new Promise<Blob | null>((res) => out.toBlob(res, "image/png"));
    if (!blob) return;
    setSaving(true);
    const r = await run(() => api.putMask(project.id, photo.id, blob));
    setSaving(false);
    if (r) setDirty(false);
  };

  return (
    <div>
      <div className="toolbar">
        <div className="segmented">
          <button className={tool === "add" ? "active" : ""} onClick={() => setTool("add")}>
            Paint baby
          </button>
          <button className={tool === "erase" ? "active" : ""} onClick={() => setTool("erase")}>
            Erase
          </button>
        </div>
        <label>
          Brush <input type="range" min={2} max={Math.round(a.width / 8)} value={brush} onChange={(e) => setBrush(+e.target.value)} />
        </label>
        <span className="muted small">{photo.mask_edited ? "Edited by hand" : "Automatic"}</span>
        <span className="spacer" />
        {dirty && <button onClick={() => setDirty(false)}>Discard</button>}
        {photo.mask_edited && !dirty && (
          <button onClick={() => confirm("Throw away your mask edits and use the automatic mask?") && run(() => api.resetMask(project.id, photo.id))}>
            Use automatic mask
          </button>
        )}
        <button className="primary" disabled={!dirty || saving} onClick={save}>
          {saving ? "Saving…" : "Save mask"}
        </button>
      </div>
      <p className="muted small">
        Paint over the baby and anything else that isn’t blanket (hands, toys, shadows). Masked areas are never used to
        rebuild other photos. In this photo’s own result they stay as they are.
      </p>
      <div className="stage mask-stage">
        {photo.homography ? (
          <img src={urls.render(project, photo.id, { view: "analysis", blackout: false, reconstruct: false })} alt="" draggable={false} />
        ) : (
          <div className="muted">This photo needs to be aligned before its mask can be edited.</div>
        )}
        <canvas
          ref={canvas}
          width={a.width}
          height={a.height}
          className="mask-canvas"
          onPointerDown={(e) => {
            const p = pos(e);
            last.current = p;
            stroke(p, p);
            setDirty(true);
            e.currentTarget.setPointerCapture(e.pointerId);
          }}
          onPointerMove={(e) => {
            if (!last.current) return;
            const p = pos(e);
            stroke(last.current, p);
            last.current = p;
          }}
          onPointerUp={() => (last.current = null)}
          onPointerCancel={() => (last.current = null)}
        />
      </div>
    </div>
  );
}
