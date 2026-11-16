import { useRef, useState } from "react";
import { api, urls, type Photo } from "../api";
import type { Ctx } from "./ProjectPage";

export function qualityBadge(ph: Photo, isRef: boolean) {
  if (isRef) return <span className="badge ref">Reference</span>;
  if (ph.status === "error") return <span className="badge bad">Error</span>;
  if (ph.status !== "aligned") return <span className="badge">{ph.status}</span>;
  if (!ph.homography) return <span className="badge bad">Not aligned</span>;
  const q = ph.alignment?.quality;
  if (q === "ok") return <span className="badge good">Aligned</span>;
  if (ph.approved) return <span className="badge good">Approved</span>;
  return <span className="badge warn">Needs review</span>;
}

export default function PhotosTab({ project, run }: Ctx) {
  const input = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [rejected, setRejected] = useState<{ name: string; reason: string }[]>([]);
  const [drag, setDrag] = useState(false);

  const upload = async (files: File[]) => {
    if (!files.length) return;
    setUploading(true);
    const r = await run(() => api.upload(project.id, files));
    setUploading(false);
    if (r) {
      setRejected(r.rejected);
      run(async () => r.project);
    }
  };

  const makeRef = (ph: Photo) => {
    if (
      project.reference_id &&
      !confirm(
        `Use "${ph.original_name}" as the reference? Every photo will be re-aligned to it, and the blanket outline, crop, landmarks and mask edits will be reset.`,
      )
    )
      return;
    run(() => api.setReference(project.id, ph.id));
  };

  const remove = async (ph: Photo) => {
    if (!confirm(`Remove "${ph.original_name}" from this project?`)) return;
    await run(() => api.deletePhoto(project.id, ph.id));
    run(() => api.getProject(project.id));
  };

  return (
    <section>
      <div
        className={`dropzone ${drag ? "over" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          upload(Array.from(e.dataTransfer.files));
        }}
      >
        <p>Drop photos here, or</p>
        <button className="primary" disabled={uploading} onClick={() => input.current?.click()}>
          {uploading ? "Uploading…" : "Choose photos"}
        </button>
        <input
          ref={input}
          type="file"
          multiple
          accept="image/*,.heic,.heif"
          hidden
          onChange={(e) => {
            upload(Array.from(e.target.files ?? []));
            e.target.value = "";
          }}
        />
        <p className="muted small">JPEG, PNG, WebP, TIFF or HEIC. Originals are stored unchanged.</p>
      </div>
      {rejected.length > 0 && (
        <div className="banner warn">
          Some files were skipped:
          <ul>
            {rejected.map((r, i) => (
              <li key={i}>
                {r.name}: {r.reason}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="row spread">
        <p className="muted">
          The <strong>reference</strong> photo defines the shared perspective. Pick the one with the flattest, most
          complete view of the blanket.
        </p>
        <div className="row">
          <button disabled={!project.photos.length} onClick={() => run(() => api.process(project.id))}>
            Process new photos
          </button>
          <button
            disabled={!project.photos.length}
            onClick={() => confirm("Re-run automatic alignment for every photo? Manual landmarks are kept.") && run(() => api.process(project.id, true))}
          >
            Realign all
          </button>
        </div>
      </div>

      <div className="grid">
        {project.photos.map((ph) => {
          const isRef = ph.id === project.reference_id;
          return (
            <figure key={ph.id} className={`card ${isRef ? "is-ref" : ""}`}>
              <img src={urls.thumb(project, ph.id)} alt={ph.original_name} loading="lazy" />
              <figcaption>
                <div className="name" title={ph.original_name}>
                  {ph.original_name}
                </div>
                <div className="row">
                  {qualityBadge(ph, isRef)}
                  {ph.alignment?.method === "manual" && <span className="badge">Landmarks</span>}
                </div>
                {ph.error && <div className="small bad-text">{ph.error}</div>}
                <div className="row">
                  {!isRef && (
                    <button className="link" disabled={project.processing} onClick={() => makeRef(ph)}>
                      Set as reference
                    </button>
                  )}
                  <button className="link danger" disabled={project.processing} onClick={() => remove(ph)}>
                    Remove
                  </button>
                </div>
              </figcaption>
            </figure>
          );
        })}
      </div>
    </section>
  );
}
