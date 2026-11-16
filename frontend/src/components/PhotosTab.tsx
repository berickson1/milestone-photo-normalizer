import { useRef, useState } from "react";
import { api, photoLabel, urls, type Photo } from "../api";
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
  const [editing, setEditing] = useState<{ id: string; value: string } | null>(null);
  const [dragId, setDragId] = useState<string | null>(null);
  const [dropAt, setDropAt] = useState<number | null>(null);
  const [pendingOrder, setPendingOrder] = useState<string[] | null>(null);

  const byId = new Map(project.photos.map((p) => [p.id, p]));
  const ordered =
    pendingOrder && pendingOrder.length === project.photos.length && pendingOrder.every((id) => byId.has(id))
      ? pendingOrder.map((id) => byId.get(id)!)
      : project.photos;

  const reorder = async (ids: string[]) => {
    if (ids.every((id, i) => id === project.photos[i]?.id)) return;
    setPendingOrder(ids);
    await run(() => api.reorderPhotos(project.id, ids));
    setPendingOrder(null);
  };

  const moveTo = (id: string, slot: number) => {
    const ids = ordered.map((p) => p.id);
    const from = ids.indexOf(id);
    ids.splice(from, 1);
    ids.splice(slot > from ? slot - 1 : slot, 0, id);
    reorder(ids);
  };

  const sortByName = () => {
    const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });
    reorder([...ordered].sort((x, y) => collator.compare(photoLabel(x), photoLabel(y))).map((p) => p.id));
  };

  const saveName = async () => {
    if (!editing) return;
    const ph = byId.get(editing.id);
    const value = editing.value.trim();
    setEditing(null);
    if (!ph || value === photoLabel(ph)) return;
    await run(() => api.updatePhoto(project.id, ph.id, { name: value && value !== ph.original_name ? value : null }));
  };

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
        `Use "${photoLabel(ph)}" as the reference? Every photo will be re-aligned to it, and the blanket outline, crop, landmarks and mask edits will be reset.`,
      )
    )
      return;
    run(() => api.setReference(project.id, ph.id));
  };

  const remove = async (ph: Photo) => {
    if (!confirm(`Remove "${photoLabel(ph)}" from this project?`)) return;
    await run(() => api.deletePhoto(project.id, ph.id));
    run(() => api.getProject(project.id));
  };

  return (
    <section>
      <div
        className={`dropzone ${drag ? "over" : ""}`}
        onDragOver={(e) => {
          if (!e.dataTransfer.types.includes("Files")) return;
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
          <button disabled={project.photos.length < 2} onClick={sortByName} title="Order photos by their names, treating numbers naturally (2 before 10)">
            Sort by name
          </button>
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

      {project.photos.length > 1 && (
        <p className="muted small">
          Drag photos (or use ◀ ▶) to set the export order. Click a name to rename it; exported files are named after it.
        </p>
      )}
      <div className="grid" onDragEnd={() => (setDragId(null), setDropAt(null))}>
        {ordered.map((ph, i) => {
          const isRef = ph.id === project.reference_id;
          const label = photoLabel(ph);
          const dropClass = dragId && dropAt === i ? "drop-before" : dragId && dropAt === i + 1 ? "drop-after" : "";
          return (
            <figure
              key={ph.id}
              className={`card ${isRef ? "is-ref" : ""} ${dragId === ph.id ? "dragging" : ""} ${dropClass}`}
              draggable={editing?.id !== ph.id}
              onDragStart={(e) => {
                setDragId(ph.id);
                e.dataTransfer.effectAllowed = "move";
                e.dataTransfer.setData("application/x-photo-id", ph.id);
              }}
              onDragOver={(e) => {
                if (!dragId) return;
                e.preventDefault();
                const box = e.currentTarget.getBoundingClientRect();
                setDropAt(e.clientX < box.left + box.width / 2 ? i : i + 1);
              }}
              onDrop={(e) => {
                if (!dragId) return;
                e.preventDefault();
                e.stopPropagation();
                if (dropAt !== null) moveTo(dragId, dropAt);
                setDragId(null);
                setDropAt(null);
              }}
            >
              <img src={urls.thumb(project, ph.id)} alt={label} loading="lazy" draggable={false} />
              <figcaption>
                {editing?.id === ph.id ? (
                  <input
                    autoFocus
                    value={editing.value}
                    maxLength={120}
                    placeholder={ph.original_name}
                    aria-label="Photo name"
                    onChange={(e) => setEditing({ id: ph.id, value: e.target.value })}
                    onFocus={(e) => e.target.select()}
                    onBlur={saveName}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") e.currentTarget.blur();
                      if (e.key === "Escape") setEditing(null);
                    }}
                  />
                ) : (
                  <button
                    className="name name-edit"
                    title={ph.name ? `${label}\nOriginal file: ${ph.original_name}\nClick to rename` : `${label}\nClick to rename`}
                    onClick={() => setEditing({ id: ph.id, value: label })}
                  >
                    <span className="muted">{i + 1}.</span> {label}
                  </button>
                )}
                {ph.name && (
                  <div className="name muted small" title={ph.original_name}>
                    {ph.original_name}
                  </div>
                )}
                <div className="row">
                  {qualityBadge(ph, isRef)}
                  {ph.alignment?.method === "manual" && <span className="badge">Landmarks</span>}
                </div>
                {ph.error && <div className="small bad-text">{ph.error}</div>}
                <div className="row">
                  <button className="link" disabled={i === 0} onClick={() => moveTo(ph.id, i - 1)} aria-label={`Move ${label} earlier`} title="Move earlier">
                    ◀
                  </button>
                  <button
                    className="link"
                    disabled={i === ordered.length - 1}
                    onClick={() => moveTo(ph.id, i + 2)}
                    aria-label={`Move ${label} later`}
                    title="Move later"
                  >
                    ▶
                  </button>
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
