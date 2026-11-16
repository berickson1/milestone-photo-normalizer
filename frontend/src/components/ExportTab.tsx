import { useEffect, useState } from "react";
import { api, urls, type Readiness } from "../api";
import type { Ctx } from "./ProjectPage";

export default function ExportTab({ project }: Ctx) {
  const [r, setR] = useState<Readiness | null>(null);
  const [loading, setLoading] = useState(false);
  const idle = project.job.state === "idle" && !project.processing;

  useEffect(() => {
    setLoading(true);
    api
      .readiness(project.id)
      .then(setR, () => setR(null))
      .finally(() => setLoading(false));
  }, [project.id, project.revision, idle]);

  const blocking = r ? r.issues.filter((i) => i.blocking).length + r.photos.reduce((n, p) => n + p.issues.filter((i) => i.blocking).length, 0) : 0;
  const width = project.output_width ?? project.default_output_width;

  return (
    <section className="narrow">
      <p>
        Exports a ZIP of same-size PNGs{width ? ` (${width} px wide)` : ""}, one per photo, in upload order. Originals are
        never changed.
      </p>
      {loading && !r && <p className="muted">Checking…</p>}
      {r && (
        <>
          {r.ready ? (
            <div className="banner ok">Everything is ready to export.</div>
          ) : (
            <div className="banner warn">
              {blocking} thing{blocking === 1 ? "" : "s"} to resolve before exporting. Fix them on the Review tab.
            </div>
          )}
          <ul className="checklist">
            {r.issues.map((i) => (
              <li key={i.code} className={i.blocking ? "blocking" : ""}>
                {i.message}
              </li>
            ))}
            {r.photos.map((p) => (
              <li key={p.photo_id} className={p.issues.some((i) => i.blocking) ? "blocking" : p.issues.length ? "note" : "done"}>
                <strong>{p.name}</strong>
                {p.issues.length === 0 ? (
                  <span className="muted"> Ready</span>
                ) : (
                  <ul>
                    {p.issues.map((i) => (
                      <li key={i.code} className={i.blocking ? "blocking" : "note"}>
                        {i.message}
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
          <a className={`button primary ${r.ready ? "" : "disabled"}`} href={r.ready ? urls.exportZip(project) : undefined} aria-disabled={!r.ready} download>
            Download ZIP
          </a>
        </>
      )}
    </section>
  );
}
