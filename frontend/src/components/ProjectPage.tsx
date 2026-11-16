import { useCallback, useEffect, useState } from "react";
import { api, type Project } from "../api";
import PhotosTab from "./PhotosTab";
import SetupTab from "./SetupTab";
import ReviewTab from "./ReviewTab";
import ExportTab from "./ExportTab";

export type Ctx = {
  project: Project;
  setProject: (p: Project) => void;
  /** Runs an API call, shows errors in the page banner, and applies a returned project. */
  run: <T>(fn: () => Promise<T>) => Promise<T | undefined>;
};

const TABS = ["photos", "setup", "review", "export"] as const;
type Tab = (typeof TABS)[number];
const LABELS: Record<Tab, string> = { photos: "1. Photos", setup: "2. Blanket & crop", review: "3. Review", export: "4. Export" };

export default function ProjectPage({ id }: { id: string }) {
  const [project, setProject] = useState<Project | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("photos");

  useEffect(() => {
    api.getProject(id).then(setProject, (e) => setError(e.message));
  }, [id]);

  const busy = !!project && (project.processing || project.job.state !== "idle");
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(() => {
      api.getProject(id).then(setProject, () => {});
    }, 1000);
    return () => clearInterval(t);
  }, [busy, id]);

  const run = useCallback(async <T,>(fn: () => Promise<T>) => {
    setError(null);
    try {
      const r = await fn();
      if (r && typeof r === "object" && "photos" in r && "revision" in r) setProject(r as unknown as Project);
      return r;
    } catch (e) {
      setError((e as Error).message);
      return undefined;
    }
  }, []);

  if (!project) return <main className="page">{error ? <div className="banner error">{error}</div> : "Loading…"}</main>;

  const ctx: Ctx = { project, setProject, run };
  const rename = () => {
    const name = prompt("Project name", project.name);
    if (name && name.trim()) run(() => api.updateProject(project.id, { name: name.trim() }));
  };

  return (
    <main className="page">
      <div className="project-head">
        <h1>
          {project.name}{" "}
          <button className="link" onClick={rename}>
            Rename
          </button>
        </h1>
        <JobStatus project={project} />
      </div>
      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t} className={t === tab ? "active" : ""} onClick={() => setTab(t)}>
            {LABELS[t]}
          </button>
        ))}
      </nav>
      {error && (
        <div className="banner error">
          {error}{" "}
          <button className="link" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}
      {project.job.error && <div className="banner error">Processing failed: {project.job.error}</div>}
      {tab === "photos" && <PhotosTab {...ctx} />}
      {tab === "setup" && <SetupTab {...ctx} />}
      {tab === "review" && <ReviewTab {...ctx} />}
      {tab === "export" && <ExportTab {...ctx} />}
    </main>
  );
}

function JobStatus({ project }: { project: Project }) {
  const { job } = project;
  if (job.state === "idle" && !project.processing) return <span className="muted">Up to date</span>;
  const pct = job.total ? Math.round((100 * job.done) / job.total) : 0;
  return (
    <div className="job">
      <span>{job.message || (job.state === "queued" ? "Queued…" : "Processing…")}</span>
      <progress value={job.total ? job.done : undefined} max={job.total || undefined} aria-label={`${pct}%`} />
    </div>
  );
}
