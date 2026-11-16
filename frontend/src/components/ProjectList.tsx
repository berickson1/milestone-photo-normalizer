import { useEffect, useState } from "react";
import { api, type ProjectSummary } from "../api";

export default function ProjectList() {
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);

  const load = () => api.listProjects().then(setProjects, (e) => setError(e.message));
  useEffect(() => {
    load();
  }, []);

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const p = await api.createProject(name.trim() || "Milestones");
      window.location.hash = `#/p/${p.id}`;
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const remove = async (p: ProjectSummary) => {
    if (!confirm(`Delete "${p.name}" and all of its photos? This cannot be undone.`)) return;
    try {
      await api.deleteProject(p.id);
      load();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  return (
    <main className="page narrow">
      <h1>Projects</h1>
      <p className="muted">
        Each project holds one set of milestone photos taken on the same blanket. Photos stay on this server.
      </p>
      {error && <div className="banner error">{error}</div>}
      <form className="row" onSubmit={create}>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="New project name" maxLength={120} />
        <button className="primary" type="submit">
          Create project
        </button>
      </form>
      {projects === null ? (
        <p className="muted">Loading…</p>
      ) : projects.length === 0 ? (
        <p className="muted">No projects yet.</p>
      ) : (
        <ul className="project-list">
          {projects.map((p) => (
            <li key={p.id}>
              <a href={`#/p/${p.id}`}>{p.name}</a>
              <span className="muted">
                {p.photo_count} photo{p.photo_count === 1 ? "" : "s"} · {new Date(p.created_at * 1000).toLocaleDateString()}
              </span>
              <button className="link danger" onClick={() => remove(p)}>
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
