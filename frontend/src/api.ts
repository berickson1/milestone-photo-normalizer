export type Pt = [number, number];
export type Rect = { x: number; y: number; w: number; h: number };
export type Quality = "ok" | "review" | "failed";

export type Alignment = {
  method: "auto" | "manual" | "reference";
  quality: Quality;
  inliers: number;
  inlier_ratio: number;
  coverage: number;
  rms_error: number;
  ambiguous: boolean;
  reasons: string[];
};

export type Landmark = { ref: Pt; src: Pt };

export type Photo = {
  id: string;
  original_name: string;
  name: string | null;
  width: number;
  height: number;
  work_scale: number;
  status: "pending" | "queued" | "aligning" | "aligned" | "error";
  error: string | null;
  homography: number[][] | null;
  alignment: Alignment | null;
  landmarks: Landmark[];
  color: { enabled: boolean; strength: number; gains: [number, number, number] | null };
  mask_edited: boolean;
  approved: boolean;
  accept_unresolved: boolean;
};

/** The user's label for a photo, or its original filename. */
export const photoLabel = (ph: Pick<Photo, "name" | "original_name">) => ph.name || ph.original_name;

export type Wrinkles = "off" | "straighten" | "clean";

export type Analysis = { scale: number; offset_x: number; offset_y: number; width: number; height: number };

export type Project = {
  id: string;
  name: string;
  revision: number;
  reference_id: string | null;
  crop: Rect | null;
  output_width: number | null;
  blanket_polygon: Pt[] | null;
  perspective: boolean;
  blanket_quad: Pt[] | null;
  blanket_aspect: number | null;
  estimated_aspect: number | null;
  wrinkles: Wrinkles;
  photos: Photo[];
  job: { state: "idle" | "queued" | "running"; done: number; total: number; message: string; error: string | null };
  analysis?: Analysis;
  default_output_width: number | null;
  processing: boolean;
};

export type ProjectSummary = { id: string; name: string; created_at: number; photo_count: number };

export type Issue = { code: string; message: string; blocking: boolean };
export type Stats = {
  blanket_px: number;
  reconstructed_px: number;
  unresolved_px: number;
  reconstructed_fraction: number;
  unresolved_fraction: number;
  donors: { photo_id: string; px: number; name?: string; color?: string }[];
};
export type Readiness = {
  ready: boolean;
  issues: Issue[];
  photos: { photo_id: string; name: string; issues: Issue[]; stats: Stats | null }[];
  revision: number;
};

export class ApiError extends Error {}

async function request<T>(method: string, url: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method };
  if (body instanceof FormData || body instanceof Blob) {
    init.body = body;
  } else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers = { "Content-Type": "application/json" };
  }
  const res = await fetch(url, init);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (typeof data.detail === "string") msg = data.detail;
      else if (Array.isArray(data.detail)) msg = data.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* not JSON */
    }
    throw new ApiError(msg);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

const P = (id: string) => `/api/projects/${id}`;
const PH = (id: string, pid: string) => `${P(id)}/photos/${pid}`;

export const api = {
  listProjects: () => request<ProjectSummary[]>("GET", "/api/projects"),
  createProject: (name: string) => request<Project>("POST", "/api/projects", { name }),
  getProject: (id: string) => request<Project>("GET", P(id)),
  updateProject: (id: string, body: Partial<Pick<Project, "name" | "crop" | "output_width" | "blanket_polygon" | "perspective" | "blanket_quad" | "blanket_aspect" | "wrinkles">>) =>
    request<Project>("PATCH", P(id), body),
  deleteProject: (id: string) => request<void>("DELETE", P(id)),
  setReference: (id: string, photo_id: string) => request<Project>("POST", `${P(id)}/reference`, { photo_id }),
  process: (id: string, realign_all = false) => request<Project>("POST", `${P(id)}/process`, { realign_all }),
  autoBlanket: (id: string) => request<Project>("POST", `${P(id)}/blanket/auto`),
  upload: (id: string, files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    return request<{ added: string[]; rejected: { name: string; reason: string }[]; project: Project }>(
      "POST",
      `${P(id)}/photos`,
      fd,
    );
  },
  deletePhoto: (id: string, pid: string) => request<void>("DELETE", PH(id, pid)),
  updatePhoto: (
    id: string,
    pid: string,
    body: { name?: string | null; color_enabled?: boolean; color_strength?: number; approved?: boolean; accept_unresolved?: boolean },
  ) => request<Project>("PATCH", PH(id, pid), body),
  reorderPhotos: (id: string, photo_ids: string[]) => request<Project>("PUT", `${P(id)}/photos/order`, { photo_ids }),
  setLandmarks: (id: string, pid: string, landmarks: Landmark[]) =>
    request<Project>("PUT", `${PH(id, pid)}/landmarks`, { landmarks }),
  putMask: (id: string, pid: string, png: Blob) => request<Project>("PUT", `${PH(id, pid)}/mask.png`, png),
  resetMask: (id: string, pid: string) => request<Project>("DELETE", `${PH(id, pid)}/mask.png`),
  stats: (id: string, pid: string) => request<Stats>("GET", `${PH(id, pid)}/stats`),
  readiness: (id: string) => request<Readiness>("GET", `${P(id)}/readiness`),
};

type RenderOpts = { view?: "analysis" | "output"; width?: number; blackout?: boolean; reconstruct?: boolean; color?: boolean };

export const urls = {
  thumb: (p: Project, pid: string) => `${PH(p.id, pid)}/thumb.jpg`,
  source: (p: Project, pid: string) => `${PH(p.id, pid)}/source.jpg`,
  background: (p: Project) => `${P(p.id)}/background.jpg?rev=${p.revision}`,
  mask: (p: Project, pid: string) => `${PH(p.id, pid)}/mask.png?rev=${p.revision}`,
  render: (p: Project, pid: string, o: RenderOpts = {}) => {
    const q = new URLSearchParams({ rev: String(p.revision), view: o.view ?? "output" });
    if (o.width) q.set("width", String(o.width));
    for (const k of ["blackout", "reconstruct", "color"] as const) if (o[k] === false) q.set(k, "false");
    return `${PH(p.id, pid)}/render.jpg?${q}`;
  },
  provenance: (p: Project, pid: string, width: number) =>
    `${PH(p.id, pid)}/provenance.png?rev=${p.revision}&width=${width}`,
  exportZip: (p: Project) => `${P(p.id)}/export.zip`,
};

/** Reference full-resolution coordinates -> analysis canvas pixels. */
export const toAnalysis = (a: Analysis, [x, y]: Pt): Pt => [x * a.scale + a.offset_x, y * a.scale + a.offset_y];
export const fromAnalysis = (a: Analysis, [x, y]: Pt): Pt => [(x - a.offset_x) / a.scale, (y - a.offset_y) / a.scale];

export const workSize = (ph: Photo): Pt => [Math.round(ph.width * ph.work_scale), Math.round(ph.height * ph.work_scale)];

/** Client mouse position -> SVG user coordinates. */
export function svgPoint(svg: SVGSVGElement, clientX: number, clientY: number): Pt {
  const m = svg.getScreenCTM();
  if (!m) return [0, 0];
  const p = new DOMPoint(clientX, clientY).matrixTransform(m.inverse());
  return [p.x, p.y];
}
