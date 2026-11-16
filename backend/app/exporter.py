"""Export readiness checks and ZIP generation."""

import io
import re
import tempfile
import zipfile
from pathlib import Path

from . import config
from .geometry import has_frame, output_view
from .imageio import encode_png
from .render import LRU, Renderer
from .storage import Project


class ExportBlocked(Exception):
    def __init__(self, readiness: dict):
        super().__init__("Export is blocked")
        self.readiness = readiness


_readiness_cache = LRU(32)


def _issue(code: str, message: str, blocking: bool = True) -> dict:
    return {"code": code, "message": message, "blocking": blocking}


def readiness(renderer: Renderer, project: Project) -> dict:
    key = (project.id, project.revision, project.job.state)
    cached = _readiness_cache.get(key)
    if cached is not None:
        return cached

    issues = []
    if project.job.state != "idle":
        issues.append(_issue("processing", "Processing is still running"))
    if not project.photos:
        issues.append(_issue("no_photos", "Upload photos first"))
    if project.reference is None:
        issues.append(_issue("no_reference", "Choose a reference photo"))
    if not project.blanket_polygon:
        issues.append(_issue("no_blanket", "Define the blanket boundary"))
    if not has_frame(project):
        issues.append(_issue("no_crop", "Set the shared crop"))

    photos = []
    can_render = has_frame(project) and bool(project.blanket_polygon)
    view = output_view(project, config.PREVIEW_WIDTH) if can_render else None
    for ph in project.photos:
        p_issues, stats = [], None
        al = ph.alignment
        if ph.homography is None:
            msg = ph.error or (al.reasons[0] if al and al.reasons else "Not aligned yet")
            p_issues.append(_issue("not_aligned", f"Alignment failed: {msg}" if al else msg))
        elif al and al.quality != "ok" and not ph.approved:
            p_issues.append(_issue("alignment_review", "Alignment needs review: " + "; ".join(al.reasons)))
        if ph.homography is not None and view is not None:
            stats = renderer.render(project, ph.id, view).stats
            if stats["reconstructed_px"] and not ph.approved:
                p_issues.append(_issue("reconstruction_review", f"{stats['reconstructed_fraction']:.1%} of the blanket is rebuilt from other photos; review it"))
            if stats["unresolved_px"]:
                msg = f"{stats['unresolved_fraction']:.1%} of the blanket has no trustworthy source"
                if ph.accept_unresolved:
                    p_issues.append(_issue("unresolved_accepted", msg + " (accepted as black)", blocking=False))
                else:
                    p_issues.append(_issue("unresolved", msg))
        if ph.color.enabled and ph.color.gains is None and ph.homography is not None and ph.id != project.reference_id:
            p_issues.append(_issue("color_skipped", "Too little shared blanket to match color; left unchanged", blocking=False))
        photos.append({"photo_id": ph.id, "name": ph.original_name, "issues": p_issues, "stats": stats})

    ready = not any(i["blocking"] for i in issues) and not any(i["blocking"] for p in photos for i in p["issues"])
    result = {"ready": ready, "issues": issues, "photos": photos, "revision": project.revision}
    _readiness_cache.put(key, result)
    return result


def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem)[:60] or "photo"


def build_zip(renderer: Renderer, project: Project) -> Path:
    state = readiness(renderer, project)
    if not state["ready"]:
        raise ExportBlocked(state)
    view = output_view(project)
    fd, name = tempfile.mkstemp(suffix=".zip")
    path = Path(name)
    try:
        with open(fd, "wb") as f, zipfile.ZipFile(f, "w", compression=zipfile.ZIP_STORED) as zf:
            for i, ph in enumerate(project.photos, start=1):
                r = renderer.render(project, ph.id, view, full=True)
                zf.writestr(f"{i:03d}_{_safe_stem(ph.original_name)}.png", encode_png(r.image))
            zf.writestr("README.txt", _manifest(project, view.width, view.height))
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _manifest(project: Project, w: int, h: int) -> str:
    buf = io.StringIO()
    buf.write(f"{project.name}\nOutput size: {w}x{h}\nReference: {project.reference.original_name}\n\n")
    for i, ph in enumerate(project.photos, start=1):
        buf.write(f"{i:03d}  {ph.original_name}  sha256={ph.sha256}\n")
    return buf.getvalue()
