import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np
from fastapi import BackgroundTasks, Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .exporter import ExportBlocked, build_zip, readiness
from .geometry import analysis_view, default_output_width, estimate_aspect, has_frame, output_view
from .imageio import InvalidImageError, decode_gray_bytes, encode_jpeg, encode_png, write_bytes_atomic
from .jobs import JobManager
from .masks import propose_blanket
from .pipeline import Pipeline, crop_for_polygon, set_quad_from_polygon, set_reference
from .render import Renderer, donor_color, provenance_overlay
from .storage import Landmark, NotFoundError, Project, ProjectStore, Rect, dump

log = logging.getLogger(__name__)


class State:
    store: ProjectStore
    renderer: Renderer
    pipeline: Pipeline
    jobs: JobManager


state = State()


def init_state(data_dir: Path | None = None) -> State:
    state.store = ProjectStore(data_dir)
    state.renderer = Renderer(state.store)
    state.pipeline = Pipeline(state.store, state.renderer)
    state.jobs = JobManager(state.store, state.pipeline, config.MAX_CONCURRENT_JOBS)
    return state


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(state, "store"):
        init_state()
    state.jobs.recover()
    yield
    state.jobs.shutdown()


app = FastAPI(title="Milestone Photo Normalizer", lifespan=lifespan)


@app.exception_handler(NotFoundError)
async def _not_found(_: Request, exc: NotFoundError):
    return JSONResponse({"detail": "Not found"}, status_code=404)


NO_STORE = {"Cache-Control": "no-store"}


def _project(project_id: str) -> Project:
    return state.store.get(project_id)


def _photo(project: Project, photo_id: str):
    ph = project.photo(photo_id)
    if ph is None:
        raise HTTPException(404, "Photo not found")
    return ph


def _project_json(project: Project) -> dict:
    data = dump(project)
    ref = project.reference
    if ref is not None:
        av = analysis_view(ref)
        data["analysis"] = {
            "scale": ref.work_scale,
            "offset_x": float(av.matrix[0, 2]),
            "offset_y": float(av.matrix[1, 2]),
            "width": av.width,
            "height": av.height,
        }
    data["default_output_width"] = default_output_width(project) if has_frame(project) else None
    data["estimated_aspect"] = (
        round(estimate_aspect(np.array(project.blanket_quad), (ref.width, ref.height) if ref else None), 4)
        if project.blanket_quad
        else None
    )
    data["processing"] = state.jobs.is_active(project.id)
    return data


# ---- projects ---------------------------------------------------------------------
class CreateProject(BaseModel):
    name: str = Field(default="Milestones", max_length=120)


class UpdateProject(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    crop: Rect | None = None
    output_width: int | None = Field(default=None, ge=16, le=config.MAX_OUTPUT_WIDTH)
    blanket_polygon: list[tuple[float, float]] | None = None
    perspective: bool | None = None
    blanket_quad: list[tuple[float, float]] | None = None
    blanket_aspect: float | None = Field(default=None, gt=0.1, lt=10)
    wrinkles: Literal["off", "straighten", "clean"] | None = None


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/projects")
def list_projects():
    return [
        {"id": p.id, "name": p.name, "created_at": p.created_at, "photo_count": len(p.photos), "reference_id": p.reference_id}
        for p in state.store.list()
    ]


@app.post("/api/projects", status_code=201)
def create_project(body: CreateProject):
    return _project_json(state.store.create(body.name))


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    return _project_json(_project(project_id))


@app.patch("/api/projects/{project_id}")
def update_project(project_id: str, body: UpdateProject):
    fields = body.model_fields_set
    if "blanket_polygon" in fields and body.blanket_polygon is not None and len(body.blanket_polygon) < 3:
        raise HTTPException(422, "The blanket boundary needs at least 3 points")
    if "blanket_quad" in fields and body.blanket_quad is not None:
        quad = np.array(body.blanket_quad, dtype=np.float32)
        if len(quad) != 4 or not cv2.isContourConvex(quad.reshape(-1, 1, 2)) or abs(cv2.contourArea(quad)) < 1:
            raise HTTPException(422, "The blanket corners must form a convex four-sided shape")

    def fn(p: Project):
        if "name" in fields and body.name:
            p.name = body.name.strip() or p.name
        if "crop" in fields:
            p.crop = body.crop
        if "output_width" in fields:
            p.output_width = body.output_width
        if "blanket_polygon" in fields:
            p.blanket_polygon = body.blanket_polygon
            for ph in p.photos:
                ph.approved = False
        if "perspective" in fields and body.perspective is not None:
            p.perspective = body.perspective
        if "blanket_quad" in fields:
            p.blanket_quad = body.blanket_quad
            if body.blanket_quad is None:
                # Null means "fit the corners to the blanket outline again".
                set_quad_from_polygon(p)
        if "blanket_aspect" in fields:
            p.blanket_aspect = body.blanket_aspect
        if "wrinkles" in fields and body.wrinkles is not None and body.wrinkles != p.wrinkles:
            retolerance = (p.wrinkles == "off") != (body.wrinkles == "off")
            p.wrinkles = body.wrinkles
            # Alignment tolerances depend on whether wrinkles will be corrected.
            for ph in p.photos:
                if retolerance and ph.homography is not None and not ph.landmarks:
                    ph.status = "pending"

    before = _project(project_id).wrinkles
    project = state.store.mutate(project_id, fn)
    if project.reference and ("blanket_polygon" in fields or project.wrinkles != before):
        state.renderer.invalidate()
        state.jobs.request(project_id)
    elif fields & {"perspective", "blanket_quad", "blanket_aspect", "crop", "output_width"}:
        state.renderer.invalidate()
    return _project_json(project)


@app.delete("/api/projects/{project_id}", status_code=204)
def delete_project(project_id: str):
    if state.jobs.is_active(project_id):
        raise HTTPException(409, "Wait for processing to finish")
    state.store.delete(project_id)
    state.renderer.invalidate()


class SetReference(BaseModel):
    photo_id: str


@app.post("/api/projects/{project_id}/reference")
def change_reference(project_id: str, body: SetReference):
    if state.jobs.is_active(project_id):
        raise HTTPException(409, "Wait for processing to finish before changing the reference")
    project = _project(project_id)
    _photo(project, body.photo_id)
    project = state.store.mutate(project_id, lambda p: set_reference(state.store, p, body.photo_id))
    state.renderer.invalidate()
    state.jobs.request(project_id)
    return _project_json(project)


@app.post("/api/projects/{project_id}/process")
def process(project_id: str, realign_all: Annotated[bool, Body(embed=True)] = False):
    def fn(p: Project):
        for ph in p.photos:
            if realign_all or ph.status == "error":
                ph.status = "pending"

    state.store.mutate(project_id, fn)
    state.jobs.request(project_id)
    return _project_json(_project(project_id))


@app.post("/api/projects/{project_id}/blanket/auto")
def auto_blanket(project_id: str):
    project = _project(project_id)
    ref = project.reference
    if ref is None:
        raise HTTPException(409, "Choose a reference photo first")
    img, _ = state.renderer.source(project, ref, full=False)
    poly = propose_blanket(img, ref.work_scale)

    def fn(p: Project):
        p.blanket_polygon = poly
        p.crop = crop_for_polygon(poly)
        set_quad_from_polygon(p)
        for ph in p.photos:
            ph.approved = False

    project = state.store.mutate(project_id, fn)
    state.jobs.request(project_id)
    return _project_json(project)


# ---- photos -------------------------------------------------------------------------
@app.post("/api/projects/{project_id}/photos", status_code=201)
def upload_photos(project_id: str, files: list[UploadFile] = File(...)):
    project = _project(project_id)
    tmp_dir = state.store.project_dir(project.id) / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    added, rejected = [], []
    for f in files:
        fd, name = tempfile.mkstemp(dir=tmp_dir)
        tmp = Path(name)
        try:
            size = 0
            with open(fd, "wb") as out:
                while chunk := f.file.read(1 << 20):
                    size += len(chunk)
                    if size > config.MAX_UPLOAD_BYTES:
                        raise InvalidImageError(f"larger than {config.MAX_UPLOAD_BYTES // (1 << 20)} MB")
                    out.write(chunk)
            photo = state.pipeline.import_photo(project.id, tmp, f.filename or "photo")
            added.append(photo.id)
        except InvalidImageError as exc:
            rejected.append({"name": f.filename, "reason": f"Not a supported image ({exc})"})
        finally:
            tmp.unlink(missing_ok=True)
    if added:
        state.jobs.request(project.id)
    return {"added": added, "rejected": rejected, "project": _project_json(_project(project_id))}


@app.delete("/api/projects/{project_id}/photos/{photo_id}", status_code=204)
def delete_photo(project_id: str, photo_id: str):
    if state.jobs.is_active(project_id):
        raise HTTPException(409, "Wait for processing to finish")
    project = _project(project_id)
    ph = _photo(project, photo_id)
    if project.reference_id == photo_id and len(project.photos) > 1:
        raise HTTPException(409, "Choose a different reference before removing this photo")

    def fn(p: Project):
        p.photos = [x for x in p.photos if x.id != photo_id]
        if p.reference_id == photo_id:
            p.reference_id = None
            set_reference_clear(p)

    state.store.mutate(project_id, fn)
    for path in (
        state.store.original_path(project_id, ph),
        state.store.work_path(project_id, photo_id),
        state.store.thumb_path(project_id, photo_id),
        state.store.mask_path(project_id, photo_id),
        state.store.flow_path(project_id, photo_id),
    ):
        path.unlink(missing_ok=True)
    state.renderer.invalidate()
    if _project(project_id).photos:
        state.jobs.request(project_id)


def set_reference_clear(p: Project):
    p.crop = None
    p.blanket_polygon = None
    p.blanket_quad = None
    p.blanket_aspect = None
    p.output_width = None


class UpdatePhoto(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    color_enabled: bool | None = None
    color_strength: float | None = Field(default=None, ge=0, le=1)
    approved: bool | None = None
    accept_unresolved: bool | None = None


@app.patch("/api/projects/{project_id}/photos/{photo_id}")
def update_photo(project_id: str, photo_id: str, body: UpdatePhoto):
    _photo(_project(project_id), photo_id)

    def fn(p: Project):
        ph = p.photo(photo_id)
        if "name" in body.model_fields_set:
            # Blank or null clears the label back to the original filename.
            ph.name = (body.name or "").strip() or None
        if body.color_enabled is not None:
            ph.color.enabled = body.color_enabled
        if body.color_strength is not None:
            ph.color.strength = body.color_strength
        if body.approved is not None:
            if body.approved and ph.homography is None:
                raise HTTPException(409, "Align this photo before approving it")
            ph.approved = body.approved
        if body.accept_unresolved is not None:
            ph.accept_unresolved = body.accept_unresolved

    return _project_json(state.store.mutate(project_id, fn))


class PhotoOrder(BaseModel):
    photo_ids: list[str]


@app.put("/api/projects/{project_id}/photos/order")
def reorder_photos(project_id: str, body: PhotoOrder):
    """Set the display and export order. The list must name every photo exactly once."""

    def fn(p: Project):
        by_id = {ph.id: ph for ph in p.photos}
        if len(body.photo_ids) != len(by_id) or set(body.photo_ids) != set(by_id):
            raise HTTPException(409, "The photo list changed; reload and try again")
        p.photos = [by_id[i] for i in body.photo_ids]

    return _project_json(state.store.mutate(_project(project_id).id, fn))


class Landmarks(BaseModel):
    landmarks: list[Landmark]


@app.put("/api/projects/{project_id}/photos/{photo_id}/landmarks")
def set_landmarks(project_id: str, photo_id: str, body: Landmarks):
    project = _project(project_id)
    _photo(project, photo_id)
    if photo_id == project.reference_id:
        raise HTTPException(409, "The reference photo defines the alignment")
    if 0 < len(body.landmarks) < 4:
        raise HTTPException(422, "Place at least 4 landmark pairs")

    def fn(p: Project):
        ph = p.photo(photo_id)
        ph.landmarks = body.landmarks
        ph.status = "pending"
        ph.approved = False

    state.store.mutate(project_id, fn)
    state.jobs.request(project_id)
    return _project_json(_project(project_id))


def _image_response(data: bytes, media: str) -> Response:
    return Response(data, media_type=media, headers=NO_STORE)


@app.get("/api/projects/{project_id}/photos/{photo_id}/thumb.jpg")
def thumb(project_id: str, photo_id: str):
    _photo(_project(project_id), photo_id)
    return FileResponse(state.store.thumb_path(project_id, photo_id), media_type="image/jpeg")


@app.get("/api/projects/{project_id}/photos/{photo_id}/source.jpg")
def source(project_id: str, photo_id: str):
    _photo(_project(project_id), photo_id)
    return FileResponse(state.store.work_path(project_id, photo_id), media_type="image/jpeg")


@app.get("/api/projects/{project_id}/background.jpg")
def background(project_id: str):
    path = state.store.background_path(project_id)
    if not path.exists():
        raise HTTPException(404, "No combined background yet")
    return FileResponse(path, media_type="image/jpeg", headers=NO_STORE)


ViewName = Literal["analysis", "output"]


def _view(project: Project, view: ViewName, width: int | None):
    if project.reference is None:
        raise HTTPException(409, "Choose a reference photo first")
    if view == "analysis":
        return analysis_view(project.reference)
    if not has_frame(project):
        raise HTTPException(409, "Set the crop first")
    return output_view(project, min(width or config.PREVIEW_WIDTH, 2400))


def _render(project_id, photo_id, view, width, blackout, reconstruct, color):
    project = _project(project_id)
    ph = _photo(project, photo_id)
    if ph.homography is None:
        raise HTTPException(409, "This photo is not aligned")
    v = _view(project, view, width)
    if blackout and not project.blanket_polygon:
        blackout = False
    return project, state.renderer.render(project, photo_id, v, blackout=blackout, reconstruct=reconstruct, apply_color=color)


@app.get("/api/projects/{project_id}/photos/{photo_id}/render.jpg")
def render_photo(
    project_id: str,
    photo_id: str,
    view: ViewName = "output",
    width: int | None = Query(None, ge=16),
    blackout: bool = True,
    reconstruct: bool = True,
    color: bool = True,
):
    _, r = _render(project_id, photo_id, view, width, blackout, reconstruct, color)
    return _image_response(encode_jpeg(r.image, 90), "image/jpeg")


@app.get("/api/projects/{project_id}/photos/{photo_id}/provenance.png")
def provenance(project_id: str, photo_id: str, view: ViewName = "output", width: int | None = Query(None, ge=16)):
    _, r = _render(project_id, photo_id, view, width, True, True, True)
    return _image_response(encode_png(provenance_overlay(r)), "image/png")


@app.get("/api/projects/{project_id}/photos/{photo_id}/stats")
def photo_stats(project_id: str, photo_id: str, view: ViewName = "output", width: int | None = Query(None, ge=16)):
    project, r = _render(project_id, photo_id, view, width, True, True, True)
    stats = r.stats
    for i, d in enumerate(r.donors):
        for entry in stats["donors"]:
            if entry["photo_id"] == d:
                donor = project.photo(d)
                entry["name"] = donor.original_name if donor else d
                entry["color"] = "#%02x%02x%02x" % donor_color(i)
    return stats


@app.get("/api/projects/{project_id}/photos/{photo_id}/mask.png")
def get_mask(project_id: str, photo_id: str):
    project = _project(project_id)
    _photo(project, photo_id)
    if project.reference is None:
        raise HTTPException(409, "Choose a reference photo first")
    mask = state.renderer.analysis_mask(project, photo_id)
    if mask is None:
        av = analysis_view(project.reference)
        mask = np.zeros((av.height, av.width), np.uint8)
    return _image_response(encode_png(mask), "image/png")


@app.put("/api/projects/{project_id}/photos/{photo_id}/mask.png")
async def put_mask(project_id: str, photo_id: str, request: Request):
    project = _project(project_id)
    _photo(project, photo_id)
    if project.reference is None:
        raise HTTPException(409, "Choose a reference photo first")
    data = await request.body()
    try:
        mask = decode_gray_bytes(data)
    except InvalidImageError as exc:
        raise HTTPException(422, str(exc))
    av = analysis_view(project.reference)
    if mask.shape != (av.height, av.width):
        raise HTTPException(422, f"Mask must be {av.width}x{av.height}")
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    write_bytes_atomic(state.store.mask_path(project_id, photo_id), encode_png(mask))

    def fn(p: Project):
        ph = p.photo(photo_id)
        ph.mask_edited = True
        ph.approved = False

    state.store.mutate(project_id, fn)
    state.jobs.request(project_id)
    return _project_json(_project(project_id))


@app.delete("/api/projects/{project_id}/photos/{photo_id}/mask.png")
def reset_mask(project_id: str, photo_id: str):
    _photo(_project(project_id), photo_id)

    def fn(p: Project):
        ph = p.photo(photo_id)
        ph.mask_edited = False
        ph.approved = False

    state.store.mutate(project_id, fn)
    state.jobs.request(project_id)
    return _project_json(_project(project_id))


# ---- export --------------------------------------------------------------------------
@app.get("/api/projects/{project_id}/readiness")
def get_readiness(project_id: str):
    return readiness(state.renderer, _project(project_id))


@app.get("/api/projects/{project_id}/export.zip")
def export_zip(project_id: str, background_tasks: BackgroundTasks):
    project = _project(project_id)
    try:
        path = build_zip(state.renderer, project)
    except ExportBlocked as exc:
        return JSONResponse({"detail": "Export is blocked", "readiness": exc.readiness}, status_code=409)
    background_tasks.add_task(path.unlink, missing_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in project.name)[:60] or "milestones"
    return FileResponse(path, media_type="application/zip", filename=f"{safe}.zip")


if config.FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="frontend")
