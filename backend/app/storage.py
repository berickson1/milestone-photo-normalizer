import hashlib
import json
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from . import config

Quality = Literal["ok", "review", "failed"]
PhotoStatus = Literal["pending", "queued", "aligning", "aligned", "error"]

_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class Rect(BaseModel):
    x: float
    y: float
    w: float = Field(gt=0)
    h: float = Field(gt=0)


class Landmark(BaseModel):
    ref: tuple[float, float]
    src: tuple[float, float]


class Alignment(BaseModel):
    method: Literal["auto", "manual", "reference"] = "auto"
    quality: Quality = "failed"
    inliers: int = 0
    inlier_ratio: float = 0.0
    coverage: float = 0.0
    rms_error: float = 0.0
    ambiguous: bool = False
    reasons: list[str] = Field(default_factory=list)


class ColorSettings(BaseModel):
    enabled: bool = True
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    gains: tuple[float, float, float] | None = None


class Photo(BaseModel):
    id: str
    original_name: str
    stored_name: str
    sha256: str
    width: int
    height: int
    work_scale: float
    created_at: float
    status: PhotoStatus = "pending"
    error: str | None = None
    homography: list[list[float]] | None = None
    alignment: Alignment | None = None
    landmarks: list[Landmark] = Field(default_factory=list)
    color: ColorSettings = Field(default_factory=ColorSettings)
    mask_edited: bool = False
    approved: bool = False
    accept_unresolved: bool = False


class Job(BaseModel):
    state: Literal["idle", "queued", "running"] = "idle"
    done: int = 0
    total: int = 0
    message: str = ""
    error: str | None = None


class Project(BaseModel):
    id: str
    name: str
    created_at: float
    revision: int = 0
    reference_id: str | None = None
    crop: Rect | None = None
    output_width: int | None = None
    blanket_polygon: list[tuple[float, float]] | None = None
    photos: list[Photo] = Field(default_factory=list)
    job: Job = Field(default_factory=Job)

    def photo(self, photo_id: str) -> Photo | None:
        return next((p for p in self.photos if p.id == photo_id), None)

    @property
    def reference(self) -> Photo | None:
        return self.photo(self.reference_id) if self.reference_id else None


class NotFoundError(KeyError):
    pass


def new_id() -> str:
    return uuid.uuid4().hex


def valid_id(value: str) -> bool:
    return bool(_ID_RE.match(value))


class ProjectStore:
    """Filesystem-backed project storage. All mutations go through `mutate` under a per-project lock."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root or config.DATA_DIR) / "projects"
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def lock(self, project_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(project_id, threading.RLock())

    def project_dir(self, project_id: str) -> Path:
        if not valid_id(project_id):
            raise NotFoundError(project_id)
        return self.root / project_id

    def original_path(self, project_id: str, photo: Photo) -> Path:
        return self.project_dir(project_id) / "originals" / photo.stored_name

    def work_path(self, project_id: str, photo_id: str) -> Path:
        return self.project_dir(project_id) / "work" / f"{photo_id}.jpg"

    def thumb_path(self, project_id: str, photo_id: str) -> Path:
        return self.project_dir(project_id) / "thumbs" / f"{photo_id}.jpg"

    def mask_path(self, project_id: str, photo_id: str) -> Path:
        return self.project_dir(project_id) / "masks" / f"{photo_id}.png"

    def background_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "background.jpg"

    def list(self) -> list[Project]:
        projects = []
        for path in sorted(self.root.glob("*/project.json")):
            try:
                projects.append(self.get(path.parent.name))
            except (NotFoundError, ValueError):
                continue
        return sorted(projects, key=lambda p: p.created_at, reverse=True)

    def get(self, project_id: str) -> Project:
        path = self.project_dir(project_id) / "project.json"
        if not path.exists():
            raise NotFoundError(project_id)
        with self.lock(project_id):
            return Project.model_validate_json(path.read_text(encoding="utf-8"))

    def _write(self, project: Project) -> None:
        path = self.project_dir(project.id) / "project.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(project.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

    def create(self, name: str) -> Project:
        project = Project(id=new_id(), name=name.strip() or "Untitled", created_at=time.time())
        d = self.project_dir(project.id)
        for sub in ("originals", "work", "thumbs", "masks"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        self._write(project)
        return project

    def delete(self, project_id: str) -> None:
        d = self.project_dir(project_id)
        if not d.exists():
            raise NotFoundError(project_id)
        with self.lock(project_id):
            shutil.rmtree(d)

    def mutate(self, project_id: str, fn: Callable[[Project], None], bump: bool = True) -> Project:
        with self.lock(project_id):
            project = self.get(project_id)
            fn(project)
            if bump:
                project.revision += 1
            self._write(project)
            return project


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(model: BaseModel) -> dict:
    return json.loads(model.model_dump_json())
