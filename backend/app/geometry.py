"""Coordinate spaces.

* Source full-res: upright pixels of an original photo.
* Reference full-res: source full-res space of the reference photo; homographies map into it.
  Crop and blanket polygon are stored here.
* Analysis canvas: reference space scaled to the reference working copy, plus a margin on
  every side. Foreground masks are stored here.
* Output: the shared crop resized to the requested width.
"""

from dataclasses import dataclass

import numpy as np

from . import config
from .storage import Photo, Project


@dataclass(frozen=True)
class View:
    matrix: np.ndarray  # reference full-res -> view pixels
    width: int
    height: int


def scale_matrix(s: float) -> np.ndarray:
    return np.diag([s, s, 1.0])


def analysis_view(ref: Photo) -> View:
    s = ref.work_scale
    m = config.ANALYSIS_MARGIN
    ox, oy = ref.width * s * m, ref.height * s * m
    mat = np.array([[s, 0, ox], [0, s, oy], [0, 0, 1]], dtype=np.float64)
    return View(mat, round(ref.width * s * (1 + 2 * m)), round(ref.height * s * (1 + 2 * m)))


def default_output_width(project: Project) -> int:
    assert project.crop is not None
    return int(min(config.MAX_OUTPUT_WIDTH, max(16, round(project.crop.w))))


def output_view(project: Project, width: int | None = None) -> View:
    crop = project.crop
    if crop is None:
        raise ValueError("Crop is not set")
    width = int(width or project.output_width or default_output_width(project))
    k = width / crop.w
    mat = np.array([[k, 0, -k * crop.x], [0, k, -k * crop.y], [0, 0, 1]], dtype=np.float64)
    return View(mat, width, max(1, round(crop.h * k)))


def homography(photo: Photo) -> np.ndarray | None:
    return None if photo.homography is None else np.array(photo.homography, dtype=np.float64)


def project_points(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    return hom[:, :2] / hom[:, 2:3]


def source_corners(photo: Photo) -> np.ndarray:
    w, h = photo.width, photo.height
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
