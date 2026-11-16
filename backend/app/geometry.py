"""Coordinate spaces.

* Source full-res: upright pixels of an original photo.
* Reference full-res: source full-res space of the reference photo; homographies map into it.
  Crop, blanket polygon and blanket quad are stored here.
* Analysis canvas: reference space scaled to the reference working copy, plus a margin on
  every side. Foreground masks are stored here.
* Flow view: the analysis canvas downscaled; wrinkle displacement fields are stored here (deform.py).
* Output: with perspective on, the blanket quad mapped to an upright rectangle of the blanket's
  aspect ratio plus a small margin; otherwise the shared crop. Either way resized to the requested width.
"""

from dataclasses import dataclass

import cv2
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
    if uses_quad(project):
        q = np.array(project.blanket_quad, dtype=np.float64)
        top = np.linalg.norm(q[1] - q[0])
        bottom = np.linalg.norm(q[2] - q[3])
        return int(min(config.MAX_OUTPUT_WIDTH, max(16, round(max(top, bottom) * (1 + 2 * QUAD_MARGIN)))))
    assert project.crop is not None
    return int(min(config.MAX_OUTPUT_WIDTH, max(16, round(project.crop.w))))


# Black border kept around the straightened blanket so slightly bulging edges are not cut off.
QUAD_MARGIN = 0.02


def uses_quad(project: Project) -> bool:
    return bool(project.perspective and project.blanket_quad and len(project.blanket_quad) == 4)


def has_frame(project: Project) -> bool:
    return uses_quad(project) or project.crop is not None


def quad_aspect(project: Project) -> float:
    if project.blanket_aspect:
        return float(project.blanket_aspect)
    ref = project.reference
    size = (ref.width, ref.height) if ref else None
    return estimate_aspect(np.array(project.blanket_quad, dtype=np.float64), size)


def output_view(project: Project, width: int | None = None) -> View:
    width = int(width or project.output_width or default_output_width(project))
    if uses_quad(project):
        aspect = quad_aspect(project)
        inner_w = width / (1 + 2 * QUAD_MARGIN)
        inner_h = inner_w / aspect
        m = QUAD_MARGIN * inner_w
        dst = np.float32([[m, m], [m + inner_w, m], [m + inner_w, m + inner_h], [m, m + inner_h]])
        mat = cv2.getPerspectiveTransform(np.float32(project.blanket_quad), dst).astype(np.float64)
        return View(mat / mat[2, 2], width, max(1, round(inner_h + 2 * m)))
    crop = project.crop
    if crop is None:
        raise ValueError("Crop is not set")
    k = width / crop.w
    mat = np.array([[k, 0, -k * crop.x], [0, k, -k * crop.y], [0, 0, 1]], dtype=np.float64)
    return View(mat, width, max(1, round(crop.h * k)))


def quad_from_polygon(poly) -> list[tuple[float, float]]:
    """Four outermost corners (TL, TR, BR, BL) of a roughly rectangular outline."""
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    s, d = p.sum(axis=1), p[:, 0] - p[:, 1]
    corners = [p[np.argmin(s)], p[np.argmax(d)], p[np.argmax(s)], p[np.argmin(d)]]
    return [(float(x), float(y)) for x, y in corners]


def estimate_aspect(quad: np.ndarray, image_size: tuple[int, int] | None) -> float:
    """Width/height of the real rectangle seen as `quad` (TL, TR, BR, BL).

    Uses the single-view rectangle method of Zhang & He (assumes square pixels and a centered
    principal point); falls back to the mean side ratio when the focal length is unrecoverable
    (near fronto-parallel views).
    """
    q = np.asarray(quad, dtype=np.float64)
    top = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    side = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    fallback = float(top / side) if side > 0 else 1.0
    if image_size is None:
        return fallback
    w, h = image_size
    c = np.array([w / 2, h / 2])
    m1, m2, m4, m3 = (np.append(pt - c, 1.0) for pt in q)  # TL, TR, BR, BL
    den2 = np.dot(np.cross(m2, m4), m3)
    den3 = np.dot(np.cross(m3, m4), m2)
    if abs(den2) < 1e-9 or abs(den3) < 1e-9:
        return fallback
    k2 = np.dot(np.cross(m1, m4), m3) / den2
    k3 = np.dot(np.cross(m1, m4), m2) / den3
    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1
    if abs(n2[2] * n3[2]) < 1e-9:
        return fallback
    f2 = -(n2[0] * n3[0] + n2[1] * n3[1]) / (n2[2] * n3[2])
    if not np.isfinite(f2) or f2 <= 0:
        return fallback
    f = np.sqrt(f2)
    if not 0.3 * max(w, h) <= f <= 6 * max(w, h):
        return fallback
    a_inv = np.diag([1 / f2, 1 / f2, 1.0])
    ratio2 = (n2 @ a_inv @ n2) / (n3 @ a_inv @ n3)
    if not np.isfinite(ratio2) or ratio2 <= 0:
        return fallback
    aspect = float(np.sqrt(ratio2))
    # Guard against wild estimates from noisy corners.
    return aspect if 0.5 * fallback <= aspect <= 2 * fallback else fallback


def homography(photo: Photo) -> np.ndarray | None:
    return None if photo.homography is None else np.array(photo.homography, dtype=np.float64)


def project_points(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    return hom[:, :2] / hom[:, 2:3]


def source_corners(photo: Photo) -> np.ndarray:
    w, h = photo.width, photo.height
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
