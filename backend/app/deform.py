"""Non-rigid wrinkle correction on top of the planar alignment.

After the homography, residual misalignment on a flat-ish blanket comes from wrinkles and folds.
For every aligned photo we estimate a smooth dense displacement field to the reference (on the
"flow view", a downscaled analysis canvas), then subtract the mean field over all photos. Wrinkles
differ from photo to photo, so the mean geometry is flatter than any single photo, and every photo
(including the reference) is warped onto it.

A field D stored for a photo means: the straightened image at flow-view pixel q shows the
homography-warped photo at q + D(q).
"""

from pathlib import Path

import cv2
import numpy as np

from .geometry import View, analysis_view, scale_matrix
from .storage import Photo

FLOW_MAX_DIM = 1000
SMOOTH_FRAC = 0.012  # Gaussian sigma of the field, as a fraction of the flow view width
MAX_SHIFT_FRAC = 0.04
MIN_SUPPORT = 0.05


def flow_view(ref: Photo) -> View:
    v = analysis_view(ref)
    s = min(1.0, FLOW_MAX_DIM / max(v.width, v.height))
    return View(scale_matrix(s) @ v.matrix, max(1, round(v.width * s)), max(1, round(v.height * s)))


def _texture(gray: np.ndarray) -> np.ndarray:
    """Local-contrast normalized image so lighting differences do not drive the flow."""
    g = gray.astype(np.float32)
    mean = cv2.GaussianBlur(g, (0, 0), 6)
    sd = np.sqrt(cv2.GaussianBlur((g - mean) ** 2, (0, 0), 6)) + 4.0
    return np.clip((g - mean) / sd * 40 + 128, 0, 255).astype(np.uint8)


def _smooth(field: np.ndarray, weight: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalized convolution: weighted Gaussian average that fills unsupported areas."""
    w = cv2.GaussianBlur(weight, (0, 0), sigma)
    out = np.empty_like(field)
    for c in range(field.shape[2]):
        out[..., c] = cv2.GaussianBlur(field[..., c] * weight, (0, 0), sigma) / np.maximum(w, 1e-6)
    return out, w


def pairwise_flow(ref_gray: np.ndarray, img_gray: np.ndarray, confidence: np.ndarray) -> np.ndarray | None:
    """Smooth field F with ref(q) ~ img(q + F(q)); zero where there is no support."""
    h, w = ref_gray.shape
    if np.count_nonzero(confidence) < 0.02 * h * w:
        return None
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    a, b = _texture(ref_gray), _texture(img_gray)
    flow = dis.calc(a, b, None)
    # Reject pixels whose warped appearance still disagrees (occlusions, bad matches).
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    back = cv2.remap(b, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = cv2.GaussianBlur(np.abs(a.astype(np.float32) - back.astype(np.float32)), (0, 0), 3)
    weight = (confidence & (err < 30)).astype(np.float32)
    sigma = SMOOTH_FRAC * w
    smooth, support = _smooth(flow, weight, sigma)
    # Fade to zero where nearby evidence is thin, using a wider fill as a bridge.
    wide, wide_support = _smooth(flow, weight, sigma * 4)
    near = np.clip(support / MIN_SUPPORT, 0, 1)[..., None]
    far = np.clip(wide_support / MIN_SUPPORT, 0, 1)[..., None]
    field = near * smooth + (1 - near) * far * wide
    limit = MAX_SHIFT_FRAC * w
    mag = np.linalg.norm(field, axis=2, keepdims=True)
    field *= np.minimum(1.0, limit / np.maximum(mag, 1e-6))
    return field.astype(np.float32)


def group_fields(fields: list[np.ndarray | None]) -> list[np.ndarray | None]:
    """Per-photo displacement onto the mean geometry: D_i = F_i - mean(F)."""
    present = [f for f in fields if f is not None]
    if not present:
        return [None] * len(fields)
    mean = np.mean(present, axis=0)
    return [None if f is None else (f - mean).astype(np.float32) for f in fields]


def save_field(path: Path, field: np.ndarray, homography: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, d=field.astype(np.float16), h=np.asarray(homography, np.float64))
    tmp.replace(path)


def load_field(path: Path, homography: np.ndarray) -> np.ndarray | None:
    """The stored field, or None when missing or computed for a different alignment."""
    try:
        with np.load(path) as data:
            if not np.allclose(data["h"], homography, atol=1e-6):
                return None
            return data["d"].astype(np.float32)
    except (OSError, KeyError, ValueError):
        return None


def _apply(m: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    den = m[2, 0] * xs + m[2, 1] * ys + m[2, 2]
    return (
        ((m[0, 0] * xs + m[0, 1] * ys + m[0, 2]) / den).astype(np.float32),
        ((m[1, 0] * xs + m[1, 1] * ys + m[1, 2]) / den).astype(np.float32),
    )


def warp_deformed(
    img: np.ndarray,
    src_to_ref: np.ndarray,
    src_scale: float,
    view: View,
    fview: View,
    field: np.ndarray,
    strip: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a source image into `view` through the homography plus the flow-view field.

    Returns (image, valid) like a plain perspective warp. Processes row strips to bound memory.
    """
    out = np.zeros((view.height, view.width) + img.shape[2:], img.dtype)
    valid = np.zeros((view.height, view.width), np.uint8)
    ones = np.full(img.shape[:2], 255, np.uint8)
    to_flow = fview.matrix @ np.linalg.inv(view.matrix)
    flow_to_src = scale_matrix(src_scale) @ np.linalg.inv(src_to_ref) @ np.linalg.inv(fview.matrix)
    xs_row = np.arange(view.width, dtype=np.float64)
    for y0 in range(0, view.height, strip):
        y1 = min(view.height, y0 + strip)
        xs, ys = np.meshgrid(xs_row, np.arange(y0, y1, dtype=np.float64))
        fx, fy = _apply(to_flow, xs, ys)
        d = cv2.remap(field, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sx, sy = _apply(flow_to_src, (fx + d[..., 0]).astype(np.float64), (fy + d[..., 1]).astype(np.float64))
        out[y0:y1] = cv2.remap(img, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid[y0:y1] = cv2.remap(ones, sx, sy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, valid
