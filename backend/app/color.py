"""Conservative white-balance/exposure matching in linear light, applied through per-channel LUTs."""

import cv2
import numpy as np

from .storage import ColorSettings

_X = np.arange(256, dtype=np.float64) / 255.0
SRGB_TO_LINEAR = np.where(_X <= 0.04045, _X / 12.92, ((_X + 0.055) / 1.055) ** 2.4)

GAIN_MIN, GAIN_MAX = 0.4, 2.5
MIN_SAMPLES = 500


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def effective_gains(color: ColorSettings) -> np.ndarray:
    if not color.enabled or color.gains is None:
        return np.ones(3)
    return np.power(np.array(color.gains, dtype=np.float64), color.strength)


def gain_lut(gains) -> np.ndarray:
    gains = np.asarray(gains, dtype=np.float64)
    lut = np.empty((256, 1, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, 0, c] = np.round(linear_to_srgb(SRGB_TO_LINEAR * gains[c]) * 255.0).astype(np.uint8)
    return lut


def apply_gains(img: np.ndarray, gains) -> np.ndarray:
    gains = np.asarray(gains, dtype=np.float64)
    if np.allclose(gains, 1.0):
        return img
    return cv2.LUT(img, gain_lut(gains))


def usable_pixels(img: np.ndarray) -> np.ndarray:
    """Pixels that are neither clipped highlights nor deep shadows."""
    return (img.max(axis=2) < 245) & (img.min(axis=2) > 12)


def estimate_gains(
    target: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    lo: float = GAIN_MIN,
    hi: float = GAIN_MAX,
    min_samples: int = MIN_SAMPLES,
) -> np.ndarray | None:
    """Per-channel gains that map `target` blanket colors to `reference` (both already aligned)."""
    t = cv2.GaussianBlur(target, (7, 7), 0)
    r = cv2.GaussianBlur(reference, (7, 7), 0)
    sel = mask & usable_pixels(t) & usable_pixels(r)
    if np.count_nonzero(sel) < min_samples:
        return None
    t_lin = SRGB_TO_LINEAR[t[sel]]
    r_lin = SRGB_TO_LINEAR[r[sel]]
    gains = np.median(r_lin / np.maximum(t_lin, 1e-4), axis=0)
    return np.clip(gains, lo, hi)
