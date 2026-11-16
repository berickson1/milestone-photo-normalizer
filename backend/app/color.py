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


FIELD_WIDTH = 200  # working width for smooth gain fields
FIELD_MAX_LOG = float(np.log(GAIN_MAX))


def fill_sparse(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Push-pull interpolation: a smooth dense field through sparse weighted samples."""
    pyramid = [(values * weights[..., None], weights)]
    while max(pyramid[-1][1].shape) > 1:
        v, w = pyramid[-1]
        size = (max(1, w.shape[1] // 2), max(1, w.shape[0] // 2))
        pyramid.append((cv2.resize(v, size, interpolation=cv2.INTER_AREA).reshape(size[1], size[0], -1), cv2.resize(w, size, interpolation=cv2.INTER_AREA)))
    v, w = pyramid[-1]
    est = v / np.maximum(w, 1e-6)[..., None]
    for v, w in reversed(pyramid[:-1]):
        up = cv2.resize(est, (w.shape[1], w.shape[0]), interpolation=cv2.INTER_LINEAR).reshape(v.shape)
        a = np.clip(w * 4.0, 0.0, 1.0)[..., None]
        est = a * (v / np.maximum(w, 1e-6)[..., None]) + (1 - a) * up
    return est


def _resample(field: np.ndarray, shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    """Bilinearly resample a coarse field covering `shape` at the pixels of `box` (y0, y1, x0, x1)."""
    h, w = shape
    y0, y1, x0, x1 = box
    fh, fw = field.shape[:2]
    gx = ((np.arange(x0, x1, dtype=np.float32) + 0.5) * fw / w - 0.5)[None, :].repeat(y1 - y0, 0)
    gy = ((np.arange(y0, y1, dtype=np.float32) + 0.5) * fh / h - 0.5)[:, None].repeat(x1 - x0, 1)
    out = cv2.remap(field.astype(np.float32), gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(y1 - y0, x1 - x0, -1)


def _masked_area(img: np.ndarray, mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img * mask[..., None], size, interpolation=cv2.INTER_AREA).reshape(size[1], size[0], -1)


def seam_transfer(
    source: np.ndarray, source_valid: np.ndarray, target: np.ndarray, samples: np.ndarray, region: np.ndarray
) -> np.ndarray | None:
    """Recolor `source` within `region` so it continues `target` seamlessly.

    Lighting and haze vary across photos, so one global gain leaves visible steps. Along the
    seam (`samples`, where both show the same blanket), a local linear model is fitted per
    channel in linear light: a gain matching local means and a contrast factor matching local
    relative spread (restoring washed-out print). Both are extended smoothly over the region.
    Returns the recolored copy, or None without enough samples.
    """
    h, w = samples.shape
    pre = (min(w, 4 * FIELD_WIDTH), max(1, round(h * min(w, 4 * FIELD_WIDTH) / w)))
    lin = SRGB_TO_LINEAR.astype(np.float32)
    sw = min(pre[0], 2 * FIELD_WIDTH)
    size = (sw, max(4, round(pre[1] * sw / pre[0])))
    src = cv2.resize(source, pre, interpolation=cv2.INTER_AREA) if pre[0] != w else source
    tgt = cv2.resize(target, pre, interpolation=cv2.INTER_AREA) if pre[0] != w else target
    smp = cv2.resize(samples.astype(np.uint8), pre, interpolation=cv2.INTER_NEAREST) > 0
    sel = (smp & usable_pixels(src) & usable_pixels(tgt)).astype(np.float32)
    if np.count_nonzero(sel) < 50:
        return None
    s_lin, t_lin = lin[src], lin[tgt]
    sigma = max(1.0, 0.02 * size[0])
    blur = lambda a: cv2.GaussianBlur(a, (0, 0), sigma).reshape(a.shape)
    wt = cv2.resize(sel, size, interpolation=cv2.INTER_AREA)
    wb = np.maximum(blur(wt), 1e-6)[..., None]
    ms = blur(_masked_area(s_lin, sel, size)) / wb
    mt = blur(_masked_area(t_lin, sel, size)) / wb
    vs = np.maximum(blur(_masked_area(s_lin * s_lin, sel, size)) / wb - ms * ms, 0)
    vt = np.maximum(blur(_masked_area(t_lin * t_lin, sel, size)) / wb - mt * mt, 0)
    cov = blur(_masked_area(s_lin * t_lin, sel, size)) / wb - ms * mt
    ok = (wb[..., 0] > 0.2) & (ms.min(axis=2) > 1e-3) & (mt.min(axis=2) > 1e-3)
    if np.count_nonzero(ok) < 5:
        return None
    # Local affine model per channel: t ≈ mt + alpha * (s - ms). Without trustworthy texture the
    # slope is the plain gain mt/ms; where the print correlates it also matches relative spread,
    # so washed-out donor colors regain saturation.
    corr = cov / np.sqrt(vs * vt + 1e-12)
    trust = np.clip((corr - 0.3) / 0.4, 0, 1)
    gain = np.exp(np.clip(np.log(np.maximum(mt, 1e-6) / np.maximum(ms, 1e-6)), -FIELD_MAX_LOG, FIELD_MAX_LOG))
    rel = np.sqrt((vt + 1e-8) / (vs + 1e-8)) / gain
    alpha = gain * np.clip(1 + trust * (rel - 1), 0.6, 2.0)
    textured = ok[..., None] & (vs > 2e-4)
    weights_m = np.where(ok, wt, 0).astype(np.float32)
    weights_a = weights_m * np.where(textured, trust, 0).mean(axis=2)
    ms = fill_sparse(np.where(ok[..., None], ms, 0), weights_m)
    mt = fill_sparse(np.where(ok[..., None], mt, 0), weights_m)
    alpha = fill_sparse(np.where(ok[..., None], alpha, 1), weights_m * 0.05 + weights_a)
    smooth = max(1.0, 0.015 * size[0])
    sm = lambda a: cv2.GaussianBlur(a, (0, 0), smooth).reshape(a.shape)
    ms, mt, alpha = sm(ms), sm(mt), sm(alpha)

    out = source.copy()
    if not region.any():
        return out
    ys, xs = np.nonzero(region)
    box = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    sub = region[box[0] : box[1], box[2] : box[3]]
    a = _resample(alpha, (h, w), box)[sub]
    m_s = _resample(ms, (h, w), box)[sub]
    m_t = _resample(mt, (h, w), box)[sub]
    d = lin[source[box[0] : box[1], box[2] : box[3]][sub]]
    patch = out[box[0] : box[1], box[2] : box[3]]
    patch[sub] = np.round(linear_to_srgb(np.clip(m_t + a * (d - m_s), 0, 1)) * 255.0).astype(np.uint8)
    return out


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
