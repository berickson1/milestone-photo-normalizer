"""Automatic proposals for the blanket boundary and per-photo foreground (baby, markers, props)."""

import cv2
import numpy as np

from .geometry import View, scale_matrix

MIN_COMPONENT_FRACTION = 0.002
DIFF_THRESHOLD = 28.0
NCC_THRESHOLD = 0.5
SMOOTH_VAR = 25.0
WINDOW = 11
SUPPORT_FRACTION = 0.6
SUPPORT_WINDOW = 9
# Shading-tolerant comparison (crease shadows): chromaticity within a plausible brightness ratio.
CHROMA_THRESHOLD = 0.02
MAX_SHADE_RATIO = 2.5


def propose_blanket(ref_rgb: np.ndarray, ref_scale: float) -> list[tuple[float, float]]:
    """Polygon (reference full-res coordinates) for the blanket in the reference working image."""
    small_scale = min(1.0, 600 / max(ref_rgb.shape[:2]))
    small = cv2.resize(ref_rgb, None, fx=small_scale, fy=small_scale, interpolation=cv2.INTER_AREA) if small_scale < 1 else ref_rgb
    h, w = small.shape[:2]
    inset = 0.04
    fallback = [(w * inset, h * inset), (w * (1 - inset), h * inset), (w * (1 - inset), h * (1 - inset)), (w * inset, h * (1 - inset))]
    poly = fallback
    try:
        mask = np.zeros((h, w), np.uint8)
        rect = (round(w * inset), round(h * inset), round(w * (1 - 2 * inset)), round(h * (1 - 2 * inset)))
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 0.2 * w * h:
                hull = cv2.convexHull(c)
                approx = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True).reshape(-1, 2)
                if len(approx) >= 3:
                    poly = [tuple(map(float, p)) for p in approx]
    except cv2.error:
        pass
    k = 1.0 / (small_scale * ref_scale)
    return [(x * k, y * k) for x, y in poly]


def detect_foreground(
    images: list[np.ndarray],
    valids: list[np.ndarray],
    blanket: np.ndarray,
    shading_tolerant: bool = False,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Flag pixels where an aligned photo is not part of the dominant agreeing group.

    Two photos agree at a pixel when colors are close and, unless both are locally smooth, the
    blanket print correlates. Babies rarely match each other in both color and structure, so even
    when most photos have a baby over the same spot, the photos showing bare blanket form the
    dominant group. Ties between groups flag everyone (conservative). Pixels covered by fewer
    than two photos carry no evidence and stay unflagged; users review masks before export.

    With `shading_tolerant`, brightness changes that keep chromaticity (crease shadows) are not
    treated as objects.

    Returns one boolean mask per image and the combined background (RGB).
    """
    n = len(images)
    h, w = images[0].shape[:2]
    labs = [cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_RGB2LAB), (5, 5), 0).astype(np.float32) for im in images]
    valid = np.stack(valids)
    count = valid.sum(axis=0)
    win = (WINDOW, WINDOW)
    shades: list[tuple[np.ndarray, np.ndarray]] = []
    if shading_tolerant:
        for im in images:
            rgb = cv2.GaussianBlur(im, (5, 5), 0).astype(np.float32) + 1.0
            total = rgb.sum(axis=2)
            shades.append((rgb / total[..., None], total))
    stats = []
    for lab in labs:
        g = lab[:, :, 0]
        mu = cv2.boxFilter(g, -1, win)
        var = np.maximum(cv2.boxFilter(g * g, -1, win) - mu * mu, 0)
        stats.append((g, mu, var))

    support = np.zeros((n, h, w), np.int16)
    for i in range(n):
        gi, mui, vari = stats[i]
        for j in range(i + 1, n):
            both = valid[i] & valid[j]
            if not both.any():
                continue
            gj, muj, varj = stats[j]
            close = np.linalg.norm(labs[i] - labs[j], axis=2) < DIFF_THRESHOLD
            cov = cv2.boxFilter(gi * gj, -1, win) - mui * muj
            ncc = cov / np.sqrt(vari * varj + 1e-3)
            smooth = (vari < SMOOTH_VAR) & (varj < SMOOTH_VAR)
            agree = both & close & ((ncc > NCC_THRESHOLD) | smooth)
            if shading_tolerant:
                # Shadows scale brightness but keep chromaticity, however sharp the crease is.
                (ci, ti), (cj, tj) = shades[i], shades[j]
                ratio = ti / tj
                shaded = (np.linalg.norm(ci - cj, axis=2) < CHROMA_THRESHOLD) & (ratio < MAX_SHADE_RATIO) & (ratio > 1 / MAX_SHADE_RATIO)
                agree |= both & shaded
            support[i] += agree
            support[j] += agree

    support[~valid] = -1
    # Pairwise tests are noisy per pixel; average support locally before comparing photos.
    support = np.stack([np.where(valid[i], cv2.blur(np.maximum(support[i], 0).astype(np.float32), (SUPPORT_WINDOW, SUPPORT_WINDOW)), -1.0) for i in range(n)])
    best = support.max(axis=0)
    members = (support >= SUPPORT_FRACTION * best).sum(axis=0)
    # One agreeing group of size k gives its members support up to k-1; far more members means rival groups.
    tie = members > 1.5 * (best + 1)
    evidence = (count >= 2)[None] & valid
    # Real photos agree noisily (lighting, residual wrinkles), so members of the dominant group only
    # need a clear share of the best support; objects have near-zero support.
    raw_fg = evidence & ((support < SUPPORT_FRACTION * best[None]) | (best[None] <= 0) | tie[None])

    bg_sum = np.zeros((h, w, 3), np.float64)
    bg_n = np.zeros((h, w), np.float64)
    for i in range(n):
        sel = valid[i] & ~raw_fg[i]
        bg_sum[sel] += images[i][sel]
        bg_n[sel] += 1
    background = (bg_sum / np.maximum(bg_n, 1)[:, :, None]).astype(np.uint8)

    masks = []
    min_area = MIN_COMPONENT_FRACTION * h * w
    dil = max(3, round(0.02 * w))
    zone = cv2.dilate(blanket.astype(np.uint8), np.ones((dil * 2 + 1, dil * 2 + 1), np.uint8)) > 0
    for i in range(n):
        m = (raw_fg[i] & zone).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros_like(m)
        for j in range(1, num):
            if stats[j, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == j] = 1
        # Fill holes so interiors of flagged objects are excluded too.
        contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(keep)
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
        filled = cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil * 2 + 1, dil * 2 + 1)))
        masks.append(filled > 0)
    return masks, background


def downscaled(view: View, max_dim: int) -> View:
    s = min(1.0, max_dim / max(view.width, view.height))
    return View(scale_matrix(s) @ view.matrix, max(1, round(view.width * s)), max(1, round(view.height * s)))
