"""Automatic proposals for the blanket boundary and per-photo foreground (baby, markers, props)."""

import os
from concurrent.futures import ThreadPoolExecutor

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
SHIFT_RADIUS = 0.008  # residual misalignment tolerated when comparing photos, as a fraction of width
DENSE_SHIFT_RADIUS = 0.0015  # same, once wrinkle correction has aligned photos densely
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
    shift_radius: float = SHIFT_RADIUS,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Flag pixels where an aligned photo is not part of the dominant agreeing group.

    Two photos agree at a pixel when colors are close and, unless both are locally smooth, the
    blanket print correlates. Babies rarely match each other in both color and structure, so even
    when most photos have a baby over the same spot, the photos showing bare blanket form the
    dominant group. Ties between groups flag everyone (conservative). Pixels covered by fewer
    than two photos carry no evidence and stay unflagged; users review masks before export.

    With `shading_tolerant`, brightness changes that keep chromaticity (crease shadows) are not
    treated as objects.

    Photos are compared with up to `shift_radius` (fraction of width) of residual misalignment,
    so textured blanket that is a few pixels off still agrees.

    Returns one boolean mask per image and the combined background (RGB).
    """
    h, w = images[0].shape[:2]
    # Only pixels near the blanket can be flagged; skip the rest of the frame.
    dil = max(3, round(0.02 * w))
    zone = cv2.dilate(blanket.astype(np.uint8), np.ones((dil * 2 + 1, dil * 2 + 1), np.uint8)) > 0
    if not zone.any():
        zone = np.ones((h, w), bool)
    ys, xs = np.nonzero(zone)
    radius = max(1, round(shift_radius * w))
    pad = dil + WINDOW + SUPPORT_WINDOW + radius
    crop = (slice(max(0, ys.min() - pad), min(h, ys.max() + pad + 1)), slice(max(0, xs.min() - pad), min(w, xs.max() + pad + 1)))
    masks_c, bg_c = _detect(
        [im[crop] for im in images], [v[crop] for v in valids], zone[crop], shading_tolerant, radius, h * w, dil
    )
    bg_sum = np.zeros((h, w, 3), np.float64)
    bg_n = np.zeros((h, w), np.float64)
    for im, v in zip(images, valids):
        bg_sum[v] += im[v]
        bg_n[v] += 1
    background = (bg_sum / np.maximum(bg_n, 1)[:, :, None]).astype(np.uint8)
    background[crop] = bg_c
    masks = []
    for m in masks_c:
        full = np.zeros((h, w), bool)
        full[crop] = m
        masks.append(full)
    return masks, background


def _detect(images, valids, zone, shading_tolerant, radius, full_area, dil):
    n = len(images)
    h, w = images[0].shape[:2]
    labs = [cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_RGB2LAB), (5, 5), 0).astype(np.float32) for im in images]
    planes = [[np.ascontiguousarray(lab[:, :, k]) for k in range(3)] for lab in labs]
    valid = np.stack(valids)
    count = valid.sum(axis=0)
    win = (WINDOW, WINDOW)
    shades: list[tuple[np.ndarray, np.ndarray]] = []
    if shading_tolerant:
        for im in images:
            rgb = cv2.GaussianBlur(im, (5, 5), 0).astype(np.float32) + 1.0
            total = rgb.sum(axis=2)
            chroma = rgb / total[..., None]
            shades.append(([np.ascontiguousarray(chroma[:, :, k]) for k in range(3)], total))
    stats = []
    for lab in labs:
        g = np.ascontiguousarray(lab[:, :, 0])
        mu = cv2.boxFilter(g, -1, win)
        var = np.maximum(cv2.boxFilter(g * g, -1, win) - mu * mu, 0)
        stats.append((g, mu, np.sqrt(var + 1e-3) * np.sqrt(NCC_THRESHOLD), var < SMOOTH_VAR))

    # Real blankets are never perfectly flat, so a homography leaves the print a few pixels off between
    # photos. Pixels agree when they match anywhere within a small neighbourhood of each other.
    step = max(1, radius // 2)
    offsets = [(dy, dx) for dy in range(-radius, radius + 1, step) for dx in range(-radius, radius + 1, step)]
    max_dist2 = DIFF_THRESHOLD**2
    max_chroma2 = CHROMA_THRESHOLD**2

    def sq_dist(pa, pb, si, sj):
        out = None
        for a, b in zip(pa, pb):
            d = a[si] - b[sj]
            np.multiply(d, d, out=d)
            out = d if out is None else np.add(out, d, out=out)
        return out

    def compare(i: int, j: int):
        gi, mui, sdi, smi = stats[i]
        gj, muj, sdj, smj = stats[j]
        agree_i = np.zeros((h, w), bool)
        agree_j = np.zeros((h, w), bool)
        for dy, dx in offsets:
            # Compare pixel p of photo i with pixel p + (dy, dx) of photo j, over the overlapping slices.
            si = (slice(max(0, -dy), h - max(0, dy)), slice(max(0, -dx), w - max(0, dx)))
            sj = (slice(max(0, dy), h - max(0, -dy)), slice(max(0, dx), w - max(0, -dx)))
            both = valid[i][si] & valid[j][sj]
            if not both.any():
                continue
            cov = cv2.boxFilter(gi[si] * gj[sj], -1, win)
            cov -= mui[si] * muj[sj]
            # ncc > NCC_THRESHOLD, with sd already scaled by sqrt(NCC_THRESHOLD).
            structure = cov > sdi[si] * sdj[sj]
            if dy == 0 and dx == 0:
                # Smooth areas aren't sensitive to misalignment; shifting them would let plain objects
                # match any nearby plain blanket.
                structure |= smi[si] & smj[sj]
            agree = both & (sq_dist(planes[i], planes[j], si, sj) < max_dist2) & structure
            if shading_tolerant and dy == 0 and dx == 0:
                # Shadows scale brightness but keep chromaticity, however sharp the crease is.
                (ci, ti), (cj, tj) = shades[i], shades[j]
                ratio = ti[si] / tj[sj]
                shaded = (sq_dist(ci, cj, si, sj) < max_chroma2) & (ratio < MAX_SHADE_RATIO) & (ratio > 1 / MAX_SHADE_RATIO)
                agree |= both & shaded
            agree_i[si] |= agree
            agree_j[sj] |= agree
        return i, j, agree_i, agree_j

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if (valid[i] & valid[j]).any()]
    support = np.zeros((n, h, w), np.int16)
    # NumPy and OpenCV release the GIL, so photo pairs run in parallel.
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        for i, j, agree_i, agree_j in pool.map(lambda ij: compare(*ij), pairs):
            support[i] += agree_i
            support[j] += agree_j

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
    min_area = MIN_COMPONENT_FRACTION * full_area
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
