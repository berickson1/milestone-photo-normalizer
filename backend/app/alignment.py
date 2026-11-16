"""Planar alignment of a photo to the reference using blanket features."""

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import project_points, scale_matrix
from .storage import Alignment, Landmark

MIN_INLIERS = 12
GOOD_INLIERS = 40
GOOD_INLIER_RATIO = 0.2
GOOD_COVERAGE = 0.3
MAX_RMS = 3.0
# RANSAC tolerance for wrinkled blankets, as a fraction of the working image diagonal.
WRINKLE_TOLERANCE = 0.025
GRID = 6


@dataclass
class AlignmentResult:
    homography: np.ndarray | None  # source full-res -> reference full-res
    alignment: Alignment


def _sift() -> cv2.SIFT:
    return cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)


def _features(gray: np.ndarray, mask: np.ndarray | None):
    if mask is not None:
        mask = np.where(mask > 0, 0, 255).astype(np.uint8)  # mask marks excluded pixels
    return _sift().detectAndCompute(gray, mask)


def _geometry_ok(h_work: np.ndarray, src_shape, ref_shape) -> tuple[bool, str | None]:
    sh, sw = src_shape[:2]
    rh, rw = ref_shape[:2]
    corners = project_points(h_work, [[0, 0], [sw, 0], [sw, sh], [0, sh]]).astype(np.float32)
    if not np.all(np.isfinite(corners)):
        return False, "Transform is degenerate"
    if not cv2.isContourConvex(corners.reshape(-1, 1, 2)):
        return False, "Transform folds the image"
    ratio = abs(cv2.contourArea(corners)) / float(rw * rh)
    if not 0.05 <= ratio <= 20:
        return False, "Transform changes scale implausibly"
    return True, None


def _coverage(ref_pts: np.ndarray, region: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = region
    if len(ref_pts) == 0 or x1 <= x0 or y1 <= y0:
        return 0.0
    gx = np.clip(((ref_pts[:, 0] - x0) / (x1 - x0) * GRID).astype(int), -1, GRID)
    gy = np.clip(((ref_pts[:, 1] - y0) / (y1 - y0) * GRID).astype(int), -1, GRID)
    inside = (gx >= 0) & (gx < GRID) & (gy >= 0) & (gy < GRID)
    cells, counts = np.unique(gy[inside] * GRID + gx[inside], return_counts=True)
    return float(np.count_nonzero(counts >= 2)) / (GRID * GRID)


def _work_to_full(h_work: np.ndarray, src_scale: float, ref_scale: float) -> np.ndarray:
    h = np.linalg.inv(scale_matrix(ref_scale)) @ h_work @ scale_matrix(src_scale)
    return h / h[2, 2]


def align_auto(
    ref_gray: np.ndarray,
    src_gray: np.ndarray,
    ref_scale: float,
    src_scale: float,
    ref_exclude: np.ndarray | None = None,
    src_exclude: np.ndarray | None = None,
    region: tuple[float, float, float, float] | None = None,
    wrinkle_correction: bool = False,
) -> AlignmentResult:
    """Estimate a homography between working copies; masks mark pixels to ignore (e.g. the baby).

    `region` is the reference-work-space rectangle used for spatial coverage (normally the blanket).
    With `wrinkle_correction`, residual error the wrinkle step will remove is not flagged.
    """
    max_rms = WRINKLE_TOLERANCE * np.hypot(*ref_gray.shape[:2]) / 2 if wrinkle_correction else MAX_RMS

    def failed(reason: str, **kw) -> AlignmentResult:
        return AlignmentResult(None, Alignment(quality="failed", reasons=[reason], **kw))

    kr, dr = _features(ref_gray, ref_exclude)
    ks, ds = _features(src_gray, src_exclude)
    if dr is None or ds is None or len(kr) < MIN_INLIERS or len(ks) < MIN_INLIERS:
        return failed("Not enough blanket detail was found")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(ds, dr, k=2)
    backward = {m.queryIdx: m.trainIdx for m in matcher.match(dr, ds)}
    pairs = [
        (m[0].queryIdx, m[0].trainIdx)
        for m in forward
        if len(m) == 2 and m[0].distance < 0.8 * m[1].distance and backward.get(m[0].trainIdx) == m[0].queryIdx
    ]
    if len(pairs) < MIN_INLIERS:
        return failed("Too few matching blanket features", inliers=len(pairs))

    src_pts = np.float32([ks[s].pt for s, _ in pairs])
    ref_pts = np.float32([kr[r].pt for _, r in pairs])
    rh, rw = ref_gray.shape[:2]
    region = region or (0, 0, rw, rh)
    h, mask = cv2.findHomography(src_pts, ref_pts, cv2.USAC_MAGSAC, 4.0, maxIters=10000, confidence=0.9995)
    threshold = 4.0
    if h is not None and mask is not None:
        inl0 = mask.ravel().astype(bool)
        weak = inl0.sum() < GOOD_INLIER_RATIO * len(pairs) or _coverage(ref_pts[inl0], region) < GOOD_COVERAGE
    else:
        weak = True
    if weak or wrinkle_correction:
        # Wrinkled blankets displace features by more than a flat-plane tolerance allows.
        loose = WRINKLE_TOLERANCE * np.hypot(rw, rh)
        h2, mask2 = cv2.findHomography(src_pts, ref_pts, cv2.USAC_MAGSAC, loose, maxIters=10000, confidence=0.9995)
        if h2 is not None and mask2 is not None and (mask is None or mask2.sum() > 1.3 * mask.sum()):
            h, mask, threshold = h2, mask2, loose
    if h is None or mask is None:
        return failed("No consistent perspective transform was found")
    inl = mask.ravel().astype(bool)
    n_inl = int(inl.sum())
    if n_inl < MIN_INLIERS:
        return failed("Too few consistent matches", inliers=n_inl)

    refined, _ = cv2.findHomography(src_pts[inl], ref_pts[inl], 0)
    if refined is not None:
        h = refined
    ok, why = _geometry_ok(h, src_gray.shape, ref_gray.shape)
    if not ok:
        return failed(why or "Implausible transform", inliers=n_inl)

    err = np.linalg.norm(project_points(h, src_pts[inl]) - ref_pts[inl], axis=1)
    rms = float(np.sqrt(np.mean(err**2)))
    coverage = _coverage(ref_pts[inl], region)
    ratio = n_inl / len(pairs)

    # Repeating prints can produce a second, different plausible transform.
    ambiguous = False
    rest = ~inl
    if rest.sum() >= MIN_INLIERS:
        h2, mask2 = cv2.findHomography(src_pts[rest], ref_pts[rest], cv2.USAC_MAGSAC, threshold, maxIters=5000)
        if h2 is not None and mask2 is not None and int(mask2.sum()) >= max(MIN_INLIERS, 0.4 * n_inl):
            sh, sw = src_gray.shape[:2]
            c = [[0, 0], [sw, 0], [sw, sh], [0, sh]]
            disp = np.linalg.norm(project_points(h, c) - project_points(h2, c), axis=1).max()
            ambiguous = disp > 0.03 * np.hypot(rw, rh)

    reasons = []
    if n_inl < GOOD_INLIERS:
        reasons.append(f"Only {n_inl} consistent matches")
    if ratio < GOOD_INLIER_RATIO:
        reasons.append("Most feature matches were inconsistent")
    if coverage < GOOD_COVERAGE:
        reasons.append("Matches cover only part of the blanket")
    if rms > max_rms:
        reasons.append(f"High residual error ({rms:.1f}px)")
    if ambiguous:
        reasons.append("Repeating pattern allows another alignment")

    alignment = Alignment(
        method="auto",
        quality="review" if reasons else "ok",
        inliers=n_inl,
        inlier_ratio=round(ratio, 3),
        coverage=round(coverage, 3),
        rms_error=round(rms, 2),
        ambiguous=ambiguous,
        reasons=reasons,
    )
    return AlignmentResult(_work_to_full(h, src_scale, ref_scale), alignment)


def align_manual(landmarks: list[Landmark], src_size: tuple[int, int], ref_size: tuple[int, int]) -> AlignmentResult:
    if len(landmarks) < 4:
        return AlignmentResult(None, Alignment(method="manual", quality="failed", reasons=["At least 4 landmark pairs are required"]))
    src = np.float64([lm.src for lm in landmarks])
    ref = np.float64([lm.ref for lm in landmarks])
    h, _ = cv2.findHomography(src, ref, 0)
    if h is None:
        return AlignmentResult(None, Alignment(method="manual", quality="failed", reasons=["Landmarks are degenerate (collinear or duplicated)"]))
    h = h / h[2, 2]
    sw, sh = src_size
    rw, rh = ref_size
    ok, why = _geometry_ok(h, (sh, sw), (rh, rw))
    if not ok:
        return AlignmentResult(None, Alignment(method="manual", quality="failed", reasons=[why or "Implausible transform"]))
    err = np.linalg.norm(project_points(h, src) - ref, axis=1)
    rms = float(np.sqrt(np.mean(err**2)))
    reasons = []
    if len(landmarks) > 4 and rms > 0.01 * np.hypot(rw, rh):
        reasons.append(f"Landmarks disagree by {rms:.0f}px; check their placement")
    return AlignmentResult(
        h,
        Alignment(method="manual", quality="review" if reasons else "ok", inliers=len(landmarks), inlier_ratio=1.0, coverage=1.0, rms_error=round(rms, 2), reasons=reasons),
    )
