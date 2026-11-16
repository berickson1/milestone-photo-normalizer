import cv2
import numpy as np

from app.alignment import align_auto, align_manual
from app.color import apply_gains, estimate_gains
from app.geometry import project_points
from app.storage import Landmark

from synth import PHOTO_H, PHOTO_W, blanket_texture, camera, render_scene

TEX = blanket_texture()


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def corner_error(h_est, h_true):
    c = [[0, 0], [PHOTO_W, 0], [PHOTO_W, PHOTO_H], [0, PHOTO_H]]
    return np.linalg.norm(project_points(h_est, c) - project_points(h_true, c), axis=1).max()


def test_recovers_known_perspective_despite_baby():
    g_ref = camera()
    g_src = camera(dx=60, dy=-40, rot_deg=12, scale=1.05, persp=0.00025)
    ref = render_scene(TEX, g_ref, 1, baby_at=(0.45, 0.5))
    src = render_scene(TEX, g_src, 2, baby_at=(0.6, 0.45))
    result = align_auto(gray(ref), gray(src), 1.0, 1.0)
    assert result.homography is not None
    assert result.alignment.quality == "ok", result.alignment.reasons
    assert corner_error(result.homography, g_ref @ np.linalg.inv(g_src)) < 2.0


def test_scales_homography_from_working_copies():
    g_ref, g_src = camera(), camera(dx=-50, rot_deg=-8)
    ref = render_scene(TEX, g_ref, 1)
    src = render_scene(TEX, g_src, 2)
    s = 0.5
    small = lambda im: cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    result = align_auto(gray(small(ref)), gray(small(src)), s, s)
    assert corner_error(result.homography, g_ref @ np.linalg.inv(g_src)) < 4.0


def test_unrelated_image_fails():
    rng = np.random.default_rng(3)
    noise = cv2.GaussianBlur(rng.integers(0, 255, (PHOTO_H, PHOTO_W), dtype=np.uint8), (9, 9), 0)
    result = align_auto(gray(render_scene(TEX, camera(), 1)), noise, 1.0, 1.0)
    assert result.alignment.quality == "failed"
    assert result.homography is None


def test_manual_landmarks():
    h_true = camera(dx=30, rot_deg=5) @ np.linalg.inv(camera())
    src = [(100, 100), (1400, 120), (1350, 1000), (150, 1050), (800, 600)]
    ref = project_points(h_true, src)
    lms = [Landmark(src=s, ref=tuple(r)) for s, r in zip(src, ref)]
    result = align_manual(lms, (PHOTO_W, PHOTO_H), (PHOTO_W, PHOTO_H))
    assert result.alignment.quality == "ok"
    assert corner_error(result.homography, h_true) < 0.5
    assert align_manual(lms[:3], (PHOTO_W, PHOTO_H), (PHOTO_W, PHOTO_H)).homography is None


def test_color_gain_estimation_ignores_masked_pixels():
    ref = render_scene(TEX, camera(), 1, baby_at=None)
    shifted = render_scene(TEX, camera(), 1, baby_at=None, gain=(1.3, 1.0, 0.75))
    mask = np.ones(ref.shape[:2], bool)
    shifted_with_junk = shifted.copy()
    shifted_with_junk[:300] = (255, 0, 0)
    mask_excluding_junk = mask.copy()
    mask_excluding_junk[:300] = False
    gains = estimate_gains(shifted_with_junk, ref, mask_excluding_junk)
    np.testing.assert_allclose(gains, [1 / 1.3, 1.0, 1 / 0.75], rtol=0.05)
    corrected = apply_gains(shifted, gains)
    assert np.abs(corrected.astype(int) - ref.astype(int))[mask].mean() < 3
