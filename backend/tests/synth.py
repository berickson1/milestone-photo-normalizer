"""Synthetic milestone-blanket scenes with known geometry."""

import cv2
import numpy as np

BLANKET_W, BLANKET_H = 1000, 750
PHOTO_W, PHOTO_H = 1600, 1200


def blanket_texture(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((BLANKET_H, BLANKET_W, 3), (236, 226, 210), np.uint8)
    for _ in range(140):
        c = tuple(int(v) for v in rng.integers(40, 230, 3))
        x, y = int(rng.integers(0, BLANKET_W)), int(rng.integers(0, BLANKET_H))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.circle(img, (x, y), int(rng.integers(6, 30)), c, -1, cv2.LINE_AA)
        elif kind == 1:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(8, 50)), y + int(rng.integers(8, 50))), c, -1)
        else:
            cv2.putText(img, str(int(rng.integers(0, 99))), (x, y), cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.6, 1.6)), c, 2, cv2.LINE_AA)
    noise = rng.normal(0, 4, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def floor(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(70, 6, (PHOTO_H // 8, PHOTO_W // 8, 3))
    return cv2.resize(np.clip(base, 0, 255).astype(np.uint8), (PHOTO_W, PHOTO_H), interpolation=cv2.INTER_CUBIC)


def camera(dx: float = 0, dy: float = 0, rot_deg: float = 0, scale: float = 1.2, persp: float = 0) -> np.ndarray:
    """Homography from blanket coordinates to photo pixels."""
    cx, cy = BLANKET_W / 2, BLANKET_H / 2
    t1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
    a = np.deg2rad(rot_deg)
    r = np.array([[np.cos(a) * scale, -np.sin(a) * scale, 0], [np.sin(a) * scale, np.cos(a) * scale, 0], [0, 0, 1]])
    p = np.array([[1, 0, 0], [0, 1, 0], [persp, 0, 1]], float)
    t2 = np.array([[1, 0, PHOTO_W / 2 + dx], [0, 1, PHOTO_H / 2 + dy], [0, 0, 1]], float)
    return t2 @ p @ r @ t1


def wrinkle(texture: np.ndarray, amp: float, phase: float, wavelength: float = 260.0, crease: float = 0.0) -> np.ndarray:
    """Smoothly displace the blanket print (wrinkles) and optionally darken a diagonal crease band."""
    h, w = texture.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    mx = xs + amp * np.sin(2 * np.pi * ys / wavelength + phase)
    my = ys + amp * np.sin(2 * np.pi * xs / wavelength + phase + 1.3)
    out = cv2.remap(texture, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if crease:
        band = np.exp(-(((xs - ys * 0.8 - 200 - 150 * phase) / 25.0) ** 2))
        out = np.clip(out * (1 - crease * band[..., None]), 0, 255).astype(np.uint8)
    return out


def render_scene(texture: np.ndarray, g: np.ndarray, seed: int, baby_at=(0.5, 0.5), gain=(1.0, 1.0, 1.0)) -> np.ndarray:
    tex = texture.copy()
    if baby_at is not None:
        rng = np.random.default_rng(seed)
        bx, by = int(baby_at[0] * BLANKET_W), int(baby_at[1] * BLANKET_H)
        skin = tuple(int(v) for v in rng.integers(150, 220, 3))
        cv2.ellipse(tex, (bx, by), (150, 95), float(rng.uniform(0, 180)), 0, 360, skin, -1, cv2.LINE_AA)
        cv2.circle(tex, (bx + 120, by - 20), 60, skin, -1, cv2.LINE_AA)
    out = floor(seed)
    warped = cv2.warpPerspective(tex, g, (PHOTO_W, PHOTO_H), flags=cv2.INTER_LINEAR)
    m = cv2.warpPerspective(np.full(tex.shape[:2], 255, np.uint8), g, (PHOTO_W, PHOTO_H), flags=cv2.INTER_LINEAR) > 127
    out[m] = warped[m]
    lin = (out / 255.0) ** 2.2 * np.array(gain)
    return np.clip(lin ** (1 / 2.2) * 255, 0, 255).astype(np.uint8)


def blanket_corners_in_photo(g: np.ndarray) -> np.ndarray:
    c = np.array([[0, 0], [BLANKET_W, 0], [BLANKET_W, BLANKET_H], [0, BLANKET_H]], float)
    h = np.hstack([c, np.ones((4, 1))]) @ g.T
    return h[:, :2] / h[:, 2:3]


def encode(rgb: np.ndarray, ext: str = ".png") -> bytes:
    ok, buf = cv2.imencode(ext, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()
