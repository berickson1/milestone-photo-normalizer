"""Perspective straightening and wrinkle correction, end to end."""

import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.geometry import estimate_aspect, output_view, project_points, quad_from_polygon
from synth import BLANKET_H, BLANKET_W, blanket_corners_in_photo, blanket_texture, camera, encode, render_scene, wrinkle

TEX = blanket_texture()
CAMS = [
    camera(rot_deg=4, persp=0.00025),
    camera(dx=40, rot_deg=10, scale=1.1),
    camera(dx=-30, dy=20, rot_deg=-3, persp=0.0002),
    camera(dx=20, dy=-30, rot_deg=6, scale=1.15),
]
BABIES = [(0.3, 0.5), (0.7, 0.4), (0.5, 0.72), (0.42, 0.3)]
# Equally spaced phases: the mean wrinkle displacement over the set is zero, i.e. a flat blanket.
PHASES = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]


def wait_idle(client, pid, timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        p = client.get(f"/api/projects/{pid}").json()
        if p["job"]["state"] == "idle" and not p["processing"]:
            return p
        time.sleep(0.2)
    raise AssertionError("processing did not finish")


@pytest.fixture()
def client(tmp_path):
    main.init_state(tmp_path)
    with TestClient(main.app) as c:
        yield c


def test_quad_and_aspect_helpers():
    poly = [(10, 12), (500, 5), (520, 200), (515, 390), (8, 400), (0, 200)]
    assert quad_from_polygon(poly) == [(10, 12), (500, 5), (515, 390), (8, 400)]
    # Fronto-parallel views carry no focal-length information; fall back to side lengths.
    assert estimate_aspect(np.array([[0, 0], [400, 0], [400, 300], [0, 300]], float), (1000, 800)) == pytest.approx(4 / 3)


def _setup(client, crease=0.0):
    pid = client.post("/api/projects", json={"name": "Wrinkles"}).json()["id"]
    files = []
    for i, (g, baby, ph) in enumerate(zip(CAMS, BABIES, PHASES)):
        tex = wrinkle(TEX, 9.0, ph, crease=crease)
        files.append(("files", (f"{i}.png", encode(render_scene(tex, g, 20 + i, baby_at=baby)), "image/png")))
    client.post(f"/api/projects/{pid}/photos", files=files)
    wait_idle(client, pid)
    corners = blanket_corners_in_photo(CAMS[0])
    r = client.patch(
        f"/api/projects/{pid}",
        json={"blanket_polygon": corners.tolist(), "blanket_quad": corners.tolist(), "blanket_aspect": BLANKET_W / BLANKET_H, "output_width": 700},
    )
    assert r.status_code == 200
    wait_idle(client, pid)
    return pid


def _errors(pid):
    """Mean abs error of each output vs the flat, baby-free blanket, away from babies and edges."""
    project = main.state.store.get(pid)
    view = output_view(project)
    to_out = view.matrix @ CAMS[0]
    truth = cv2.warpPerspective(TEX, to_out, (view.width, view.height))
    inside = cv2.warpPerspective(np.full((BLANKET_H, BLANKET_W), 255, np.uint8), to_out, (view.width, view.height)) > 0
    inside = cv2.erode(inside.astype(np.uint8), np.ones((25, 25))) > 0
    babies = np.zeros_like(inside)
    for b in BABIES:
        x, y = project_points(to_out, [(b[0] * BLANKET_W + 50, b[1] * BLANKET_H)])[0]
        cv2.circle(babies.view(np.uint8), (round(x), round(y)), round(0.3 * view.width), 1, -1)
    errs = []
    for ph in project.photos:
        img = main.state.renderer.render(project, ph.id, view).image
        own = inside & ~babies
        errs.append(np.abs(img.astype(float) - truth.astype(float)).mean(axis=2)[own].mean())
    return np.array(errs), view, to_out


def test_perspective_maps_blanket_to_rectangle(client):
    pid = _setup(client)
    _, view, to_out = _errors(pid)
    to_out = to_out / to_out[2, 2]
    # The blanket plane lands on the output as an axis-aligned scaled rectangle (no skew or keystone).
    corners = project_points(to_out, [(0, 0), (BLANKET_W, 0), (BLANKET_W, BLANKET_H), (0, BLANKET_H)])
    assert abs(corners[0, 1] - corners[1, 1]) < 0.5 and abs(corners[0, 0] - corners[3, 0]) < 0.5
    assert abs(corners[2, 1] - corners[3, 1]) < 0.5 and abs(corners[1, 0] - corners[2, 0]) < 0.5
    w, h = corners[1, 0] - corners[0, 0], corners[3, 1] - corners[0, 1]
    assert w / h == pytest.approx(BLANKET_W / BLANKET_H, rel=1e-3)
    assert w < view.width and h < view.height


def test_straighten_reduces_wrinkle_distortion(client):
    pid = _setup(client)
    off, *_ = _errors(pid)
    client.patch(f"/api/projects/{pid}", json={"wrinkles": "straighten"})
    wait_idle(client, pid)
    assert all(main.state.store.flow_path(pid, p.id).exists() for p in main.state.store.get(pid).photos)
    on, *_ = _errors(pid)
    assert on.mean() < 0.7 * off.mean(), (off, on)
    assert np.all(on < off), (off, on)


def _baby_patches(pid):
    project = main.state.store.get(pid)
    view = output_view(project)
    to_out = view.matrix @ CAMS[0]
    out = []
    for ph, b in zip(project.photos, BABIES):
        x, y = project_points(to_out, [(b[0] * BLANKET_W, b[1] * BLANKET_H)])[0]
        img = main.state.renderer.render(project, ph.id, view).image
        out.append(img[round(y) - 6 : round(y) + 6, round(x) - 6 : round(x) + 6].astype(float))
    return out


def _crease_errors(pid):
    """Mean abs error vs the flat blanket inside each photo's own crease band."""
    project = main.state.store.get(pid)
    view = output_view(project)
    to_out = view.matrix @ CAMS[0]
    truth = cv2.warpPerspective(TEX, to_out, (view.width, view.height)).astype(float)
    ys, xs = np.mgrid[0:BLANKET_H, 0:BLANKET_W]
    errs = []
    for ph, phase in zip(project.photos, PHASES):
        band = (np.abs(xs - ys * 0.8 - 200 - 150 * phase) < 20).astype(np.uint8) * 255
        m = cv2.warpPerspective(band, to_out, (view.width, view.height)) > 127
        m = cv2.erode(m.astype(np.uint8), np.ones((3, 3))) > 0
        if m.sum() < 500:
            continue
        img = main.state.renderer.render(project, ph.id, view).image.astype(float)
        errs.append(np.abs(img - truth).mean(axis=2)[m].mean())
    return np.array(errs)


def test_clean_composite_removes_crease_shadows(client):
    pid = _setup(client, crease=0.5)
    client.patch(f"/api/projects/{pid}", json={"wrinkles": "straighten"})
    wait_idle(client, pid)
    straight, *_ = _errors(pid)
    straight_crease = _crease_errors(pid)
    babies_before = _baby_patches(pid)
    client.patch(f"/api/projects/{pid}", json={"wrinkles": "clean"})
    wait_idle(client, pid)
    clean, *_ = _errors(pid)
    clean_crease = _crease_errors(pid)
    assert len(straight_crease) >= 2
    assert np.all(clean_crease < 0.5 * straight_crease), (straight_crease, clean_crease)
    assert clean.mean() <= straight.mean(), (straight, clean)
    # Each photo keeps its own baby rather than the composite blanket.
    for before, after in zip(babies_before, _baby_patches(pid)):
        assert np.abs(before - after).mean() < 6
