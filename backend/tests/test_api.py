import hashlib
import io
import time
import zipfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.geometry import analysis_view, output_view, project_points
from synth import BLANKET_H, BLANKET_W, blanket_corners_in_photo, blanket_texture, camera, encode, render_scene

TEX = blanket_texture()
G_REF = camera(rot_deg=4)
SCENES = [
    ("ref.jpg", G_REF, (0.5, 0.5)),
    ("week2.jpg", camera(dx=40, rot_deg=10, scale=1.1), (0.55, 0.5)),
    ("week3.png", camera(dx=-30, dy=20, rot_deg=-3, persp=0.0002), (0.5, 0.55)),
    ("week4.jpg", camera(dx=20, dy=-30, rot_deg=6, scale=1.15), (0.75, 0.5)),
    # Left ~40% of the blanket is outside this frame.
    ("partial.jpg", camera(dx=-700, rot_deg=2), (0.7, 0.5)),
]


def wait_idle(client, pid, timeout=180):
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


def test_full_workflow(client, tmp_path):
    pid = client.post("/api/projects", json={"name": "Baby"}).json()["id"]
    uploads = []
    for i, (name, g, baby) in enumerate(SCENES):
        data = encode(render_scene(TEX, g, seed=10 + i, baby_at=baby), ".png" if name.endswith(".png") else ".jpg")
        uploads.append(("files", (name, data, "image/jpeg")))
    uploads.append(("files", ("notes.txt", b"not an image", "text/plain")))
    res = client.post(f"/api/projects/{pid}/photos", files=uploads).json()
    assert len(res["added"]) == 5
    assert res["rejected"][0]["name"] == "notes.txt"
    project = wait_idle(client, pid)
    ref_id, *_, partial_id = [p["id"] for p in project["photos"]]
    assert project["reference_id"] == ref_id
    assert all(p["homography"] is not None for p in project["photos"]), [p["alignment"] for p in project["photos"]]

    # Use the true blanket outline; the crop is its bounding box so its corners fall outside the blanket.
    corners = blanket_corners_in_photo(G_REF)
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    client.patch(
        f"/api/projects/{pid}",
        json={"blanket_polygon": corners.tolist(), "crop": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}, "output_width": 800},
    )
    project = wait_idle(client, pid)

    ready = client.get(f"/api/projects/{pid}/readiness").json()
    assert not ready["ready"]
    partial = next(p for p in ready["photos"] if p["photo_id"] == partial_id)
    assert partial["stats"]["reconstructed_px"] > 0
    assert partial["stats"]["unresolved_px"] == 0
    assert any(i["code"] == "reconstruction_review" for i in partial["issues"])
    assert client.get(f"/api/projects/{pid}/export.zip").status_code == 409

    for p in project["photos"]:
        client.patch(f"/api/projects/{pid}/photos/{p['id']}", json={"approved": True})
    assert client.get(f"/api/projects/{pid}/readiness").json()["ready"]
    r = client.get(f"/api/projects/{pid}/export.zip")
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    pngs = sorted(n for n in zf.namelist() if n.endswith(".png"))
    assert len(pngs) == 5
    images = [cv2.cvtColor(cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for n in pngs]
    assert len({im.shape for im in images}) == 1

    project_model = main.state.store.get(pid)
    # Every baby, including the reference photo's, is proposed as foreground.
    av = analysis_view(project_model.reference)
    for (name, _, baby), ph in zip(SCENES, project_model.photos):
        mask = cv2.imread(str(main.state.store.mask_path(pid, ph.id)), cv2.IMREAD_GRAYSCALE)
        x, y = project_points(av.matrix @ G_REF, [(baby[0] * BLANKET_W, baby[1] * BLANKET_H)])[0]
        assert mask[round(y), round(x)] == 255, name

    view = output_view(project_model)
    assert images[0].shape[:2] == (view.height, view.width)
    to_out = view.matrix @ G_REF
    inside = cv2.warpPerspective(np.full((BLANKET_H, BLANKET_W), 255, np.uint8), to_out, (view.width, view.height)) > 0
    outside = ~(cv2.dilate(inside.astype(np.uint8), np.ones((5, 5))) > 0)
    for im in images:
        assert im[outside].max() == 0

    # Rebuilt blanket must match the clean blanket, not a donor's baby.
    truth = cv2.warpPerspective(TEX, to_out, (view.width, view.height))
    g_partial = SCENES[-1][1]
    frame = cv2.warpPerspective(np.full((1200, 1600), 255, np.uint8), to_out @ np.linalg.inv(g_partial), (view.width, view.height)) > 0
    rebuilt = inside & ~(cv2.dilate(frame.astype(np.uint8), np.ones((9, 9))) > 0)
    rebuilt = cv2.erode(rebuilt.astype(np.uint8), np.ones((5, 5))) > 0
    assert rebuilt.sum() > 0.2 * inside.sum()
    diff = np.abs(images[-1].astype(int) - truth.astype(int)).max(axis=2)[rebuilt]
    assert np.median(diff) < 20
    assert np.mean(diff > 40) < 0.02

    # Originals are stored unchanged.
    for (name, *_), up, ph in zip(SCENES, uploads, project_model.photos):
        stored = main.state.store.original_path(pid, ph).read_bytes()
        assert hashlib.sha256(stored).hexdigest() == hashlib.sha256(up[1][1]).hexdigest() == ph.sha256

    # Reopening the project reproduces the same render.
    before = client.get(f"/api/projects/{pid}/photos/{partial_id}/render.jpg").content
    main.init_state(tmp_path)
    after = client.get(f"/api/projects/{pid}/photos/{partial_id}/render.jpg").content
    assert before == after


def test_unresolved_blanket_blocks_until_accepted(client):
    pid = client.post("/api/projects", json={"name": "Two"}).json()["id"]
    files = [
        ("files", ("a.jpg", encode(render_scene(TEX, G_REF, 1), ".jpg"), "image/jpeg")),
        ("files", ("b.jpg", encode(render_scene(TEX, camera(dx=-700), 2, baby_at=(0.75, 0.5)), ".jpg"), "image/jpeg")),
    ]
    client.post(f"/api/projects/{pid}/photos", files=files)
    project = wait_idle(client, pid)
    # A blanket polygon wider than anything photographed leaves pixels without any source.
    corners = blanket_corners_in_photo(G_REF)
    c = corners.mean(axis=0)
    big = (corners - c) * 1.25 + c
    x0, y0 = big.min(axis=0)
    x1, y1 = big.max(axis=0)
    client.patch(f"/api/projects/{pid}", json={"blanket_polygon": big.tolist(), "crop": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}})
    project = wait_idle(client, pid)
    b = project["photos"][1]["id"]
    for p in project["photos"]:
        client.patch(f"/api/projects/{pid}/photos/{p['id']}", json={"approved": True})
    ready = client.get(f"/api/projects/{pid}/readiness").json()
    issues = next(p for p in ready["photos"] if p["photo_id"] == b)["issues"]
    assert any(i["code"] == "unresolved" and i["blocking"] for i in issues)
    for p in project["photos"]:
        client.patch(f"/api/projects/{pid}/photos/{p['id']}", json={"accept_unresolved": True})
    assert client.get(f"/api/projects/{pid}/readiness").json()["ready"]


def test_mask_edit_resets_approval_and_persists(client):
    pid = client.post("/api/projects", json={"name": "Mask"}).json()["id"]
    files = [("files", (f"{i}.jpg", encode(render_scene(TEX, g, i, baby_at=b), ".jpg"), "image/jpeg")) for i, (_, g, b) in enumerate(SCENES[:3])]
    client.post(f"/api/projects/{pid}/photos", files=files)
    project = wait_idle(client, pid)
    ph = project["photos"][1]["id"]
    client.patch(f"/api/projects/{pid}/photos/{ph}", json={"approved": True})
    a = project["analysis"]
    mask = np.zeros((a["height"], a["width"]), np.uint8)
    mask[10:50, 10:50] = 255
    ok, buf = cv2.imencode(".png", mask)
    r = client.put(f"/api/projects/{pid}/photos/{ph}/mask.png", content=buf.tobytes())
    assert r.status_code == 200
    project = wait_idle(client, pid)
    photo = next(p for p in project["photos"] if p["id"] == ph)
    assert photo["mask_edited"] and not photo["approved"]
    got = cv2.imdecode(np.frombuffer(client.get(f"/api/projects/{pid}/photos/{ph}/mask.png").content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(got, mask)
    bad = client.put(f"/api/projects/{pid}/photos/{ph}/mask.png", content=cv2.imencode(".png", mask[:10])[1].tobytes())
    assert bad.status_code == 422
