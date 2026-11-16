"""Rendering normalized photos, including donor-based blanket reconstruction."""

import tempfile
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import color
from .deform import flow_view, load_field, warp_deformed
from .geometry import View, analysis_view, homography, project_points, scale_matrix, source_corners
from .imageio import load_rgb, read_gray_file, read_rgb_file
from .storage import Photo, Project, ProjectStore

NONE, TARGET, PLATE = -2, -1, -3


class LRU:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
        return None

    def put(self, key, value):
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()


@dataclass
class Rendered:
    image: np.ndarray
    provenance: np.ndarray  # NONE, TARGET, or index into `donors`
    unresolved: np.ndarray
    donors: list[str] = field(default_factory=list)

    @property
    def stats(self) -> dict:
        blanket = int(np.count_nonzero(self.provenance != NONE) + np.count_nonzero(self.unresolved))
        per_donor = {pid: int(np.count_nonzero(self.provenance == i)) for i, pid in enumerate(self.donors)}
        recon = sum(per_donor.values())
        unresolved = int(np.count_nonzero(self.unresolved))
        return {
            "blanket_px": blanket,
            "reconstructed_px": recon,
            "unresolved_px": unresolved,
            "reconstructed_fraction": recon / blanket if blanket else 0.0,
            "unresolved_fraction": unresolved / blanket if blanket else 0.0,
            "donors": [{"photo_id": pid, "px": px} for pid, px in per_donor.items() if px > 0],
        }


def warp(img: np.ndarray, m: np.ndarray, view: View, interp=cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.warpPerspective(img, m, (view.width, view.height), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def polygon_mask(points: np.ndarray, view: View) -> np.ndarray:
    mask = np.zeros((view.height, view.width), np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(mask, [np.round(points * 16).astype(np.int32).reshape(-1, 1, 2)], 255, lineType=cv2.LINE_8, shift=4)
    return mask > 0


def blanket_mask(project: Project, view: View) -> np.ndarray:
    if not project.blanket_polygon:
        return np.zeros((view.height, view.width), bool)
    return polygon_mask(project_points(view.matrix, np.array(project.blanket_polygon)), view)


def _kernel(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


SEAM_FLOW_WIDTH = 1000  # working width for seam alignment
SEAM_MAX_SHIFT = 0.025  # largest local correction, as a fraction of output width


def seam_flow(donor: np.ndarray, target: np.ndarray, samples: np.ndarray, max_shift: float) -> np.ndarray | None:
    """Smooth displacement field (h, w, 2) so that donor(x + flow(x)) lines up with target(x).

    Measured by optical flow where both images show blanket (`samples`), kept only where forward
    and backward flow agree on textured print, then extended smoothly over the rest.
    """
    h, w = samples.shape
    s = min(1.0, SEAM_FLOW_WIDTH / w)
    size = (max(8, round(w * s)), max(8, round(h * s)))
    gray = [cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), size, interpolation=cv2.INTER_AREA) for im in (target, donor)]
    smp = cv2.resize(samples.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    smp = cv2.erode(smp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if np.count_nonzero(smp) < 200:
        return None
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fwd = dis.calc(gray[0], gray[1], None)
    bwd = dis.calc(gray[1], gray[0], None)
    gx, gy = np.meshgrid(np.arange(size[0], dtype=np.float32), np.arange(size[1], dtype=np.float32))
    back = cv2.remap(bwd, gx + fwd[..., 0], gy + fwd[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    consistent = np.linalg.norm(fwd + back, axis=2) < 1.0
    g = gray[0].astype(np.float32)
    mu = cv2.blur(g, (7, 7))
    textured = cv2.blur(g * g, (7, 7)) - mu * mu > 16.0
    conf = smp & consistent & textured
    if np.count_nonzero(conf) < 100:
        return None
    k = min(1.0, 200 / size[0])
    small = (max(4, round(size[0] * k)), max(4, round(size[1] * k)))
    wts = cv2.resize(conf.astype(np.float32), small, interpolation=cv2.INTER_AREA)
    vals = cv2.resize(fwd * conf[..., None], small, interpolation=cv2.INTER_AREA) / np.maximum(wts, 1e-6)[..., None]
    flow = color.fill_sparse(vals, np.where(wts > 0.05, wts, 0).astype(np.float32))
    flow = cv2.GaussianBlur(flow, (0, 0), max(1.0, 0.01 * small[0])).reshape(small[1], small[0], 2)
    flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / s
    limit = max_shift * w
    mag = np.linalg.norm(flow, axis=2, keepdims=True)
    return (flow * np.minimum(1.0, limit / np.maximum(mag, 1e-6))).astype(np.float32)


def align_donor(
    d_img: np.ndarray, usable: np.ndarray, result: np.ndarray, resolved: np.ndarray, samples: np.ndarray, region: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Locally warp a donor within `region` so its print continues the result across the seam."""
    ys, xs = np.nonzero(region)
    if not len(ys):
        return d_img, usable
    crop = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    # Show the donor inside the hole so flow isn't pulled by the hole's black edge.
    target = np.where(resolved[crop][..., None], result[crop], d_img[crop])
    flow = seam_flow(d_img[crop], target, samples[crop], SEAM_MAX_SHIFT * region.shape[1] / (xs.max() - xs.min() + 1))
    if flow is None:
        return d_img, usable
    gx, gy = np.meshgrid(np.arange(crop[1].start, crop[1].stop, dtype=np.float32), np.arange(crop[0].start, crop[0].stop, dtype=np.float32))
    mx, my = gx + flow[..., 0], gy + flow[..., 1]
    d_img, usable = d_img.copy(), usable.copy()
    d_img[crop] = cv2.remap(d_img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    usable[crop] = cv2.remap(usable.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return d_img, usable


def donor_eligible(photo: Photo, target_id: str) -> bool:
    return (
        photo.id != target_id
        and photo.homography is not None
        and photo.alignment is not None
        and (photo.alignment.quality == "ok" or photo.approved)
    )


class Renderer:
    def __init__(self, store: ProjectStore):
        self.store = store
        self._work = LRU(64)
        self._full = LRU(4)
        self._orders = LRU(256)
        self._fields = LRU(64)
        self._plates = LRU(3)

    def invalidate(self, project_id: str | None = None):
        self._work.clear()
        self._full.clear()
        self._orders.clear()
        self._fields.clear()
        self._plates.clear()

    # ---- sources -------------------------------------------------------
    def source(self, project: Project, photo: Photo, full: bool) -> tuple[np.ndarray, float]:
        key = (project.id, photo.id)
        if full:
            img = self._full.get(key)
            if img is None:
                img = load_rgb(self.store.original_path(project.id, photo))
                self._full.put(key, img)
            return img, 1.0
        img = self._work.get(key)
        if img is None:
            img = read_rgb_file(self.store.work_path(project.id, photo.id))
            self._work.put(key, img)
        return img, photo.work_scale

    def analysis_mask(self, project: Project, photo_id: str) -> np.ndarray | None:
        return read_gray_file(self.store.mask_path(project.id, photo_id))

    def foreground(self, project: Project, photo: Photo, view: View, dilate_frac: float = 0.0) -> np.ndarray:
        ref = project.reference
        mask = self.analysis_mask(project, photo.id)
        if mask is None or ref is None:
            return np.zeros((view.height, view.width), bool)
        m = view.matrix @ np.linalg.inv(analysis_view(ref).matrix)
        out = warp(mask, m, view, cv2.INTER_NEAREST) > 127
        if dilate_frac > 0 and out.any():
            out = cv2.dilate(out.astype(np.uint8), _kernel(round(dilate_frac * view.width))) > 0
        return out

    def warped(self, project: Project, photo: Photo, view: View, full: bool, apply_color: bool = True, deform: bool = True):
        h = homography(photo)
        if h is None:
            raise ValueError(f"Photo {photo.original_name} is not aligned")
        img, s = self.source(project, photo, full)
        disp = self.wrinkle_field(project, photo) if deform else None
        if disp is not None:
            out, valid = warp_deformed(img, h, s, view, flow_view(project.reference), disp)
        else:
            m = view.matrix @ h @ np.linalg.inv(scale_matrix(s))
            out = warp(img, m, view)
            valid = warp(np.full(img.shape[:2], 255, np.uint8), m, view, cv2.INTER_NEAREST)
        # Trim interpolation fringe at the photo's edges.
        valid = cv2.erode(valid, _kernel(max(1, round(view.width / 800)))) > 0
        if apply_color:
            out = color.apply_gains(out, color.effective_gains(photo.color))
        return out, valid

    def wrinkle_field(self, project: Project, photo: Photo) -> np.ndarray | None:
        """Wrinkle displacement field for the photo, when wrinkle correction is on and current."""
        if project.wrinkles == "off" or project.reference is None or photo.homography is None:
            return None
        path = self.store.flow_path(project.id, photo.id)
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            return None
        key = (project.id, photo.id, stamp)
        hit = self._fields.get(key)
        if hit is None:
            hit = (load_field(path, homography(photo)),)
            self._fields.put(key, hit)
        return hit[0]

    # ---- clean composite -------------------------------------------------
    def clean_plate(self, project: Project, view: View, full: bool) -> tuple[np.ndarray, np.ndarray]:
        """Per-pixel median of the blanket across all trusted photos, ignoring babies and props.

        Crease shadows and wrinkles vary between photos, so the median shows a clean blanket.
        Returns (image, valid). Warped photos are spilled to a temporary memmap to bound memory.
        """
        key = (project.id, project.revision, view.matrix.tobytes(), view.width, view.height, full)
        hit = self._plates.get(key)
        if hit is not None:
            return hit
        photos = [p for p in project.photos if p.homography is not None and p.alignment is not None and (p.alignment.quality == "ok" or p.approved or p.id == project.reference_id)]
        shape = (view.height, view.width)
        plate = np.zeros(shape + (3,), np.uint8)
        plate_valid = np.zeros(shape, bool)
        if photos:
            blanket = blanket_mask(project, view)
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                stack = np.lib.format.open_memmap(Path(tmp) / "stack.npy", mode="w+", dtype=np.uint8, shape=(len(photos),) + shape + (4,))
                for i, p in enumerate(photos):
                    img, valid = self.warped(project, p, view, full)
                    usable = valid & blanket & ~self.foreground(project, p, view, 0.012)
                    stack[i, ..., :3] = img
                    stack[i, ..., 3] = usable
                rows = max(8, int(4_000_000 // max(1, view.width * len(photos))))
                for y0 in range(0, shape[0], rows):
                    chunk = np.array(stack[:, y0 : y0 + rows])  # copy so no view keeps the file mapped
                    use = chunk[..., 3] > 0
                    vals = chunk[..., :3].astype(np.float32)
                    vals[~use] = np.nan
                    any_use = use.any(axis=0)
                    with np.errstate(all="ignore"), warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        med = np.nanmedian(vals, axis=0)
                    plate[y0 : y0 + rows][any_use] = np.round(med[any_use]).astype(np.uint8)
                    plate_valid[y0 : y0 + rows] = any_use
                del stack
        result = (plate, plate_valid)
        self._plates.put(key, result)
        return result

    # ---- donor selection ------------------------------------------------
    def donor_order(self, project: Project, target: Photo) -> list[Photo]:
        """Greedy donor ranking computed once at analysis resolution for consistent previews and exports."""
        key = (project.id, project.revision, target.id)
        cached = self._orders.get(key)
        if cached is not None:
            return [project.photo(pid) for pid in cached]
        ref = project.reference
        candidates = [p for p in project.photos if donor_eligible(p, target.id)]
        if ref is None or not candidates or target.homography is None:
            return []
        view = analysis_view(ref)
        scale = min(1.0, 600 / max(view.width, view.height))
        small = View(scale_matrix(scale) @ view.matrix, max(1, round(view.width * scale)), max(1, round(view.height * scale)))

        def frame(p: Photo) -> np.ndarray:
            return polygon_mask(project_points(small.matrix @ homography(p), source_corners(p)), small)

        remaining = blanket_mask(project, small) & ~frame(target)
        usable = {p.id: frame(p) & ~self.foreground(project, p, small, 0.01) for p in candidates}
        rank = {p.id: (0 if p.alignment.quality == "ok" else 1, p.alignment.rms_error) for p in candidates}
        order: list[str] = []
        while remaining.any():
            gains = {pid: int(np.count_nonzero(remaining & u)) for pid, u in usable.items() if pid not in order}
            if not gains:
                break
            best = min(gains, key=lambda pid: (-gains[pid], rank[pid]))
            if gains[best] == 0:
                break
            order.append(best)
            remaining &= ~usable[best]
        # Remaining donors catch pixels that differ at full resolution.
        order += sorted((pid for pid in usable if pid not in order), key=lambda pid: rank[pid])
        self._orders.put(key, order)
        return [project.photo(pid) for pid in order]

    # ---- rendering -------------------------------------------------------
    def render(
        self,
        project: Project,
        photo_id: str,
        view: View,
        *,
        full: bool = False,
        blackout: bool = True,
        reconstruct: bool = True,
        apply_color: bool = True,
    ) -> Rendered:
        target = project.photo(photo_id)
        if target is None:
            raise KeyError(photo_id)
        img, valid = self.warped(project, target, view, full, apply_color)
        shape = (view.height, view.width)
        if not blackout:
            prov = np.where(valid, TARGET, NONE).astype(np.int16)
            return Rendered(img, prov, np.zeros(shape, bool))

        blanket = blanket_mask(project, view)
        result = np.zeros_like(img)
        own = blanket & valid
        result[own] = img[own]
        prov = np.full(shape, NONE, np.int16)
        prov[own] = TARGET
        remaining = blanket & ~valid
        donors_used: list[str] = []

        if project.wrinkles == "clean" and reconstruct:
            plate, plate_valid = self.clean_plate(project, view, full)
            area = blanket & plate_valid
            if area.any():
                # Keep the target's own pixels on and just around the baby/props, feathered into the plate.
                keep = self.foreground(project, target, view, 0.012).astype(np.float32)
                if keep.any():
                    feather = max(1.0, 0.006 * view.width)
                    keep = np.maximum(keep, cv2.GaussianBlur(keep, (0, 0), feather))
                    keep = np.clip(keep * 1.6, 0, 1)
                w_plate = np.where(valid, 1.0 - keep, 1.0)[area][:, None]
                result[area] = np.round(result[area] * (1 - w_plate) + plate[area] * w_plate).astype(np.uint8)
                plate_px = np.zeros_like(area)
                plate_px[area] = w_plate[:, 0] >= 0.5
                prov[plate_px] = PLATE
                remaining &= ~plate_valid

        if reconstruct and remaining.any():
            fg_t = self.foreground(project, target, view)
            ring_zone = cv2.dilate(remaining.astype(np.uint8), _kernel(round(0.03 * view.width))) > 0
            ring = ring_zone & own & ~fg_t
            feather = max(2, round(0.004 * view.width))
            for donor in self.donor_order(project, target):
                if not remaining.any():
                    break
                d_img, d_valid = self.warped(project, donor, view, full, apply_color)
                usable = d_valid & ~self.foreground(project, donor, view, 0.01)
                take = remaining & usable
                if not take.any():
                    continue
                near = cv2.dilate(take.astype(np.uint8), _kernel(round(0.025 * view.width))) > 0
                resolved = prov != NONE
                d_img, usable = align_donor(d_img, usable, result, resolved, near & resolved & ~fg_t & usable, near)
                take = remaining & usable
                if not take.any():
                    continue
                # Match the donor's local brightness and contrast to the surrounding result along
                # the seam, so lighting gradients and haze don't leave a visible step.
                samples = near & resolved & ~fg_t & usable
                matched = color.seam_transfer(d_img, usable, result, samples, cv2.dilate(take.astype(np.uint8), _kernel(feather + 1)) > 0)
                if matched is not None:
                    d_img = matched
                else:
                    overlap = ring & usable
                    local = color.estimate_gains(d_img, result, overlap, 0.8, 1.25, 50) if overlap.any() else None
                    if local is not None:
                        d_img = color.apply_gains(d_img, local)
                idx = len(donors_used)
                donors_used.append(donor.id)
                result[take] = d_img[take]
                prov[take] = idx
                remaining &= ~take
                # Cross-fade a narrow band of the target into the donor to hide the seam.
                dist = cv2.distanceTransform((~take).astype(np.uint8), cv2.DIST_L2, 3)
                band = (dist > 0) & (dist <= feather) & (prov == TARGET) & usable & ~fg_t
                if band.any():
                    w = (0.5 * (1.0 - dist[band] / feather))[:, None]
                    result[band] = np.round(result[band] * (1 - w) + d_img[band] * w).astype(np.uint8)

        unresolved = remaining if reconstruct else np.zeros(shape, bool)
        return Rendered(result, prov, unresolved, donors_used)


DONOR_COLORS = [
    (66, 135, 245), (245, 166, 35), (80, 200, 120), (190, 90, 220), (40, 200, 200),
    (240, 220, 60), (255, 120, 160), (150, 110, 80), (120, 160, 255), (170, 230, 90),
]


def donor_color(i: int) -> tuple[int, int, int]:
    return DONOR_COLORS[i % len(DONOR_COLORS)]


def provenance_overlay(r: Rendered) -> np.ndarray:
    """RGBA overlay: donor pixels tinted per donor, unresolved pixels hatched magenta."""
    h, w = r.provenance.shape
    out = np.zeros((h, w, 4), np.uint8)
    for i in range(len(r.donors)):
        sel = r.provenance == i
        out[sel] = (*donor_color(i), 120)
    yy, xx = np.mgrid[0:h, 0:w]
    hatch = ((xx + yy) // 6) % 2 == 0
    out[r.unresolved & hatch] = (255, 0, 255, 220)
    out[r.unresolved & ~hatch] = (255, 0, 255, 90)
    return out
