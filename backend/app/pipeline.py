"""Import and batch processing: alignment, foreground proposals, and color estimation."""

import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from . import alignment, color, config, deform, masks
from .geometry import analysis_view, estimate_aspect, homography, quad_from_polygon, scale_matrix
from .imageio import encode_jpeg, encode_png, load_rgb, resize_max, write_bytes_atomic
from .render import Renderer, blanket_mask
from .storage import Alignment, Photo, Project, ProjectStore, Rect, new_id, sha256_file

log = logging.getLogger(__name__)
Progress = Callable[[int, int, str], None]
FIELD_FG_DILATE = 0.03  # foreground margin kept out of the wrinkle-flow evidence, as a fraction of width


class ProcessingAborted(Exception):
    pass


class Pipeline:
    def __init__(self, store: ProjectStore, renderer: Renderer):
        self.store = store
        self.renderer = renderer

    # ---- import ---------------------------------------------------------------
    def import_photo(self, project_id: str, tmp_path: Path, original_name: str) -> Photo:
        """Validate and register an upload. The original file is stored byte-for-byte unchanged."""
        rgb = load_rgb(tmp_path)
        photo_id = new_id()
        ext = Path(original_name).suffix.lower()[:10] or ".img"
        photo = Photo(
            id=photo_id,
            original_name=Path(original_name).name[:200] or "photo",
            stored_name=f"{photo_id}{ext}",
            sha256=sha256_file(tmp_path),
            width=rgb.shape[1],
            height=rgb.shape[0],
            work_scale=1.0,
            created_at=time.time(),
        )
        work, scale = resize_max(rgb, config.WORK_MAX_DIM)
        photo.work_scale = scale
        thumb, _ = resize_max(rgb, config.THUMB_MAX_DIM)
        write_bytes_atomic(self.store.work_path(project_id, photo_id), encode_jpeg(work, 95))
        write_bytes_atomic(self.store.thumb_path(project_id, photo_id), encode_jpeg(thumb, 85))
        dest = self.store.original_path(project_id, photo)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_path), dest)

        def add(p: Project):
            p.photos.append(photo)
            if p.reference_id is None:
                set_reference(self.store, p, photo.id)

        self.store.mutate(project_id, add)
        return photo

    # ---- processing ----------------------------------------------------------
    def process(self, project_id: str, progress: Progress) -> None:
        project = self.store.get(project_id)
        ref = project.reference
        if ref is None:
            progress(0, 0, "Choose a reference photo")
            return
        ref_id = ref.id
        todo = [p for p in project.photos if p.status in ("pending", "queued", "aligning", "error")]
        total = len(todo) + 1
        if todo:
            self.store.mutate(project_id, lambda p: [setattr(x, "status", "queued") for x in p.photos if x.id in {t.id for t in todo}], bump=False)

        for i, photo in enumerate(todo):
            progress(i, total, f"Aligning {photo.original_name}")
            self._set_status(project_id, photo.id, "aligning")
            try:
                result = self.align(self.store.get(project_id), photo.id)
            except ProcessingAborted:
                return
            except Exception as exc:  # keep processing the rest of the batch
                log.exception("alignment failed")
                self._apply_alignment(project_id, ref_id, photo.id, None, Alignment(quality="failed", reasons=[str(exc)]), error=str(exc))
                continue
            self._apply_alignment(project_id, ref_id, photo.id, result.homography, result.alignment)

        progress(total - 1, total, "Detecting babies and props, matching colors")
        self.refresh_analysis(project_id, realign=bool(todo))
        progress(total, total, "Done")

    def _set_status(self, project_id: str, photo_id: str, status: str):
        def fn(p: Project):
            ph = p.photo(photo_id)
            if ph:
                ph.status = status

        self.store.mutate(project_id, fn, bump=False)

    def _apply_alignment(self, project_id, ref_id, photo_id, h, al: Alignment, error: str | None = None):
        def fn(p: Project):
            ph = p.photo(photo_id)
            if ph is None or p.reference_id != ref_id:
                return
            changed = (ph.homography is None) != (h is None) or (
                h is not None and not np.allclose(np.array(ph.homography), h, atol=1e-6)
            )
            ph.homography = None if h is None else np.asarray(h).tolist()
            ph.alignment = al
            ph.status = "error" if error else "aligned"
            ph.error = error
            if changed:
                ph.approved = False

        self.store.mutate(project_id, fn)

    def _work_gray(self, project: Project, photo: Photo) -> np.ndarray:
        img, _ = self.renderer.source(project, photo, full=False)
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def align(self, project: Project, photo_id: str, use_masks: bool = False) -> alignment.AlignmentResult:
        ref = project.reference
        photo = project.photo(photo_id)
        if ref is None or photo is None:
            raise ProcessingAborted()
        if photo.id == ref.id:
            return alignment.AlignmentResult(np.eye(3), Alignment(method="reference", quality="ok", coverage=1.0, inlier_ratio=1.0))
        if photo.landmarks:
            return alignment.align_manual(photo.landmarks, (photo.width, photo.height), (ref.width, ref.height))

        ref_ex = src_ex = None
        if use_masks:
            ref_ex = self._exclusion_in_source(project, ref)
            src_ex = self._exclusion_in_source(project, photo)
        region = None
        if project.blanket_polygon:
            pts = np.array(project.blanket_polygon) * ref.work_scale
            region = (*pts.min(axis=0), *pts.max(axis=0))
        return alignment.align_auto(
            self._work_gray(project, ref),
            self._work_gray(project, photo),
            ref.work_scale,
            photo.work_scale,
            ref_ex,
            src_ex,
            region,
            wrinkle_correction=project.wrinkles != "off",
        )

    def _exclusion_in_source(self, project: Project, photo: Photo) -> np.ndarray | None:
        """Foreground mask mapped back into the photo's working image."""
        mask = self.renderer.analysis_mask(project, photo.id)
        h = homography(photo)
        if mask is None or h is None or project.reference is None:
            return None
        av = analysis_view(project.reference)
        src_to_analysis = av.matrix @ h @ np.linalg.inv(scale_matrix(photo.work_scale))
        w = round(photo.width * photo.work_scale)
        hh = round(photo.height * photo.work_scale)
        return cv2.warpPerspective(mask, np.linalg.inv(src_to_analysis), (w, hh), flags=cv2.INTER_NEAREST)

    # ---- masks & color --------------------------------------------------------
    def refresh_analysis(self, project_id: str, realign: bool = False) -> None:
        project = self.store.get(project_id)
        ref = project.reference
        if ref is None:
            return
        if not project.blanket_polygon:
            ref_img, _ = self.renderer.source(project, ref, full=False)
            poly = masks.propose_blanket(ref_img, ref.work_scale)

            def set_blanket(p: Project):
                if not p.blanket_polygon:
                    p.blanket_polygon = poly
                if p.crop is None:
                    p.crop = crop_for_polygon(poly)

            project = self.store.mutate(project_id, set_blanket)
        if not project.blanket_quad and project.blanket_polygon:
            project = self.store.mutate(project_id, set_quad_from_polygon)

        self._estimate_colors(project_id, use_masks=False)
        self._detect_foreground(project_id)
        if realign:
            self._realign_with_masks(project_id)
            self._detect_foreground(project_id)
        if self.store.get(project_id).wrinkles != "off":
            self._compute_fields(project_id)
            self._detect_foreground(project_id)
            self._compute_fields(project_id, exclude_foreground=True)
            self._detect_foreground(project_id)
        self._estimate_colors(project_id, use_masks=True)

    def _compute_fields(self, project_id: str, exclude_foreground: bool = False) -> None:
        """Wrinkle displacement fields onto the photos' mean geometry (see deform.py).

        With exclude_foreground, the (dilated) foreground masks are left out of the flow evidence so
        the field over the baby is interpolated from the surrounding blanket instead of following it.
        """
        project = self.store.get(project_id)
        ref = project.reference
        photos = self._aligned_photos(project)
        if ref is None or ref.homography is None or len(photos) < 2:
            return
        fview = deform.flow_view(ref)
        blanket = blanket_mask(project, fview)
        grays, oks = {}, {}
        for p in photos:
            img, valid = self.renderer.warped(project, p, fview, full=False, apply_color=False, deform=False)
            grays[p.id] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # On the first pass foreground masks are unreliable; the flow's photometric check rejects most of the baby.
            oks[p.id] = valid & blanket
            if exclude_foreground:
                oks[p.id] &= ~self.renderer.foreground(project, p, fview, FIELD_FG_DILATE)
        fields = []
        for p in photos:
            if p.id == ref.id:
                fields.append(np.zeros((fview.height, fview.width, 2), np.float32))
                continue
            fields.append(deform.pairwise_flow(grays[ref.id], grays[p.id], oks[ref.id] & oks[p.id]))
        for p, d in zip(photos, deform.group_fields(fields)):
            path = self.store.flow_path(project_id, p.id)
            if d is None:
                path.unlink(missing_ok=True)
            else:
                deform.save_field(path, d, np.array(p.homography))
        self.store.mutate(project_id, lambda p: None)

    def _aligned_photos(self, project: Project) -> list[Photo]:
        return [p for p in project.photos if p.homography is not None]

    def _analysis_small(self, project: Project):
        return masks.downscaled(analysis_view(project.reference), 800)

    def _detect_foreground(self, project_id: str) -> None:
        project = self.store.get(project_id)
        photos = self._aligned_photos(project)
        if not photos:
            return
        view = self._analysis_small(project)
        imgs, valids = [], []
        for p in photos:
            img, valid = self.renderer.warped(project, p, view, full=False)
            imgs.append(img)
            valids.append(valid)
        fg, background = masks.detect_foreground(imgs, valids, blanket_mask(project, view), shading_tolerant=project.wrinkles == "clean")
        av = analysis_view(project.reference)
        bg_full = cv2.resize(background, (av.width, av.height), interpolation=cv2.INTER_LINEAR)
        write_bytes_atomic(self.store.background_path(project_id), encode_jpeg(bg_full, 90))
        current = self.store.get(project_id)
        for p, m in zip(photos, fg):
            cur = current.photo(p.id)
            if cur is None or cur.mask_edited:
                continue
            up = cv2.resize(m.astype(np.uint8) * 255, (av.width, av.height), interpolation=cv2.INTER_NEAREST)
            write_bytes_atomic(self.store.mask_path(project_id, p.id), encode_png(up))
        self.store.mutate(project_id, lambda p: None)

    def _realign_with_masks(self, project_id: str) -> None:
        project = self.store.get(project_id)
        ref_id = project.reference_id
        for photo in project.photos:
            if photo.homography is None or photo.landmarks or photo.id == ref_id:
                continue
            try:
                result = self.align(project, photo.id, use_masks=True)
            except Exception:
                log.exception("masked realignment failed")
                continue
            old = photo.alignment
            better = result.homography is not None and (
                old is None
                or _quality_rank(result.alignment.quality) < _quality_rank(old.quality)
                or (result.alignment.quality == old.quality and result.alignment.rms_error <= old.rms_error * 1.05)
            )
            if better:
                self._apply_alignment(project_id, ref_id, photo.id, result.homography, result.alignment)

    def _estimate_colors(self, project_id: str, use_masks: bool) -> None:
        project = self.store.get(project_id)
        ref = project.reference
        if ref is None or ref.homography is None:
            return
        view = self._analysis_small(project)
        ref_img, ref_valid = self.renderer.warped(project, ref, view, full=False, apply_color=False)
        blanket = blanket_mask(project, view)
        ref_fg = self.renderer.foreground(project, ref, view) if use_masks else np.zeros_like(blanket)
        gains: dict[str, tuple[float, float, float] | None] = {}
        for p in self._aligned_photos(project):
            if p.id == ref.id:
                gains[p.id] = (1.0, 1.0, 1.0)
                continue
            img, valid = self.renderer.warped(project, p, view, full=False, apply_color=False)
            fg = self.renderer.foreground(project, p, view) if use_masks else np.zeros_like(blanket)
            g = color.estimate_gains(img, ref_img, blanket & valid & ref_valid & ~fg & ~ref_fg)
            gains[p.id] = None if g is None else tuple(round(float(x), 4) for x in g)

        def fn(pr: Project):
            for ph in pr.photos:
                if ph.id in gains:
                    ph.color.gains = gains[ph.id]

        self.store.mutate(project_id, fn)


def _quality_rank(q: str) -> int:
    return {"ok": 0, "review": 1, "failed": 2}[q]


def crop_for_polygon(poly) -> Rect:
    pts = np.array(poly, dtype=np.float64)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return Rect(x=float(x0), y=float(y0), w=float(max(1.0, x1 - x0)), h=float(max(1.0, y1 - y0)))


def set_quad_from_polygon(project: Project) -> None:
    if not project.blanket_polygon:
        return
    project.blanket_quad = quad_from_polygon(project.blanket_polygon)
    ref = project.reference
    size = (ref.width, ref.height) if ref else None
    project.blanket_aspect = round(estimate_aspect(np.array(project.blanket_quad), size), 4)


def set_reference(store: ProjectStore, project: Project, photo_id: str) -> None:
    """Switch the reference; everything defined in reference space must be rebuilt."""
    project.reference_id = photo_id
    project.crop = None
    project.blanket_polygon = None
    project.blanket_quad = None
    project.blanket_aspect = None
    project.output_width = None
    for ph in project.photos:
        ph.homography = None
        ph.alignment = None
        ph.landmarks = []
        ph.color.gains = None
        ph.mask_edited = False
        ph.approved = False
        ph.accept_unresolved = False
        ph.status = "pending"
        ph.error = None
        store.mask_path(project.id, ph.id).unlink(missing_ok=True)
        store.flow_path(project.id, ph.id).unlink(missing_ok=True)
    store.background_path(project.id).unlink(missing_ok=True)
