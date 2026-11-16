"""Bounded background processing with per-project coalescing and restart recovery."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from .pipeline import Pipeline
from .storage import NotFoundError, Project, ProjectStore

log = logging.getLogger(__name__)


class JobManager:
    def __init__(self, store: ProjectStore, pipeline: Pipeline, max_workers: int):
        self.store = store
        self.pipeline = pipeline
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mpn-job")
        self._lock = threading.Lock()
        self._active: set[str] = set()
        self._dirty: set[str] = set()

    def request(self, project_id: str) -> None:
        """Process a project now, or once more after its current run finishes."""
        with self._lock:
            if project_id in self._active:
                self._dirty.add(project_id)
                return
            self._active.add(project_id)
        self._set_job(project_id, state="queued", message="Waiting to start", error=None)
        self.executor.submit(self._run, project_id)

    def is_active(self, project_id: str) -> bool:
        with self._lock:
            return project_id in self._active

    def _set_job(self, project_id: str, **fields) -> None:
        def fn(p: Project):
            for k, v in fields.items():
                setattr(p.job, k, v)

        try:
            self.store.mutate(project_id, fn, bump=False)
        except NotFoundError:
            pass

    def _run(self, project_id: str) -> None:
        while True:
            error = None
            try:
                self._set_job(project_id, state="running", done=0, total=0, message="Starting")
                self.pipeline.process(
                    project_id, lambda d, t, m: self._set_job(project_id, state="running", done=d, total=t, message=m)
                )
            except NotFoundError:
                pass
            except Exception as exc:
                log.exception("processing failed for %s", project_id)
                error = str(exc) or exc.__class__.__name__
            with self._lock:
                if project_id in self._dirty and error is None:
                    self._dirty.discard(project_id)
                    continue
                self._dirty.discard(project_id)
                self._active.discard(project_id)
            self._set_job(project_id, state="idle", message="Failed" if error else "Idle", error=error)
            return

    def recover(self) -> None:
        """Requeue work interrupted by a restart."""
        for project in self.store.list():
            interrupted = project.job.state != "idle" or any(p.status in ("queued", "aligning") for p in project.photos)
            if not interrupted:
                continue

            def reset(p: Project):
                for ph in p.photos:
                    if ph.status in ("queued", "aligning"):
                        ph.status = "pending"

            self.store.mutate(project.id, reset, bump=False)
            self.request(project.id)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
