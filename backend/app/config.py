import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("MPN_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
MAX_CONCURRENT_JOBS = max(1, int(os.environ.get("MPN_MAX_JOBS", "2")))
MAX_UPLOAD_BYTES = int(os.environ.get("MPN_MAX_UPLOAD_MB", "100")) * 1024 * 1024
FRONTEND_DIST = Path(
    os.environ.get("MPN_FRONTEND_DIST", Path(__file__).resolve().parents[2] / "frontend" / "dist")
)

# Longest side of the cached working copy used for matching and previews.
WORK_MAX_DIM = 1600
THUMB_MAX_DIM = 320
# Extra space around the reference frame in the analysis canvas so the blanket
# and crop may extend past the edges of the reference photo.
ANALYSIS_MARGIN = 0.15
PREVIEW_WIDTH = 640
MAX_OUTPUT_WIDTH = 8000
