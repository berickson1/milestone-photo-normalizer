# Milestone Photo Normalizer

A self-hosted web app for baby milestone photos taken on the same milestone blanket. Upload the set, pick one photo as
the reference, and the app warps every photo to that perspective. It uses one shared crop, matches colour to the
reference, blacks out everything outside the blanket, and fills blanket areas missing from one photo using the others.

Photos never leave your machine. The originals are stored byte-for-byte and never changed.

## How it works

1. **Photos:** upload the set. The first photo becomes the reference; change it to the one with the flattest, most
   complete view of the blanket. Each photo is aligned automatically (SIFT features + MAGSAC homography), and
   uncertain alignments are flagged for review. Click a photo's name to rename it (e.g. "Month 3"), and drag photos
   (or use ◀ ▶, or **Sort by name**) to set their order.
2. **Blanket & crop:** the blanket outline and its four corners are proposed automatically. Drag them to fix them,
   and set the output width if you want something other than the default.
   - **Straighten perspective** (on by default) stretches the four corners into a true rectangle, removing camera
     skew. The blanket's real width ÷ height is estimated from the reference photo; type the real value if you know
     it. Turn it off to use a plain rectangular crop of the reference view instead.
   - **Wrinkles:**
     - *Leave as photographed:* only the flat (homography) alignment.
     - *Straighten the pattern:* each photo is also warped smoothly (optical flow, excluding the baby) so the printed
       pattern lines up with the average shape of all photos. Wrinkles that differ between photos flatten out.
     - *Clean composite:* the blanket is rebuilt from a per-pixel median of all straightened photos, which removes
       creases and their shadows. Only the baby and props come from each photo. This needs good baby masks; check
       them on the Review tab.
3. **Review:** for each photo:
   - **Result:** the final image. "Highlight rebuilt areas" tints each region by the photo it came from and hatches
     anything with no usable source.
   - **Compare to reference:** overlay or blink the aligned photo against the reference to spot misalignment.
   - **Baby mask:** areas covered by the baby (or hands, toys, shadows) are found by comparing photos. Masked areas are
     never used to rebuild other photos. Paint or erase to correct the mask.
   - **Fix alignment:** if automatic alignment fails or is wrong, click 4+ matching blanket points in the reference and
     the photo.
   - **Colour:** turn colour matching on or off, or reduce its strength.
   - **Sign-off:** photos with uncertain alignment or rebuilt areas must be marked as reviewed before export. If part of
     the blanket appears in no photo, accept leaving it black.
4. **Export:** download a ZIP of same-size PNGs in your chosen order, named `001_<name>.png` and so on, plus a
   `README.txt` listing each original filename and checksum.

Rebuilt areas only ever use real pixels from your other photos, warped into place and colour-matched locally. Nothing
is generated or invented.

## Running it

Requirements: Python 3.11+ and Node 20+.

```powershell
# Backend
cd backend
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python

# Frontend (built once; the backend serves it)
cd ..\frontend
npm install
npm run build

# Start
cd ..\backend
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

There is no login. Keep it bound to `127.0.0.1`, or put it behind a reverse proxy with authentication before exposing
it on a network.

For frontend development, run the backend as above and `npm run dev` in `frontend/`. Vite proxies `/api` to port 8000.

### Configuration

| Variable              | Default         | Meaning                                         |
| --------------------- | --------------- | ----------------------------------------------- |
| `MPN_DATA_DIR`        | `backend/data`  | Where projects, originals and masks are stored. |
| `MPN_MAX_JOBS`        | `2`             | Projects processed in parallel.                 |
| `MPN_MAX_UPLOAD_MB`   | `100`           | Per-file upload limit.                          |
| `MPN_FRONTEND_DIST`   | `frontend/dist` | Built frontend to serve.                        |

HEIC/HEIF photos are supported through `pillow-heif`.

### Tests

```powershell
cd backend
.venv\Scripts\python -m pytest -q
```

The tests use synthetic blanket photos with known camera geometry. They check alignment accuracy, masking, rebuilding
and export end to end.

## Limitations

- Wrinkle straightening corrects smooth distortions only. Sharp folds, and wrinkles under the baby, are
  interpolated from the surrounding blanket. The clean composite can show a soft patch where a photo's baby mask is
  too generous (that photo's own blanket is kept there).
- Baby detection compares photos against each other. It needs at least two photos covering a spot, and it can miss
  parts of the baby that look like the blanket. Check the masks for photos that donate rebuilt areas.
- Anything outside the blanket outline is blacked out, including parts of the baby that extend past the blanket.
- Colour matching is a single per-photo white-balance/exposure correction, not local relighting.
