from oemer.ete import extract
from oemer import staffline_extraction as _sle
from oemer import layers as _layers
from argparse import Namespace
import asyncio
import hashlib
import json
import shutil
import uuid
import os
import warnings
from pathlib import Path
from music21 import converter
import cv2
import numpy as np
from pdf2image import convert_from_path
from scipy.signal import find_peaks

OMR_TEMP_BASE = "/tmp/aiaol_omr"
os.makedirs(OMR_TEMP_BASE, exist_ok=True)

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif"}

# Name of the per-image cache manifest written after a successful OMR run.
_CACHE_MANIFEST = "cache.json"


# ---------------------------------------------------------------------------
# Monkey-patches for oemer bugs — oemer crashes on images where its NN
# predicts zero staff pixels in any zone.  Three functions are patched:
#
#   1. extract_line  — crashes with "index 0 out of bounds" (empty peaks)
#   2. init_zones    — crashes with "max() arg is empty" (no staff pixels)
#   3. extract       — top-level; guard against empty all_staffs
# ---------------------------------------------------------------------------

_original_extract_line = _sle.extract_line

def _safe_extract_line(pred, x_offset=0, line_threshold=0.8):
    sub_ys, _ = np.where(pred > 0)
    if len(sub_ys) == 0:
        return np.array([]), np.zeros(len(pred))

    count = np.zeros(len(pred), dtype=np.uint16)
    for y in sub_ys:
        count[y] += 1
    count_ext = np.insert(count, [0, len(count)], [0, 0])
    std = np.std(count_ext)
    if std == 0:
        return np.array([]), count.astype(np.float64)

    norm = (count_ext - np.mean(count_ext)) / std
    centers, _ = find_peaks(norm, height=line_threshold, distance=8, prominence=1)
    if len(centers) == 0:
        return np.array([]), norm[1:-1]

    return _original_extract_line(pred, x_offset=x_offset, line_threshold=line_threshold)

_sle.extract_line = _safe_extract_line

# --- patch init_zones -----------------------------------------------------

_original_init_zones = _sle.init_zones

def _safe_init_zones(staff_pred, splits=8):
    ys, xs = np.where(staff_pred > 0)
    if len(ys) == 0 or len(xs) == 0:
        raise ValueError("No staff pixels detected — the image may not contain readable music notation.")
    return _original_init_zones(staff_pred, splits=splits)

_sle.init_zones = _safe_init_zones

# --- patch staff_extract (top-level) --------------------------------------

_original_staff_extract = _sle.extract

def _safe_staff_extract(*args, **kwargs):
    staff_pred = _layers.get_layer('staff_pred')
    if np.sum(staff_pred) == 0:
        raise ValueError("No staff pixels detected — the image may not contain readable music notation.")
    return _original_staff_extract(*args, **kwargs)

_sle.extract = _safe_staff_extract


# ---------------------------------------------------------------------------
# Directory layout
#
#   /tmp/aiaol_omr/
#   └── <job_id>/                  ← one upload job
#       └── images/
#           ├── <img_id>/          ← one image / PDF-page container
#           │   ├── source.png     ← input image fed to oemer
#           │   ├── cache.json     ← manifest written on success
#           │   ├── result.musicxml
#           │   └── result.mid
#           └── <img_id>/
#               └── …
# ---------------------------------------------------------------------------

def _get_job_dir(job_id: str) -> str:
    """Root directory for a single upload job."""
    return os.path.join(OMR_TEMP_BASE, job_id)


def _get_images_dir(job_id: str) -> str:
    """Container for all per-image sub-directories."""
    return os.path.join(_get_job_dir(job_id), "images")


def _get_img_dir(job_id: str, img_id: str) -> str:
    """Isolated working directory for one image/page."""
    return os.path.join(_get_images_dir(job_id), img_id)


# ---------------------------------------------------------------------------
# Per-image cache manifest
# ---------------------------------------------------------------------------

def _write_img_cache(img_dir: str, source_hash: str,
                     musicxml_path: str | None, midi_path: str | None) -> None:
    """Persist a JSON manifest so future requests can skip re-processing."""
    manifest: dict = {
        "source_hash":   source_hash,
        "musicxml_path": musicxml_path,
        "midi_path":     midi_path,
    }
    with open(os.path.join(img_dir, _CACHE_MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=2)


def _read_img_cache(img_dir: str) -> dict | None:
    """Return the manifest dict when a valid cache entry exists, else None.

    Validity requires:
      - cache.json is present and parseable
      - both musicxml_path and midi_path still exist on disk
    """
    manifest_path = os.path.join(img_dir, _CACHE_MANIFEST)
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path) as fh:
            data = json.load(fh)
        musicxml_ok = bool(data.get("musicxml_path")) and os.path.exists(data["musicxml_path"])
        midi_ok     = bool(data.get("midi_path"))     and os.path.exists(data["midi_path"])
        if musicxml_ok and midi_ok:
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _hash_file(path: str) -> str:
    """Return a 16-char SHA-256 hex digest of a file (used as cache key)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Filesystem search helper
# ---------------------------------------------------------------------------

def _find_file(directory: str, extension: str) -> str | None:
    if not directory or not os.path.exists(directory):
        return None
    for path in Path(directory).rglob(f"*{extension}"):
        return str(path)
    return None


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def _prepare_pages(input_path: str, job_id: str) -> list[tuple[str, str]]:
    """Expand the input file into per-image (img_id, png_path) pairs.

    PDFs   → one pair per page, PNG written to its own img_id directory.
    Images → one pair; the original file path is used (no copy made).
    oemer expects raw, unmodified images — no preprocessing applied.
    """
    ext = os.path.splitext(input_path)[-1].lower()
    images_dir = _get_images_dir(job_id)
    os.makedirs(images_dir, exist_ok=True)

    if ext == ".pdf":
        pages = convert_from_path(input_path, dpi=300)
        if not pages:
            raise ValueError(f"No pages found in PDF: {input_path}")
        result: list[tuple[str, str]] = []
        for page in pages:
            img_id  = str(uuid.uuid4())
            img_dir = os.path.join(images_dir, img_id)
            os.makedirs(img_dir, exist_ok=True)
            png_path = os.path.join(img_dir, "source.png")
            img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            cv2.imwrite(png_path, img)
            result.append((img_id, png_path))
        return result

    elif ext in SUPPORTED_IMAGE_EXTS:
        # Single image: give it its own img_id dir but keep the original path.
        img_id  = str(uuid.uuid4())
        img_dir = os.path.join(images_dir, img_id)
        os.makedirs(img_dir, exist_ok=True)
        return [(img_id, input_path)]

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _run_extract(args: Namespace) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        extract(args)


def _run_music21_convert(musicxml_path: str, output_dir: str) -> str | None:
    try:
        score = converter.parse(musicxml_path)
        midi_path = os.path.join(output_dir, "result.mid")
        score.write("midi", fp=midi_path)
        return midi_path
    except Exception:
        return None



async def run_omr(input_path: str, job_id: str | None = None) -> dict:
    """Run OMR on *input_path*, returning per-image results.

    Each image / PDF page is processed in its own UUID container directory
    under <job_dir>/images/<img_id>/.  A cache.json manifest is written on
    success so that re-uploads of identical content skip oemer entirely.

    Returns
    -------
    dict with keys:
      job_id         – str
      output_dir     – str  (job root)
      page_count     – int
      pages          – list[dict]  (one per image/page)
        .page            – 1-based page number
        .img_id          – UUID of the image container directory
        .musicxml_path   – str | None
        .midi_path       – str | None
        .cached          – bool
        .errors          – str | None
      musicxml_path  – str | None  (first successful page, backwards compat)
      midi_path      – str | None  (first successful page, backwards compat)
      cached         – bool  (True only when every page was served from cache)
      errors         – str | None
    """
    if job_id is None:
        job_id = str(uuid.uuid4())

    job_dir = _get_job_dir(job_id)
    os.makedirs(job_dir, exist_ok=True)

    loop = asyncio.get_event_loop()

    try:
        # Expand the input file into (img_id, png_path) pairs.
        img_pairs: list[tuple[str, str]] = await loop.run_in_executor(
            None, _prepare_pages, input_path, job_id
        )

        page_results: list[dict] = []
        all_cached = True

        for page_num, (img_id, img_path) in enumerate(img_pairs, start=1):
            img_dir = _get_img_dir(job_id, img_id)

            # ── Per-image cache check ────────────────────────────────────
            cached_entry = _read_img_cache(img_dir)
            if cached_entry:
                page_results.append({
                    "page":          page_num,
                    "img_id":        img_id,
                    "musicxml_path": cached_entry["musicxml_path"],
                    "midi_path":     cached_entry["midi_path"],
                    "cached":        True,
                    "errors":        None,
                })
                continue  # skip oemer — already done

            all_cached = False  # at least one image needs processing

            # ── Run oemer ────────────────────────────────────────────────
            args = Namespace(
                img_path=img_path,
                output_path=img_dir,
                use_tf=False,
                save_cache=False,
                without_deskew=False,
            )
            try:
                await loop.run_in_executor(None, _run_extract, args)

                musicxml_path = (
                    _find_file(img_dir, ".musicxml") or _find_file(img_dir, ".xml")
                )
                midi_path: str | None = None
                if musicxml_path:
                    midi_path = await loop.run_in_executor(
                        None, _run_music21_convert, musicxml_path, img_dir
                    )

                # Hash the source image and persist the manifest.
                source_hash = await loop.run_in_executor(None, _hash_file, img_path)
                _write_img_cache(img_dir, source_hash, musicxml_path, midi_path)

                page_results.append({
                    "page":          page_num,
                    "img_id":        img_id,
                    "musicxml_path": musicxml_path,
                    "midi_path":     midi_path,
                    "cached":        False,
                    "errors":        None,
                })

            except Exception as page_err:
                # One failed page must not abort the remaining pages.
                page_results.append({
                    "page":          page_num,
                    "img_id":        img_id,
                    "musicxml_path": None,
                    "midi_path":     None,
                    "cached":        False,
                    "errors":        str(page_err),
                })

        successful = [p for p in page_results if p["musicxml_path"]]
        return {
            "job_id":        job_id,
            "output_dir":    job_dir,
            "page_count":    len(img_pairs),
            "pages":         page_results,
            # Convenience fields — first successful page (backwards compat).
            "musicxml_path": successful[0]["musicxml_path"] if successful else None,
            "midi_path":     successful[0]["midi_path"]     if successful else None,
            "cached":        all_cached,
            "errors":        None if successful else "No pages produced MusicXML output.",
        }

    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        return {
            "job_id":        job_id,
            "musicxml_path": None,
            "midi_path":     None,
            "output_dir":    None,
            "page_count":    0,
            "pages":         [],
            "cached":        False,
            "errors":        str(e),
        }



def cleanup_omr_output(output_dir: str) -> None:
    if output_dir and os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)