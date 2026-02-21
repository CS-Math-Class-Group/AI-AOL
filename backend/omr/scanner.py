from oemer.ete import extract
from oemer import staffline_extraction as _sle
from oemer import layers as _layers
from argparse import Namespace
import asyncio
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


# ---------------------------------------------------------------------------
# Monkey-patches for oemer bugs — oemer crashes on images where its NN
# predicts zero staff pixels in any zone.  Three functions are patched:
#
#   1. extract_line  — crashes with "index 0 out of bounds" (empty peaks)
#   2. init_zones    — crashes with "max() arg is empty" (no staff pixels)
#   3. extract       — top-level; guard against empty all_staffs
# ---------------------------------------------------------------------------

# --- patch extract_line ---------------------------------------------------

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



def _get_job_dir(job_id: str) -> str:
    return os.path.join(OMR_TEMP_BASE, job_id)


def _job_is_cached(job_id: str) -> bool:
    job_dir = _get_job_dir(job_id)
    has_midi = _find_file(job_dir, ".mid") is not None
    has_xml = (_find_file(job_dir, ".musicxml") or _find_file(job_dir, ".xml")) is not None
    return has_midi and has_xml


def _find_file(directory: str, extension: str) -> str | None:
    if not directory or not os.path.exists(directory):
        return None
    for path in Path(directory).rglob(f"*{extension}"):
        return str(path)
    return None



def _prepare_pages(input_path: str, job_dir: str) -> list[str]:
    """Convert every PDF page to a PNG and return the list of paths.
    For image files a single-element list is returned.
    oemer expects raw, unmodified images — no preprocessing applied."""
    ext = os.path.splitext(input_path)[-1].lower()

    if ext == ".pdf":
        pages = convert_from_path(input_path, dpi=300)
        if not pages:
            raise ValueError(f"No pages found in PDF: {input_path}")
        img_paths: list[str] = []
        for i, page in enumerate(pages):
            png_path = os.path.join(job_dir, f"page_{i + 1}.png")
            img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            cv2.imwrite(png_path, img)
            img_paths.append(png_path)
        return img_paths
    elif ext in SUPPORTED_IMAGE_EXTS:
        return [input_path]
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
    if job_id is None:
        job_id = str(uuid.uuid4())

    job_dir = _get_job_dir(job_id)

    # idempotency: return cached result if this job already finished
    if _job_is_cached(job_id):
        musicxml_path = _find_file(job_dir, ".musicxml") or _find_file(job_dir, ".xml")
        midi_path = _find_file(job_dir, ".mid")
        return {
            "job_id": job_id,
            "musicxml_path": musicxml_path,
            "midi_path": midi_path,
            "output_dir": job_dir,
            "cached": True,
            "errors": None,
        }

    os.makedirs(job_dir, exist_ok=True)
    loop = asyncio.get_event_loop()

    try:
        img_paths: list[str] = await loop.run_in_executor(
            None, _prepare_pages, input_path, job_dir
        )

        page_results: list[dict] = []
        for i, img_path in enumerate(img_paths):
            page_dir = os.path.join(job_dir, f"page_{i + 1}")
            os.makedirs(page_dir, exist_ok=True)
            args = Namespace(
                img_path=img_path,
                output_path=page_dir,
                use_tf=False,
                save_cache=False,
                without_deskew=False,
            )
            try:
                await loop.run_in_executor(None, _run_extract, args)
                musicxml_path = (
                    _find_file(page_dir, ".musicxml") or _find_file(page_dir, ".xml")
                )
                midi_path: str | None = None
                if musicxml_path:
                    midi_path = await loop.run_in_executor(
                        None, _run_music21_convert, musicxml_path, page_dir
                    )
                page_results.append(
                    {
                        "page": i + 1,
                        "musicxml_path": musicxml_path,
                        "midi_path": midi_path,
                        "errors": None,
                    }
                )
            except Exception as page_err:
                page_results.append(
                    {
                        "page": i + 1,
                        "musicxml_path": None,
                        "midi_path": None,
                        "errors": str(page_err),
                    }
                )

        successful = [p for p in page_results if p["musicxml_path"]]
        return {
            "job_id": job_id,
            "output_dir": job_dir,
            "page_count": len(img_paths),
            "pages": page_results,
            # convenience fields — first successful page for backwards compat
            "musicxml_path": successful[0]["musicxml_path"] if successful else None,
            "midi_path": successful[0]["midi_path"] if successful else None,
            "cached": False,
            "errors": None if successful else "No pages produced MusicXML output.",
        }

    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        return {
            "job_id": job_id,
            "musicxml_path": None,
            "midi_path": None,
            "output_dir": None,
            "page_count": 0,
            "pages": [],
            "cached": False,
            "errors": str(e),
        }



def cleanup_omr_output(output_dir: str) -> None:
    if output_dir and os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    return None