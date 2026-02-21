from oemer.ete import extract
from argparse import Namespace
import asyncio
import shutil
import uuid
import os
from pathlib import Path
from music21 import converter
import cv2
import numpy as np
from pdf2image import convert_from_path

OMR_TEMP_BASE = "/tmp/aiaol_omr"
os.makedirs(OMR_TEMP_BASE, exist_ok=True)

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif"}



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



def _prepare_image(input_path: str, job_dir: str) -> str:
    """Convert PDF to PNG if needed, otherwise return original path.
    oemer expects raw, unmodified images — no preprocessing applied."""
    ext = os.path.splitext(input_path)[-1].lower()

    if ext == ".pdf":
        pages = convert_from_path(input_path, dpi=300)
        if not pages:
            raise ValueError(f"No pages found in PDF: {input_path}")
        png_path = os.path.join(job_dir, "page.png")
        img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
        cv2.imwrite(png_path, img)
        return png_path
    elif ext in SUPPORTED_IMAGE_EXTS:
        return input_path
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _run_extract(args: Namespace) -> None:
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
        img_path = await loop.run_in_executor(None, _prepare_image, input_path, job_dir)

        args = Namespace(
            img_path=img_path,
            output_path=job_dir,
            use_tf=False,           # use tensorflow or not
            save_cache=False,       # do not persist intermediate cache
            without_deskew=False,   # allow deskew correction
        )
        await loop.run_in_executor(None, _run_extract, args)

    # Locate oemer output
        musicxml_path = _find_file(job_dir, ".musicxml") or _find_file(job_dir, ".xml")

        if musicxml_path is None:
            return {
                "job_id": job_id,
                "musicxml_path": None,
                "midi_path": None,
                "output_dir": job_dir,
                "cached": False,
                "errors": "oemer did not produce a MusicXML output file.",
            }

    # Convert MusicXML → MIDI (music21 — blocking)
        midi_path = await loop.run_in_executor(
            None, _run_music21_convert, musicxml_path, job_dir
        )

        return {
            "job_id": job_id,
            "musicxml_path": musicxml_path,
            "midi_path": midi_path,
            "output_dir": job_dir,
            "cached": False,
            "errors": None,
        }

    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        return {
            "job_id": job_id,
            "musicxml_path": None,
            "midi_path": None,
            "output_dir": None,
            "cached": False,
            "errors": str(e),
        }



def cleanup_omr_output(output_dir: str) -> None:
    if output_dir and os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    return None