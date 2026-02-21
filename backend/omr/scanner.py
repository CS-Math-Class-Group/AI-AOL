from oemer.ete import extract
from argparse import Namespace
import asyncio
import shutil
import uuid
import os
from pathlib import Path
from backend.omr.preprocessor import preprocess_image
from music21 import converter

OMR_TEMP_BASE = "/tmp/aiaol_omr"
os.makedirs(OMR_TEMP_BASE, exist_ok=True)



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



def _run_preprocess(input_path: str, preprocessed_path: str) -> None:
    preprocess_image(input_path, preprocessed_path)


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
    preprocessed_path = os.path.join(job_dir, "preprocessed.png")
    loop = asyncio.get_event_loop()

    try:
    # Preprocess image (cv2 — blocking I/O + CPU)
        await loop.run_in_executor(None, _run_preprocess, input_path, preprocessed_path)

    # Run oemer OMR engine (heavy CPU — must not block event loop)
        args = Namespace(
            img_path=preprocessed_path,
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