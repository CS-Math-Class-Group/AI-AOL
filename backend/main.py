from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, time
from backend.omr.scanner import run_omr, cleanup_omr_output
import uuid
import xmltodict


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FILE_TTL_SECONDS = 1800
file_store: dict[str, dict] = {}


def store_file(file_id: str, entry: dict) -> None:
    file_store[file_id] = {
        **entry,
        "expires_at": time.time() + FILE_TTL_SECONDS,
    }


def get_file(file_id: str) -> dict | None:
    entry = file_store.get(file_id)
    if not entry:
        return None
    if time.time() > entry["expires_at"]:
        cleanup_omr_output(entry["output_dir"])
        del file_store[file_id]
        return None
    return entry

@app.get("/")
async def root():
    return JSONResponse(content={
        "status_code": 200,
        "message": "AI-AOL API is running.",
    })


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1] or ".png"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_in:
        temp_in.write(await file.read())
        temp_in.flush()
        input_path = temp_in.name

    try:
        file_id = str(uuid.uuid4())
        result = await run_omr(input_path, job_id=file_id)

        if result["errors"]:
            raise HTTPException(status_code=422, detail=result["errors"])

        store_file(file_id, {
            "musicxml_path": result["musicxml_path"],
            "midi_path":     result["midi_path"],
            "output_dir":    result["output_dir"],
            "page_count":    result.get("page_count", 1),
            "pages":         result.get("pages", []),
        })

        return JSONResponse(content={
            "status_code":  200,
            "file_id":      file_id,
            "page_count":   result.get("page_count", 1),
            "has_musicxml": result["musicxml_path"] is not None,
            "has_midi":     result["midi_path"] is not None,
            "cached":       result["cached"],
            "pages": [
                {
                    "page":         p["page"],
                    "img_id":       p.get("img_id"),
                    "has_musicxml": p["musicxml_path"] is not None,
                    "has_midi":     p["midi_path"] is not None,
                    "cached":       p.get("cached", False),
                    "errors":       p["errors"],
                }
                for p in result.get("pages", [])
            ],
            "expires_in":   f"{FILE_TTL_SECONDS//60} minutes",
        })

    finally:
        os.remove(input_path)


@app.get("/midi/{file_id}")
async def stream_midi(file_id: str):
    entry = get_file(file_id)
    if not entry:
        raise HTTPException(status_code=404, detail="File ID not found or expired.")

    midi_path = entry.get("midi_path")
    if not midi_path or not os.path.exists(midi_path):
        raise HTTPException(status_code=404, detail="MIDI file not available.")

    def iter_file():
        with open(midi_path, "rb") as f:
            yield from f

    return StreamingResponse(
        iter_file(),
        media_type="audio/midi",
        headers={"Content-Disposition": "inline; filename=result.mid"},
    )


@app.get("/midi/{file_id}/{page}")
async def stream_midi_page(file_id: str, page: int):
    """Stream the MIDI for a specific page (1-based) of a multi-page upload."""
    entry = get_file(file_id)
    if not entry:
        raise HTTPException(status_code=404, detail="File ID not found or expired.")

    pages: list[dict] = entry.get("pages", [])
    page_entry = next((p for p in pages if p["page"] == page), None)
    if page_entry is None:
        raise HTTPException(status_code=404, detail=f"Page {page} not found.")

    midi_path = page_entry.get("midi_path")
    if not midi_path or not os.path.exists(midi_path):
        raise HTTPException(status_code=404, detail=f"MIDI for page {page} is not available.")

    def iter_file():
        with open(midi_path, "rb") as f:
            yield from f

    return StreamingResponse(
        iter_file(),
        media_type="audio/midi",
        headers={"Content-Disposition": f"inline; filename=page_{page}.mid"},
    )


@app.get("/musicxml/{file_id}")
async def stream_xml(file_id: str):
    entry = get_file(file_id)
    if not entry:
        raise HTTPException(status_code=404, detail="File ID not found or expired.")

    musicxml_path = entry.get("musicxml_path")
    if not musicxml_path or not os.path.exists(musicxml_path):
        raise HTTPException(status_code=404, detail="MusicXML file not available.")

    def iter_file():
        with open(musicxml_path, "rb") as f:
            yield from f

    return StreamingResponse(
        iter_file(),
        media_type="application/xml",
        headers={"Content-Disposition": "inline; filename=result.musicxml"},
    )


@app.get("/musicxml/{file_id}/{page}")
async def stream_xml_page(file_id: str, page: int):
    """Stream the MusicXML for a specific page (1-based) of a multi-page upload."""
    entry = get_file(file_id)
    if not entry:
        raise HTTPException(status_code=404, detail="File ID not found or expired.")

    pages: list[dict] = entry.get("pages", [])
    page_entry = next((p for p in pages if p["page"] == page), None)
    if page_entry is None:
        raise HTTPException(status_code=404, detail=f"Page {page} not found.")

    musicxml_path = page_entry.get("musicxml_path")
    if not musicxml_path or not os.path.exists(musicxml_path):
        raise HTTPException(status_code=404, detail=f"MusicXML for page {page} is not available.")

    def iter_file():
        with open(musicxml_path, "rb") as f:
            yield from f

    return StreamingResponse(
        iter_file(),
        media_type="application/xml",
        headers={"Content-Disposition": f"inline; filename=page_{page}.musicxml"},
    )

@app.get("/musicxml/json/{file_id}")
async def get_musicxml_json(file_id: str):
    entry = get_file(file_id)
    if not entry:
        raise HTTPException(status_code=404, detail="File ID not found or expired.")

    musicxml_path = entry.get("musicxml_path")
    if not musicxml_path or not os.path.exists(musicxml_path):
        raise HTTPException(status_code=404, detail="MusicXML file not available.")

    with open(musicxml_path, "r", encoding="utf-8") as f:
        xml_content = f.read()
        xml_dict = xmltodict.parse(xml_content)

    return JSONResponse(content=xml_dict)

@app.get("/download/midi/{file_id}")
async def download_midi(file_id: str):
    entry = get_file(file_id)
    if not entry:
        raise HTTPException(status_code=404, detail="File ID not found or expired.")

    midi_path = entry.get("midi_path")
    if not midi_path or not os.path.exists(midi_path):
        raise HTTPException(status_code=404, detail="MIDI file not available.")

    return FileResponse(midi_path, media_type="audio/midi", filename="result.mid")


@app.get("/download/musicxml/{file_id}")
async def download_musicxml(file_id: str):
    entry = get_file(file_id)
    if not entry:
        raise HTTPException(status_code=404, detail="File ID not found or expired.")

    musicxml_path = entry.get("musicxml_path")
    if not musicxml_path or not os.path.exists(musicxml_path):
        raise HTTPException(status_code=404, detail="MusicXML file not available.")

    return FileResponse(musicxml_path, media_type="application/xml", filename="result.musicxml")


@app.delete("/cleanup/{file_id}")
async def cleanup(file_id: str):
    entry = file_store.pop(file_id, None)
    if not entry:
        raise HTTPException(status_code=404, detail="File ID not found.")

    cleanup_omr_output(entry["output_dir"])
    return JSONResponse(content={"status_code": 200, "message": f"Cleaned up {file_id}."})
