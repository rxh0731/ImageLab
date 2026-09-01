from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .processing import process_image


ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

app = FastAPI(title="ImageLab", version="0.1.0")
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ImageLab"}


@app.post("/api/process")
async def process(
    file: UploadFile = File(...),
    image_type: str = Form("rubbing"),
    mode: str = Form("conservative"),
    keep_faint: bool = Form(True),
) -> dict:
    allowed = {"image/jpeg", "image/png", "image/tiff", "image/webp", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=415, detail="仅支持 JPG、PNG、TIFF、WEBP 或 BMP 图片")
    suffix = Path(file.filename or "image.png").suffix.lower() or ".png"
    job_id = uuid.uuid4().hex[:12]
    source = UPLOADS / f"{job_id}{suffix}"
    with source.open("wb") as target:
        shutil.copyfileobj(file.file, target)
    try:
        result = process_image(source, OUTPUTS / job_id, image_type, mode, keep_faint)
    except Exception as exc:
        source.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"图片处理失败：{exc}") from exc
    result["job_id"] = job_id
    result["original"] = source.name
    result["output_base"] = f"/outputs/{job_id}/"
    (OUTPUTS / job_id / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")

