import os
import tempfile
from typing import Any, Dict, List, Tuple

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from core.analysis import analyse_video
from core.intervals import merge_boolean_runs, invert_intervals
from core.rendering import render_edited_video

app = FastAPI(title="VISTA API", version="1.0.0")

# Allow React dev server to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    # Stream to disk (avoid reading huge file into RAM)
    with open(tmp_path, "wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return tmp_path

def _build_keep_intervals(df: pd.DataFrame) -> List[Tuple[float, float]]:
    if df.empty:
        return []
    seg_len = float(df["t_end"].iloc[0] - df["t_start"].iloc[0])
    times = df["t_start"].tolist()
    remove_flags = (~df["keep"]).tolist()

    remove_intervals = merge_boolean_runs(times, remove_flags, seg_len)
    total_start = float(df["t_start"].min())
    total_end = float(df["t_end"].max())

    keep_intervals = invert_intervals(total_start, total_end, remove_intervals)
    return keep_intervals

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyse")
def analyse(
    video: UploadFile = File(...),
    sample_fps: float = Form(2.0),
    segment_seconds: float = Form(1.0),
    dark_thresh: float = Form(45.0),
    blur_thresh: float = Form(60.0),
    shake_thresh: float = Form(1.5),
    dark_extent_pct: float = Form(40.0),
    blur_extent_pct: float = Form(30.0),
    shake_extent_pct: float = Form(50.0),
) -> JSONResponse:
    try:
        in_path = _save_upload_to_temp(video)

        df = analyse_video(
            video_path=in_path,
            sample_fps=sample_fps,
            segment_seconds=segment_seconds,
            dark_thresh=dark_thresh,
            blur_thresh=blur_thresh,
            shake_thresh=shake_thresh,
            dark_extent_pct=dark_extent_pct,
            blur_extent_pct=blur_extent_pct,
            shake_extent_pct=shake_extent_pct,
        )

        keep_intervals = _build_keep_intervals(df)

        # Convert dataframe to JSON-safe types
        rows: List[Dict[str, Any]] = df.to_dict(orient="records")

        # Clean up input temp file
        try:
            os.remove(in_path)
        except OSError:
            pass

        return JSONResponse(
            {
                "rows": rows,
                "summary": {
                    "segments": int(len(df)),
                    "kept": int(df["keep"].sum()) if "keep" in df else 0,
                    "removed": int((~df["keep"]).sum()) if "keep" in df else 0,
                    "removed_pct": float((1 - df["keep"].mean()) * 100) if "keep" in df and len(df) else 0.0,
                },
                "keep_intervals": keep_intervals,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/render")
def render(
    video: UploadFile = File(...),
    sample_fps: float = Form(2.0),
    segment_seconds: float = Form(1.0),
    dark_thresh: float = Form(45.0),
    blur_thresh: float = Form(60.0),
    shake_thresh: float = Form(1.5),
    dark_extent_pct: float = Form(40.0),
    blur_extent_pct: float = Form(30.0),
    shake_extent_pct: float = Form(50.0),
) -> FileResponse:
    """
    Convenience endpoint: analyse + render in one call.
    Returns the cleaned MP4 as a file download.
    """
    in_path = None
    out_path = None

    try:
        in_path = _save_upload_to_temp(video)

        df = analyse_video(
            video_path=in_path,
            sample_fps=sample_fps,
            segment_seconds=segment_seconds,
            dark_thresh=dark_thresh,
            blur_thresh=blur_thresh,
            shake_thresh=shake_thresh,
            dark_extent_pct=dark_extent_pct,
            blur_extent_pct=blur_extent_pct,
            shake_extent_pct=shake_extent_pct,
        )
        keep_intervals = _build_keep_intervals(df)

        if not keep_intervals:
            raise RuntimeError("No usable segments found to keep.")

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        render_edited_video(
            video_path=in_path,
            keep_intervals=keep_intervals,
            out_path=out_path,
        )

        # We can't delete out_path before sending it; cleanup can be manual or left to OS temp cleanup
        return FileResponse(
            out_path,
            media_type="video/mp4",
            filename="vista_edited_output.mp4",
        )

    except Exception as e:
        # Cleanup
        if out_path and os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        if in_path and os.path.exists(in_path):
            try:
                os.remove(in_path)
            except OSError:
                pass
