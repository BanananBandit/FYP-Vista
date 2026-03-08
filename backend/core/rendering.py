import os
import tempfile
import subprocess
from typing import List, Tuple

from core.preprocessing import sanitize_video_for_moviepy


def run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "FFmpeg failed")[-3000:])


def render_edited_video(video_path: str, keep_intervals: List[Tuple[float, float]], out_path: str) -> None:
    """
    Faster render path:
    1. Sanitize input to a standard MP4
    2. Extract each keep interval with ffmpeg
    3. Concatenate all kept clips with ffmpeg
    """
    if not keep_intervals:
        raise RuntimeError("No usable segments found to keep.")

    safe_input = sanitize_video_for_moviepy(video_path)

    temp_dir = tempfile.mkdtemp(prefix="vista_render_")
    segment_paths = []

    try:
        for i, (start, end) in enumerate(keep_intervals):
            duration = end - start
            if duration <= 0.15:
                continue

            seg_path = os.path.join(temp_dir, f"segment_{i:03d}.mp4")

            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(start),
                "-i", safe_input,
                "-t", str(duration),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "veryfast",
                "-movflags", "+faststart",
                seg_path,
            ]
            run_ffmpeg(cmd)
            segment_paths.append(seg_path)

        if not segment_paths:
            raise RuntimeError("No valid kept segments were produced.")

        concat_file = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for path in segment_paths:
                f.write(f"file '{path}'\n")

        concat_cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            out_path,
        ]
        run_ffmpeg(concat_cmd)

    finally:
        # optional cleanup can stay simple for now
        pass