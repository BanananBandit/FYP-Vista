import tempfile
import subprocess

def sanitize_video_for_moviepy(in_path: str) -> str:
    """
    Normalise to a clean MP4 so MoviePy/ffmpeg can read iPhone MOVs reliably:
    - Keep first video stream and first audio stream only
    - Encode video to H.264 and audio to AAC
    - Drop extra/unknown streams (metadata, extra audio tracks etc.)
    """
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        out_path,
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tail = (p.stderr or "")[-2000:]
        raise RuntimeError("FFmpeg sanitize failed:\n" + tail)

    return out_path
