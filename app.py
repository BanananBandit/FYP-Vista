import os
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from moviepy import VideoFileClip, concatenate_videoclips



# ----------------------------
# Metrics
# ----------------------------
def frame_brightness_bgr(frame_bgr: np.ndarray) -> float:
    """Mean brightness in [0..255] via Y (luma)."""
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    return float(np.mean(y))


def frame_blur_score_bgr(frame_bgr: np.ndarray) -> float:
    """Blur score via variance of Laplacian (higher = sharper)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def shake_score_optical_flow(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """
    Camera shake proxy: mean optical flow magnitude between consecutive frames.
    Higher = more motion (potential shake).
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


@dataclass
class SegmentScore:
    t_start: float
    t_end: float
    brightness: float
    blur: float
    shake: float
    is_dark: bool
    is_blurry: bool
    is_shaky: bool
    keep: bool
    reason: str


def merge_boolean_runs(times: List[float], flags: List[bool], seg_len: float) -> List[Tuple[float, float]]:
    merged = []
    run_start = None
    for i, flag in enumerate(flags):
        t0 = times[i]
        if flag and run_start is None:
            run_start = t0
        if (not flag) and run_start is not None:
            merged.append((run_start, t0))
            run_start = None
    if run_start is not None and times:
        merged.append((run_start, times[-1] + seg_len))
    merged = [(max(0.0, a), max(0.0, b)) for a, b in merged if b > a]
    return merged


def invert_intervals(total_start: float, total_end: float, remove: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if total_end <= total_start:
        return []
    remove_sorted = sorted((max(total_start, a), min(total_end, b)) for a, b in remove)
    remove_sorted = [(a, b) for a, b in remove_sorted if b > a]

    keep = []
    cur = total_start
    for a, b in remove_sorted:
        if a > cur:
            keep.append((cur, a))
        cur = max(cur, b)
    if cur < total_end:
        keep.append((cur, total_end))

    # Drop tiny segments (avoid awkward cuts)
    keep = [(a, b) for a, b in keep if (b - a) > 0.15]
    return keep


def analyse_video(
    video_path: str,
    sample_fps: float,
    segment_seconds: float,
    dark_thresh: float,
    blur_thresh: float,
    shake_thresh: float,
    dark_extent_pct: float,
    blur_extent_pct: float,
    shake_extent_pct: float,
) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = float(frame_count / fps) if frame_count > 0 else 0.0
    if duration <= 0.1:
        raise RuntimeError("Video duration looks too small or unreadable.")

    sample_dt = 1.0 / max(0.1, sample_fps)
    seg_len = max(0.5, float(segment_seconds))

    seg_starts = np.arange(0.0, duration, seg_len).tolist()

    rows: List[SegmentScore] = []

    for t0 in seg_starts:
        t1 = min(duration, t0 + seg_len)

        brightness_vals = []
        blur_vals = []
        shake_vals = []

        prev_gray: Optional[np.ndarray] = None

        t = t0
        while t < t1:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                t += sample_dt
                continue

            brightness_vals.append(frame_brightness_bgr(frame))
            blur_vals.append(frame_blur_score_bgr(frame))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Downscale to make optical flow faster + more stable
            gray_small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            if prev_gray is not None:
                shake_vals.append(shake_score_optical_flow(prev_gray, gray_small))
            prev_gray = gray_small

            t += sample_dt

        if len(brightness_vals) == 0:
            mean_brightness = 0.0
            mean_blur = 0.0
            mean_shake = float("inf")  # treat as unusable (no readable frames)
        else:
            mean_brightness = float(np.mean(brightness_vals))
            mean_blur = float(np.mean(blur_vals))
            mean_shake = float(np.mean(shake_vals)) if len(shake_vals) else 0.0

        # Flags based on thresholds
        is_dark = mean_brightness < dark_thresh
        is_blurry = mean_blur < blur_thresh
        is_shaky = mean_shake > shake_thresh  # NOTE: higher shake score is worse

        # "To an extent" knobs:
        # - for dark & blur (lower is worse): only remove if far beyond threshold
        # - for shake (higher is worse): only remove if far beyond threshold
        dark_remove_cut = dark_thresh * (1.0 - dark_extent_pct / 100.0)
        blur_remove_cut = blur_thresh * (1.0 - blur_extent_pct / 100.0)
        shake_remove_cut = shake_thresh * (1.0 + shake_extent_pct / 100.0)

        remove_dark = is_dark and (mean_brightness < dark_remove_cut)
        remove_blur = is_blurry and (mean_blur < blur_remove_cut)
        remove_shake = is_shaky and (mean_shake > shake_remove_cut)

        keep = not (remove_dark or remove_blur or remove_shake)

        reasons = []
        if remove_dark:
            reasons.append("too dark")
        if remove_blur:
            reasons.append("too blurry")
        if remove_shake:
            reasons.append("too shaky")

        reason = "KEEP" if keep else ("REMOVE: " + " + ".join(reasons))

        rows.append(
            SegmentScore(
                t_start=float(t0),
                t_end=float(t1),
                brightness=mean_brightness,
                blur=mean_blur,
                shake=mean_shake,
                is_dark=is_dark,
                is_blurry=is_blurry,
                is_shaky=is_shaky,
                keep=keep,
                reason=reason,
            )
        )

    cap.release()
    return pd.DataFrame([r.__dict__ for r in rows])


def render_edited_video(video_path: str, keep_intervals: List[Tuple[float, float]], out_path: str) -> None:
    clip = VideoFileClip(video_path)

    subclips = []
    for a, b in keep_intervals:
        a2 = max(0.0, min(a, clip.duration))
        b2 = max(0.0, min(b, clip.duration))
        if b2 - a2 > 0.15:
            subclips.append(clip.subclip(a2, b2))

    if not subclips:
        raise RuntimeError("No usable segments found to keep.")

    final = concatenate_videoclips(subclips, method="compose")
    final.write_videofile(
        out_path,
        codec="libx264",
        audio_codec="aac",
        fps=clip.fps,
        threads=2,
        verbose=False,
        logger=None,
    )
    clip.close()
    final.close()


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Video Quality Cleaner Demo", layout="wide")
st.title("Localhost Website Demo: Dark + Blur + Shake Detection → Edited Video")

st.markdown(
    """
This is a computer-vision demo: we sample frames, compute quality metrics (brightness, blur, camera shake),
remove unusable segments, and export a cleaned video.
"""
)

uploaded = st.file_uploader("Upload video (mp4/mov/m4v)", type=["mp4", "mov", "m4v"])

colA, colB = st.columns([1, 1])
with colA:
    st.subheader("Analysis settings")
    sample_fps = st.slider("Sample rate (fps)", 0.5, 10.0, 2.0, 0.5)
    segment_seconds = st.slider("Segment length (seconds)", 0.5, 5.0, 1.0, 0.5)

    dark_thresh = st.slider("Darkness threshold (mean brightness)", 0.0, 255.0, 45.0, 1.0)
    blur_thresh = st.slider("Blur threshold (Laplacian variance)", 0.0, 500.0, 60.0, 1.0)
    shake_thresh = st.slider("Shake threshold (optical flow magnitude)", 0.0, 10.0, 1.5, 0.1)

    st.caption("Brightness ↑ better. Blur score ↑ sharper. Shake score ↑ worse (more motion).")

with colB:
    st.subheader('"To an extent" removal controls')
    dark_extent = st.slider("Dark extent (%) — remove only if *very* dark", 0, 90, 40, 5)
    blur_extent = st.slider("Blur extent (%) — remove only if *very* blurry", 0, 90, 30, 5)
    shake_extent = st.slider("Shake extent (%) — remove only if *very* shaky", 0, 200, 50, 10)

analyse_btn = st.button("Analyse video", type="primary", disabled=(uploaded is None))

if uploaded is not None:
    if "tmp_in" not in st.session_state:
        st.session_state.tmp_in = None

    if st.session_state.tmp_in is None or st.session_state.get("uploaded_name") != uploaded.name:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tmp.write(uploaded.read())
        tmp.flush()
        tmp.close()

        st.session_state.tmp_in = tmp.name
        st.session_state.uploaded_name = uploaded.name
        st.session_state.analysis_df = None
        st.session_state.keep_intervals = None
        st.session_state.tmp_out = None

    st.video(st.session_state.tmp_in)

if analyse_btn and uploaded is not None:
    with st.spinner("Analysing…"):
        df = analyse_video(
            st.session_state.tmp_in,
            sample_fps=sample_fps,
            segment_seconds=segment_seconds,
            dark_thresh=dark_thresh,
            blur_thresh=blur_thresh,
            shake_thresh=shake_thresh,
            dark_extent_pct=dark_extent,
            blur_extent_pct=blur_extent,
            shake_extent_pct=shake_extent,
        )
    st.session_state.analysis_df = df

if st.session_state.get("analysis_df") is not None:
    df = st.session_state.analysis_df

    st.subheader("Results")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Segments", int(len(df)))
    k2.metric("Kept", int(df["keep"].sum()))
    k3.metric("Removed", int((~df["keep"]).sum()))
    k4.metric("Removed %", f"{(1 - df['keep'].mean())*100:.1f}%")

    st.dataframe(
        df[["t_start", "t_end", "brightness", "blur", "shake", "keep", "reason"]],
        use_container_width=True
    )

    seg_len = float(df["t_end"].iloc[0] - df["t_start"].iloc[0]) if len(df) else 1.0
    times = df["t_start"].tolist()
    remove_flags = (~df["keep"]).tolist()
    remove_intervals = merge_boolean_runs(times, remove_flags, seg_len)

    total_start = float(df["t_start"].min())
    total_end = float(df["t_end"].max())
    keep_intervals = invert_intervals(total_start, total_end, remove_intervals)

    st.session_state.keep_intervals = keep_intervals

    with st.expander("Keep intervals (what remains in the edited video)"):
        st.write(keep_intervals)

    st.download_button(
        "Download analysis report (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="video_quality_report.csv",
        mime="text/csv",
    )

    st.subheader("Render edited video")
    render_btn = st.button("Render edited output", disabled=(st.session_state.keep_intervals is None))

    if render_btn:
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out_file.close()
        st.session_state.tmp_out = out_file.name

        with st.spinner("Rendering…"):
            render_edited_video(st.session_state.tmp_in, st.session_state.keep_intervals, st.session_state.tmp_out)

    if st.session_state.get("tmp_out") is not None and os.path.exists(st.session_state.tmp_out):
        st.success("Edited video created.")
        st.video(st.session_state.tmp_out)

        with open(st.session_state.tmp_out, "rb") as f:
            st.download_button(
                "Download edited video (MP4)",
                data=f.read(),
                file_name="edited_output.mp4",
                mime="video/mp4",
            )
