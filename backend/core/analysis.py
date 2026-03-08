from typing import Optional, List

import cv2
import numpy as np
import pandas as pd

from models.segment import SegmentScore

def frame_brightness_bgr(frame_bgr: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    return float(np.mean(ycrcb[:, :, 0]))

def frame_blur_score_bgr(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def shake_score_optical_flow(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))

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

        brightness_vals, blur_vals, shake_vals = [], [], []
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
            gray_small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            if prev_gray is not None:
                shake_vals.append(shake_score_optical_flow(prev_gray, gray_small))
            prev_gray = gray_small
            t += sample_dt

        if not brightness_vals:
            mean_brightness = 0.0
            mean_blur = 0.0
            mean_shake = float("inf")
        else:
            mean_brightness = float(np.mean(brightness_vals))
            mean_blur = float(np.mean(blur_vals))
            mean_shake = float(np.mean(shake_vals)) if shake_vals else 0.0

        is_dark = mean_brightness < dark_thresh
        is_blurry = mean_blur < blur_thresh
        is_shaky = mean_shake > shake_thresh

        dark_remove_cut = dark_thresh * (1.0 - dark_extent_pct / 100.0)
        blur_remove_cut = blur_thresh * (1.0 - blur_extent_pct / 100.0)
        shake_remove_cut = shake_thresh * (1.0 + shake_extent_pct / 100.0)

        remove_dark = is_dark and (mean_brightness < dark_remove_cut)
        remove_blur = is_blurry and (mean_blur < blur_remove_cut)
        remove_shake = is_shaky and (mean_shake > shake_remove_cut)

        keep = not (remove_dark or remove_blur or remove_shake)

        reasons = []
        if remove_dark: reasons.append("too dark")
        if remove_blur: reasons.append("too blurry")
        if remove_shake: reasons.append("too shaky")
        reason = "KEEP" if keep else "REMOVE: " + " + ".join(reasons)

        rows.append(SegmentScore(
            t_start=float(t0), t_end=float(t1),
            brightness=mean_brightness, blur=mean_blur, shake=mean_shake,
            is_dark=is_dark, is_blurry=is_blurry, is_shaky=is_shaky,
            keep=keep, reason=reason
        ))

    cap.release()
    return pd.DataFrame([r.__dict__ for r in rows])
