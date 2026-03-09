import numpy as np
import cv2

from core.analysis import (
    frame_brightness_bgr,
    frame_blur_score_bgr,
    shake_score_optical_flow,
)


def test_brightness_black_vs_white():
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    white = np.full((100, 100, 3), 255, dtype=np.uint8)

    black_score = frame_brightness_bgr(black)
    white_score = frame_brightness_bgr(white)

    assert white_score > black_score
    assert black_score < 10
    assert white_score > 240


def test_brightness_mid_grey():
    grey = np.full((100, 100, 3), 128, dtype=np.uint8)
    score = frame_brightness_bgr(grey)

    assert 120 <= score <= 136


def test_blur_score_sharp_vs_blurred():
    sharp = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(sharp, (50, 50), (150, 150), (255, 255, 255), -1)

    blurred = cv2.GaussianBlur(sharp, (21, 21), 0)

    sharp_score = frame_blur_score_bgr(sharp)
    blurred_score = frame_blur_score_bgr(blurred)

    assert sharp_score > blurred_score


def test_blur_score_uniform_image_is_low():
    flat = np.full((200, 200, 3), 127, dtype=np.uint8)
    score = frame_blur_score_bgr(flat)

    assert score < 1


def test_shake_score_identical_frames_is_near_zero():
    frame = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(frame, (60, 60), 20, 255, -1)

    score = shake_score_optical_flow(frame, frame)

    assert score >= 0
    assert score < 0.01


def test_shake_score_moved_frame_is_higher():
    frame1 = np.zeros((120, 120), dtype=np.uint8)
    frame2 = np.zeros((120, 120), dtype=np.uint8)

    cv2.circle(frame1, (50, 60), 20, 255, -1)
    cv2.circle(frame2, (70, 60), 20, 255, -1)

    still_score = shake_score_optical_flow(frame1, frame1)
    moved_score = shake_score_optical_flow(frame1, frame2)

    assert moved_score > still_score


def test_shake_score_small_motion_vs_large_motion():
    frame1 = np.zeros((120, 120), dtype=np.uint8)
    small_move = np.zeros((120, 120), dtype=np.uint8)
    large_move = np.zeros((120, 120), dtype=np.uint8)

    cv2.rectangle(frame1, (30, 40), (60, 70), 255, -1)
    cv2.rectangle(small_move, (35, 40), (65, 70), 255, -1)
    cv2.rectangle(large_move, (70, 40), (100, 70), 255, -1)

    small_score = shake_score_optical_flow(frame1, small_move)
    large_score = shake_score_optical_flow(frame1, large_move)

    assert large_score > small_score