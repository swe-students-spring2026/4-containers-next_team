# pylint: disable=no-member,too-many-locals
"""Preprocessing helpers to convert camera frames to sign_mnist-like input."""

from typing import Dict, Tuple

import cv2
import numpy as np


def _center_crop(frame: np.ndarray, crop_ratio: float = 0.65) -> np.ndarray:
    """Crop the center square region of the frame."""
    height, width = frame.shape[:2]
    side = min(height, width)
    crop_side = max(1, int(side * crop_ratio))

    center_y = height // 2
    center_x = width // 2

    top = max(center_y - crop_side // 2, 0)
    left = max(center_x - crop_side // 2, 0)
    bottom = min(top + crop_side, height)
    right = min(left + crop_side, width)

    return frame[top:bottom, left:right]


def _largest_foreground_region(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest external contour region."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return binary

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return cv2.bitwise_and(binary, mask)


def _pad_to_square(image: np.ndarray, pad_value: int = 0) -> np.ndarray:
    """Pad an image to a square canvas."""
    height, width = image.shape[:2]
    side = max(height, width)

    square = np.full((side, side), pad_value, dtype=np.uint8)

    y_offset = (side - height) // 2
    x_offset = (side - width) // 2
    square[y_offset : y_offset + height, x_offset : x_offset + width] = image

    return square


def _resize_and_center(binary: np.ndarray, output_size: int = 28) -> np.ndarray:
    """Resize foreground with aspect ratio preserved and center on output canvas."""
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((output_size, output_size), dtype=np.uint8)

    x, y, width, height = cv2.boundingRect(coords)
    roi = binary[y : y + height, x : x + width]

    max_inner = output_size - 4
    scale = min(max_inner / max(width, 1), max_inner / max(height, 1))
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((output_size, output_size), dtype=np.uint8)

    x_offset = (output_size - new_width) // 2
    y_offset = (output_size - new_height) // 2
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized

    return canvas


def preprocess_frame(
    frame: np.ndarray, invert: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Convert a raw camera frame into a hand-focused 28x28 image and pixel mapping."""
    crop = _center_crop(frame, crop_ratio=0.65)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio = float(np.mean(binary > 0))
    if white_ratio > 0.5:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    binary = _largest_foreground_region(binary)
    binary = _pad_to_square(binary, pad_value=0)
    gray_28 = _resize_and_center(binary, output_size=28)

    if invert:
        gray_28 = cv2.bitwise_not(gray_28)

    pixels = gray_28.flatten().astype(np.uint8)
    pixel_dict = {f"pixel{i + 1}": int(value) for i, value in enumerate(pixels)}

    return crop, gray_28, pixel_dict
