"""Shared utilities: config loading, logging, image I/O, validation."""
import logging
import os

import cv2
import numpy as np
import yaml


def load_config(path: str) -> dict:
    """Load a YAML configuration file.

    Inputs: path to YAML file.
    Outputs: dict of configuration values.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a simple console format.

    Inputs: level string (e.g. "INFO", "DEBUG").
    Outputs: None.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_image_gray(path: str) -> np.ndarray:
    """Load an image as grayscale.

    Inputs: image file path.
    Outputs: 2D uint8 ndarray (H, W).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read image (unsupported/corrupt file): {path}")
    return img


def load_image_color(path: str) -> np.ndarray:
    """Load an image in BGR color.

    Inputs: image file path.
    Outputs: 3D uint8 ndarray (H, W, 3) in BGR order.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read image (unsupported/corrupt file): {path}")
    return img


def validate_image_pair_shapes(img_left: np.ndarray, img_right: np.ndarray) -> None:
    """Validate that left and right images have identical shapes.

    Inputs: two image arrays.
    Outputs: None; raises ValueError on shape mismatch.
    """
    if img_left.shape != img_right.shape:
        raise ValueError(
            f"Left/right image shapes differ: {img_left.shape} vs {img_right.shape}"
        )
