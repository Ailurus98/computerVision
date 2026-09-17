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
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {path}: {exc}") from exc
    required = {
        "chessboard": ("pattern_size", "square_size_mm"),
        "calibration": ("min_images", "left_images_dir", "right_images_dir"),
        "stereo": ("num_disparities", "block_size"),
        "paths": ("output_dir",),
    }
    for section, keys in required.items():
        if not isinstance(config, dict) or not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section {section}: {path}")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing configuration key {section}.{key}: {path}")
    minimum = config["calibration"]["min_images"]
    if type(minimum) is not int or minimum <= 0:
        raise ValueError("calibration.min_images must be a positive integer")
    return config


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
