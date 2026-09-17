from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config


@pytest.fixture(scope="session")
def settings():
    return load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")


@pytest.fixture(scope="session")
def chessboards(tmp_path_factory, settings):
    root = tmp_path_factory.mktemp("chessboards")
    left, right = root / "left", root / "right"
    left.mkdir()
    right.mkdir()
    cols, rows = settings["chessboard"]["pattern_size"]
    square = settings["chessboard"]["square_size_mm"]
    K = np.array([[700., 0., 320.], [0., 700., 240.], [0., 0., 1.]])
    rng = np.random.default_rng(10)
    for index in range(settings["calibration"]["min_images"] + 2):
        rotation = rng.uniform(-0.25, 0.25, 3)
        translation = np.array([-105., -65., 650.]) + rng.uniform(-20., 20., 3)
        for directory, offset in ((left, 0.), (right, -60.)):
            image = np.full((960, 1280), 180, np.uint8)
            for y in range(rows + 1):
                for x in range(cols + 1):
                    polygon = np.array([[x-1, y-1, 0], [x, y-1, 0],
                                        [x, y, 0], [x-1, y, 0]], np.float32) * square
                    pixels, _ = cv2.projectPoints(
                        polygon, rotation, translation + [offset, 0., 0.], K, None
                    )
                    pixels = np.round(pixels.reshape(-1, 2) * 2).astype(np.int32)
                    cv2.fillConvexPoly(image, pixels, 0 if (x + y) % 2 == 0 else 255)
            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
            assert cv2.imwrite(str(directory / f"{index:02d}.png"), image)
    return left, right


@pytest.fixture(scope="session")
def stereo_pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("stereo")
    rng = np.random.default_rng(23)
    left = rng.integers(0, 256, (480, 640), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-16] = left[:, 16:]
    paths = root / "left.png", root / "right.png"
    for path, image in zip(paths, (left, right)):
        assert cv2.imwrite(str(path), image)
    return paths
