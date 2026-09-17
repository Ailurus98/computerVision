import logging

import cv2
import numpy as np
import pytest
import yaml

from src.calibration import calibrate_camera, collect_stereo_points, list_images
from src.errors import CalibrationError
from src.main import main
from src.stereo_calibration import stereo_calibrate
from src.utils import load_config, load_image_color, load_image_gray, validate_image_pair_shapes


def test_calibration_accuracy(chessboards, settings):
    paths = list_images(str(chessboards[0]))
    result = calibrate_camera(paths, tuple(settings["chessboard"]["pattern_size"]),
                              settings["chessboard"]["square_size_mm"])
    assert result["K"].shape == (3, 3)
    assert result["rms_error"] < 1.0
    assert result["image_size"] == (640, 480)


def test_insufficient_corners(chessboards, settings):
    paths = list_images(str(chessboards[0]))[:settings["calibration"]["min_images"] - 1]
    with pytest.raises(CalibrationError):
        calibrate_camera(paths, tuple(settings["chessboard"]["pattern_size"]),
                         settings["chessboard"]["square_size_mm"])


def test_missing_corners_logged(tmp_path, settings, caplog):
    path = tmp_path / "blank.png"
    assert cv2.imwrite(str(path), np.zeros((480, 640), np.uint8))
    with caplog.at_level(logging.WARNING), pytest.raises(CalibrationError):
        calibrate_camera([str(path)], tuple(settings["chessboard"]["pattern_size"]),
                         settings["chessboard"]["square_size_mm"])
    assert str(path) in caplog.text


@pytest.mark.parametrize("loader", [load_image_color, load_image_gray])
def test_image_io_errors(tmp_path, loader):
    missing = tmp_path / "missing.png"
    with pytest.raises(FileNotFoundError, match="missing.png"):
        loader(str(missing))
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(IOError, match="corrupt.png"):
        loader(str(corrupt))


def test_shape_error():
    with pytest.raises(ValueError):
        validate_image_pair_shapes(np.zeros((2, 3)), np.zeros((3, 2)))


def test_bad_config(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("chessboard: [", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_cli_missing_config(caplog, tmp_path):
    assert main(["calibrate", "--config", str(tmp_path / "missing.yaml")]) == 1
    assert "missing.yaml" in caplog.text
    assert "Traceback" not in caplog.text


def test_stereo_matching_and_units(chessboards, settings):
    paths_l, paths_r = (list_images(str(p)) for p in chessboards)
    pattern = tuple(settings["chessboard"]["pattern_size"])
    square = settings["chessboard"]["square_size_mm"]
    with pytest.raises(CalibrationError, match="matching filenames"):
        collect_stereo_points(paths_l, paths_r[:-1], pattern, square)
    obj, left, right, size = collect_stereo_points(paths_l, paths_r, pattern, square)
    K = np.array([[700., 0., 320.], [0., 700., 240.], [0., 0., 1.]])
    result = stereo_calibrate(K, np.zeros(5), K.copy(), np.zeros(5), obj, left, right, size)
    assert result["R"].shape == (3, 3)
    assert np.linalg.norm(result["T"]) == pytest.approx(60., abs=2.)


def test_cli_end_to_end(chessboards, stereo_pair, settings, tmp_path):
    config = {key: dict(value) for key, value in settings.items()}
    output = tmp_path / "output"
    config["paths"]["output_dir"] = str(output)
    config["calibration"]["left_images_dir"] = str(chessboards[0])
    config["calibration"]["right_images_dir"] = str(chessboards[1])
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["calibrate", "--config", str(path)]) == 0
    with np.load(output / "calibration.npz", allow_pickle=False) as archive:
        assert archive["rms_left"] < 1.
        assert archive["K_left"].shape == (3, 3)
    assert main(["reconstruct", "--left", str(stereo_pair[0]), "--right",
                 str(stereo_pair[1]), "--config", str(path)]) == 0
    for name in ("rectified_left.png", "rectified_right.png", "disparity.png",
                 "depth.npy", "pointcloud.ply", "pointcloud_preview.png"):
        assert (output / name).stat().st_size > 0
    depth = np.load(output / "depth.npy")
    assert depth.shape == (480, 640)
    assert depth.dtype == np.float32
