import cv2
import numpy as np
import pytest

from src.disparity import compute_disparity
from src.depth import disparity_to_depth


def test_known_disparity(stereo_pair, settings):
    left, right = (cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in stereo_pair)
    result = compute_disparity(left, right, settings["stereo"])
    assert result.shape == left.shape
    assert result.dtype == np.float32
    assert np.median(result[:, 180:-30]) == pytest.approx(16., abs=0.5)


@pytest.mark.parametrize("value", [0, -16, 17, 1.5, True])
def test_invalid_num_disparities(value):
    image = np.zeros((100, 320), np.uint8)
    with pytest.raises(ValueError, match="positive multiple of 16"):
        compute_disparity(image, image, {"numDisparities": value})


def test_yaml_parameter_mapping(monkeypatch):
    captured = {}

    class Matcher:
        def compute(self, left, right):
            return np.full(left.shape, 32, np.int16)

    def create(**kwargs):
        captured.update(kwargs)
        return Matcher()

    monkeypatch.setattr(cv2, "StereoSGBM_create", create)
    image = np.zeros((60, 100), np.uint8)
    result = compute_disparity(image, image, {"num_disparities": 32, "block_size": 7})
    assert captured["numDisparities"] == 32
    assert captured["blockSize"] == 7
    assert captured["P1"] == 8 * 3 * 7 ** 2
    assert captured["P2"] == 32 * 3 * 7 ** 2
    assert np.all(result == 2.)


def test_depth_invalid_disparities():
    result = disparity_to_depth(np.array([[0., -1., 2.]], np.float32), 700., 60.)
    assert np.isinf(result[0, :2]).all()
    assert result[0, 2] == 21000.
    assert result.dtype == np.float32


@pytest.mark.parametrize("focal,baseline", [(0., 60.), (700., 0.), (np.nan, 60.)])
def test_depth_invalid_calibration(focal, baseline):
    with pytest.raises(ValueError):
        disparity_to_depth(np.ones((2, 2)), focal, baseline)
