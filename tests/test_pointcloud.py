import numpy as np
import pytest

from src.pointcloud import build_point_cloud, reproject_to_3d, save_ply


def test_cloud_masks_nonfinite_and_invalid_disparity():
    points = np.ones((2, 3, 3), np.float32)
    points[0, 0, 2] = np.inf
    points[0, 1, 0] = np.nan
    colors = np.full((2, 3, 3), [10, 20, 30], np.uint8)
    disparity = np.array([[2., 2., 2.], [0., 1., 4.]], np.float32)
    cloud, rgb = build_point_cloud(points, colors, disparity)
    assert np.isfinite(cloud).all()
    assert len(cloud) == len(rgb) == 2
    np.testing.assert_array_equal(rgb, [[30, 20, 10], [30, 20, 10]])


def test_reprojection():
    Q = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                  [0., 0., 0., 700.], [0., 0., 1. / 60., 0.]])
    points = reproject_to_3d(np.full((2, 2), 10., np.float32), Q)
    assert points.shape == (2, 2, 3)
    np.testing.assert_allclose(points[:, :, 2], 4200.)


def test_ply_roundtrip(tmp_path):
    path = tmp_path / "cloud.ply"
    points = np.array([[1., 2., 3.], [4., 5., 6.]], np.float32)
    colors = np.array([[255, 0, 10], [20, 30, 40]], np.uint8)
    save_ply(str(path), points, colors)
    header, payload = path.read_bytes().split(b"end_header\n", 1)
    assert b"format binary_little_endian 1.0" in header
    assert b"element vertex 2" in header
    vertices = np.frombuffer(payload, dtype=[("xyz", "<f4", (3,)), ("rgb", "u1", (3,))])
    np.testing.assert_array_equal(vertices["xyz"], points)
    np.testing.assert_array_equal(vertices["rgb"], colors)


def test_cloud_shape_error():
    with pytest.raises(ValueError):
        build_point_cloud(np.zeros((2, 2, 3)), np.zeros((1, 2, 3)), np.zeros((2, 2)))


def test_ply_count_error(tmp_path):
    with pytest.raises(ValueError):
        save_ply(str(tmp_path / "bad.ply"), np.ones((2, 3)), np.ones((1, 3)))
