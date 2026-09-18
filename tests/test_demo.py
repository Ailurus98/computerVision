import io

import cv2
import numpy as np
import pytest
import yaml

from src import demo, main, visualize
from src.utils import load_config


def test_paths_relative_to_config(settings, tmp_path, monkeypatch):
    config = {key: dict(value) for key, value in settings.items()}
    config["paths"]["output_dir"] = "results"
    config["calibration"]["left_images_dir"] = "left"
    config["calibration"]["right_images_dir"] = "right"
    path = tmp_path / "portable.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)
    result = load_config(path)
    assert result["paths"]["output_dir"] == str(tmp_path / "results")
    assert result["calibration"]["left_images_dir"] == str(tmp_path / "left")


@pytest.fixture
def scene_folder(tmp_path):
    folder = tmp_path / "scenes" / "Fake-perfect"
    folder.mkdir(parents=True)
    image = np.full((1000, 1000, 3), 128, np.uint8)[:, :, ::-1]
    for filename in ("im0.png", "im1.png"):
        assert cv2.imwrite(str(folder / filename), image)
    (folder / "calib.txt").write_text(
        "cam0=[721.5377 0 609.5593; 0 721.5377 172.854; 0 0 1]\n"
        "doffs=131.111\nbaseline=193.001\nndisp=272\nwidth=1000\nheight=1000\n", encoding="utf-8")
    return folder

@pytest.mark.parametrize("scene", ["1", "2", "3"])
def test_scene_popup_order(scene, settings, scene_folder, tmp_path, monkeypatch):
    events = []
    folders = (tmp_path / "calibration" / "left", tmp_path / "calibration" / "right")
    for side in folders:
        side.mkdir(parents=True)
    monkeypatch.setattr(demo, "cache_directory", lambda: tmp_path)
    monkeypatch.setattr(demo, "prepare_scenes", lambda root: {demo.SCENES[scene][1]: scene_folder})
    monkeypatch.setattr(demo, "prepare_chessboards", lambda root: folders)
    monkeypatch.setattr(visualize, "enable_interactive", lambda: True)
    monkeypatch.setattr(visualize, "show_input_pair", lambda *args: events.append("photos"))
    monkeypatch.setattr(main, "run_calibrate", lambda *args: events.append("calibrate"))

    def reconstruct(*args, **kwargs):
        events.append("reconstruct")
        left = np.zeros((4, 4, 3), np.uint8)
        raw = np.full((4, 4), 8.0, np.float32)
        stages = {"left": left, "right": left, "raw": raw, "filtered": raw, "depth": raw}
        return np.ones((2, 3), np.float32), np.ones((2, 3), np.uint8), stages

    monkeypatch.setattr(main, "run_scene_reconstruction", reconstruct)
    monkeypatch.setattr(visualize, "show_intermediate_stages", lambda *args: events.append("stages"))
    monkeypatch.setattr(visualize, "show_point_cloud_interactive", lambda *args: events.append("3d"))
    config = {key: dict(value) for key, value in settings.items()}
    config["calibration"]["left_images_dir"] = str(folders[0])
    config["calibration"]["right_images_dir"] = str(folders[1])
    demo.run_demo(config, scene)
    assert events == ["photos", "calibrate", "reconstruct", "stages", "3d"]


def test_invalid_scene(settings):
    with pytest.raises(ValueError, match="1, 2, or 3"):
        demo.run_demo(settings, "4")


def test_no_arguments_selects_demo(monkeypatch):
    calls = []
    monkeypatch.setattr(demo, "run_demo", lambda *args: calls.append(args))
    assert main.main([]) == 0
    assert calls[0][1:] == (None, False, None)


def test_cache_without_network(chessboards, tmp_path, monkeypatch):
    folders = tmp_path / "calibration" / "left", tmp_path / "calibration" / "right"
    for source, folder in zip(chessboards, folders):
        folder.mkdir(parents=True)
        for index in (1, 2, 3):
            (folder / f"{index:02d}.jpg").write_bytes((source / f"{index:02d}.png").read_bytes())
    monkeypatch.setattr(demo, "INDICES", (1, 2, 3))
    monkeypatch.setattr(demo.urllib.request, "urlopen",
                        lambda *args, **kwargs: pytest.fail("Unexpected network request"))
    assert demo.prepare_chessboards(tmp_path) == folders


def test_empty_cloud_preview(tmp_path):
    path = tmp_path / "empty.png"
    visualize.show_point_cloud_preview(np.empty((0, 3)), np.empty((0, 3)), str(path))
    assert path.stat().st_size > 0


def test_scenes_offline_missing(tmp_path):
    with pytest.raises(OSError, match="python src/main.py fetch"):
        demo.prepare_scenes(tmp_path)


def test_scenes_offline_cached(tmp_path, monkeypatch):
    image = np.full((1000, 1000, 3), 128, np.uint8)
    for _, name in demo.SCENES.values():
        folder = tmp_path / "scenes" / name
        folder.mkdir(parents=True)
        assert cv2.imwrite(str(folder / "im0.png"), image)
        assert cv2.imwrite(str(folder / "im1.png"), image)
        (folder / "calib.txt").write_text(
            "cam0=[1000 0 500; 0 1000 500; 0 0 1]\n"
            "doffs=50\nbaseline=100\nndisp=64\nwidth=1000\nheight=1000\n", encoding="utf-8")
    monkeypatch.setattr(demo.urllib.request, "urlopen",
                        lambda *args, **kwargs: pytest.fail("Unexpected network request"))
    folders = demo.prepare_scenes(tmp_path)
    assert folders["Motorcycle-perfect"] == tmp_path / "scenes" / "Motorcycle-perfect"


def test_chessboards_offline_missing(tmp_path):
    with pytest.raises(OSError, match="python src/main.py fetch"):
        demo.prepare_chessboards(tmp_path)


def test_fetch_command_downloads_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(demo, "prepare_scenes", lambda root, allow_download=False: calls.append(("scenes", allow_download)) or {})
    monkeypatch.setattr(demo, "prepare_chessboards", lambda root, allow_download=False: calls.append(("boards", allow_download)) or (None, None))
    assert main.main(["fetch"]) == 0
    assert calls == [("scenes", True), ("boards", True)]
