from pathlib import Path

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


@pytest.mark.parametrize("scene", ["1", "2", "3"])
def test_scene_popup_order(scene, settings, tmp_path, chessboards, monkeypatch):
    events = []
    monkeypatch.setattr(demo, "cache_directory", lambda: tmp_path)
    folders = tmp_path / "left", tmp_path / "right"
    for source, folder in zip(chessboards, folders):
        folder.mkdir()
        index = demo.SCENES[scene][1]
        (folder / f"{index}.jpg").write_bytes((source / f"{index}.png").read_bytes())
    monkeypatch.setattr(demo, "prepare_reference_data", lambda root: folders)
    monkeypatch.setattr(visualize, "enable_interactive", lambda: True)
    monkeypatch.setattr(visualize, "show_input_pair", lambda *args: events.append("photos"))
    monkeypatch.setattr(main, "run_calibrate", lambda *args: events.append("calibrate"))

    def reconstruct(*args):
        events.append("reconstruct")
        return np.ones((2, 3)), np.ones((2, 3), np.uint8)

    monkeypatch.setattr(main, "run_reconstruct", reconstruct)
    monkeypatch.setattr(visualize, "show_point_cloud_interactive", lambda *args: events.append("3d"))
    demo.run_demo(settings, scene)
    assert events == ["photos", "calibrate", "reconstruct", "3d"]


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
    assert demo.prepare_reference_data(tmp_path) == folders


def test_empty_cloud_preview(tmp_path):
    path = tmp_path / "empty.png"
    visualize.show_point_cloud_preview(np.empty((0, 3)), np.empty((0, 3)), str(path))
    assert path.stat().st_size > 0
