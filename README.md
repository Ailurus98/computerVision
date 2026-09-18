# Stereo Depth Estimation & 3D Reconstruction from a Calibrated Camera Pair

**Course:** CSE3010 – Computer Vision (VIT) | **Type:** Build-Your-Own-Project
**Syllabus mapping:** Module 2 – Depth Estimation and Multi-Camera Views · Indicative Experiments 2 (Camera Calibration), 3 (Projection), 4 (Depth map from stereo pair), 5 (3D model from stereo pair)

## 1. Overview
This project implements a classical (non–deep-learning) binocular stereo vision pipeline. It takes a pair of images captured by two horizontally-offset cameras and reconstructs a 3D point cloud of the scene, covering the full geometric pipeline taught in Module 2: single-camera calibration, stereo calibration, epipolar rectification, dense disparity estimation, and 3D reprojection.

The full scope, boundaries, and explicit exclusions of this project are defined in [`statement.md`](./statement.md) and, at implementation-spec level, in [`implementation.md`](./implementation.md). Read those before extending or grading this project — nothing outside that scope is in play here.

## 2. Features (Functional Modules)
1. **Camera Calibration** — computes intrinsic parameters and lens-distortion coefficients of each camera from multiple images of a planar chessboard pattern (`cv2.findChessboardCorners` + `cv2.calibrateCamera`).
2. **Stereo Calibration & Rectification** — jointly calibrates the camera pair (rotation `R`, translation `T`, essential/fundamental matrices) and rectifies both images so corresponding epipolar lines become horizontal (`cv2.stereoCalibrate`, `cv2.stereoRectify`).
3. **Disparity & Depth Estimation** — computes a dense disparity map with Semi-Global Block Matching (`cv2.StereoSGBM`) and converts it to metric depth using `Z = f·B / d`.
4. **3D Point Cloud Reconstruction & Visualization** — reprojects the depth map into 3D coordinates (`cv2.reprojectImageTo3D`), colors each point from the left image, exports a `.ply` file, and renders a preview.

## 3. Technologies / Tools Used
- Python 3.10+
- OpenCV (`opencv-python`, `opencv-contrib-python`) — calibration, rectification, stereo matching
- NumPy — array/matrix operations
- Matplotlib — 2D disparity/depth heatmaps and a static 3D point-cloud preview
- PyYAML — configuration file parsing
- pytest — unit tests
- *(optional, not required to run the tool)* Open3D — interactive point-cloud viewer

## 4. Project Structure
```
stereo-3d-reconstruction/
├── README.md
├── statement.md
├── PROJECT_REPORT.md
├── implementation.md
├── requirements.txt
├── config/
│   └── config.yaml
├── data/
│   ├── calibration/left/    # chessboard images, camera L
│   ├── calibration/right/   # chessboard images, camera R
│   └── stereo_pair/         # test left.png / right.png
├── src/
│   ├── calibration.py
│   ├── stereo_calibration.py
│   ├── disparity.py
│   ├── depth.py
│   ├── pointcloud.py
│   ├── visualize.py
│   ├── utils.py
│   └── main.py
├── tests/
│   ├── test_calibration.py
│   ├── test_disparity.py
│   └── test_pointcloud.py
└── output/
    ├── rectified_left.png / rectified_right.png
    ├── disparity.png
    ├── depth.npy
    └── pointcloud.ply
```

## 5. Installation
```bash
git clone <your-repo-url>
cd stereo-3d-reconstruction
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 6. How to Run
**Step 1 — Calibrate both cameras:**
```bash
python src/main.py calibrate --config config/config.yaml
```

**Step 2 — Run the full stereo → depth → point cloud pipeline:**
```bash
python src/main.py reconstruct --left data/stereo_pair/left.png --right data/stereo_pair/right.png --config config/config.yaml
```

Outputs are written to `output/`. `pointcloud.ply` can be opened in MeshLab, CloudCompare, or Open3D.

## 7. Testing
```bash
pytest tests/ -v
```
Covers: chessboard corner detection on sample images, disparity map shape/range sanity checks, and point-cloud generation (no NaNs/Infs, correct point count).

## 8. Screenshots / Sample Output
*(To be added after running the pipeline on the sample dataset — see `PROJECT_REPORT.md`, Section 10, for what each image should show.)*

## 9. Scope Note
This is a **classical geometric computer-vision project** (calibration, epipolar geometry, block-matching stereo). It does **not** include deep learning, video/real-time processing, object detection/recognition, or a GUI. See `statement.md` for the complete in-scope / out-of-scope list.
