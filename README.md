# Stereo Depth Estimation & 3D Reconstruction from a Calibrated Camera Pair

**Course:** CSE3010 – Computer Vision (VIT) | **Type:** Build-Your-Own-Project (Flipped Course Evaluation)  
**Syllabus Mapping:** Module 2 – Depth Estimation and Multi-Camera Views · Indicative Experiments 2 (Camera Calibration), 3 (Projection), 4 (Depth map from stereo pair), 5 (3D model from stereo pair)

---

## 1. Overview
This project implements a classical (non–deep-learning) binocular stereo vision pipeline in Python. Taking a pair of horizontally offset images, it reconstructs a metric 3D point cloud of the scene, implementing the complete classical geometric chain taught in Module 2: single-camera calibration, stereo calibration and rectification, dense disparity estimation with SGBM and Left-Right filtering, depth computation, outlier cleaning, and 3D point cloud generation.

---

## 2. Features & Functional Modules
The pipeline is modularized into 10 single-responsibility Python modules in `src/`:

1. **Camera Calibration (`calibration.py`)** — Computes intrinsic matrices ($K$) and lens-distortion parameters from chessboard calibration grids (`cv2.findChessboardCorners` + `cv2.cornerSubPix` + `cv2.calibrateCamera`).
2. **Stereo Calibration & Rectification (`stereo_calibration.py`)** — Jointly calibrates camera pairs (rotation $R$, translation $T$, essential/fundamental matrices) and computes rectification maps so epipolar lines are horizontally aligned (`cv2.stereoCalibrate`, `cv2.stereoRectify`).
3. **Filtered Disparity Estimation (`disparity.py`)** — Computes dense pixel disparity using CPU-based Semi-Global Block Matching (`cv2.StereoSGBM`), reinforced with Left-Right consistency validation and 3x3 median residual filtering.
4. **Metric Depth Conversion (`depth.py`)** — Transforms pixel disparities into metric depth maps via $Z = \frac{f \cdot B}{d + doffs}$.
5. **3D Point Cloud Generation (`pointcloud.py`)** — Reprojects depth into metric 3D spatial coordinates ($X, Y, Z$), applies Z-percentile capping and statistical std-gating outlier filters, and exports binary `.ply` point cloud files (`cv2.reprojectImageTo3D`).
6. **Visualization & Headless Rendering (`visualize.py`)** — Renders 2D disparity/depth heatmaps and interactive/headless 3D matplotlib scatter point cloud previews.
7. **Middlebury Demo & Data Handler (`demo.py`)** — Manages automated fetching and execution for Middlebury 2014 benchmark scenes (Motorcycle, Adirondack, Jadeplant).
8. **Pipeline Orchestrator & CLI (`main.py`)** — Entry point providing CLI subcommands (`fetch`, `demo`, `calibrate`, `reconstruct`).
9. **Shared Utilities (`utils.py`)** — Configuration validation (`config.yaml`), image I/O, logging setup, and shape assertions.
10. **Error Handling (`errors.py`)** — Custom exception classes (e.g., `CalibrationError`).

---

## 3. Technologies Used
- **Python 3.10+**
- **OpenCV (`opencv-python`)** — Chessboard corner detection, camera calibration, stereo rectification, and SGBM stereo matching.
- **NumPy** — Vectorized matrix operations, depth calculations, and point cloud filtering.
- **Matplotlib** — Disparity/depth heatmaps and interactive/headless 3D point cloud visualization.
- **PyYAML** — Configuration file loading and parameter management.
- **pytest** — Comprehensive unit test suite (38 unit tests).

---

## 4. Project Structure
```
computerVision/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── src/
│   ├── calibration.py
│   ├── stereo_calibration.py
│   ├── disparity.py
│   ├── depth.py
│   ├── pointcloud.py
│   ├── visualize.py
│   ├── utils.py
│   ├── demo.py
│   ├── errors.py
│   └── main.py
├── tests/
│   ├── conftest.py
│   ├── test_calibration.py
│   ├── test_demo.py
│   ├── test_disparity.py
│   └── test_pointcloud.py
└── output/
    ├── calibration.npz
    ├── rectified_left.png / rectified_right.png
    ├── disparity.png
    ├── depth.npy
    ├── pointcloud.ply
    └── pointcloud_preview.png
```

---

## 5. Installation

```bash
# 1. Clone repository
git clone https://github.com/Ailurus98/computerVision.git
cd computerVision

# 2. Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 6. How to Run

The application provides four CLI subcommands via `src/main.py`:

### Option A: Portable Demo (Middlebury Datasets)

1. **One-Time Data Fetch (Offline Preparation):**
   ```bash
   python src/main.py fetch
   ```
   Downloads sample Middlebury 2014 benchmark scenes (`Motorcycle-perfect`, `Adirondack-perfect`, `Jadeplant-perfect`) and chessboard images into the local user cache.

2. **Run Interactive Demo:**
   ```bash
   python src/main.py demo --scene 1
   ```
   * `--scene 1`: Motorcycle (vehicle)
   * `--scene 2`: Adirondack (outdoor backyard)
   * `--scene 3`: Jadeplant (plant)

3. **Run Headless Mode:**
   ```bash
   python src/main.py demo --scene 1 --headless --output output/
   ```

### Option B: Custom Calibration & Reconstruction Pipeline

1. **Step 1 — Calibrate Binocular Cameras:**
   ```bash
   python src/main.py calibrate --config config/config.yaml
   ```
   Reads chessboard image pairs from paths specified in `config.yaml`, computes $K_1, K_2, dist_1, dist_2, R, T$, and saves `calibration.npz` to the output folder.

2. **Step 2 — Reconstruct Depth & Point Cloud:**
   ```bash
   python src/main.py reconstruct --left path/to/left.png --right path/to/right.png --config config/config.yaml
   ```
   Applies rectification maps, computes disparity and depth, and exports `pointcloud.ply`, `disparity.png`, `depth.npy`, and `pointcloud_preview.png`.

---

## 7. Testing

Run the full pytest suite (38 unit tests):

```bash
pytest tests/ -v
```

**Test Coverage Highlights:**
- Single & stereo camera calibration accuracy and minimum chessboard corner assertions.
- Stereo matching parameter validation and error handling for invalid input shapes/types.
- Point cloud Z-percentile capping, statistical outlier removal, and binary `.ply` export verification.
- Middlebury demo dataset caching and offline execution workflows.

---

## 8. Output Artifacts

Running the reconstruction pipeline populates `output/` with:
- `disparity.png`: Jet-colormap disparity map (pixels).
- `depth.npy`: 2D metric depth array (millimeters).
- `pointcloud.ply`: Binary Little-Endian 3D point cloud file (viewable in MeshLab, CloudCompare, or Open3D).
- `pointcloud_preview.png`: 3D scatter render preview.
- `steps/`: (Optional) Intermediate rectified left/right images and raw vs. filtered disparity maps.

---

## 9. Scope & Exclusions
This project is strictly focused on **classical geometric computer vision** (pinhole geometry, epipolar constraints, SGBM stereo matching). As defined in the project scope:
- **In-Scope:** Camera calibration, epipolar rectification, classical SGBM disparity estimation, metric depth reprojection, outlier filtering, PLY point cloud export.
- **Out-of-Scope:** Machine learning / deep learning stereo models (NeRF, StereoNet), real-time video processing, structure-from-motion (SfM), Poisson mesh generation.
