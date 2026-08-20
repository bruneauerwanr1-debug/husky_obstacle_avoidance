# Husky Vision-Based Obstacle Avoidance & Reactive Navigation

Autonomous reactive obstacle avoidance, vision-based emergency stop, and comprehensive experimental evaluation suite developed for the **Clearpath Husky A200** mobile robot under **ROS 2**.

---

## 1. System Overview

This repository provides three vision-based obstacle perception and reactive navigation architectures designed for real-time robotic safety:

1. **CPU Dense Optical Flow (Farneback)** : Motion-field estimation, statistical ego-motion cancellation, 3-zone lateral balance, and predictive evasion.
2. **Deep Optical Flow (RAFT)** : High-precision recurrent optical flow on GPU, directional blob coherence filtering, and dynamic trajectory extrapolation.
3. **Predictive Object Detection (YOLOv26 + Ground Homography)** : Semantic obstacle classification, metric ground-plane projection ($Z=0$), Velocity Obstacle (VO) side selection, and Time-to-Collision (TTC) forecasting.

```
                       ┌────────────────────────────┐
                       │    RGB Camera (RealSense)  │
                       └─────────────┬──────────────┘
                                     │ /camera/color/image_raw
                                     ▼
                       ┌────────────────────────────┐
                       │    Obstacle Avoidance Node │
                       │ (Farneback / RAFT / YOLO)  │
                       └──────┬──────────────┬──────┘
  /avoidance_cmd_vel         │              │ /emergency_stop
  (Priority 50)              │              │ (Priority 255 Lock)
                             ▼              ▼
                       ┌────────────────────────────┐
/joy_teleop/cmd_vel ─►│         twist_mux          │
(Priority 10)         └─────────────┬──────────────┘
                                     │ /cmd_vel_in
                                     ▼
                       ┌────────────────────────────┐
                       │   Command Filter & Safety  │
                       └─────────────┬──────────────┘
                                     │ /cmd_vel
                                     ▼
                       ┌────────────────────────────┐
                       │    Husky Motor Base        │
                       └────────────────────────────┘
```

---

## 2. Quick Navigation

| Component | Directory | Description |
| :--- | :--- | :--- |
| 🚀 **Functional Deployment** | [`nodes/README.md`](file:///nodes/README.md) | Robot bringup, `twist_mux` setup, node execution, and algorithm summaries |
| 📊 **Experimental Evaluation** | [`evaluation/README.md`](file:///evaluation/README.md) | Benchmark protocol (S1.1–S5.3), statistical tests, and bag analysis |
| 📷 **Camera Distortion Analysis** | [`evaluation/camera_distortion/`](file:///evaluation/camera_distortion/README.md) | RealSense D435i optical quality, reprojection RMSE, and rectilinearity |
| 📐 **Homography Benchmark** | [`evaluation/homography_benchmark/`](file:///evaluation/homography_benchmark/README.md) | Ground-plane metric comparison ($H_{old}$ 4-tag vs $H_{new}$ 8-tag) |
| 📈 **Comparative Figures** | [`evaluation/plots/`](file:///evaluation/plots/) | Publication-ready comparative figures (Figures 01 to 13) |

---

## 3. Repository Structure

```text
husky_obstacle_avoidance/
├── .gitignore                          # Ignores __pycache__, logs, raw bags, and temporary artifacts
├── README.md                           # Main repository overview & quick links
│
├── config/                             # Robot configuration & calibration
│   ├── base.launch-collision.py        # Base bringup launch file with collision multiplexing
│   ├── homography_calibration.json     # Calibrated 8-AprilTag ground homography matrix
│   └── twist_mux.yaml                  # Priority & emergency lock configuration for twist_mux
│
├── launch/                             # ROS 2 unified launch files
│   └── avoidance.launch.py             # Unified launch file (farneback, raft, yolo, homography)
│
├── nodes/                              # Functional obstacle avoidance & calibration nodes
│   ├── README.md                       # Functional deployment guide
│   ├── optical_flow_farneback.py       # Farneback CPU dense optical flow avoidance node
│   ├── raft_avoidance_node.py          # RAFT GPU deep optical flow avoidance node
│   ├── raft_flow_core.py               # RAFT core inference & risk estimation module
│   ├── predictive_yolo.py              # YOLOv26 + Homography predictive distance avoidance node
│   └── homography_node.py              # Live AprilTag ground homography calibrator
│
└── evaluation/                         # Evaluation, benchmarking, camera & homography analysis
    ├── README.md                       # Evaluation guide & benchmark matrix (S1.1 to S5.3)
    │
    ├── bag_analysis/                   # Rosbag processing & comparative performance tools
    │   ├── extract_bag.py              # Extracts ROS 2 rosbags to synchronized CSVs
    │   ├── analyze_bag_csv.py          # Computes comparative metrics & generates figures 01-09
    │   ├── generate_scientific_deep_dive.py # Computes statistical tests (Mann-Whitney U) & figures 10-13
    │   ├── visualize_bag_analysis.py   # Interactive GUI & trajectory visualization dashboard
    │   ├── perf_monitor.py             # Real-time resource (CPU, GPU, latency) profiler
    │   └── split_trials.py             # Automatic trial segmentation for continuous recordings
    │
    ├── camera_distortion/              # Camera intrinsics, distortion & ground rectilinearity evaluation
    │   ├── README.md                   # Camera evaluation guide & scientific report
    │   ├── distortion_evaluator_node.py # Live ROS 2 node evaluating chessboard/tag distortion
    │   ├── homography_distortion_evaluator_node.py # Ground metric error evaluation (raw vs undistorted)
    │   ├── camera_analysis.py          # Standalone analysis & figure generator
    │   ├── run_camera_evaluation.sh    # Automated evaluation runner script
    │   ├── results/                    # Metric test logs & report
    │   └── plots/                      # Reprojection RMSE, rectilinearity & homography plots
    │
    ├── homography_benchmark/           # Homography model comparison ($H_{old}$ 4-tag vs $H_{new}$ 8-tag)
    │   ├── README.md                   # Homography benchmark report and usage
    │   ├── yolo_homography_benchmark_node.py # ROS 2 node comparing live detections under $H_{old}$ vs $H_{new}$
    │   ├── analyze_homography_benchmark.py   # Statistical analysis and figure generator
    │   ├── run_benchmark_analysis.sh   # Automated benchmark runner script
    │   ├── results/                    # Benchmark CSV datasets & logs
    │   └── plots/                      # BEV spatial error map, correlation scatter & boxplots
    │
    └── plots/                          # Consolidated experimental benchmark plots
        ├── schema_farnebackv9_en.png   # Pipeline architecture schema
        ├── comparatif_trials/          # 🔬 Micro-granularity (N=111 individual segmented trials, Figures 01-13)
        └── comparatif_bags/            # 📦 Macro-granularity (N=22 unsegmented rosbag sessions, Figures 01-10)
```

---

## 4. Quick Start

### 1. Robot Base Bringup
Ensure that `config/twist_mux.yaml` is copied to your robot's configuration, then execute:

```bash
cd SSM-Nav2/ssm-nav2
source install/setup.bash
ros2 launch husky_base base.launch-collision.py
```

### 2. Launch an Obstacle Avoidance Node
```bash
# Option A: Farneback Dense Optical Flow (CPU)
ros2 run husky_obstacle_avoidance optical_flow_farneback.py

# Option B: RAFT Deep Optical Flow (GPU)
ros2 run husky_obstacle_avoidance raft_avoidance_node.py

# Option C: YOLOv26 Predictive Avoidance (GPU)
ros2 run husky_obstacle_avoidance predictive_yolo.py

# Or via unified launch:
ros2 launch husky_obstacle_avoidance avoidance.launch.py method:=farneback
```

### 3. Run Benchmark Analysis
```bash
# Extract rosbag to CSV:
python3 evaluation/bag_analysis/extract_bag.py /path/to/bag_folder

# Compute benchmark metrics & statistical tests:
python3 evaluation/bag_analysis/analyze_bag_csv.py /path/to/extracted_csvs
python3 evaluation/bag_analysis/generate_scientific_deep_dive.py

# Launch interactive visualization dashboard:
python3 evaluation/bag_analysis/visualize_bag_analysis.py
```

---

## 5. Summary of Key Experimental Results

- **Safety & Clearance**: Both YOLO ($d_{min} = 1.05\text{ m}$) and Farneback ($d_{min} = 0.88\text{ m}$) maintain robust safety distances across standard obstacles (S1.1).
- **Domain Boundaries**: YOLO outperforms on semantic targets and pedestrian cut-ins (S2.1), while Optical Flow successfully handles out-of-vocabulary obstacles (S5.1) where YOLO fails.
- **Hardware Efficiency**: Farneback runs at **$18\text{--}20\text{ Hz}$ on CPU** with $0\text{ MB}$ VRAM, making it ideal for compute-constrained platforms. RAFT provides superior flow quality at $12\text{ Hz}$ on GPU.
- **Sensor Calibration**: The Intel RealSense D435i exhibits subpixel reprojection accuracy ($0.274\text{ px}$) and sub-centimeter ground homography ($2.96\text{ mm}$ error), confirming that runtime software undistortion is unnecessary.

---

## 6. License

This project is licensed under the **MIT License**.
