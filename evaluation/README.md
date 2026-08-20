# Obstacle Avoidance Benchmark & Experimental Evaluation Suite

This directory contains the experimental benchmarking protocols, rosbag analysis pipelines, statistical evaluation suites, camera distortion analyses, homography calibration comparisons, and publication-quality plots for the **Clearpath Husky A200** obstacle avoidance systems.

---

## 1. Experimental Scenarios Matrix (S1.1 — S5.3)

The benchmark systematically assesses the three avoidance paradigms:
- **P1**: YOLOv8 + Ground Homography + BotSort Tracking (GPU)
- **P2**: Farneback Dense Optical Flow (CPU)
- **P3**: RAFT Deep Optical Flow (GPU)

| Scenario | Scenario Name & Description | Variable / Stress Parameter | Scientific Hypothesis & Baseline Evaluation |
| :--- | :--- | :--- | :--- |
| **S1.1** | **Standard Geometric Obstacle**<br>Cardboard box / traffic cone in direct path | Active avoidance node | **Baseline Reference**: Evaluates minimum clearance margin $d_{min}$ and curvature regularity during steady-state evasion. |
| **S1.2** | **Thin Obstacle / Vertical Pole**<br>Narrow cylinder ($⌀5\text{ cm}$) | Object diameter / texture | **Spatial Resolution**: Tests isolated optical flow peak capture vs. YOLO bounding box regression on low-pixel-width targets. |
| **S1.3 ★** | **Dark / Featureless Surface**<br>Matte black foam board | Texture / reflectance | **Hypothesis $H_2$**: Expected failure of Farneback gradient approximation vs. robust depth/feature recovery by RAFT and YOLO. |
| **S2.1** | **Lateral Pedestrian Crossing (Cut-in)**<br>Pedestrian crossing corridor at $0.8\text{ m/s}$ | Target velocity ($0.8\text{ m/s}$) | **Dynamic Forecasting**: Evaluates Time-to-Collision (TTC) anticipation, tracking ID stability (BotSort), and timely deceleration. |
| **S2.2** | **Frontal Pedestrian Approach**<br>Pedestrian walking directly toward robot ($0.5\text{ m/s}$) | Approach velocity ($0.5\text{ m/s}$) | **Relative Motion Accuracy**: Tests TTC estimation accuracy and emergency stop triggering on dynamic head-on targets. |
| **S3.1 ★** | **Low Ambient Illumination**<br>Dim ambient lighting ($< 50\text{ lux}$) | Scene illuminance | **Illumination Robustness**: Assesses optical descriptor degradation and noise floor amplification under low-light conditions. |
| **S3.2** | **Textured Ground & Visual Clutter**<br>High-contrast floor patterns + AprilTags | Ground texture noise | **Ground Segmentation**: Validates the effectiveness of homography-based ground masking ($H$) and tag removal filters. |
| **S4.1 ★** | **High Robot Velocity**<br>Robot cruising at $v_x = 0.8\text{ m/s}$ | Forward speed $v_x$ | **Latency Impact**: Measures the effect of algorithmic and inference latency on stopping distance and safety margins at high speed. |
| **S4.2** | **In-Place Pure Rotation**<br>Operator spins robot ($\omega_z = 0.5\text{ rad/s}$) | Angular velocity $\omega_z$ | **Ego-Motion Compensation**: Verifies rotation bypass to ensure zero false-positive E-STOPs during in-place pivoting. |
| **S5.1 ★** | **Out-of-Vocabulary Obstacle**<br>Unusual partition, open door, blank foam | Class semantics | **Hypothesis $H_1$**: Tests semantic blind spots in YOLO (open-set failure) vs. universal geometric perception by optical flow. |
| **S5.2** | **Narrow Chicane Navigation**<br>Staggered obstacles with $1.1\text{ m}$ clearance | Spatial confinement | **Anti-Deadlock**: Assesses cumulative yaw ceiling ($\Delta\psi_{cum} \le 1.3\text{ rad}$) and avoidance without oscillatory jamming. |
| **S5.3** | **Sudden Occlusion & Disappearance**<br>Obstacle momentarily hidden or masked | Spatial continuity | **Memory Retention**: Evaluates 15-second tracking memory buffer and re-association stability upon reappearance. |

> ★ **Key Critical Benchmark Scenarios**: Directly test domain boundaries, failure modes, and comparative hypotheses ($H_1, H_2$).

---

## 2. Directory Structure & Two-Tier Evaluation Granularity

The evaluation figures are structured into two distinct analysis levels in `plots/`:

```text
evaluation/
├── README.md                          # This comprehensive evaluation guide
│
├── bag_analysis/                      # Rosbag processing & comparative performance tools
│   ├── extract_bag.py                 # Extracts ROS 2 rosbags into synchronized CSVs
│   ├── analyze_bag_csv.py             # Computes comparative metrics & generates figures 01-09
│   ├── generate_scientific_deep_dive.py # Computes statistical tests (Mann-Whitney U) & figures 10-13
│   ├── visualize_bag_analysis.py      # Interactive GUI & trajectory visualization dashboard
│   ├── perf_monitor.py                # Standalone CPU/GPU/VRAM/latency logger
│   └── split_trials.py                # Automated trial segmentation for continuous recordings
│
├── camera_distortion/                 # Camera intrinsics, distortion & ground rectilinearity evaluation
│   ├── README.md                      # Consolidated camera scientific report & test protocol
│   ├── distortion_evaluator_node.py   # Real-time ROS 2 node evaluating chessboard/tag distortion
│   ├── homography_distortion_evaluator_node.py # Ground metric accuracy evaluation (raw vs undistorted)
│   ├── camera_analysis.py             # Standalone evaluation & figure generator
│   ├── run_camera_evaluation.sh       # Automated evaluation execution script
│   ├── results/                       # Test logs & consolidated markdown reports
│   └── plots/                         # Reprojection RMSE, rectilinearity & homography plots
│
├── homography_benchmark/              # Ground homography model comparison (4-tag $H_{old}$ vs 8-tag $H_{new}$)
│   ├── README.md                      # Homography comparison report & metric analysis
│   ├── yolo_homography_benchmark_node.py # ROS 2 node comparing live detections under $H_{old}$ vs $H_{new}$
│   ├── analyze_homography_benchmark.py   # Statistical analysis & figure generator
│   ├── run_benchmark_analysis.sh      # Automated benchmark execution script
│   ├── results/                       # JSON calibration matrices & benchmark CSV logs
│   └── plots/                         # BEV spatial error map, correlation scatter & boxplots
│
└── plots/
    ├── schema_farnebackv9_en.png      # Pipeline architecture schema
    │
    ├── comparatif_trials/             # 🔬 MICRO-GRANULARITY (N = 111 Individual Trials)
    │   ├── 01_comparative_bars.png    # Safety & kinematics bar chart comparison across trials
    │   ├── 02_distance_scatter.png    # Minimum clearance vs approach velocity scatter
    │   ├── 03_ttc_scatter.png         # TTC vs approach velocity correlation
    │   ├── 04_heatmap.png             # Full 111-trial x Metric performance heatmap
    │   ├── 05_node_summary.png        # Radar / scorecard summary of node profiles
    │   ├── 07_resources_summary.png   # CPU, GPU, RAM, VRAM resource consumption
    │   ├── 08_depth_vs_detection_errors.png # Depth ground truth vs detection error
    │   ├── 09_distance_error_by_range.png # Metric error breakdown by distance bracket
    │   ├── 10_statistical_significance_tests.png # Mann-Whitney U test p-values & effect sizes
    │   ├── 11_kinematics_and_directional_bias.png # Turning bias (Left/Right) & angular jerk
    │   ├── 12_yolo_object_classes_analysis.png # COCO obstacle class distance & accuracy
    │   ├── 13_scenario_repeatability_stability.png # Inter-trial coefficient of variation (CV%)
    │   └── recap_trials.csv           # Master CSV with 111 segmented trial records
    │
    └── comparatif_bags/               # 📦 MACRO-GRANULARITY (N = 22 Full Recording Sessions)
        ├── 01_comparative_bars.png    # High-level session comparison
        ├── 02_distance_scatter.png    # Session-averaged distance scatter
        ├── 03_ttc_scatter.png         # Session-averaged TTC scatter
        ├── 04_heatmap.png             # 22-session aggregated heatmap
        ├── 05_node_summary.png        # Session-level node scorecard
        ├── 07_resources_summary.png   # Session-level computational footprint
        ├── 08_depth_vs_detection_errors.png # Session depth error comparison
        ├── 09_distance_error_by_range.png # Session distance bracket breakdown
        ├── 10_statistical_significance_tests.png # Session-level significance testing
        └── recap_bags.csv             # Master CSV with 22 unsegmented session records
```

---

## 3. Understanding the Two Granularity Levels

### A. Trial-Level Analysis (`comparatif_trials/`) — Micro-Granularity ($N=111$ Trials)
- **Methodology**: In practice, an operator records continuous rosbags that encompass multiple successive runs towards an obstacle (e.g., 3 to 6 approach attempts per session). Using `split_trials.py`, each continuous rosbag is automatically segmented into isolated trial windows based on forward velocity thresholds ($|v_x| > 0.03\text{ m/s}$) and stationary boundaries.
- **Scientific Value**:
  1. **Accurate Safety Metrics**: Measures the true minimum clearance distance ($d_{min}$) for each independent maneuver without averaging out stationary pause periods.
  2. **Statistical Power**: Expands the dataset to **$N=111$ independent trials**, providing the sample size required for valid non-parametric hypothesis testing (**Mann-Whitney U tests**, Figure 10).
  3. **Repeatability Assessment**: Enables computation of the Coefficient of Variation (**CV%**, Figure 13) across repeated runs of identical scenarios.
  4. **Kinematic Precision**: Isolates active avoidance duration, cumulative yaw ($\Delta\psi_{cum}$), and angular jerk RMS for each specific evasion curve (Figure 11).

### B. Bag-Level Analysis (`comparatif_bags/`) — Macro-Granularity ($N=22$ Sessions)
- **Methodology**: Evaluates each recorded rosbag file as a single uninterrupted timeline ($N=22$) without sub-segmentation.
- **Scientific Value**:
  1. **Continuous Endurance**: Assesses how the node behaves over an extended multi-run session (long-term memory buffer, cumulative CPU/RAM stability).
  2. **Quick Executive Overview**: Provides a concise, high-level summary per test file for rapid triage and sanity checking.

---

## 4. Primary Performance Indicators & Statistical Methods

### Quantitative Metrics
1. **Safety Clearance ($d_{min}$)**: Minimum Euclidean distance (m) recorded between the robot base and the obstacle throughout the maneuver.
2. **Time-to-Collision ($TTC$)**: Estimated time remaining before impact ($TTC = d / v_{rel}$), evaluated against ground truth depth.
3. **Motion Smoothness ($\Delta\psi_{cum}$ & Jerk RMS)**:
   - Cumulative yaw rotation: $\Delta\psi_{cum} = \int |\omega_z| dt$ (rad)
   - Angular Jerk RMS: $\sqrt{\frac{1}{T}\int (\frac{d\alpha_z}{dt})^2 dt}$ ($\text{rad/s}^2$)
4. **Computational Footprint**: Mean CPU load (%), RAM usage (MB), GPU compute (%), VRAM allocation (MB), and frame processing rate (FPS / Hz).

### Statistical Significance Testing
- **Non-Parametric Mann-Whitney U Test**: Two-sided hypothesis test evaluating whether performance differences between Farneback (CPU) and YOLO (GPU) are statistically significant ($p < 0.05^*$, $p < 0.01^{**}$, $p < 0.001^{***}$).
- **Repeatability (CV%)**: Coefficient of Variation across multiple runs per scenario:
  $$\text{CV} = \frac{\sigma}{\mu} \times 100\%$$

---

## 5. How to Run the Evaluation Tools

### Step 1: Recording a Benchmark Rosbag
On the robot, record all required telemetry topics during an experimental run:

```bash
ros2 bag record -o bag_P1_S1.1_01 \
  /odometry/filtered /cmd_vel /cmd_vel_in /avoidance_cmd_vel \
  /joy_teleop/cmd_vel /emergency_stop \
  /movement_command /optical_flow_zones \
  /ground_detections /flow_diagnostics \
  /camera/color/image_raw /camera/color/camera_info \
  /camera/aligned_depth_to_color/image_raw \
  /tf /tf_static
```

### Step 2: Extracting Rosbag Data to CSV
Extract raw topics from a recorded rosbag into synchronized CSV files:

```bash
python3 evaluation/bag_analysis/extract_bag.py /path/to/bag_P1_S1.1_01
```

### Step 3: Segmenting Continuous Sessions into Trials
Automatically detect and extract individual trial runs:

```bash
python3 evaluation/bag_analysis/split_trials.py /path/to/extracted_csv_folder --all
```

### Step 4: Generating Comparative Figures & Metrics
```bash
# Generate primary benchmark metrics and figures 01-09:
python3 evaluation/bag_analysis/analyze_bag_csv.py /path/to/bag_analysis_dir

# Generate statistical significance tests & deep-dive figures 10-13:
python3 evaluation/bag_analysis/generate_scientific_deep_dive.py
```

### Step 5: Interactive Trajectory & Telemetry Visualizer
Launch the interactive dashboard to inspect 2D trajectories, velocity profiles, and avoidance states:

```bash
python3 evaluation/bag_analysis/visualize_bag_analysis.py
```

---

## 6. Camera & Homography Benchmarks

### Camera Distortion Evaluation (`camera_distortion/`)
- **Objective**: Quantify RealSense D435i optical distortion, focal length stability, and evaluate whether real-time undistortion is necessary.
- **Key Result**: Factory intrinsics show subpixel reprojection error (**$0.274\text{ px}$**) and mean edge deviation of **$0.61\text{ px}$**. Real-time software undistortion provides an insignificant metric gain of only **$+2.29\%$ ($0.07\text{ mm}$)**, proving that running on raw frames saves CPU/GPU resources with zero loss of precision.

### Homography Matrix Comparison (`homography_benchmark/`)
- **Objective**: Compare baseline 4-tag homography ($H_{old}$) against regularized 8-tag calibration ($H_{new}$).
- **Key Result**: $H_{new}$ achieves sub-centimeter ground localization (**$2.96\text{ mm}$**) across the robot's driving corridor and eliminates the $40\text{ cm}$ perspective drift observed at distances $> 3\text{ m}$ with $H_{old}$.
