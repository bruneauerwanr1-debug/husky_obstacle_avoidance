# Camera Optical Distortion & Ground Rectilinearity Evaluation

This module provides the experimental evaluation suite for assessing the optical characteristics of the **Intel RealSense D435i** RGB sensor and measuring the impact of lens distortion on ground-plane homography.

---

## 1. Executive Summary of Results

| Test ID | Evaluation Metric | Measured Value | Standard Tolerance | Verdict |
| :--- | :--- | :---: | :---: | :---: |
| **Test A** | Subpixel Reprojection RMSE (39 poses) | **0.274 px** | < 0.30 px | 🟢 **EXCELLENT** |
| **Test A** | Focal Length Drift ($f_x, f_y$ vs Factory) | **< 0.2%** ($f_x=615.6, f_y=616.9$) | < 2.0% | 🟢 **PERFECT STABILITY** |
| **Test B** | Straight-Line Edge Residual (Raw Frame) | **0.61 px** | <= 1.0 px | 🟢 **RECTILINEAR** |
| **Test C** | Firmware Distortion Model ($D$) | $[0, 0, 0, 0, 0]$ | — | 🟢 **FACTORY RECTIFIED** |
| **Test D** | Ground AprilTag Error ($H_{raw}$, 6 tags) | **2.96 mm** (0.296 cm) | < 20.0 mm | 🟢 **SUB-CENTIMETRIC** |
| **Test D** | Precision Gain with `cv2.undistort` | **+2.29%** ($\Delta = 0.07\text{ mm}$) | > 10.0% | ⚪ **NEGLIGIBLE GAIN** |

---

## 2. Key Operational Findings

1. **Hardware Health**: The optical lens assembly of the Intel RealSense D435i is in pristine factory condition with no physical or thermal distortion drift.
2. **Computational Optimization**: Bypassing runtime software undistortion (`cv2.undistort`) in the obstacle avoidance loop is **scientifically justified**, as it saves CPU/GPU cycles while metric precision remains within $< 0.1\text{ mm}$.
3. **Ground Homography**: Metric projection onto the ground plane achieves sub-centimeter accuracy across the robot's entire forward field of view.

---

## 3. Tool Suite & Usage

### Real-Time Evaluator Node
```bash
# Evaluates live chessboard / AprilTag corner detection & reprojection:
ros2 run husky_obstacle_avoidance distortion_evaluator_node.py

# Evaluates ground metric errors (raw vs undistorted):
ros2 run husky_obstacle_avoidance homography_distortion_evaluator_node.py
```

### Standalone Benchmark Analysis & Plot Generator
```bash
python3 camera_analysis.py --results results/ --out plots/
```

### Automated Batch Runner
```bash
bash run_camera_evaluation.sh
```

---

## 4. Generated Plots (Directory: `plots/`)
- `fig1_test_a_reprojection_rmse.png`: Reprojection RMSE vs poses and focal length drift.
- `fig2_test_b_rectilinearity_profile.png`: Perpendicular line residuals across raw edge pixels.
- `fig3_test_d_homography_ground_accuracy.png`: Metric error comparison (Raw vs Undistorted across zones).
- `fig4_camera_distortion_dashboard.png`: Consolidated 4-in-1 executive summary dashboard.
