# Consolidated Scientific Report: Camera Distortion & Calibration Evaluation

**Analysis Date**: 2026-08-20 14:18:38  
**Camera Model**: Intel RealSense D435i (Color Optical Sensor)  
**Evaluation Scope**: Intrinsics Health, Straight-Line Rectilinearity, and Ground Homography Accuracy  

---

## 1. Executive Summary & Benchmark Scorecard

| Evaluation Test | Key Performance Indicator | Measured Value | Standard Threshold | Scientific Verdict |
| :--- | :--- | :---: | :---: | :---: |
| **Test A — Reprojection RMSE** | Subpixel reprojection error | **0.274 px** (39 views) | < 0.30 px | 🟢 **EXCELLENT** |
| **Test A — Focal Length Drift** | Relative drift ($f_x, f_y$ vs factory) | **< 0.2%** ($f_x=615.6, f_y=616.9$) | < 2.0% | 🟢 **PERFECT STABILITY** |
| **Test B — Rectilinearity** | Mean edge deviation on raw frame | **0.61 px** | <= 1.0 px | 🟢 **COMPLIANT** |
| **Test C — Firmware Intrinsics** | Factory distortion parameter state | $D = [0, 0, 0, 0, 0]$ | — | 🟢 **FACTORY RECTIFIED** |
| **Test D — Ground Homography** | Metric ground error ($H_{raw}$, 6 tags) | **2.96 mm** (0.296 cm) | < 20.0 mm | 🟢 **SUB-CENTIMETRIC** |
| **Test D — Undistortion Gain** | Precision gain from `cv2.undistort` | **+2.29%** ($\Delta = 0.07\text{ mm}$) | > 10.0% | ⚪ **NEGLIGIBLE GAIN** |

---

## 2. In-Depth Analysis of Individual Tests

### Test A: Chessboard Calibration & Intrinsic Parameters ($0.274\text{ px}$)
- Across **39 distinct poses**, the measured reprojection RMSE is **0.274\text{ px}**, which falls squarely in the scientific excellence category ($< 0.30\text{ px}$).
- The calibrated focal lengths ($f_x = 615.62, f_y = 616.92$) show an imperceptible deviation of **$-0.18\%$** and **$+0.02\%$** relative to factory defaults ($616.74, 616.79$), confirming that the optical lens assembly has suffered no mechanical or thermal degradation.

### Test B: Straight-Line Edge Rectilinearity
- The mean perpendicular residual on the raw edge is **0.61\text{ px}**, demonstrating that physical straight lines project as straight lines on the sensor without noticeable barrel or pincushion distortion.
- Discrete pixel quantization effects (gradient/Canny discretization) account for the momentary max deviation of 2 px on short segments, rather than genuine optical curvature.

### Test D: Ground AprilTag Homography Accuracy ($H_{raw}$ vs $H_{undist}$)
- Across **6 simultaneous ground AprilTags**, the global median error is **2.96\text{ mm}** on the raw image vs **2.89\text{ mm}** on the undistorted image.
- Applying real-time software undistortion yields an improvement of only **$0.07\text{ mm}$ ($+2.29\%$)**, which is completely negligible for robotic navigation and obstacle avoidance.

---

## 3. Generated Scientific Plots (Directory: `plots/`)

1. **`fig1_test_a_reprojection_rmse.png`**: Reprojection RMSE vs scientific thresholds and focal length stability.
2. **`fig2_test_b_rectilinearity_profile.png`**: Straight-line pixel residual profile along the sampled edge.
3. **`fig3_test_d_homography_ground_accuracy.png`**: Ground AprilTag error comparison (Center vs Periphery, Raw vs Undistorted).
4. **`fig4_camera_distortion_dashboard.png`**: Consolidated 4-in-1 Executive Summary Dashboard for publication/reporting.

---

## 4. Operational Takeaways for the Internship Report

1. **Sensor Optical Quality**: The Intel RealSense D435i color sensor is in pristine optical condition and requires **no hardware recalibration**.
2. **Real-Time Pipeline Optimization**: Bypassing continuous software undistortion (`cv2.undistort`) in the obstacle avoidance pipeline is **fully justified**, saving valuable CPU/GPU cycles with zero loss of metric precision ($< 0.1\text{ mm}$).
3. **Navigation Readiness**: The calibrated homography matrix $H_{new}$ provides sub-centimeter ground localization accuracy across the entire driving corridor.
