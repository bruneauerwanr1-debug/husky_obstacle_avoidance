# Ground-Plane Homography Benchmark ($H_{old}$ vs $H_{new}$)

This module compares the baseline 4-AprilTag homography calibration matrix ($H_{old}$) against the high-definition 8-AprilTag calibration matrix ($H_{new}$) using 6,384 YOLOv8 obstacle detections.

---

## 1. Executive Summary of Results

| Statistical Indicator | Value | Practical Impact on Robot Navigation |
| :--- | :---: | :--- |
| **Median Difference $\Delta \|H_{old} - H_{new}\|$** | **7.30 cm** | High consistency in the immediate vicinity of the robot ($< 2\text{ m}$). |
| **90th Percentile (P90)** | **40.10 cm** | 90% of detections exhibit less than $40.1\text{ cm}$ discrepancy. |
| **Danger Zone (< 1.0 m) Difference** | **3.60 cm** | Emergency braking zone is highly reliable across both models. |
| **Warning Zone (1.0 - 2.0 m) Difference** | **0.90 cm** | Active deceleration triggers with negligible variation ($< 1\text{ cm}$). |
| **Long Range (> 3.5 m) Discrepancy** | **32.70 cm** | $H_{old}$ exhibits progressive underestimation due to lack of distant calibration points. |

---

## 2. Distance Bracket Breakdown

| Distance Range | Detections | % Total | Median $\Delta$ (cm) | P90 $\Delta$ (cm) | Navigation Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **< 1.0 m (Danger Zone)** | 323 | 5.1% | **3.6 cm** | 11.8 cm | Immediate emergency stop compliant |
| **1.0 - 2.0 m (Warning Zone)** | 2,716 | 42.5% | **0.9 cm** | 9.2 cm | Stable predictive deceleration alert |
| **2.0 - 3.5 m (Medium Range)** | 417 | 6.5% | **12.1 cm** | 23.0 cm | Linear tracking preserved |
| **> 3.5 m (Long Range)** | 2,928 | 45.9% | **32.7 cm** | 40.2 cm | $H_{old}$ diverges outside 4-tag perimeter |

---

## 3. Tool Suite & Usage

### Live Benchmark Node
```bash
# Compares live YOLO detections projected simultaneously via H_old and H_new:
ros2 run husky_obstacle_avoidance yolo_homography_benchmark_node.py
```

### Statistical Analysis & Plot Generator
```bash
python3 analyze_homography_benchmark.py --csv results/yolo_homography_benchmark_20260819_143836.csv --out plots/
```

### Automated Benchmark Runner
```bash
bash run_benchmark_analysis.sh
```

---

## 4. Generated Plots (Directory: `plots/`)
- `fig1_distance_correlation_scatter.png`: Parity correlation ($y=x$) between $H_{old}$ and $H_{new}$.
- `fig2_divergence_vs_distance.png`: Metric divergence $\Delta D = f(D)$ with inter-percentile envelope.
- `fig3_spatial_bev_error_map.png`: 2D Bird's-Eye View spatial error heatmap around the Husky footprint.
- `fig4_class_comparison_boxplots.png`: Discrepancy boxplots across COCO obstacle classes.
- `fig5_error_distribution_hist_kde.png`: Probability density function (KDE) and error distribution.
- `fig6_executive_summary_dashboard.png`: Consolidated 4-in-1 executive summary dashboard.
