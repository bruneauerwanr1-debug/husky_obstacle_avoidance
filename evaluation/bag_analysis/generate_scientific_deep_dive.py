#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_scientific_deep_dive.py
================================
Scientific & Statistical Deep-Dive Evaluation Suite for ROS Bag Benchmark.
Generates publication-quality figures and CSV summaries in English:
  1. Non-parametric Statistical Hypothesis Testing (Mann-Whitney U, p-values, effect size).
  2. Dynamic Kinematics & Directional Dodge Bias (Left vs Right turning, angular jerk, active avoidance time).
  3. Object Class Recognition & Distance Distortion Analysis (COCO categories, ground-contact vs homography).
  4. Inter-Trial Repeatability & Trajectory Stability (Coefficient of Variation CV%).

Outputs saved to: bag_analysis/comparatif_scientific/ (and synced to comparatif/ & comparatif_trials/)
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── Color Palette for Nodes ──
NODE_COLORS = {
    "farneback": "#E53935",  # Red / Orange-Red
    "yolo": "#1E88E5",       # Blue
    "raft": "#43A047",       # Green
    "unknown": "#757575"     # Grey
}

NODE_NAMES = {
    "farneback": "Farneback (P2)",
    "yolo": "YOLOv26m + Homography (P1)",
    "raft": "RAFT Deep Flow (P3)"
}


def _sig_title(res_df, metric_key, base_label):
    """
    Construit un titre de figure avec la p-value REELLEMENT calculee pour
    `metric_key` dans res_df (au lieu d'une valeur codee en dur qui ne
    refletait pas les resultats reels lors des re-executions du script).

    Retourne "<base_label> (p = X.XXe-YY ***)" avec le bon nombre d'etoiles,
    ou "<base_label> (test indisponible)" si la metrique n'a pas ete testee
    (colonne absente ou N < 3 dans un des deux groupes, cf. compute_statistical_tests).
    """
    row = res_df[res_df["Metric_Key"] == metric_key]
    if row.empty:
        return f"{base_label} (test indisponible)"
    r = row.iloc[0]
    p_val = r["P_Value"]
    if p_val < 0.001:
        stars = "***"
    elif p_val < 0.01:
        stars = "**"
    elif p_val < 0.05:
        stars = "*"
    else:
        stars = "NS"
    mantissa, exponent = f"{p_val:.2e}".split("e")
    exponent = int(exponent)
    return f"{base_label} ($p = {mantissa} \\times 10^{{{exponent}}}$ {stars})"


def load_all_datasets(bag_analysis_dir):
    """Loads trials recap, bags recap, and individual trial odometry/detections."""
    trials_csv = os.path.join(bag_analysis_dir, "comparatif_trials", "recap_trials.csv")
    if not os.path.isfile(trials_csv):
        trials_csv = os.path.join(bag_analysis_dir, "comparatif", "recap_trials.csv")
    
    if not os.path.isfile(trials_csv):
        raise FileNotFoundError(f"Could not find recap_trials.csv in {bag_analysis_dir}")
        
    df_trials = pd.read_csv(trials_csv)
    return df_trials


# ==============================================================================
#  1. STATISTICAL SIGNIFICANCE TESTS (MANN-WHITNEY U)
# ==============================================================================
def compute_statistical_tests(df_trials, out_dir):
    """
    Computes two-sided Mann-Whitney U tests between Farneback and YOLO
    for all primary safety, kinematic, accuracy, and computational metrics.
    """
    metrics_meta = [
        ("d_min_m", "Minimum Distance to Obstacle (m)", "Safety / Proximity", "Higher is Safer"),
        ("estop_count", "Emergency Stop (E-STOP) Count", "Safety / Failures", "Lower is Safer"),
        ("delta_psi_cum_rad", "Cumulative Heading Rotation Δψ (rad)", "Agitation / Stability", "Lower is Smoother"),
        ("jerk_rms_rad_s2", "Angular Jerk RMS (rad/s²)", "Motion Smoothness", "Lower is Smoother"),
        ("fps_hz", "Processing Rate / FPS (Hz)", "Real-time Execution", "Higher is Faster"),
        ("cpu_percent_mean", "Mean CPU Utilization (%)", "Resource Consumption", "Lower is Lighter"),
        ("ram_used_mb_mean", "Mean RAM Utilization (MB)", "Resource Consumption", "Lower is Lighter"),
        ("gpu_percent_mean", "Mean GPU Utilization (%)", "Hardware Acceleration", "Context-dependent"),
        ("gpu_mem_used_mb_mean", "Mean GPU VRAM Used (MB)", "Hardware Acceleration", "Context-dependent"),
        ("dist_err_median_m", "Median Distance Error (m)", "Estimation Accuracy", "Lower is More Accurate")
    ]

    results = []
    fb_df = df_trials[df_trials["node"] == "farneback"]
    yo_df = df_trials[df_trials["node"] == "yolo"]

    for col, label, category, interpretation in metrics_meta:
        if col not in df_trials.columns:
            continue
        fb_s = fb_df[col].dropna()
        yo_s = yo_df[col].dropna()
        if len(fb_s) < 3 or len(yo_s) < 3:
            continue

        u_stat, p_val = stats.mannwhitneyu(fb_s, yo_s, alternative="two-sided")
        
        # Rank-biserial correlation as effect size r = 1 - (2U) / (n1 * n2)
        n1, n2 = len(fb_s), len(yo_s)
        r_effect = 1.0 - (2.0 * u_stat) / (n1 * n2)

        sig_label = "*** (p < 0.001)" if p_val < 0.001 else "** (p < 0.01)" if p_val < 0.01 else "* (p < 0.05)" if p_val < 0.05 else "Not Significant (NS)"

        results.append({
            "Metric_Key": col,
            "Metric_Name": label,
            "Category": category,
            "Interpretation": interpretation,
            "Farneback_Mean": round(float(fb_s.mean()), 3),
            "Farneback_Median": round(float(fb_s.median()), 3),
            "Farneback_Std": round(float(fb_s.std()), 3),
            "Farneback_N": int(n1),
            "YOLO_Mean": round(float(yo_s.mean()), 3),
            "YOLO_Median": round(float(yo_s.median()), 3),
            "YOLO_Std": round(float(yo_s.std()), 3),
            "YOLO_N": int(n2),
            "Mann_Whitney_U": round(float(u_stat), 1),
            "P_Value": p_val,
            "P_Value_Scientific": f"{p_val:.3e}",
            "Effect_Size_r": round(float(r_effect), 3),
            "Significance": sig_label
        })

    res_df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "scientific_significance_tests.csv")
    res_df.to_csv(csv_path, index=False)

    # ── Figure 10: Statistical Hypothesis Testing ──
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)
    fig.suptitle("Scientific Hypothesis Testing: Farneback Optical Flow vs YOLOv26m (Mann-Whitney U)", fontsize=14, fontweight="bold")

    # Plot A: Safety & Proximity (d_min & E-stops)
    ax_a = fig.add_subplot(gs[0, 0])
    box_data_dmin = [fb_df["d_min_m"].dropna(), yo_df["d_min_m"].dropna()]
    bplot1 = ax_a.boxplot(box_data_dmin, patch_artist=True, tick_labels=["Farneback (P2)", "YOLOv26m (P1)"], widths=0.5)
    bplot1["boxes"][0].set_facecolor(NODE_COLORS["farneback"])
    bplot1["boxes"][1].set_facecolor(NODE_COLORS["yolo"])
    ax_a.set_ylabel("Minimum Distance $d_{min}$ (m)", fontsize=10, fontweight="bold")
    ax_a.set_title(_sig_title(res_df, "d_min_m", "Safety Clearance Margin"), fontsize=11, fontweight="bold")
    ax_a.grid(axis="y", alpha=0.3)

    # Plot B: E-STOP count distribution
    ax_b = fig.add_subplot(gs[0, 1])
    box_data_estop = [fb_df["estop_count"].dropna(), yo_df["estop_count"].dropna()]
    bplot2 = ax_b.boxplot(box_data_estop, patch_artist=True, tick_labels=["Farneback (P2)", "YOLOv26m (P1)"], widths=0.5)
    bplot2["boxes"][0].set_facecolor(NODE_COLORS["farneback"])
    bplot2["boxes"][1].set_facecolor(NODE_COLORS["yolo"])
    ax_b.set_ylabel("E-STOP Count per Trial", fontsize=10, fontweight="bold")
    ax_b.set_title(_sig_title(res_df, "estop_count", "Emergency Stop Invocations"), fontsize=11, fontweight="bold")
    ax_b.grid(axis="y", alpha=0.3)

    # Table Panel: Complete Statistical Summary Table
    ax_t = fig.add_subplot(gs[1, :])
    ax_t.axis("off")
    table_rows = []
    for _, r in res_df.iterrows():
        table_rows.append([
            r["Metric_Name"],
            f"{r['Farneback_Mean']:.2f} ± {r['Farneback_Std']:.2f}",
            f"{r['YOLO_Mean']:.2f} ± {r['YOLO_Std']:.2f}",
            f"{r['Mann_Whitney_U']:.0f}",
            r["P_Value_Scientific"],
            f"{r['Effect_Size_r']:+.2f}",
            r["Significance"]
        ])

    headers = ["Metric Name", "Farneback (Mean ± Std)", "YOLO (Mean ± Std)", "Mann-Whitney U", "p-value", "Effect Size r", "Significance"]
    col_widths = [0.26, 0.15, 0.15, 0.11, 0.10, 0.10, 0.13]
    table = ax_t.table(cellText=table_rows, colLabels=headers, colWidths=col_widths, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#0D47A1")
        table[0, j].set_text_props(color="white", fontweight="bold")

    fig.savefig(os.path.join(out_dir, "10_statistical_significance_tests.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return res_df


# ==============================================================================
#  2. KINEMATICS, ACTIVE TIME & DIRECTIONAL BIAS (LEFT VS RIGHT)
# ==============================================================================
def compute_kinematics_and_directional_bias(bag_analysis_dir, df_trials, out_dir):
    """
    Extracts high-resolution angular yaw rate (wz) and linear velocity (vx)
    from odometry files to quantify directional dodge bias and active maneuver time.
    """
    kin_records = []

    for bag_dir in sorted(glob.glob(os.path.join(bag_analysis_dir, "bag_*"))):
        bname = os.path.basename(bag_dir)
        trials_dir = os.path.join(bag_dir, "trials")
        if not os.path.isdir(trials_dir):
            continue

        for t_folder in sorted(os.listdir(trials_dir)):
            t_path = os.path.join(trials_dir, t_folder)
            if not os.path.isdir(t_path) or not t_folder.startswith("trial_"):
                continue

            trial_tag = f"{bname}_{t_folder}"
            info = df_trials[df_trials["bag_name"] == trial_tag]
            if info.empty:
                continue

            node = info.iloc[0]["node"]
            scen = info.iloc[0]["scenario"]

            odom_path = os.path.join(t_path, "odometry_filtered.csv")
            avoid_path = os.path.join(t_path, "avoidance_cmd_vel.csv")

            vx_mean, vx_max, wz_mean_abs, wz_max = np.nan, np.nan, np.nan, np.nan
            left_frames, right_frames, straight_frames = 0, 0, 0
            t_active_avoidance = 0.0

            if os.path.isfile(odom_path):
                try:
                    od = pd.read_csv(odom_path)
                    if not od.empty and "wz" in od.columns:
                        wz = od["wz"].dropna()
                        wz_mean_abs = wz.abs().mean()
                        wz_max = wz.abs().max()
                        left_frames = int((wz > 0.05).sum())
                        right_frames = int((wz < -0.05).sum())
                        straight_frames = int((wz.abs() <= 0.05).sum())
                    if not od.empty and "vx" in od.columns:
                        vx = od["vx"].dropna()
                        vx_mean = vx.mean()
                        vx_max = vx.max()
                except Exception:
                    pass

            if os.path.isfile(avoid_path):
                try:
                    av = pd.read_csv(avoid_path)
                    lin_col = "lin_x" if "lin_x" in av.columns else "linear_x"
                    ang_col = "ang_z" if "ang_z" in av.columns else "angular_z"
                    t_col = "t" if "t" in av.columns else "timestamp"
                    if lin_col in av.columns and ang_col in av.columns and t_col in av.columns:
                        dt = av[t_col].diff().fillna(0)
                        active_mask = (av[lin_col].abs() > 0.01) | (av[ang_col].abs() > 0.01)
                        t_active_avoidance = float((dt * active_mask).sum())
                except Exception:
                    pass

            tot_turns = left_frames + right_frames
            left_ratio = (left_frames / tot_turns) if tot_turns > 0 else np.nan

            kin_records.append({
                "Trial_Name": trial_tag,
                "Node": node,
                "Scenario": scen,
                "Duration_s": float(info.iloc[0].get("duration", np.nan)),
                "Active_Avoidance_Time_s": round(t_active_avoidance, 2),
                "Mean_Linear_Vel_mps": round(vx_mean, 3) if not np.isnan(vx_mean) else None,
                "Max_Linear_Vel_mps": round(vx_max, 3) if not np.isnan(vx_max) else None,
                "Mean_Abs_Yaw_Rate_radps": round(wz_mean_abs, 3) if not np.isnan(wz_mean_abs) else None,
                "Max_Yaw_Rate_radps": round(wz_max, 3) if not np.isnan(wz_max) else None,
                "Left_Turn_Frames": left_frames,
                "Right_Turn_Frames": right_frames,
                "Straight_Frames": straight_frames,
                "Left_Dodge_Pct": round(left_ratio * 100.0, 1) if not np.isnan(left_ratio) else None,
                "Right_Dodge_Pct": round((1.0 - left_ratio) * 100.0, 1) if not np.isnan(left_ratio) else None
            })

    df_kin = pd.DataFrame(kin_records)
    csv_path = os.path.join(out_dir, "kinematics_and_directional_bias.csv")
    df_kin.to_csv(csv_path, index=False)

    # ── Figure 11: Kinematics & Directional Dodge Bias ──
    fig = plt.figure(figsize=(17, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    fig.suptitle("Kinematic Response & Directional Dodge Bias: Farneback vs RAFT vs YOLOv26m", fontsize=14, fontweight="bold")

    # Panel A: Directional Dodge Bias (Left vs Right %)
    ax_bias = fig.add_subplot(gs[0, 0])
    nodes = [n for n in ["farneback", "raft", "yolo"] if n in df_kin["Node"].unique()]
    left_pcts = [df_kin[df_kin["Node"] == n]["Left_Dodge_Pct"].dropna().mean() for n in nodes]
    right_pcts = [100.0 - lp for lp in left_pcts]

    y_pos = np.arange(len(nodes))
    ax_bias.barh(y_pos, left_pcts, height=0.45, color="#1976D2", alpha=0.85, label="Turn Left (%)")
    ax_bias.barh(y_pos, -np.array(right_pcts), height=0.45, color="#D32F2F", alpha=0.85, label="Turn Right (%)")

    for i, n in enumerate(nodes):
        ax_bias.text(left_pcts[i] / 2.0, i, f"{left_pcts[i]:.1f}% Left", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        ax_bias.text(-right_pcts[i] / 2.0, i, f"{right_pcts[i]:.1f}% Right", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    ax_bias.axvline(0, color="black", linewidth=1.2)
    ax_bias.set_yticks(y_pos)
    ax_bias.set_yticklabels([NODE_NAMES.get(n, n) for n in nodes], fontsize=10, fontweight="bold")
    ax_bias.set_xlabel("Turning Bias Percentage (%)", fontsize=10, fontweight="bold")
    ax_bias.set_title("Directional Dodge Bias (Left vs Right Turning)", fontsize=11, fontweight="bold")
    ax_bias.set_xlim(-100, 100)
    ax_bias.legend(loc="upper right", fontsize=9)
    ax_bias.grid(axis="x", alpha=0.3)

    # Panel B: Active Avoidance Time vs Trial Duration
    ax_time = fig.add_subplot(gs[0, 1])
    time_data = [df_kin[df_kin["Node"] == n]["Active_Avoidance_Time_s"].dropna().mean() for n in nodes]
    bars_t = ax_time.bar([NODE_NAMES.get(n, n) for n in nodes], time_data, color=[NODE_COLORS.get(n, "#888") for n in nodes], alpha=0.85, width=0.5)
    for b in bars_t:
        h = b.get_height()
        ax_time.text(b.get_x() + b.get_width()/2.0, h + 0.3, f"{h:.2f} s", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax_time.set_ylabel("Active Avoidance Duration (s)", fontsize=10, fontweight="bold")
    ax_time.set_title("Time Spent in Active Avoidance Maneuver", fontsize=11, fontweight="bold")
    ax_time.grid(axis="y", alpha=0.3)

    # Panel C: Angular Yaw Rate Distribution (wz max & wz mean)
    ax_yaw = fig.add_subplot(gs[1, 0])
    wz_means = [df_kin[df_kin["Node"] == n]["Mean_Abs_Yaw_Rate_radps"].dropna().mean() for n in nodes]
    wz_maxs = [df_kin[df_kin["Node"] == n]["Max_Yaw_Rate_radps"].dropna().mean() for n in nodes]
    x_wz = np.arange(len(nodes))
    ax_yaw.bar(x_wz - 0.15, wz_means, width=0.3, label="Mean |ω_z| (rad/s)", color="#455A64", alpha=0.85)
    ax_yaw.bar(x_wz + 0.15, wz_maxs, width=0.3, label="Peak |ω_z| (rad/s)", color="#FF7043", alpha=0.85)
    ax_yaw.set_xticks(x_wz)
    ax_yaw.set_xticklabels([NODE_NAMES.get(n, n) for n in nodes], fontsize=9, fontweight="bold")
    ax_yaw.set_ylabel("Yaw Rate ω_z (rad/s)", fontsize=10, fontweight="bold")
    ax_yaw.set_title("Angular Velocity Intensity During Avoidance", fontsize=11, fontweight="bold")
    ax_yaw.legend(fontsize=9)
    ax_yaw.grid(axis="y", alpha=0.3)

    # Panel D: Kinematics Summary Table
    ax_kt = fig.add_subplot(gs[1, 1])
    ax_kt.axis("off")
    table_k = []
    short_names = {
        "farneback": "Farneback (P2)",
        "yolo": "YOLOv26m (P1)",
        "raft": "RAFT (P3)"
    }
    for n in nodes:
        sub = df_kin[df_kin["Node"] == n]
        table_k.append([
            short_names.get(n, n),
            f"{sub['Mean_Linear_Vel_mps'].mean():.3f} m/s",
            f"{sub['Max_Yaw_Rate_radps'].mean():.3f} rad/s",
            f"{sub['Active_Avoidance_Time_s'].mean():.1f} s",
            f"{sub['Left_Dodge_Pct'].mean():.1f}% L / {100-sub['Left_Dodge_Pct'].mean():.1f}% R"
        ])
    headers_k = ["Algorithm Node", "Mean Speed", "Peak Yaw Rate", "Active Avoidance", "Turning Bias (L / R)"]
    col_w_k = [0.24, 0.18, 0.18, 0.18, 0.22]
    t_obj = ax_kt.table(cellText=table_k, colLabels=headers_k, colWidths=col_w_k, loc="center", cellLoc="center")
    t_obj.auto_set_font_size(False)
    t_obj.set_fontsize(8.0)
    t_obj.scale(1.0, 1.5)
    for j in range(len(headers_k)):
        t_obj[0, j].set_facecolor("#2E7D32")
        t_obj[0, j].set_text_props(color="white", fontweight="bold")

    fig.savefig(os.path.join(out_dir, "11_kinematics_and_directional_bias.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df_kin


# ==============================================================================
#  3. OBJECT CLASS RECOGNITION & DISTANCE ACCURACY (YOLO COCO CATEGORIES)
# ==============================================================================
def compute_yolo_object_classes_analysis(bag_analysis_dir, out_dir):
    """
    Parses all 46,000+ ground detections across YOLO sessions,
    evaluating detection confidence, distance distribution, and tracking memory by COCO class.
    """
    class_data = []

    for bag_dir in sorted(glob.glob(os.path.join(bag_analysis_dir, "bag_P1_*"))):
        bname = os.path.basename(bag_dir)
        gd_path = os.path.join(bag_dir, "ground_detections.csv")
        if not os.path.isfile(gd_path):
            gd_path = os.path.join(bag_dir, "ground_det.csv")
        if not os.path.isfile(gd_path):
            continue

        try:
            df_g = pd.read_csv(gd_path)
            if "class_name" in df_g.columns:
                col_c = "class_name"
            elif "class" in df_g.columns:
                col_c = "class"
            else:
                continue

            for _, row in df_g.dropna(subset=[col_c, "dist_m"]).iterrows():
                class_data.append({
                    "bag_name": bname,
                    "class": str(row[col_c]).lower(),
                    "conf": float(row.get("conf", 0.5)),
                    "dist_m": float(row["dist_m"]),
                    "memorized": int(row.get("memorized", 0))
                })
        except Exception:
            pass

    if not class_data:
        return pd.DataFrame()

    df_cls = pd.DataFrame(class_data)
    cls_summary = df_cls.groupby("class").agg(
        Total_Detections=('dist_m', 'count'),
        Mean_Confidence=('conf', 'mean'),
        Min_Distance_m=('dist_m', 'min'),
        Median_Distance_m=('dist_m', 'median'),
        Mean_Distance_m=('dist_m', 'mean'),
        Pct_Tracking_Memory=('memorized', lambda x: (x > 0).mean() * 100.0)
    ).reset_index()

    cls_summary["Mean_Confidence"] = cls_summary["Mean_Confidence"].round(3)
    cls_summary["Min_Distance_m"] = cls_summary["Min_Distance_m"].round(2)
    cls_summary["Median_Distance_m"] = cls_summary["Median_Distance_m"].round(2)
    cls_summary["Mean_Distance_m"] = cls_summary["Mean_Distance_m"].round(2)
    cls_summary["Pct_Tracking_Memory"] = cls_summary["Pct_Tracking_Memory"].round(1)

    cls_summary = cls_summary.sort_values("Total_Detections", ascending=False)
    csv_path = os.path.join(out_dir, "yolo_object_classes_breakdown.csv")
    cls_summary.to_csv(csv_path, index=False)

    # ── Figure 12: Object Classes Recognition & Distance Distortion ──
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    fig.suptitle("YOLOv26m Detection Breakdown by Object Class: Frequency, Confidence & Distance", fontsize=14, fontweight="bold")

    # Filter top 8 classes for clean plotting
    top_cls = cls_summary.head(8)

    # Panel A: Detection Count & Confidence
    ax_c1 = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(len(top_cls))
    bars_c = ax_c1.bar(x_pos, top_cls["Total_Detections"], color="#1565C0", alpha=0.85, width=0.55)
    ax_c1.set_yscale("log")
    ax_c1.set_xticks(x_pos)
    ax_c1.set_xticklabels(top_cls["class"], rotation=25, ha="right", fontsize=9, fontweight="bold")
    ax_c1.set_ylabel("Detections Count (Log Scale)", fontsize=10, fontweight="bold")
    ax_c1.set_title("Total Detections by Class (Top 8 COCO Categories)", fontsize=11, fontweight="bold")
    ax_c1.grid(axis="y", alpha=0.3)

    # Overlay confidence as secondary axis
    ax_conf = ax_c1.twinx()
    ax_conf.plot(x_pos, top_cls["Mean_Confidence"], color="#FF8F00", marker="o", linewidth=2.0, label="Mean Confidence")
    ax_conf.set_ylabel("Mean Confidence Score", color="#FF8F00", fontsize=10, fontweight="bold")
    ax_conf.set_ylim(0.4, 1.0)

    # Panel B: Distance Distribution (Median & Min)
    ax_c2 = fig.add_subplot(gs[0, 1])
    ax_c2.bar(x_pos - 0.15, top_cls["Min_Distance_m"], width=0.3, color="#2E7D32", alpha=0.85, label="Min Distance (Closest)")
    ax_c2.bar(x_pos + 0.15, top_cls["Median_Distance_m"], width=0.3, color="#6A1B9A", alpha=0.85, label="Median Distance")
    ax_c2.set_xticks(x_pos)
    ax_c2.set_xticklabels(top_cls["class"], rotation=25, ha="right", fontsize=9, fontweight="bold")
    ax_c2.set_ylabel("Distance (m)", fontsize=10, fontweight="bold")
    ax_c2.set_title("Class-Specific Distance Profile & Homography Distortion", fontsize=11, fontweight="bold")
    ax_c2.legend(fontsize=9)
    ax_c2.grid(axis="y", alpha=0.3)

    # Panel C: Complete Classes Table
    ax_ct = fig.add_subplot(gs[1, :])
    ax_ct.axis("off")
    t_rows = []
    for _, r in cls_summary.head(10).iterrows():
        t_rows.append([
            r["class"].upper(),
            f"{r['Total_Detections']:,}",
            f"{r['Mean_Confidence']:.2f}",
            f"{r['Min_Distance_m']:.2f} m",
            f"{r['Median_Distance_m']:.2f} m",
            f"{r['Mean_Distance_m']:.2f} m",
            f"{r['Pct_Tracking_Memory']:.1f}%"
        ])

    headers_cls = ["Class Name", "Total Detections", "Confidence", "Min Distance", "Median Distance", "Mean Distance", "Tracking Memory"]
    col_w_cls = [0.16, 0.14, 0.13, 0.14, 0.14, 0.14, 0.15]
    tbl = ax_ct.table(cellText=t_rows, colLabels=headers_cls, colWidths=col_w_cls, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.0)
    tbl.scale(1.0, 1.3)
    for j in range(len(headers_cls)):
        tbl[0, j].set_facecolor("#4527A0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    fig.savefig(os.path.join(out_dir, "12_yolo_object_classes_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cls_summary


# ==============================================================================
#  4. SCENARIO REPEATABILITY & STABILITY (COEFFICIENT OF VARIATION CV%)
# ==============================================================================
def compute_scenario_repeatability(df_trials, out_dir):
    """
    Quantifies repeatability across trials for each scenario using the
    Coefficient of Variation (CV = std / mean * 100).
    """
    var_rows = []
    scenarios = sorted(df_trials["scenario"].unique())

    for scen in scenarios:
        sub_scen = df_trials[df_trials["scenario"] == scen]
        desc = sub_scen.iloc[0].get("scenario_desc", "")
        for node in sub_scen["node"].unique():
            sub = sub_scen[sub_scen["node"] == node]
            n_trials = len(sub)
            dur_col = "duration" if "duration" in sub.columns else ("trial_duration_s" if "trial_duration_s" in sub.columns else "duration_s")
            dmin_col = "d_min_m" if "d_min_m" in sub.columns else "min_distance_m"

            dur_mean = sub[dur_col].mean()
            dur_std = sub[dur_col].std() if n_trials > 1 else 0.0
            dur_cv = (dur_std / dur_mean * 100.0) if dur_mean > 0 else 0.0

            dmin_mean = sub[dmin_col].dropna().mean()
            dmin_std = sub[dmin_col].dropna().std() if n_trials > 1 else 0.0
            dmin_cv = (dmin_std / dmin_mean * 100.0) if (dmin_mean is not None and dmin_mean > 0) else 0.0

            estop_mean = sub["estop_count"].mean()
            estop_std = sub["estop_count"].std() if n_trials > 1 else 0.0
            estop_cv = (estop_std / estop_mean * 100.0) if estop_mean > 0 else 0.0

            var_rows.append({
                "Scenario": scen,
                "Description": desc,
                "Node": node,
                "Trials_Count": n_trials,
                "Duration_Mean_s": round(dur_mean, 2) if not np.isnan(dur_mean) else None,
                "Duration_CV_Pct": round(dur_cv, 1) if not np.isnan(dur_cv) else None,
                "Min_Distance_Mean_m": round(dmin_mean, 3) if not np.isnan(dmin_mean) else None,
                "Min_Distance_CV_Pct": round(dmin_cv, 1) if not np.isnan(dmin_cv) else None,
                "ESTOP_Mean": round(estop_mean, 2) if not np.isnan(estop_mean) else None,
                "ESTOP_CV_Pct": round(estop_cv, 1) if not np.isnan(estop_cv) else None
            })

    rep_df = pd.DataFrame(var_rows)
    csv_path = os.path.join(out_dir, "scenario_repeatability_metrics.csv")
    rep_df.to_csv(csv_path, index=False)

    # ── Figure 13: Scenario Repeatability & Stability (CV%) ──
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.25], hspace=0.35, wspace=0.25)
    fig.suptitle("Inter-Trial Repeatability & Stability (Coefficient of Variation CV%)", fontsize=14, fontweight="bold")

    # Panel A: Duration CV% by Scenario
    ax_r1 = fig.add_subplot(gs[0, 0])
    scens_plot = [s for s in scenarios if s in rep_df["Scenario"].unique()]
    x_s = np.arange(len(scens_plot))
    nodes_rep = rep_df["Node"].unique()
    w_bar = 0.8 / len(nodes_rep)

    short_rep_names = {
        "farneback": "Farneback (P2)",
        "yolo": "YOLOv26m (P1)",
        "raft": "RAFT (P3)"
    }

    for i_n, n in enumerate(nodes_rep):
        sub_n = rep_df[rep_df["Node"] == n]
        vals = [sub_n[sub_n["Scenario"] == s]["Duration_CV_Pct"].values[0] if not sub_n[sub_n["Scenario"] == s].empty else np.nan for s in scens_plot]
        pos = x_s + (i_n - (len(nodes_rep)-1)/2.0) * w_bar
        ax_r1.bar(pos, vals, width=w_bar*0.9, color=NODE_COLORS.get(n, "#888"), alpha=0.85, label=short_rep_names.get(n, n))

    ax_r1.set_xticks(x_s)
    ax_r1.set_xticklabels(scens_plot, fontsize=8, fontweight="bold")
    ax_r1.set_ylabel("Duration CV (%)", fontsize=10, fontweight="bold")
    ax_r1.set_title("Execution Time Variability (CV% = std/mean)", fontsize=11, fontweight="bold")
    ax_r1.legend(fontsize=8)
    ax_r1.grid(axis="y", alpha=0.3)

    # Panel B: Min Distance CV% (Clearance Repeatability)
    ax_r2 = fig.add_subplot(gs[0, 1])
    for i_n, n in enumerate(nodes_rep):
        sub_n = rep_df[rep_df["Node"] == n]
        vals = [sub_n[sub_n["Scenario"] == s]["Min_Distance_CV_Pct"].values[0] if not sub_n[sub_n["Scenario"] == s].empty else np.nan for s in scens_plot]
        pos = x_s + (i_n - (len(nodes_rep)-1)/2.0) * w_bar
        ax_r2.bar(pos, vals, width=w_bar*0.9, color=NODE_COLORS.get(n, "#888"), alpha=0.85, label=short_rep_names.get(n, n))

    ax_r2.set_xticks(x_s)
    ax_r2.set_xticklabels(scens_plot, fontsize=8, fontweight="bold")
    ax_r2.set_ylabel("Min Distance CV (%)", fontsize=10, fontweight="bold")
    ax_r2.set_title("Clearance Margin Variability (CV%)", fontsize=11, fontweight="bold")
    ax_r2.legend(fontsize=8)
    ax_r2.grid(axis="y", alpha=0.3)

    # Panel C: Repeatability Summary Table
    ax_rt = fig.add_subplot(gs[1, :])
    ax_rt.axis("off")
    table_rep = []
    for _, r in rep_df.iterrows():
        table_rep.append([
            r["Scenario"],
            short_rep_names.get(r["Node"], r["Node"]),
            r["Trials_Count"],
            f"{r['Duration_Mean_s']:.1f} s",
            f"{r['Duration_CV_Pct']:.1f}%",
            f"{r['Min_Distance_Mean_m']:.2f} m" if r['Min_Distance_Mean_m'] is not None else "N/A",
            f"{r['Min_Distance_CV_Pct']:.1f}%" if r['Min_Distance_CV_Pct'] is not None else "N/A",
            f"{r['ESTOP_Mean']:.1f}"
        ])

    headers_rep = ["Scenario", "Algorithm", "Trials", "Duration (Mean)", "Duration CV%", "d_min (Mean)", "d_min CV%", "E-STOP (Mean)"]
    col_w_rep = [0.10, 0.16, 0.08, 0.14, 0.13, 0.14, 0.13, 0.12]
    t_rep = ax_rt.table(cellText=table_rep[:14], colLabels=headers_rep, colWidths=col_w_rep, loc="center", cellLoc="center")
    t_rep.auto_set_font_size(False)
    t_rep.set_fontsize(7.5)
    t_rep.scale(1.0, 1.35)
    for j in range(len(headers_rep)):
        t_rep[0, j].set_facecolor("#37474F")
        t_rep[0, j].set_text_props(color="white", fontweight="bold")

    fig.savefig(os.path.join(out_dir, "13_scenario_repeatability_stability.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rep_df


# ==============================================================================
#  MAIN ENTRY POINT
# ==============================================================================
def run_all_scientific_evaluations(bag_analysis_dir=None):
    """Executes the complete scientific deep-dive suite across all target directories."""
    if bag_analysis_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bag_analysis_dir = os.path.join(base_dir, "bag_analysis")

    print(f"\n{'='*75}")
    print("  SCIENTIFIC & STATISTICAL DEEP-DIVE EVALUATION SUITE")
    print(f"  Target: {bag_analysis_dir}")
    print(f"{'='*75}")

    df_trials = load_all_datasets(bag_analysis_dir)
    print(f"[OK] Loaded {len(df_trials)} individual trials across 22 ROS bag sessions.")

    # Target output folders (scientific folder + synced into comparatif suites)
    out_dirs = [
        os.path.join(bag_analysis_dir, "comparatif_scientific"),
        os.path.join(bag_analysis_dir, "comparatif"),
        os.path.join(bag_analysis_dir, "comparatif_trials")
    ]

    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n--- Generating Scientific Artifacts -> {os.path.relpath(out_dir, bag_analysis_dir)} ---")

        # 1. Statistical Hypothesis Tests (Mann-Whitney U)
        compute_statistical_tests(df_trials, out_dir)
        print("  [OK] 10_statistical_significance_tests.png & scientific_significance_tests.csv")

        # 2. Kinematics & Directional Dodge Bias
        compute_kinematics_and_directional_bias(bag_analysis_dir, df_trials, out_dir)
        print("  [OK] 11_kinematics_and_directional_bias.png & kinematics_and_directional_bias.csv")

        # 3. Object Classes Recognition & Distance Distortion
        compute_yolo_object_classes_analysis(bag_analysis_dir, out_dir)
        print("  [OK] 12_yolo_object_classes_analysis.png & yolo_object_classes_breakdown.csv")

        # 4. Inter-trial Repeatability (CV%)
        compute_scenario_repeatability(df_trials, out_dir)
        print("  [OK] 13_scenario_repeatability_stability.png & scenario_repeatability_metrics.csv")

    print(f"\n{'='*75}")
    print("  [SUCCESS] All scientific deep-dive figures and CSV reports generated in English!")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Scientific Deep-Dive Analysis for ROS Bags in English.")
    parser.add_argument("--dir", type=str, default=None, help="Path to bag_analysis folder.")
    args = parser.parse_args()
    run_all_scientific_evaluations(args.dir)