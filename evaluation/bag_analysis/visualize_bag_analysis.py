#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_bag_analysis.py -- Visualisation et analyse statistique des CSV de tests ROS
======================================================================================

Traite automatiquement tous les dossiers de bag_analysis/, génère :
  1. Des graphiques par dossier de test (distance, TTC, commandes, trajectoire, ressources)
  2. Des graphiques comparatifs par scénario (bar charts, scatter, heatmap)
  3. Un tableau récapitulatif CSV

Fusionne les CSV de performance système (resource_*.csv) avec les données ROS.

USAGE
-----
    py visualize_bag_analysis.py              # traite tout bag_analysis/
    py visualize_bag_analysis.py --folder bag_P1_S1_01   # un seul dossier
    py visualize_bag_analysis.py --no-comparative         # sans les comparatifs

SORTIES
-------
    bag_analysis/<bag_name>/plots/   -> graphiques par test
    bag_analysis/comparatif/         -> graphiques et tableaux comparatifs
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

# ═══════════════════════════════════════════════════════════════════════
#  CHEMINS ET CONSTANTES
# ═══════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_BAG_ANALYSIS = os.path.join(BASE_DIR, "bag_analysis")

# Seuils du nœud d'évitement (pour annotation des graphiques)
DISTANCE_WARN_M = 1.5
DISTANCE_DANGER_M = 0.8
TTC_WARN_S = 3.0

# Description des scénarios pour légendes (English)
SCENARIO_DESCRIPTIONS = {
    "S1":   "Static obstacle (generic)",
    "S1.1": "Static cone/box (measured position)",
    "S1.2": "Thin pole ⌀5cm",
    "S1.3": "Matte black panel (low texture)",
    "S2":   "Dynamic obstacle (generic)",
    "S2.1": "Crossing pedestrian ~0.8 m/s",
    "S2.2": "Frontal approaching pedestrian ~0.5 m/s",
    "S3.1": "Low illumination <50 lux",
    "S3.2": "AprilTags on ground visible",
    "S4.1": "High forward speed vx≈0.8 m/s",
    "S4.2": "Pure rotation, clear path ahead",
    "S5.1": "Non-COCO object (door/partition)",
    "S5.2": "Slalom 1.1m between two obstacles",
    "S5.3": "Appearing/disappearing obstacle (15s retention)",
}

# Style global
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

# Couleurs par node
NODE_COLORS = {
    "yolo": "#2196F3",
    "farneback": "#FF9800",
    "raft": "#4CAF50",
    "unknown": "#9E9E9E",
}


# ═══════════════════════════════════════════════════════════════════════
#  PARSING DU NOM DE DOSSIER
# ═══════════════════════════════════════════════════════════════════════

def _build_avoidance_mask_local(odom_df, avoid_df, timeout_s=0.3):
    """
    Reproduit le fenetrage utilise dans analyse-bag-test.py::build_avoidance_mask :
    regroupe les messages /avoidance_cmd_vel en episodes (nouvel episode si
    l'ecart entre deux commandes depasse timeout_s, le watchdog du node),
    puis retourne un masque booleen (indexe comme odom_df) True uniquement
    pendant ces episodes. Necessaire ici car recalculer delta_psi/jerk sur
    la duree ENTIERE du trial (comme avant) inclut les phases sans evitement
    et desaligne ces metriques de celles du protocole (cf. summary.json).
    """
    mask = pd.Series(False, index=odom_df.index)
    if avoid_df is None or avoid_df.empty or odom_df.empty or "t" not in avoid_df.columns:
        return mask
    avoid_t = avoid_df["t"].dropna().sort_values().to_numpy()
    if avoid_t.size == 0:
        return mask
    episodes = []
    start = avoid_t[0]
    prev = avoid_t[0]
    for t in avoid_t[1:]:
        if t - prev > timeout_s:
            episodes.append((start, prev + timeout_s))
            start = t
        prev = t
    episodes.append((start, prev + timeout_s))
    for (s, e) in episodes:
        mask |= (odom_df["t"] >= s) & (odom_df["t"] <= e)
    return mask


def parse_bag_name(bag_name):
    """
    Extrait node, scénario, trial depuis le nom du dossier ou du trial.
    Exemples :
        bag_P1_S1_01        -> node=yolo, scenario=S1, trial=01
        bag_P2_S1.1_01      -> node=farneback, scenario=S1.1, trial=01
        bag_P3_S1.1_01      -> node=raft, scenario=S1.1, trial=01
        bag_P1_S1_01_trial_02 -> node=yolo, scenario=S1, trial=02
    """
    m = re.search(r"P(\d+)_S([\d.]+)_(\d+)", bag_name)
    if not m:
        return {"node": "unknown", "scenario": "NA", "trial": 1, "bag_name": bag_name}

    p_num = int(m.group(1))
    scenario = f"S{m.group(2)}"
    trial = int(m.group(3))
    node_map = {1: "yolo", 2: "farneback", 3: "raft"}
    node = node_map.get(p_num, "unknown")

    # Extraire le numero de trial si present a la fin (ex: _trial_03)
    m_sub = re.search(r"trial_(\d+)", bag_name)
    if m_sub:
        trial = int(m_sub.group(1))

    return {"node": node, "scenario": scenario, "trial": trial, "bag_name": bag_name}


# ═══════════════════════════════════════════════════════════════════════
#  CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════

def load_csv_safe(path):
    """Charge un CSV s'il existe, sinon retourne None."""
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception:
        return None


def load_bag_data(bag_folder):
    """Charge toutes les données d'un dossier bag."""
    data = {}

    # CSV ROS standard
    csv_names = {
        "odom": "odometry_filtered.csv",
        "cmd_vel": "cmd_vel.csv",
        "cmd_vel_in": "cmd_vel_in.csv",
        "avoidance": "avoidance_cmd_vel.csv",
        "joy": "joy_teleop_cmd_vel.csv",
        "estop": "emergency_stop.csv",
        "flow_diag": "flow_diagnostics.csv",
        "ground_det": "ground_detections.csv",
        "depth": "camera_aligned_depth_to_color_image_raw.csv",
        "movement_cmd": "movement_command.csv",
        "optical_flow": "optical_flow_zones.csv",
        "homography": "homography_matrix.csv",
        "robot_pos": "robot_ground_position.csv",
    }

    for key, fname in csv_names.items():
        data[key] = load_csv_safe(os.path.join(bag_folder, fname))

    # CSV de ressources (nom variable : resource_*.csv)
    resource_files = glob.glob(os.path.join(bag_folder, "resource*.csv"))
    if resource_files:
        try:
            data["resources"] = pd.read_csv(resource_files[0])
        except Exception:
            data["resources"] = None
    else:
        data["resources"] = None

    # Summary JSON
    summary_path = os.path.join(bag_folder, "summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, "r") as f:
            data["summary"] = json.load(f)
    else:
        data["summary"] = {}

    return data


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS GRAPHIQUES
# ═══════════════════════════════════════════════════════════════════════

def shade_estop(ax, estop_df, alpha=0.12):
    """Ombre les périodes E-STOP en rouge."""
    if estop_df is None:
        return
    estop = estop_df.sort_values("t")
    bool_vals = estop["value"].astype(str).str.strip().map(
        {"True": True, "False": False, "1": True, "0": False}).fillna(False)

    in_estop = False
    t_start = None
    for i, (_, row) in enumerate(estop.iterrows()):
        v = bool_vals.iloc[i]
        if v and not in_estop:
            in_estop = True
            t_start = row["t"]
        elif not v and in_estop:
            in_estop = False
            ax.axvspan(t_start, row["t"], color="red", alpha=alpha, zorder=0)
    if in_estop:
        ax.axvspan(t_start, estop["t"].iloc[-1], color="red", alpha=alpha, zorder=0)


def shade_avoidance_episodes(ax, avoid_df, timeout=0.5, alpha=0.10):
    """Ombre les épisodes d'évitement en orange."""
    if avoid_df is None or avoid_df.empty:
        return []
    t = avoid_df["t"].sort_values().to_numpy()
    episodes = []
    start = t[0]
    prev = t[0]
    for ti in t[1:]:
        if ti - prev > timeout:
            episodes.append((start, prev))
            start = ti
        prev = ti
    episodes.append((start, prev))
    for s, e in episodes:
        ax.axvspan(s, e + 0.1, color="orange", alpha=alpha, zorder=0)
    return episodes


def get_distance_series(data, node, max_dist=10.0):
    """Retourne (t, distance) depuis le topic approprie selon le node, filtre a < max_dist."""
    # Priorite : flow_diagnostics.min_dist_m
    fd = data.get("flow_diag")
    if fd is not None and "min_dist_m" in fd.columns:
        valid = fd.dropna(subset=["min_dist_m"])
        valid = valid[valid["min_dist_m"] < max_dist]
        if not valid.empty:
            return valid["t"].to_numpy(), valid["min_dist_m"].to_numpy(), "flow_diagnostics"

    # Repli YOLO
    gd = data.get("ground_det")
    if gd is not None and "dist_m" in gd.columns:
        valid = gd.dropna(subset=["dist_m"])
        valid = valid[valid["dist_m"] < max_dist]
        if not valid.empty:
            # Garder le plus proche par frame
            closest = valid.groupby("t")["dist_m"].min().reset_index()
            return closest["t"].to_numpy(), closest["dist_m"].to_numpy(), "ground_detections"

    # Repli Farneback
    mc = data.get("movement_cmd")
    if mc is not None and "dist_m" in mc.columns:
        valid = mc.dropna(subset=["dist_m"])
        valid = valid[valid["dist_m"] < max_dist]
        if not valid.empty:
            return valid["t"].to_numpy(), valid["dist_m"].to_numpy(), "movement_command"

    return None, None, None


# ═══════════════════════════════════════════════════════════════════════
#  GRAPHIQUES PAR DOSSIER
# ═══════════════════════════════════════════════════════════════════════

def plot_distance_comparison(data, info, out_dir):
    """
    Plot 1 : Distance estimée (node) vs vérité terrain (depth camera).
    """
    t_node, d_node, source = get_distance_series(data, info["node"])
    depth = data.get("depth")

    if t_node is None and depth is None:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    scenario_desc = SCENARIO_DESCRIPTIONS.get(info["scenario"], info["scenario"])
    ax.set_title(f"Distance to Obstacle -- {info['bag_name']}\n"
                 f"{info['scenario']} ({scenario_desc}) | Node: {info['node']}")

    if t_node is not None:
        color = NODE_COLORS.get(info["node"], "#333")
        ax.plot(t_node, d_node, linewidth=1, alpha=0.9, color=color,
                label=f"Estimated distance ({source})")

    if depth is not None and "d_min_depth_m" in depth.columns:
        valid_depth = depth.dropna(subset=["d_min_depth_m"])
        valid_depth = valid_depth[valid_depth["d_min_depth_m"] < 10.0]
        if not valid_depth.empty:
            ax.plot(valid_depth["t"], valid_depth["d_min_depth_m"],
                    linewidth=0.8, alpha=0.7, color="#E91E63",
                    label="Ground truth (depth camera)")

    # Seuils
    ax.axhline(DISTANCE_DANGER_M, color="red", linestyle="--", linewidth=0.8,
               alpha=0.6, label=f"DANGER threshold ({DISTANCE_DANGER_M}m)")
    ax.axhline(DISTANCE_WARN_M, color="#FFC107", linestyle="--", linewidth=0.8,
               alpha=0.6, label=f"WARNING threshold ({DISTANCE_WARN_M}m)")

    shade_estop(ax, data.get("estop"))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(out_dir, "01_distance_comparison.png"))
    plt.close(fig)


def plot_ttc(data, info, out_dir):
    """
    Plot 2 : TTC au cours du temps.
    """
    fd = data.get("flow_diag")
    if fd is None or "ttc" not in fd.columns:
        return
    valid = fd.dropna(subset=["ttc"])
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    scenario_desc = SCENARIO_DESCRIPTIONS.get(info["scenario"], info["scenario"])
    ax.set_title(f"Time-To-Collision (TTC) -- {info['bag_name']}\n"
                 f"{info['scenario']} ({scenario_desc}) | Node: {info['node']}")

    ax.plot(valid["t"], valid["ttc"], linewidth=1, alpha=0.9,
            color=NODE_COLORS.get(info["node"], "#333"), label="Estimated TTC (s)")

    # Vérité terrain TTC si disponible via depth
    depth = data.get("depth")
    if depth is not None and "d_min_depth_m" in depth.columns:
        dd = depth.dropna(subset=["d_min_depth_m"]).sort_values("t")
        if len(dd) >= 5:
            t = dd["t"].to_numpy()
            d = dd["d_min_depth_m"].to_numpy()
            d_smooth = pd.Series(d).rolling(3, center=True, min_periods=1).mean().to_numpy()
            dt = np.diff(t)
            dd_diff = np.diff(d_smooth)
            valid_dt = dt > 1e-6
            rate = np.full_like(dt, np.nan)
            rate[valid_dt] = dd_diff[valid_dt] / dt[valid_dt]
            approaching = rate < -1e-3
            ttc_real = np.full_like(dt, np.nan)
            ttc_real[approaching] = -d_smooth[:-1][approaching] / rate[approaching]
            mask_valid = np.isfinite(ttc_real) & (ttc_real > 0) & (ttc_real < 30)
            if mask_valid.any():
                ax.plot(t[:-1][mask_valid], ttc_real[mask_valid],
                        linewidth=0.8, alpha=0.7, color="#E91E63",
                        label="Real TTC (depth, finite diff.)", marker=".", markersize=2)

    ax.axhline(TTC_WARN_S, color="#FFC107", linestyle="--", linewidth=0.8,
               alpha=0.6, label=f"TTC WARN threshold ({TTC_WARN_S}s)")

    shade_estop(ax, data.get("estop"))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("TTC (s)")
    ax.set_ylim(0, min(30, valid["ttc"].quantile(0.98) * 1.2) if len(valid) > 5 else 30)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(out_dir, "02_ttc.png"))
    plt.close(fig)


def plot_commands(data, info, out_dir):
    """
    Plot 3 : Commandes de vitesse superposées + zones EMERGENCY.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    scenario_desc = SCENARIO_DESCRIPTIONS.get(info["scenario"], info["scenario"])
    fig.suptitle(f"Velocity Commands -- {info['bag_name']}\n"
                 f"{info['scenario']} ({scenario_desc}) | Node: {info['node']}",
                 fontsize=12, fontweight="bold")

    topics = {
        "joy": ("/joy_teleop/cmd_vel", "#4CAF50", 0.5),
        "cmd_vel_in": ("/cmd_vel_in", "#9C27B0", 0.5),
        "avoidance": ("/avoidance_cmd_vel", "#FF5722", 0.8),
        "cmd_vel": ("/cmd_vel", "#2196F3", 0.7),
    }

    for key, (label, color, alpha) in topics.items():
        df = data.get(key)
        if df is not None and "t" in df.columns:
            ax1.plot(df["t"], df["lin_x"], linewidth=0.7, alpha=alpha,
                     color=color, label=f"{label} lin_x")
            ax2.plot(df["t"], df["ang_z"], linewidth=0.7, alpha=alpha,
                     color=color, label=f"{label} ang_z")

    for ax in (ax1, ax2):
        shade_estop(ax, data.get("estop"))
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel("Linear velocity (m/s)")
    ax2.set_ylabel("Angular velocity (rad/s)")
    ax2.set_xlabel("Time (s)")

    fig.savefig(os.path.join(out_dir, "03_commands.png"))
    plt.close(fig)


def plot_state_timeline(data, info, out_dir):
    """
    Plot 4 : Timeline des états du node + distance.
    """
    fd = data.get("flow_diag")
    mc = data.get("movement_cmd")

    # Chercher la source d'états
    state_df = None
    if fd is not None and "state" in fd.columns:
        state_df = fd[["t", "state"]].dropna(subset=["state"])
    elif mc is not None and "state" in mc.columns:
        state_df = mc[["t", "state"]].dropna(subset=["state"])

    if state_df is None or state_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 3]})
    scenario_desc = SCENARIO_DESCRIPTIONS.get(info["scenario"], info["scenario"])
    fig.suptitle(f"State Timeline -- {info['bag_name']}\n"
                 f"{info['scenario']} ({scenario_desc}) | Node : {info['node']}",
                 fontsize=12, fontweight="bold")

    # Map des couleurs d'état
    state_colors = {
        "CLEAR": "#4CAF50",
        "WARNING": "#FFC107",
        "PREDICTING": "#FF9800",
        "EMERGENCY": "#F44336",
        "RECOVERY": "#9C27B0",
        "AVOIDING": "#FF5722",
    }

    # Dessiner les états comme des barres horizontales
    states = state_df.sort_values("t")
    for i in range(len(states) - 1):
        s = states.iloc[i]
        s_next = states.iloc[i + 1]
        color = state_colors.get(str(s["state"]).upper(), "#9E9E9E")
        ax1.axvspan(s["t"], s_next["t"], color=color, alpha=0.6)

    # Légende
    used_states = set(states["state"].dropna().str.upper().unique())
    legend_patches = [Patch(facecolor=state_colors.get(s, "#9E9E9E"), alpha=0.6, label=s)
                      for s in sorted(used_states) if s in state_colors]
    ax1.legend(handles=legend_patches, loc="upper right", fontsize=7, ncol=3)
    ax1.set_ylabel("State")
    ax1.set_yticks([])

    # Distance sur l'axe 2
    t_node, d_node, source = get_distance_series(data, info["node"])
    if t_node is not None:
        ax2.plot(t_node, d_node, linewidth=0.8, alpha=0.8,
                 color=NODE_COLORS.get(info["node"], "#333"),
                 label=f"Distance ({source})")
    depth = data.get("depth")
    if depth is not None and "d_min_depth_m" in depth.columns:
        valid_depth = depth.dropna(subset=["d_min_depth_m"])
        if not valid_depth.empty:
            ax2.plot(valid_depth["t"], valid_depth["d_min_depth_m"],
                     linewidth=0.6, alpha=0.6, color="#E91E63",
                     label="Depth camera")

    ax2.axhline(DISTANCE_DANGER_M, color="red", linestyle="--", linewidth=0.7, alpha=0.5)
    ax2.axhline(DISTANCE_WARN_M, color="#FFC107", linestyle="--", linewidth=0.7, alpha=0.5)
    ax2.set_ylabel("Distance (m)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.savefig(os.path.join(out_dir, "04_state_timeline.png"))
    plt.close(fig)


def plot_trajectory(data, info, out_dir):
    """
    Plot 5 : Trajectoire 2D colorée par vitesse, avec épisodes d'évitement.
    """
    odom = data.get("odom")
    if odom is None:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    scenario_desc = SCENARIO_DESCRIPTIONS.get(info["scenario"], info["scenario"])
    ax.set_title(f"2D Trajectory -- {info['bag_name']}\n"
                 f"{info['scenario']} ({scenario_desc}) | Node: {info['node']}")

    x = odom["x"].to_numpy()
    y = odom["y"].to_numpy()
    vx = np.abs(odom["vx"].to_numpy())

    # Colormap par vitesse
    scatter = ax.scatter(x, y, c=vx, s=2, cmap="viridis", alpha=0.7, zorder=2)
    plt.colorbar(scatter, ax=ax, label="|vx| (m/s)", shrink=0.8)

    # Marquer les épisodes d'évitement
    avoid = data.get("avoidance")
    if avoid is not None and not avoid.empty:
        t_avoid = set(avoid["t"].to_numpy())
        # Trouver les points odom les plus proches des timestamps d'évitement
        odom_t = odom["t"].to_numpy()
        avoid_mask = np.zeros(len(odom_t), dtype=bool)
        for ta in t_avoid:
            idx = np.argmin(np.abs(odom_t - ta))
            avoid_mask[idx] = True
        ax.scatter(x[avoid_mask], y[avoid_mask], s=8, color="red",
                   alpha=0.4, zorder=3, label="Active avoidance")

    # Point de départ et fin
    ax.plot(x[0], y[0], "go", markersize=10, label="Start", zorder=4)
    ax.plot(x[-1], y[-1], "rs", markersize=10, label="End", zorder=4)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(out_dir, "05_trajectory.png"))
    plt.close(fig)


def plot_resources(data, info, out_dir):
    """
    Plot 6 : Ressources système (CPU, RAM, GPU, VRAM).
    """
    res = data.get("resources")
    if res is None:
        return

    # Le CSV resource a des timestamps Unix, on les convertit en temps relatif
    if "t_unix" in res.columns:
        t0 = res["t_unix"].min()
        res = res.copy()
        res["t_rel"] = res["t_unix"] - t0
        t_col = "t_rel"
    elif "t" in res.columns:
        t_col = "t"
    else:
        return

    has_gpu = ("gpu_percent" in res.columns and res["gpu_percent"].notna().any())
    n_plots = 4 if has_gpu else 2

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    scenario_desc = SCENARIO_DESCRIPTIONS.get(info["scenario"], info["scenario"])
    fig.suptitle(f"System Resources -- {info['bag_name']}\n"
                 f"{info['scenario']} ({scenario_desc}) | Node: {info['node']}",
                 fontsize=12, fontweight="bold")

    # CPU
    if "cpu_percent" in res.columns:
        axes[0].plot(res[t_col], res["cpu_percent"], linewidth=0.8, color="#2196F3")
        axes[0].fill_between(res[t_col], 0, res["cpu_percent"], alpha=0.2, color="#2196F3")
        axes[0].set_ylabel("CPU (%)")
        axes[0].set_ylim(0, 105)
        axes[0].grid(True, alpha=0.3)
        mean_cpu = res["cpu_percent"].mean()
        axes[0].axhline(mean_cpu, color="#1565C0", linestyle="--", linewidth=0.8,
                        alpha=0.5, label=f"Mean = {mean_cpu:.1f}%")
        axes[0].legend(fontsize=8)

    # RAM
    if "ram_used_mb" in res.columns:
        axes[1].plot(res[t_col], res["ram_used_mb"], linewidth=0.8, color="#4CAF50")
        axes[1].fill_between(res[t_col], 0, res["ram_used_mb"], alpha=0.2, color="#4CAF50")
        axes[1].set_ylabel("RAM (MB)")
        axes[1].grid(True, alpha=0.3)

    if has_gpu:
        # GPU
        axes[2].plot(res[t_col], res["gpu_percent"], linewidth=0.8, color="#FF9800")
        axes[2].fill_between(res[t_col], 0, res["gpu_percent"], alpha=0.2, color="#FF9800")
        axes[2].set_ylabel("GPU (%)")
        axes[2].set_ylim(0, 105)
        axes[2].grid(True, alpha=0.3)
        mean_gpu = res["gpu_percent"].dropna().mean()
        axes[2].axhline(mean_gpu, color="#E65100", linestyle="--", linewidth=0.8,
                        alpha=0.5, label=f"Mean = {mean_gpu:.1f}%")
        axes[2].legend(fontsize=8)

        # VRAM
        if "gpu_mem_used_mb" in res.columns:
            axes[3].plot(res[t_col], res["gpu_mem_used_mb"], linewidth=0.8, color="#9C27B0")
            axes[3].fill_between(res[t_col], 0, res["gpu_mem_used_mb"],
                                 alpha=0.2, color="#9C27B0")
            axes[3].set_ylabel("VRAM (MB)")
            axes[3].grid(True, alpha=0.3)
            if "gpu_mem_total_mb" in res.columns:
                total = res["gpu_mem_total_mb"].iloc[0]
                axes[3].axhline(total, color="#6A1B9A", linestyle="--",
                                linewidth=0.8, alpha=0.5, label=f"Total = {total:.0f} MB")
                axes[3].legend(fontsize=8)

    axes[-1].set_xlabel("Time (s)")

    fig.savefig(os.path.join(out_dir, "06_resources.png"))
    plt.close(fig)


def plot_homography_vs_depth(data, info, out_dir):
    """
    Plot 7 : Comparaison de la distance et position Homography vs Depth Camera.
    """
    robot_pos = data.get("robot_pos")
    depth = data.get("depth")
    t_node, d_node, source = get_distance_series(data, info["node"], max_dist=10.0)

    has_homo = robot_pos is not None and not robot_pos.empty and "x" in robot_pos.columns and "y" in robot_pos.columns
    has_depth = depth is not None and not depth.empty and "d_min_depth_m" in depth.columns

    if not has_homo and not has_depth and t_node is None:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1.2, 1]})
    scenario_desc = SCENARIO_DESCRIPTIONS.get(info["scenario"], info["scenario"])
    fig.suptitle(f"Homography vs Depth Camera -- {info['bag_name']}\n"
                 f"{info['scenario']} ({scenario_desc}) | Node: {info['node']}",
                 fontsize=12, fontweight="bold")

    # 1. Haut : Distances au cours du temps
    if t_node is not None:
        color = NODE_COLORS.get(info["node"], "#333")
        ax1.plot(t_node, d_node, linewidth=1.2, alpha=0.9, color=color,
                 label=f"Estimated distance ({source})")

    if has_depth:
        valid_depth = depth.dropna(subset=["d_min_depth_m"])
        valid_depth = valid_depth[valid_depth["d_min_depth_m"] < 10.0]
        if not valid_depth.empty:
            ax1.plot(valid_depth["t"], valid_depth["d_min_depth_m"],
                     linewidth=1.0, alpha=0.8, color="#E91E63",
                     label="Depth Camera measurement (central ROI)")

    if has_homo:
        valid_hp = robot_pos.dropna(subset=["x", "y"]).copy()
        valid_hp["dist_homo"] = np.sqrt(valid_hp["x"].astype(float)**2 + valid_hp["y"].astype(float)**2)
        valid_hp = valid_hp[valid_hp["dist_homo"] < 10.0]
        if not valid_hp.empty:
            ax1.plot(valid_hp["t"], valid_hp["dist_homo"],
                     linewidth=1.2, linestyle="-.", alpha=0.8, color="#9C27B0",
                     label="Homography distance (ground pos norm)")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Distance (m)")
    ax1.set_ylim(0, 10.0)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    shade_estop(ax1, data.get("estop"))

    # 2. Bas : Trajectoire Sol / Homographie (X vs Y)
    odom = data.get("odom")
    if odom is not None and not odom.empty and "x" in odom.columns and "y" in odom.columns:
        ax2.plot(odom["x"], odom["y"], color="#2196F3", linewidth=1.2, alpha=0.7,
                 label="Odometry trajectory (EKF)")
        ax2.scatter(odom["x"].iloc[0], odom["y"].iloc[0], color="green", s=50, marker="o", label="Start Odom")
        ax2.scatter(odom["x"].iloc[-1], odom["y"].iloc[-1], color="red", s=50, marker="x", label="End Odom")

    if has_homo:
        valid_hp = robot_pos.dropna(subset=["x", "y"])
        if not valid_hp.empty:
            ax2.scatter(valid_hp["x"], valid_hp["y"], c=valid_hp["t"], cmap="viridis",
                        s=15, alpha=0.6, label="Homography Ground Positions")

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Ground Plane: Odometry vs Homography Trajectory", fontsize=10)
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axis("equal")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_homography_vs_depth.png"))
    plt.close(fig)


def load_homography_calibration():
    """Charge la calibration d'homographie si disponible."""
    possible_paths = [
        os.path.join(BASE_DIR, "homography", "homography_calibration.json"),
        os.path.join(SCRIPT_DIR, "..", "homography", "homography_calibration.json"),
    ]
    for p in possible_paths:
        if os.path.isfile(p):
            try:
                with open(p, "r") as f:
                    calib = json.load(f)
                H = np.array(calib["H"]).reshape(3, 3)
                H_inv = np.linalg.inv(H)
                w = calib["metadata"].get("bird_eye_w", 300)
                h = calib["metadata"].get("bird_eye_h", 250)
                scale = calib["metadata"].get("bird_eye_scale", 0.01)
                return H, H_inv, w, h, scale
            except Exception:
                pass
    return None, None, 300, 250, 0.01


def ground_to_cam_pixel(x_m, y_m, H_inv, calib_w=300, calib_h=250, calib_scale=0.01):
    """
    Projette un point sol (x_m, y_m) vers les coordonnees pixel (u, v) de la camera
    via l'inverse de la matrice d'homographie H.
    """
    if H_inv is None:
        return None, None
    q_inv = np.zeros(3, dtype=np.float64)
    q_inv[0] = (y_m + calib_w * calib_scale / 2.0) / calib_scale
    q_inv[1] = calib_h - x_m / calib_scale
    q_inv[2] = 1.0
    p = H_inv @ q_inv
    if abs(p[2]) < 1e-6:
        return None, None
    u = p[0] / p[2]
    v = p[1] / p[2]
    return float(u), float(v)


def compute_depth_vs_detection_errors(bag_folder, data, max_dist=10.0, max_dt=0.25):
    """
    Compare l'estimation de distance du node actif avec la verite terrain de la camera de profondeur.

    1. Pour YOLO (ground_detections.csv) :
       - Si depth_frames.npz est disponible : re-projette chaque obstacle (x_m, y_m)
         en pixel camera (u, v) via H^-1 et extrait la profondeur mediane dans un patch local 20x20 px.
         Calcule la distance 3D euclidienne reelle.
       - Repli : compare dist_m a la profondeur centrale du CSV.

    2. Pour Farneback / RAFT (flow_diagnostics.csv ou movement_command.csv) :
       - Si depth_frames.npz est disponible : extrait la profondeur dans le tiers d'image actif
         (Gauche [0..213], Centre [213..427], Droite [427..640]).
       - Repli : compare min_dist_m ou dist_m au d_min_depth_m scalaire.

    Retourne un dictionnaire avec mean, median, std, n_matched, n_good_20cm, pct_good_20cm, method.
    """
    info = parse_bag_name(os.path.basename(bag_folder))
    node = info.get("node", "unknown")

    # Charger matrices et camera info
    H, H_inv, calib_w, calib_h, calib_scale = load_homography_calibration()
    cam_info_path = os.path.join(bag_folder, "camera_info.json")
    fx, fy, cx, cy = 616.7, 616.8, 321.1, 236.6
    if os.path.isfile(cam_info_path):
        try:
            ci = json.load(open(cam_info_path))
            fx, fy, cx, cy = ci.get("fx", fx), ci.get("fy", fy), ci.get("cx", cx), ci.get("cy", cy)
        except Exception:
            pass

    # Verifier si les frames de profondeur brutes sont disponibles
    npz_path = os.path.join(bag_folder, "depth_frames.npz")
    ts_path = os.path.join(bag_folder, "depth_timestamps.csv")
    has_npz = os.path.isfile(npz_path) and os.path.isfile(ts_path)

    depth_npz = None
    depth_ts_arr = None
    if has_npz:
        try:
            depth_npz = np.load(npz_path)
            depth_ts_df = pd.read_csv(ts_path)
            depth_ts_arr = depth_ts_df["t"].to_numpy()
        except Exception:
            depth_npz = None
            depth_ts_arr = None

    errors = []
    method_used = "scalar_roi"

    # ── Cas 1 : YOLO avec detections sol (ground_detections.csv) ──
    gd_df = data.get("ground_det")
    if gd_df is not None and not gd_df.empty and "dist_m" in gd_df.columns and "x_m" in gd_df.columns:
        valid_gd = gd_df.dropna(subset=["x_m", "y_m", "dist_m"]).copy()
        valid_gd = valid_gd[valid_gd["dist_m"] < max_dist]

        if depth_npz is not None and depth_ts_arr is not None and H_inv is not None and not valid_gd.empty:
            method_used = "pixel_homography_coupling"
            for _, row in valid_gd.iterrows():
                t_det = row["t"]
                x_m, y_m, dist_m = row["x_m"], row["y_m"], row["dist_m"]

                u, v = ground_to_cam_pixel(x_m, y_m, H_inv, calib_w, calib_h, calib_scale)
                if u is None or not (0 <= u < 640 and 0 <= v < 480):
                    continue

                idx = np.argmin(np.abs(depth_ts_arr - t_det))
                if abs(depth_ts_arr[idx] - t_det) > max_dt:
                    continue

                frame_key = f"frame_{idx:05d}"
                if frame_key not in depth_npz:
                    continue
                depth_frame = depth_npz[frame_key]

                # Patch 21x21 autour de (u, v)
                u_i, v_i = int(round(u)), int(round(v))
                w_rad = 10
                u_min, u_max = max(0, u_i - w_rad), min(640, u_i + w_rad + 1)
                v_min, v_max = max(0, v_i - w_rad), min(480, v_i + w_rad + 1)

                patch = depth_frame[v_min:v_max, u_min:u_max]
                valid_patch = patch[(patch > 0.2) & (patch < max_dist)]
                if len(valid_patch) < 5:
                    continue

                z_med = float(np.median(valid_patch))
                x_c = (u - cx) * z_med / fx
                y_c = (v - cy) * z_med / fy
                d_3d = float(np.sqrt(x_c**2 + y_c**2 + z_med**2))

                errors.append(abs(dist_m - d_3d))

    # ── Cas 2 : Farneback / RAFT ou repli scalaire ──
    if len(errors) == 0:
        # Obtenir la serie temporelle de distance estimee
        t_node, d_node, src = get_distance_series(data, node, max_dist=max_dist)
        depth_df = data.get("depth")

        if t_node is not None and depth_df is not None and not depth_df.empty and "d_min_depth_m" in depth_df.columns:
            depth_clean = depth_df.dropna(subset=["d_min_depth_m"]).copy()
            depth_clean = depth_clean[depth_clean["d_min_depth_m"] < max_dist]

            if not depth_clean.empty:
                depth_t = depth_clean["t"].to_numpy()
                depth_d = depth_clean["d_min_depth_m"].to_numpy()

                for tn, dn in zip(t_node, d_node):
                    idx = np.argmin(np.abs(depth_t - tn))
                    if abs(depth_t[idx] - tn) <= max_dt:
                        errors.append(abs(dn - depth_d[idx]))

    if len(errors) == 0:
        return None

    errors = np.array(errors)
    n_good = int((errors < 0.20).sum())

    return {
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "std_error_m": float(np.std(errors)),
        "n_matched": len(errors),
        "n_good_20cm": n_good,
        "pct_good_20cm": float(n_good / len(errors) * 100),
        "errors": errors,
        "method": method_used,
    }


def generate_folder_plots(bag_folder, bag_name):
    """Génère tous les graphiques pour un dossier bag."""
    info = parse_bag_name(bag_name)
    data = load_bag_data(bag_folder)

    # Enrichir info avec le summary
    if data["summary"]:
        if info["node"] == "unknown" and "node" in data["summary"]:
            info["node"] = data["summary"]["node"]

    out_dir = os.path.join(bag_folder, "plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"  Generating plots for {bag_name} (node={info['node']}, scenario={info['scenario']})...")

    plot_distance_comparison(data, info, out_dir)
    plot_ttc(data, info, out_dir)
    plot_commands(data, info, out_dir)
    plot_state_timeline(data, info, out_dir)
    plot_trajectory(data, info, out_dir)
    plot_resources(data, info, out_dir)
    plot_homography_vs_depth(data, info, out_dir)

    # Compute depth vs detection error stats
    err_stats = compute_depth_vs_detection_errors(bag_folder, data)
    if err_stats is not None:
        print(f"    Depth vs Detection: mean={err_stats['mean_error_m']:.3f}m, "
              f"median={err_stats['median_error_m']:.3f}m, "
              f"good(<20cm)={err_stats['n_good_20cm']}/{err_stats['n_matched']} "
              f"({err_stats['pct_good_20cm']:.1f}%)")

    print(f"  [OK] Plots sauvés dans {out_dir}/")
    return info, data


# ═══════════════════════════════════════════════════════════════════════
#  GRAPHIQUES COMPARATIFS
# ═══════════════════════════════════════════════════════════════════════

def build_recap_table_bags(bag_analysis_dir):
    """
    Construit le tableau recapitulatif a l'echelle des bags complets (22 sessions).
    """
    rows = []
    folders = sorted([
        d for d in os.listdir(bag_analysis_dir)
        if os.path.isdir(os.path.join(bag_analysis_dir, d)) and not d.startswith("comparatif")
    ])

    for bag_name in folders:
        bag_folder = os.path.join(bag_analysis_dir, bag_name)
        info = parse_bag_name(bag_name)

        summary_path = os.path.join(bag_folder, "summary.json")
        if not os.path.isfile(summary_path):
            continue
        with open(summary_path, "r") as f:
            summary = json.load(f)

        row = summary.copy()
        row["bag_name"] = bag_name
        if info["node"] != "unknown":
            row["node"] = info["node"]
        if row.get("scenario") in (None, "NA", ""):
            row["scenario"] = info["scenario"]
        row["scenario_desc"] = SCENARIO_DESCRIPTIONS.get(row["scenario"], "")
        row["trial"] = info["trial"]

        # Ressources
        resource_files = glob.glob(os.path.join(bag_folder, "resource*.csv"))
        if resource_files:
            try:
                res_df = pd.read_csv(resource_files[0])
                if not res_df.empty:
                    if "cpu_percent" in res_df:
                        row["cpu_percent_mean"] = round(res_df["cpu_percent"].mean(), 1)
                        row["cpu_percent_max"] = round(res_df["cpu_percent"].max(), 1)
                    if "ram_used_mb" in res_df:
                        row["ram_used_mb_mean"] = round(res_df["ram_used_mb"].mean(), 1)
                        row["ram_used_mb_max"] = round(res_df["ram_used_mb"].max(), 1)
                    if "gpu_percent" in res_df and res_df["gpu_percent"].notna().any():
                        row["gpu_percent_mean"] = round(res_df["gpu_percent"].dropna().mean(), 1)
                        row["gpu_percent_max"] = round(res_df["gpu_percent"].dropna().max(), 1)
                    if "gpu_mem_used_mb" in res_df and res_df["gpu_mem_used_mb"].notna().any():
                        row["gpu_mem_used_mb_mean"] = round(res_df["gpu_mem_used_mb"].dropna().mean(), 1)
                        row["gpu_mem_used_mb_max"] = round(res_df["gpu_mem_used_mb"].dropna().max(), 1)
            except Exception:
                pass

        # Erreurs de distance Depth vs Node
        try:
            _data = load_bag_data(bag_folder)
            err_stats = compute_depth_vs_detection_errors(bag_folder, _data)
            if err_stats is not None:
                row["dist_err_mean_m"] = round(err_stats["mean_error_m"], 4)
                row["dist_err_median_m"] = round(err_stats["median_error_m"], 4)
                row["dist_err_std_m"] = round(err_stats["std_error_m"], 4)
                row["dist_err_n_matched"] = err_stats["n_matched"]
                row["dist_err_n_good_20cm"] = err_stats["n_good_20cm"]
                row["dist_err_pct_good_20cm"] = round(err_stats["pct_good_20cm"], 1)
        except Exception:
            pass

        rows.append(row)

    return pd.DataFrame(rows)


def build_recap_table_trials(bag_analysis_dir):
    """
    Construit le tableau recapitulatif a l'echelle des essais individuels (111 trials).
    Chaque trial_XX decoupe d'un bag est traite comme une observation unitaire.
    """
    rows = []
    folders = sorted([
        d for d in os.listdir(bag_analysis_dir)
        if os.path.isdir(os.path.join(bag_analysis_dir, d)) and not d.startswith("comparatif")
    ])

    for bag_name in folders:
        bag_folder = os.path.join(bag_analysis_dir, bag_name)
        info_parent = parse_bag_name(bag_name)

        trials_dir = os.path.join(bag_folder, "trials")
        if not os.path.isdir(trials_dir):
            continue

        # Charger stats de ressources parentes si dispo
        parent_res = {}
        resource_files = glob.glob(os.path.join(bag_folder, "resource*.csv"))
        if resource_files:
            try:
                res_df = pd.read_csv(resource_files[0])
                if not res_df.empty:
                    if "cpu_percent" in res_df:
                        parent_res["cpu_percent_mean"] = round(res_df["cpu_percent"].mean(), 1)
                        parent_res["cpu_percent_max"] = round(res_df["cpu_percent"].max(), 1)
                    if "ram_used_mb" in res_df:
                        parent_res["ram_used_mb_mean"] = round(res_df["ram_used_mb"].mean(), 1)
                        parent_res["ram_used_mb_max"] = round(res_df["ram_used_mb"].max(), 1)
                    if "gpu_percent" in res_df and res_df["gpu_percent"].notna().any():
                        parent_res["gpu_percent_mean"] = round(res_df["gpu_percent"].dropna().mean(), 1)
                        parent_res["gpu_percent_max"] = round(res_df["gpu_percent"].dropna().max(), 1)
                    if "gpu_mem_used_mb" in res_df and res_df["gpu_mem_used_mb"].notna().any():
                        parent_res["gpu_mem_used_mb_mean"] = round(res_df["gpu_mem_used_mb"].dropna().mean(), 1)
                        parent_res["gpu_mem_used_mb_max"] = round(res_df["gpu_mem_used_mb"].dropna().max(), 1)
            except Exception:
                pass

        sub_trials = sorted([
            x for x in os.listdir(trials_dir)
            if x.startswith("trial_") and os.path.isdir(os.path.join(trials_dir, x))
        ])

        for sub in sub_trials:
            trial_path = os.path.join(trials_dir, sub)
            info = parse_bag_name(f"{bag_name}_{sub}")

            # Charger summary et trial_metrics
            ts_path = os.path.join(trial_path, "summary.json")
            tm_path = os.path.join(trial_path, "trial_metrics.json")
            summary = {}
            if os.path.isfile(ts_path):
                with open(ts_path, "r") as f:
                    summary = json.load(f)
            if os.path.isfile(tm_path):
                with open(tm_path, "r") as f:
                    tm = json.load(f)
                    summary.update(tm)

            if not summary:
                continue

            row = summary.copy()
            row["bag_name"] = f"{bag_name}_{sub}"
            row["parent_bag"] = bag_name
            row["trial_id"] = sub
            if info["node"] != "unknown":
                row["node"] = info["node"]
            elif info_parent["node"] != "unknown":
                row["node"] = info_parent["node"]
            if row.get("scenario") in (None, "NA", ""):
                row["scenario"] = info_parent["scenario"]
            row["scenario_desc"] = SCENARIO_DESCRIPTIONS.get(row["scenario"], "")
            row["trial"] = info.get("trial", 1)

            # Recalculer delta_psi et jerk réels sur l'odom du trial, fenêtrés
            # sur les épisodes d'évitement (/avoidance_cmd_vel) -- cohérent
            # avec compute_metrics() dans analyse-bag-test.py. On ne recalcule
            # QUE si /avoidance_cmd_vel est disponible pour ce trial ; sinon on
            # conserve la valeur héritée de summary.json / trial_metrics.json
            # plutôt que de l'écraser par une approximation non fenêtrée
            # (sur toute la durée du trial), qui n'est pas comparable au
            # reste du protocole.
            _tdata = load_bag_data(trial_path)
            odom_df = _tdata.get("odom")
            avoid_df = _tdata.get("avoidance")
            if (odom_df is not None and not odom_df.empty and "wz" in odom_df.columns
                    and avoid_df is not None and not avoid_df.empty):
                try:
                    odom_sorted = odom_df.sort_values("t").reset_index(drop=True)
                    mask = _build_avoidance_mask_local(odom_sorted, avoid_df)
                    sub = odom_sorted[mask]
                    if len(sub) >= 2:
                        t = sub["t"].to_numpy(copy=True)
                        wz = sub["wz"].to_numpy(copy=True)
                        dt = pd.Series(t).diff().to_numpy(copy=True)
                        dt[0] = 0.0
                        d_psi = float((np.abs(wz) * dt).sum())
                        row["delta_psi_cum_rad"] = round(d_psi, 4)
                        if len(t) >= 3:
                            dwz = pd.Series(wz).diff().to_numpy(copy=True)[1:]
                            dtv = dt[1:].copy()
                            dtv[dtv == 0] = np.nan  # évite une division par 0 -> inf
                            jerk_series = dwz / dtv
                            jerk_series = jerk_series[~pd.isna(jerk_series)]
                            if jerk_series.size:
                                row["jerk_rms_rad_s2"] = round(float((jerk_series ** 2).mean() ** 0.5), 4)
                except Exception:
                    pass

            # Ajouter ressources parentes
            for k, v in parent_res.items():
                if k not in row or row[k] is None:
                    row[k] = v

            # Distance error Depth vs Node pour ce trial
            try:
                _tdata = load_bag_data(trial_path)
                err_stats = compute_depth_vs_detection_errors(trial_path, _tdata)
                if err_stats is not None:
                    row["dist_err_mean_m"] = round(err_stats["mean_error_m"], 4)
                    row["dist_err_median_m"] = round(err_stats["median_error_m"], 4)
                    row["dist_err_std_m"] = round(err_stats["std_error_m"], 4)
                    row["dist_err_n_matched"] = err_stats["n_matched"]
                    row["dist_err_n_good_20cm"] = err_stats["n_good_20cm"]
                    row["dist_err_pct_good_20cm"] = round(err_stats["pct_good_20cm"], 1)
            except Exception:
                pass

            rows.append(row)

    return pd.DataFrame(rows)


def build_recap_table(bag_analysis_dir, mode="trials"):
    """
    Construit le tableau recapitulatif selon le mode demande :
      - 'trials' (defaut) : 111 essais individuels
      - 'bags'            : 22 sessions completes
    """
    if mode == "bags":
        return build_recap_table_bags(bag_analysis_dir)
    else:
        return build_recap_table_trials(bag_analysis_dir)


def plot_comparative_bars(recap_df, out_dir):
    """
    Bar charts comparatifs : métriques clés groupées par scénario,
    barres colorées par node.
    """
    metrics = [
        ("d_min_m", "Min Distance (m)", True),
        ("ttc_min_s", "Min TTC (s)", True),
        ("estop_count", "E-STOP Count", False),
        ("n_avoidance_episodes", "Avoidance Episodes", False),
        ("delta_psi_cum_rad", "Cumulative Δψ (rad)", False),
        ("fps_hz", "Inference Rate (FPS)", True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Scenario Comparison -- Key Metrics", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for idx, (col, label, higher_better) in enumerate(metrics):
        ax = axes[idx]
        if col not in recap_df.columns:
            ax.set_visible(False)
            continue

        # Grouper par scénario et node
        plot_data = recap_df.dropna(subset=[col]).copy()
        if plot_data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue

        scenarios = sorted(plot_data["scenario"].unique())
        nodes = sorted(plot_data["node"].unique())
        x = np.arange(len(scenarios))
        width = 0.35

        for i, node in enumerate(nodes):
            node_data = plot_data[plot_data["node"] == node]
            values = [node_data[node_data["scenario"] == s][col].mean()
                      if s in node_data["scenario"].values else 0
                      for s in scenarios]
            offset = (i - (len(nodes) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width * 0.9,
                          color=NODE_COLORS.get(node, "#9E9E9E"),
                          alpha=0.8, label=node)

        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_comparative_bars.png"))
    plt.close(fig)


def plot_distance_scatter(recap_df, out_dir):
    """
    Scatter : d_min (estimee) vs d_min_depth (verite terrain) limite a 2.0m max.
    """
    df = recap_df.dropna(subset=["d_min_m", "d_min_depth_m"]).copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Estimated Min Distance vs Ground Truth (Depth Camera) [Zoom ≤ 2.0m]",
                 fontsize=12, fontweight="bold")

    for node in df["node"].unique():
        sub = df[df["node"] == node]
        ax.scatter(sub["d_min_depth_m"], sub["d_min_m"], s=60, alpha=0.75,
                   color=NODE_COLORS.get(node, "#9E9E9E"), label=node,
                   edgecolors="white", linewidth=0.8)
        # Annoter seulement si le nombre de points est raisonnable pour eviter les chevauchements
        if len(df) <= 30:
            for _, row in sub.iterrows():
                if row["d_min_depth_m"] <= 2.0 and row["d_min_m"] <= 2.0:
                    ax.annotate(row["scenario"], (row["d_min_depth_m"], row["d_min_m"]),
                                fontsize=7, alpha=0.7, ha="left",
                                xytext=(5, 3), textcoords="offset points")

    # Droite y = x
    ax.plot([0, 2.0], [0, 2.0], "k--", linewidth=1.0, alpha=0.5, label="y = x (ideal)")

    # Zone de proximite immediate 0.20m
    ax.axhspan(0, 0.20, color="red", alpha=0.06, label="Critical proximity (<0.20m)")
    ax.axvspan(0, 0.20, color="red", alpha=0.06)

    ax.set_xlabel("Depth camera distance (m)", fontsize=10)
    ax.set_ylabel("Estimated distance (m)", fontsize=10)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 2.0)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(out_dir, "02_distance_scatter.png"), dpi=150)
    plt.close(fig)


def plot_ttc_scatter(recap_df, out_dir):
    """
    Scatter : ttc_min estimé vs ttc_real_min (vérité terrain depth).
    """
    df = recap_df.dropna(subset=["ttc_min_s", "ttc_real_min_s"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Estimated Min TTC vs Real TTC (Depth, Finite Diff.)",
                 fontsize=12, fontweight="bold")

    for node in df["node"].unique():
        sub = df[df["node"] == node]
        ax.scatter(sub["ttc_real_min_s"], sub["ttc_min_s"], s=80, alpha=0.7,
                   color=NODE_COLORS.get(node, "#9E9E9E"), label=node,
                   edgecolors="white", linewidth=0.8)
        for _, row in sub.iterrows():
            ax.annotate(row["scenario"], (row["ttc_real_min_s"], row["ttc_min_s"]),
                        fontsize=7, alpha=0.7, ha="left",
                        xytext=(5, 3), textcoords="offset points")

    # Droite y = x
    ax.plot([0, 10.0], [0, 10.0], "k--", linewidth=0.8, alpha=0.4, label="y = x (ideal)")

    ax.set_xlabel("Real TTC (s)")
    ax.set_ylabel("Estimated TTC (s)")
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 10.0)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(out_dir, "03_ttc_scatter.png"))
    plt.close(fig)


def plot_heatmap(recap_df, out_dir):
    """
    Heatmap : métriques normalisées × scénarios.
    """
    metric_cols = [
        "d_min_m", "ttc_min_s", "d_min_depth_m", "ttc_error_s",
        "estop_count", "n_avoidance_episodes",
        "delta_psi_cum_rad", "jerk_rms_rad_s2",
        "fps_hz", "bag_duration_s",
    ]

    available = [c for c in metric_cols if c in recap_df.columns]
    if not available:
        return

    # Pivot : index = bag_name (ou scenario+node), columns = métriques
    pivot = recap_df.set_index("bag_name")[available].apply(pd.to_numeric, errors="coerce")
    pivot = pivot.dropna(how="all")
    if pivot.empty:
        return

    # Normaliser chaque colonne 0-1
    normalized = (pivot - pivot.min()) / (pivot.max() - pivot.min() + 1e-10)

    fig, ax = plt.subplots(figsize=(14, max(6, len(normalized) * 0.5)))
    ax.set_title("Metrics Heatmap by Test (Normalized 0-1)",
                 fontsize=12, fontweight="bold")

    cmap = LinearSegmentedColormap.from_list("custom", ["#E3F2FD", "#1565C0"], N=256)
    im = ax.imshow(normalized.values, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(available, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(normalized)))
    ax.set_yticklabels(normalized.index, fontsize=8)

    # Annoter avec les valeurs réelles
    for i in range(len(normalized)):
        for j in range(len(available)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                text = f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
                color = "white" if normalized.iloc[i, j] > 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Normalized")
    fig.savefig(os.path.join(out_dir, "04_heatmap.png"))
    plt.close(fig)


def plot_node_summary(recap_df, out_dir):
    """
    Radar / résumé par node avec les moyennes des métriques.
    """
    metrics = {
        "d_min_m": "Min Distance",
        "estop_count": "E-STOPs",
        "n_avoidance_episodes": "Avoid Episodes",
        "fps_hz": "FPS",
        "delta_psi_cum_rad": "Cumul Δψ",
    }

    available = {k: v for k, v in metrics.items() if k in recap_df.columns}
    if not available:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Node Summary -- Metric Averages",
                 fontsize=12, fontweight="bold")

    nodes = sorted(recap_df["node"].unique())
    x = np.arange(len(available))
    width = 0.8 / len(nodes)

    for i, node in enumerate(nodes):
        node_data = recap_df[recap_df["node"] == node]
        values = [node_data[col].mean() if col in node_data else 0
                  for col in available.keys()]
        offset = (i - (len(nodes) - 1) / 2) * width
        ax.bar(x + offset, values, width * 0.9,
               color=NODE_COLORS.get(node, "#9E9E9E"), alpha=0.8, label=node)

    ax.set_xticks(x)
    ax.set_xticklabels(list(available.values()), fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(out_dir, "05_node_summary.png"))
    plt.close(fig)

def plot_homography_accuracy(recap_df, out_dir):
    """
    Plot comparatif 06 : Comparaison de précision Homographie vs Depth Camera.
    """
    if "d_min_homography_m" not in recap_df.columns or "d_min_depth_m" not in recap_df.columns:
        return
    df = recap_df.dropna(subset=["d_min_homography_m", "d_min_depth_m"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Distance min Homographie vs Vérité terrain (Depth camera)",
                 fontsize=12, fontweight="bold")

    for node in df["node"].unique():
        sub = df[df["node"] == node]
        ax.scatter(sub["d_min_depth_m"], sub["d_min_homography_m"], s=80, alpha=0.7,
                   color=NODE_COLORS.get(node, "#9E9E9E"), label=node,
                   edgecolors="white", linewidth=0.8)
        for _, row in sub.iterrows():
            ax.annotate(row["scenario"], (row["d_min_depth_m"], row["d_min_homography_m"]),
                        fontsize=7, alpha=0.7, ha="left",
                        xytext=(5, 3), textcoords="offset points")

    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.4, label="y = x (idéal)")

    ax.set_xlabel("Distance Depth Camera (m)")
    ax.set_ylabel("Distance calculée Homographie (m)")
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 10.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(out_dir, "06_homography_accuracy.png"))
    plt.close(fig)


def plot_resource_consumption(recap_df, out_dir):
    """
    Plot comparatif 07 : Consommation moyenne des ressources système par node (CPU, RAM, GPU, VRAM).
    """
    res_cols = ["cpu_percent_mean", "ram_used_mb_mean", "gpu_percent_mean", "gpu_mem_used_mb_mean"]
    available_cols = [c for c in res_cols if c in recap_df.columns and recap_df[c].notna().any()]
    if not available_cols:
        return

    nodes = sorted(recap_df["node"].unique())
    # Filtrer les nodes qui ont au moins une donnée de ressource
    nodes_with_data = []
    for node in nodes:
        sub = recap_df[recap_df["node"] == node]
        if sub[available_cols].notna().any().any():
            nodes_with_data.append(node)

    if not nodes_with_data:
        return

    panels = [
        ("cpu_percent_mean", "cpu_percent_max", "CPU Load (%)", "CPU (%)", 100),
        ("ram_used_mb_mean", "ram_used_mb_max", "RAM Usage (MB)", "RAM (MB)", None),
        ("gpu_percent_mean", "gpu_percent_max", "GPU Load (%)", "GPU (%)", 100),
        ("gpu_mem_used_mb_mean", "gpu_mem_used_mb_max", "GPU VRAM (MB)", "VRAM (MB)", None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Average System Resource Consumption by Node", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for idx, (mean_col, max_col, title, ylabel, ylim) in enumerate(panels):
        ax = axes[idx]
        if mean_col not in recap_df.columns:
            ax.set_visible(False)
            continue

        x = np.arange(len(nodes_with_data))
        means = []
        maxs = []
        colors = []

        for node in nodes_with_data:
            sub = recap_df[recap_df["node"] == node]
            mean_val = sub[mean_col].dropna().mean() if not sub[mean_col].dropna().empty else 0.0
            max_val = sub[max_col].dropna().mean() if (max_col in sub.columns and not sub[max_col].dropna().empty) else mean_val
            means.append(mean_val)
            maxs.append(max_val)
            colors.append(NODE_COLORS.get(node, "#9E9E9E"))

        # Barres pour la moyenne
        bars = ax.bar(x, means, width=0.55, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2, label="Mean")

        # Afficher la valeur moyenne et max au-dessus de chaque barre
        for i, (b, m, mx) in enumerate(zip(bars, means, maxs)):
            if m > 0:
                ax.text(b.get_x() + b.get_width() / 2, m + (max(means) * 0.02 if max(means) > 0 else 0.5),
                        f"Mean: {m:.1f}\n(Peak: {mx:.1f})", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([n.upper() for n in nodes_with_data], fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        if ylim:
            ax.set_ylim(0, max(ylim, max(maxs) * 1.15 if maxs else 100))
        else:
            ax.set_ylim(0, max(maxs) * 1.25 if maxs and max(maxs) > 0 else 10)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_resources_summary.png"))
    plt.close(fig)


def plot_depth_vs_detection_errors(recap_df, out_dir, bag_analysis_dir):
    """
    Plot comparatif 08 : Distance error between depth camera ground truth
    and the node's estimated distance.
    Grouped cleanly by Scenario and Node for perfect readability on all dataset sizes.
    """
    err_cols = ["dist_err_mean_m", "dist_err_median_m", "dist_err_n_good_20cm",
                "dist_err_n_matched", "dist_err_pct_good_20cm"]
    if not all(c in recap_df.columns for c in err_cols[:2]):
        return
    df = recap_df.dropna(subset=["dist_err_mean_m"]).copy()
    if df.empty:
        return

    nodes = [n for n in ["farneback", "raft", "yolo"] if n in df["node"].values]
    if not nodes:
        nodes = sorted(df["node"].unique())

    scenarios = sorted(df["scenario"].unique(), key=lambda s: [float(x) if x.replace('.', '', 1).isdigit() else 0 for x in [s.replace('S', '')]])

    fig = plt.figure(figsize=(18, 13))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.3, 1.0, 0.6], hspace=0.38, wspace=0.22)
    fig.suptitle("Depth Camera vs Node Distance Error Analysis",
                 fontsize=14, fontweight="bold")

    # ── Panel 1: Scenario-Grouped Mean & Median Error ──
    ax1 = fig.add_subplot(gs[0, :])
    x_scen = np.arange(len(scenarios))
    n_nodes = len(nodes)
    total_group_w = 0.75
    node_w = total_group_w / n_nodes
    sub_bar_w = node_w * 0.45

    for i_node, node in enumerate(nodes):
        sub_node = df[df["node"] == node]
        offset_node = (i_node - (n_nodes - 1) / 2.0) * node_w

        mean_per_scen = []
        med_per_scen = []
        for s in scenarios:
            s_data = sub_node[sub_node["scenario"] == s]
            mean_per_scen.append(s_data["dist_err_mean_m"].mean() if not s_data.empty else np.nan)
            med_per_scen.append(s_data["dist_err_median_m"].mean() if not s_data.empty else np.nan)

        col = NODE_COLORS.get(node, "#9E9E9E")
        pos_mean = x_scen + offset_node - sub_bar_w / 2.0
        pos_med = x_scen + offset_node + sub_bar_w / 2.0

        # Mean bars (solid)
        ax1.bar(pos_mean, mean_per_scen, sub_bar_w, color=col, alpha=0.85,
                edgecolor="white", linewidth=0.6, label=f"{node.upper()} (Mean)")
        # Median bars (hatched)
        ax1.bar(pos_med, med_per_scen, sub_bar_w, color=col, alpha=0.50,
                edgecolor="white", linewidth=0.6, hatch="//", label=f"{node.upper()} (Median)")

    ax1.axhline(0.20, color="green", linestyle="--", linewidth=1.0, alpha=0.75,
                label="Target Accuracy Threshold (20 cm)")
    ax1.set_xticks(x_scen)
    ax1.set_xticklabels(scenarios, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Distance Error (m)", fontsize=10)
    ax1.set_title("Mean & Median Distance Error by Scenario & Node", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8, ncol=min(len(nodes)*2 + 1, 7))
    ax1.grid(axis="y", alpha=0.3)

    # ── Panel 2: Scenario-Grouped % Good Estimations (< 20 cm) ──
    ax2 = fig.add_subplot(gs[1, 0])
    for i_node, node in enumerate(nodes):
        sub_node = df[df["node"] == node]
        offset_node = (i_node - (n_nodes - 1) / 2.0) * node_w
        pct_per_scen = []
        for s in scenarios:
            s_data = sub_node[sub_node["scenario"] == s]
            if not s_data.empty and s_data["dist_err_n_matched"].sum() > 0:
                pct = (s_data["dist_err_n_good_20cm"].sum() / s_data["dist_err_n_matched"].sum()) * 100.0
                pct_per_scen.append(pct)
            else:
                pct_per_scen.append(np.nan)

        col = NODE_COLORS.get(node, "#9E9E9E")
        pos = x_scen + offset_node
        bars = ax2.bar(pos, pct_per_scen, node_w * 0.85, color=col, alpha=0.85,
                       edgecolor="white", linewidth=0.6, label=node.upper())
        for b in bars:
            h = b.get_height()
            if not np.isnan(h) and h > 0:
                ax2.text(b.get_x() + b.get_width()/2.0, h + 1.2,
                         f"{h:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax2.set_xticks(x_scen)
    ax2.set_xticklabels(scenarios, fontsize=9, fontweight="bold")
    ax2.set_ylabel("Good Estimations (% < 20cm)", fontsize=10)
    ax2.set_title("% Good Estimations (< 20 cm) by Scenario", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # ── Panel 3: Global Aggregated by Node ──
    ax3 = fig.add_subplot(gs[1, 1])
    node_x = np.arange(len(nodes))
    node_bar_w = 0.32
    mean_vals = [df[df["node"] == n]["dist_err_mean_m"].mean() for n in nodes]
    median_vals = [df[df["node"] == n]["dist_err_median_m"].mean() for n in nodes]
    node_cols = [NODE_COLORS.get(n, "#9E9E9E") for n in nodes]

    bars_mean = ax3.bar(node_x - node_bar_w/2, mean_vals, node_bar_w, color=node_cols, alpha=0.85,
                        edgecolor="white", linewidth=0.8, label="Mean Error (avg)")
    bars_med = ax3.bar(node_x + node_bar_w/2, median_vals, node_bar_w, color=node_cols, alpha=0.50,
                       edgecolor="white", linewidth=0.8, hatch="//", label="Median Error (avg)")
    ax3.axhline(0.20, color="green", linestyle="--", linewidth=1.0, alpha=0.75)

    for i, (mv, mdv) in enumerate(zip(mean_vals, median_vals)):
        if not np.isnan(mv):
            ax3.text(i - node_bar_w/2, mv + 0.02, f"{mv:.3f}m", ha="center", va="bottom", fontsize=8, fontweight="bold")
        if not np.isnan(mdv):
            ax3.text(i + node_bar_w/2, mdv + 0.02, f"{mdv:.3f}m", ha="center", va="bottom", fontsize=8)

    ax3.set_xticks(node_x)
    ax3.set_xticklabels([n.upper() for n in nodes], fontsize=10, fontweight="bold")
    ax3.set_ylabel("Distance Error (m)", fontsize=10)
    ax3.set_title("Overall Node Summary: Mean & Median Error", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    # ── Panel 4: Summary table ──
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    table_data = []
    for node in nodes:
        sub = df[df["node"] == node]
        total_matched = int(sub["dist_err_n_matched"].sum()) if "dist_err_n_matched" in sub else 0
        total_good = int(sub["dist_err_n_good_20cm"].sum()) if "dist_err_n_good_20cm" in sub else 0
        pct_all = (total_good / total_matched * 100) if total_matched > 0 else 0
        table_data.append([
            node.upper(),
            len(sub),
            f"{sub['dist_err_mean_m'].mean():.4f}",
            f"{sub['dist_err_median_m'].mean():.4f}",
            total_matched,
            total_good,
            f"{pct_all:.1f}%",
        ])
    # Overall row
    total_matched = int(df["dist_err_n_matched"].sum()) if "dist_err_n_matched" in df else 0
    total_good = int(df["dist_err_n_good_20cm"].sum()) if "dist_err_n_good_20cm" in df else 0
    pct_all = (total_good / total_matched * 100) if total_matched > 0 else 0
    table_data.append([
        "OVERALL",
        len(df),
        f"{df['dist_err_mean_m'].mean():.4f}",
        f"{df['dist_err_median_m'].mean():.4f}",
        total_matched,
        total_good,
        f"{pct_all:.1f}%",
    ])
    col_labels = ["Node", "Evaluations", "Mean Err (m)", "Median Err (m)",
                  "Matched Pts", "Good (<20cm)", "% Good"]
    table = ax4.table(cellText=table_data, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for j in range(len(col_labels)):
        table[len(table_data), j].set_facecolor("#E3F2FD")
        table[len(table_data), j].set_text_props(fontweight="bold")
    for i, node in enumerate(nodes):
        table[i + 1, 0].set_facecolor(NODE_COLORS.get(node, "#9E9E9E") + "33")
    ax4.set_title("Summary: Distance Error Statistics", fontsize=11, fontweight="bold", pad=10)

    fig.savefig(os.path.join(out_dir, "08_depth_vs_detection_errors.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_distance_range_breakdown(bag_analysis_dir):
    """
    Calcule la décomposition détaillée des erreurs de distance par tranche de distance
    (ex: <=2m vs >2m, et par tranches de 0.5m) sur l'ensemble des détections.
    """
    H, H_inv, w, h, s = load_homography_calibration()
    all_points = []

    folders = sorted([
        d for d in os.listdir(bag_analysis_dir)
        if os.path.isdir(os.path.join(bag_analysis_dir, d)) and not d.startswith("comparatif")
    ])

    for bag_name in folders:
        bag_folder = os.path.join(bag_analysis_dir, bag_name)
        info = parse_bag_name(bag_name)
        node = info.get("node", "unknown")
        data = load_bag_data(bag_folder)

        npz_path = os.path.join(bag_folder, "depth_frames.npz")
        ts_path = os.path.join(bag_folder, "depth_timestamps.csv")
        has_npz = os.path.isfile(npz_path) and os.path.isfile(ts_path)

        cam_info_path = os.path.join(bag_folder, "camera_info.json")
        fx, fy, cx, cy = 616.7, 616.8, 321.1, 236.6
        if os.path.isfile(cam_info_path):
            try:
                ci = json.load(open(cam_info_path))
                fx, fy, cx, cy = ci.get("fx", fx), ci.get("fy", fy), ci.get("cx", cx), ci.get("cy", cy)
            except Exception:
                pass

        if node == "yolo" and has_npz:
            gd_df = data.get("ground_det")
            if gd_df is not None and not gd_df.empty and "dist_m" in gd_df.columns:
                try:
                    depth_npz = np.load(npz_path)
                    depth_ts = pd.read_csv(ts_path)["t"].to_numpy()
                    for _, row in gd_df.dropna(subset=["x_m", "y_m", "dist_m"]).iterrows():
                        t_det = row["t"]
                        x_m, y_m, dist_m = row["x_m"], row["y_m"], row["dist_m"]
                        u, v = ground_to_cam_pixel(x_m, y_m, H_inv, w, h, s)
                        if u is None or not (0 <= u < 640 and 0 <= v < 480):
                            continue
                        idx = np.argmin(np.abs(depth_ts - t_det))
                        if abs(depth_ts[idx] - t_det) > 0.10:
                            continue
                        fkey = f"frame_{idx:05d}"
                        if fkey not in depth_npz:
                            continue
                        frame = depth_npz[fkey]
                        u_i, v_i = int(round(u)), int(round(v))
                        patch = frame[max(0, v_i-10):min(480, v_i+11), max(0, u_i-10):min(640, u_i+11)]
                        valid = patch[(patch > 0.1) & (patch < 10.0)]
                        if len(valid) == 0:
                            continue
                        Z = float(np.median(valid))
                        X_c = (u - cx) * Z / fx
                        Y_c = (v - cy) * Z / fy
                        d_3d = float(np.sqrt(X_c**2 + Y_c**2 + Z**2))
                        err = abs(dist_m - d_3d)
                        all_points.append({
                            "bag_name": bag_name,
                            "node": node,
                            "scenario": info.get("scenario", "NA"),
                            "d_estimated_m": dist_m,
                            "d_depth_m": d_3d,
                            "error_m": err,
                            "good_20cm": err < 0.20
                        })
                except Exception:
                    pass

        elif node in ("farneback", "raft"):
            flow_diag = data.get("flow_diag")
            depth_df = data.get("depth")
            if flow_diag is not None and not flow_diag.empty and depth_df is not None and not depth_df.empty:
                dist_col = None
                for c in ["min_dist_m", "distance_obstacle_m", "obstacle_distance_m", "dist_m"]:
                    if c in flow_diag.columns:
                        dist_col = c
                        break
                depth_col = None
                for c in ["d_min_depth_m", "min_distance", "depth_m"]:
                    if c in depth_df.columns:
                        depth_col = c
                        break
                t_f_col = "t" if "t" in flow_diag.columns else "timestamp"
                t_d_col = "t" if "t" in depth_df.columns else "timestamp"

                if dist_col is not None and depth_col is not None:
                    valid_flow = flow_diag.dropna(subset=[dist_col]).copy()
                    valid_depth = depth_df.dropna(subset=[depth_col]).copy()
                    if not valid_flow.empty and not valid_depth.empty:
                        for _, row in valid_flow.iterrows():
                            t_f = row[t_f_col]
                            d_est = row[dist_col]
                            if d_est <= 0 or d_est > 10.0:
                                continue
                            idx_d = np.argmin(np.abs(valid_depth[t_d_col] - t_f))
                            if abs(valid_depth[t_d_col].iloc[idx_d] - t_f) > 0.10:
                                continue
                            d_depth = valid_depth[depth_col].iloc[idx_d]
                            if d_depth <= 0 or d_depth > 10.0:
                                continue
                            err = abs(d_est - d_depth)
                            all_points.append({
                                "bag_name": bag_name,
                                "node": node,
                                "scenario": info.get("scenario", "NA"),
                                "d_estimated_m": d_est,
                                "d_depth_m": d_depth,
                                "error_m": err,
                                "good_20cm": err < 0.20
                            })

    return pd.DataFrame(all_points)


def plot_distance_range_breakdown(points_df, out_dir):
    """
    Génère le graphique 09 et le tableau CSV d'analyse des erreurs par tranche de distance.
    """
    if points_df.empty:
        return

    # Sauvegarde du CSV complet des détections couplées
    csv_raw_path = os.path.join(out_dir, "distance_coupling_raw_points.csv")
    points_df.to_csv(csv_raw_path, index=False)

    # 1. Découpage par tranches de distance
    bins = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, np.inf]
    labels = ["0.0-1.0m", "1.0-1.5m", "1.5-2.0m", "2.0-2.5m", "2.5-3.0m", "3.0-5.0m", ">5.0m"]
    points_df["range_bin"] = pd.cut(points_df["d_depth_m"], bins=bins, labels=labels)

    # Tableau récapitulatif par tranche et par node
    breakdown_rows = []
    for node in points_df["node"].unique():
        sub_node = points_df[points_df["node"] == node]
        for b in labels:
            sub = sub_node[sub_node["range_bin"] == b]
            n_pts = len(sub)
            if n_pts > 0:
                mean_e = sub["error_m"].mean()
                med_e = sub["error_m"].median()
                std_e = sub["error_m"].std()
                n_good = sub["good_20cm"].sum()
                pct_good = (n_good / n_pts) * 100.0
            else:
                mean_e, med_e, std_e, n_good, pct_good = np.nan, np.nan, np.nan, 0, 0.0

            breakdown_rows.append({
                "node": node.upper(),
                "range_bin": b,
                "points_count": n_pts,
                "mean_error_m": round(mean_e, 4) if not np.isnan(mean_e) else None,
                "median_error_m": round(med_e, 4) if not np.isnan(med_e) else None,
                "std_error_m": round(std_e, 4) if not np.isnan(std_e) else None,
                "good_20cm_count": int(n_good),
                "pct_good_20cm": round(pct_good, 1)
            })

    breakdown_df = pd.DataFrame(breakdown_rows)
    csv_breakdown_path = os.path.join(out_dir, "distance_range_breakdown.csv")
    breakdown_df.to_csv(csv_breakdown_path, index=False)

    # CSV spécifique YOLO
    yolo_df = points_df[points_df["node"] == "yolo"]
    if not yolo_df.empty:
        yolo_breakdown = breakdown_df[breakdown_df["node"] == "YOLO"]
        yolo_breakdown.to_csv(os.path.join(out_dir, "yolo_distance_range_breakdown.csv"), index=False)

    # ── Visualisation 09 ──
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1.0, 0.8], hspace=0.38, wspace=0.25)
    fig.suptitle("Distance Estimation Error Breakdown by Distance Range", fontsize=14, fontweight="bold")

    nodes = sorted(points_df["node"].unique())
    x = np.arange(len(labels))
    w_bar = 0.8 / len(nodes)

    # Panneau 1: Erreur Médiane par tranche de distance
    ax1 = fig.add_subplot(gs[0, 0])
    for i_n, node in enumerate(nodes):
        sub_b = breakdown_df[breakdown_df["node"] == node.upper()]
        meds = [sub_b[sub_b["range_bin"] == b]["median_error_m"].values[0] if not sub_b[sub_b["range_bin"] == b].empty else np.nan for b in labels]
        col = NODE_COLORS.get(node, "#9E9E9E")
        pos = x + (i_n - (len(nodes)-1)/2.0) * w_bar
        bars = ax1.bar(pos, meds, w_bar*0.9, color=col, alpha=0.85, edgecolor="white", label=node.upper())
        for b_obj in bars:
            h = b_obj.get_height()
            if not np.isnan(h) and h > 0:
                ax1.text(b_obj.get_x() + b_obj.get_width()/2.0, h + 0.05, f"{h:.2f}m", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax1.axhline(0.20, color="green", linestyle="--", linewidth=1.0, alpha=0.75, label="Target (20cm)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax1.set_ylabel("Median Error (m)", fontsize=10)
    ax1.set_title("Median Distance Error vs Distance Band", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Panneau 2: % Bonnes Estimations (< 20 cm) par tranche
    ax2 = fig.add_subplot(gs[0, 1])
    for i_n, node in enumerate(nodes):
        sub_b = breakdown_df[breakdown_df["node"] == node.upper()]
        pcts = [sub_b[sub_b["range_bin"] == b]["pct_good_20cm"].values[0] if not sub_b[sub_b["range_bin"] == b].empty else 0.0 for b in labels]
        col = NODE_COLORS.get(node, "#9E9E9E")
        pos = x + (i_n - (len(nodes)-1)/2.0) * w_bar
        bars = ax2.bar(pos, pcts, w_bar*0.9, color=col, alpha=0.85, edgecolor="white", label=node.upper())
        for b_obj in bars:
            h = b_obj.get_height()
            if not np.isnan(h) and h > 0:
                ax2.text(b_obj.get_x() + b_obj.get_width()/2.0, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax2.set_ylabel("Good Estimations (% < 20cm)", fontsize=10)
    ax2.set_title("Accuracy (% < 20cm) vs Distance Band", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # Panneau 3: Comparaison Binaire Zone Proche (<= 2.0m) vs Zone Éloignée (> 2.0m)
    ax3 = fig.add_subplot(gs[1, :])
    comp_cats = ["Near Zone (≤ 2.0 m)", "Far Zone (> 2.0 m)"]
    x_c = np.arange(len(comp_cats))
    w_c = 0.35 / len(nodes)
    for i_n, node in enumerate(nodes):
        sub_node = points_df[points_df["node"] == node]
        near_pts = sub_node[sub_node["d_depth_m"] <= 2.0]
        far_pts = sub_node[sub_node["d_depth_m"] > 2.0]

        near_med = near_pts["error_m"].median() if not near_pts.empty else 0.0
        far_med = far_pts["error_m"].median() if not far_pts.empty else 0.0
        near_pct = (near_pts["good_20cm"].mean() * 100.0) if not near_pts.empty else 0.0
        far_pct = (far_pts["good_20cm"].mean() * 100.0) if not far_pts.empty else 0.0

        col = NODE_COLORS.get(node, "#9E9E9E")
        pos = x_c + (i_n - (len(nodes)-1)/2.0) * w_c * 2.2
        bars = ax3.bar(pos, [near_med, far_med], w_c * 2.0, color=col, alpha=0.85, edgecolor="white", label=f"{node.upper()} (Median Error)")

        # Annotations textuelles détaillées
        if len(near_pts) > 0:
            ax3.text(pos[0], near_med + 0.05, f"Med: {near_med:.2f}m\n({near_pct:.1f}% good | n={len(near_pts)})", ha="center", va="bottom", fontsize=8, fontweight="bold")
        if len(far_pts) > 0:
            ax3.text(pos[1], far_med + 0.05, f"Med: {far_med:.2f}m\n({far_pct:.1f}% good | n={len(far_pts)})", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax3.axhline(0.20, color="green", linestyle="--", linewidth=1.0, alpha=0.75, label="Target (20cm)")
    ax3.set_xticks(x_c)
    ax3.set_xticklabels(comp_cats, fontsize=11, fontweight="bold")
    ax3.set_ylabel("Median Error (m)", fontsize=10)
    ax3.set_title("Operational Impact: Near Zone (≤ 2.0 m) vs Far Zone (> 2.0 m)", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    # Panneau 4: Tableau Synthétique
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    table_data = []
    for _, row in breakdown_df.iterrows():
        if row["points_count"] > 0:
            table_data.append([
                row["node"],
                row["range_bin"],
                row["points_count"],
                f"{row['mean_error_m']:.3f}m" if row['mean_error_m'] is not None else "-",
                f"{row['median_error_m']:.3f}m" if row['median_error_m'] is not None else "-",
                row["good_20cm_count"],
                f"{row['pct_good_20cm']:.1f}%"
            ])

    col_labels = ["Node", "Distance Band", "Detections", "Mean Error", "Median Error", "Good (<20cm)", "% Good"]
    if table_data:
        table = ax4.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.25)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#1565C0")
            table[0, j].set_text_props(color="white", fontweight="bold")
        for i_r, row in enumerate(table_data):
            node_name = row[0].lower()
            table[i_r + 1, 0].set_facecolor(NODE_COLORS.get(node_name, "#9E9E9E") + "33")

    ax4.set_title("Detailed Distance Band Metrics Table", fontsize=11, fontweight="bold", pad=8)
    fig.savefig(os.path.join(out_dir, "09_distance_error_by_range.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_comparative_suite(recap_df, out_dir, bag_analysis_dir, level_label=""):
    """Génère la suite complète de 9 graphiques comparatifs + CSV pour un DataFrame récapitulatif."""
    os.makedirs(out_dir, exist_ok=True)
    if recap_df.empty:
        print(f"[!] Aucune donnée pour l'analyse comparative ({level_label}). -> SKIP")
        return

    csv_name = "recap_trials.csv" if level_label == "Trials" else "recap_bags.csv"
    recap_path = os.path.join(out_dir, csv_name)
    recap_df.to_csv(recap_path, index=False)
    print(f"[OK] Tableau récapitulatif ({level_label}) : {recap_path} ({len(recap_df)} lignes)")

    # Graphiques
    plot_comparative_bars(recap_df, out_dir)
    print(f"  [OK] 01_comparative_bars.png ({level_label})")

    plot_distance_scatter(recap_df, out_dir)
    print(f"  [OK] 02_distance_scatter.png ({level_label})")

    plot_ttc_scatter(recap_df, out_dir)
    print(f"  [OK] 03_ttc_scatter.png ({level_label})")

    plot_heatmap(recap_df, out_dir)
    print(f"  [OK] 04_heatmap.png ({level_label})")

    plot_node_summary(recap_df, out_dir)
    print(f"  [OK] 05_node_summary.png ({level_label})")

    plot_homography_accuracy(recap_df, out_dir)
    print(f"  [OK] 06_homography_accuracy.png ({level_label})")

    plot_resource_consumption(recap_df, out_dir)
    print(f"  [OK] 07_resources_summary.png ({level_label})")

    plot_depth_vs_detection_errors(recap_df, out_dir, bag_analysis_dir)
    print(f"  [OK] 08_depth_vs_detection_errors.png ({level_label})")

    # Nouveau : Décomposition et tableau par tranche de distance
    points_df = compute_distance_range_breakdown(bag_analysis_dir)
    plot_distance_range_breakdown(points_df, out_dir)
    print(f"  [OK] 09_distance_error_by_range.png & distance_range_breakdown.csv ({level_label})")

    # ── Scientific Deep-Dive Suite (10-13) in English ──
    try:
        import generate_scientific_deep_dive as gsdd
        gsdd.compute_statistical_tests(recap_df, out_dir)
        print(f"  [OK] 10_statistical_significance_tests.png & scientific_significance_tests.csv ({level_label})")
        gsdd.compute_kinematics_and_directional_bias(bag_analysis_dir, recap_df, out_dir)
        print(f"  [OK] 11_kinematics_and_directional_bias.png & kinematics_and_directional_bias.csv ({level_label})")
        gsdd.compute_yolo_object_classes_analysis(bag_analysis_dir, out_dir)
        print(f"  [OK] 12_yolo_object_classes_analysis.png & yolo_object_classes_breakdown.csv ({level_label})")
        gsdd.compute_scenario_repeatability(recap_df, out_dir)
        print(f"  [OK] 13_scenario_repeatability_stability.png & scenario_repeatability_metrics.csv ({level_label})")
    except Exception as e:
        print(f"  [!] Scientific deep-dive notice: {e}")


def generate_comparative(bag_analysis_dir):
    """
    Génère les deux synthèses comparatives complètes :
      1. A l'échelle de TOUS LES ESSAIS INDIVIDUELS (111 trials)
         -> dans bag_analysis/comparatif/ et bag_analysis/comparatif_trials/
      2. A l'échelle de TOUS LES BAGS COMPLETS (22 sessions)
         -> dans bag_analysis/comparatif_bags/
    """
    print(f"\n{'='*60}")
    print("  ANALYSE COMPARATIVE GLOBALE")
    print(f"{'='*60}")

    # ── 1. SYNTHÈSE DES ESSAIS INDIVIDUELS (TRIALS) ──
    print(f"\n--- [1/2] SYNTHESE COMPARATIVE : ESSAIS INDIVIDUELS (111 TRIALS) ---")
    recap_trials_df = build_recap_table_trials(bag_analysis_dir)

    out_trials_main = os.path.join(bag_analysis_dir, "comparatif")
    out_trials_sub = os.path.join(bag_analysis_dir, "comparatif_trials")

    if not recap_trials_df.empty:
        run_comparative_suite(recap_trials_df, out_trials_main, bag_analysis_dir, level_label="Trials")
        recap_trials_df.to_csv(os.path.join(out_trials_main, "recap_scenarios.csv"), index=False)
        run_comparative_suite(recap_trials_df, out_trials_sub, bag_analysis_dir, level_label="Trials")

    # ── 2. SYNTHÈSE DES BAGS COMPLETS ──
    print(f"\n--- [2/2] SYNTHESE COMPARATIVE : BAGS COMPLETS (22 BAGS) ---")
    recap_bags_df = build_recap_table_bags(bag_analysis_dir)
    out_bags = os.path.join(bag_analysis_dir, "comparatif_bags")

    if not recap_bags_df.empty:
        run_comparative_suite(recap_bags_df, out_bags, bag_analysis_dir, level_label="Bags")

    print(f"\n[OK] Toutes les analyses comparatives (Trials & Bags) terminees !")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Visualisation et analyse des CSV de tests ROS bag",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bag-analysis-dir", default=DEFAULT_BAG_ANALYSIS,
                    help=f"Chemin du dossier bag_analysis/ (défaut: {DEFAULT_BAG_ANALYSIS})")
    ap.add_argument("--folder", default=None,
                    help="Traite un seul dossier (nom, pas chemin complet)")
    ap.add_argument("--include-trials-plots", action="store_true",
                    help="Génère aussi les 7 graphiques individuels par sous-dossier trial_XX")
    ap.add_argument("--no-per-folder", action="store_true",
                    help="Ne pas générer les graphiques par dossier")
    ap.add_argument("--no-comparative", action="store_true",
                    help="Ne pas générer les graphiques comparatifs")
    args = ap.parse_args()

    bag_dir = args.bag_analysis_dir
    if not os.path.isdir(bag_dir):
        print(f"[ERREUR] Dossier introuvable : {bag_dir}")
        sys.exit(1)

    # Graphiques par dossier
    if not args.no_per_folder:
        if args.folder:
            folders = [args.folder]
        else:
            folders = sorted([
                d for d in os.listdir(bag_dir)
                if os.path.isdir(os.path.join(bag_dir, d)) and not d.startswith("comparatif")
            ])

        print(f"[+] Génération des graphiques pour {len(folders)} dossier(s) de bag")
        for bag_name in folders:
            bag_folder = os.path.join(bag_dir, bag_name)
            if not os.path.isdir(bag_folder):
                print(f"  [!] {bag_name} introuvable -> SKIP")
                continue
            try:
                generate_folder_plots(bag_folder, bag_name)
            except Exception as e:
                print(f"  [ERREUR] {bag_name} : {e}")
                import traceback
                traceback.print_exc()

            # Optionnel : generer aussi les plots de chaque sous-trial
            if args.include_trials_plots:
                trials_dir = os.path.join(bag_folder, "trials")
                if os.path.isdir(trials_dir):
                    sub_trials = sorted([
                        x for x in os.listdir(trials_dir)
                        if x.startswith("trial_") and os.path.isdir(os.path.join(trials_dir, x))
                    ])
                    for sub in sub_trials:
                        sub_path = os.path.join(trials_dir, sub)
                        try:
                            generate_folder_plots(sub_path, f"{bag_name}_{sub}")
                        except Exception:
                            pass

    # Graphiques comparatifs
    if not args.no_comparative:
        generate_comparative(bag_dir)

    print("\n[OK] Terminé !")


if __name__ == "__main__":
    main()