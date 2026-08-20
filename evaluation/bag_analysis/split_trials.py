#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_trials.py — Découpe les CSV d'un bag ROS en essais individuels
=====================================================================

Chaque dossier bag_analysis/<bag_name>/ contient des CSV couvrant
potentiellement PLUSIEURS essais (l'opérateur lance le robot plusieurs
fois vers un obstacle dans la même session bag record).

Ce script détecte automatiquement les frontières entre essais, affiche
un aperçu graphique pour validation, et découpe tous les CSV en
sous-dossiers trial_01/, trial_02/, etc.

ALGORITHME DE DÉTECTION
-----------------------
1. Charge odometry_filtered.csv -> |vx| lissé sur fenêtre glissante
2. Crée un signal "actif" = |vx| > seuil (0.03 m/s par défaut)
3. Fusionne les régions actives proches (< merge_gap secondes)
4. Filtre les régions trop courtes (< min_duration secondes)
5. Ajoute du padding avant/après chaque essai détecté
6. Découpe tous les CSV sur ces fenêtres temporelles

USAGE
-----
    # Traite un seul dossier :
    py split_trials.py bag_analysis/bag_P1_S1_01

    # Traite tous les dossiers de bag_analysis/ :
    py split_trials.py --all

    # Avec paramètres ajustés :
    py split_trials.py --all --min-gap 8 --min-duration 3 --padding 2

    # Mode preview uniquement (pas de découpe, juste les graphiques) :
    py split_trials.py --all --preview-only

SORTIES
-------
    <bag_folder>/trials/trial_01/   -> CSV découpés pour l'essai 1
    <bag_folder>/trials/trial_02/   -> CSS découpés pour l'essai 2
    ...
    <bag_folder>/trials/trial_summary.csv -> résumé de chaque essai
    <bag_folder>/trials/split_preview.png -> graphique de validation
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.widgets import SpanSelector, Button

# ═══════════════════════════════════════════════════════════════════════
#  CHEMINS PAR DÉFAUT
# ═══════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # remonte de "Codes de test" -> évalutation
DEFAULT_BAG_ANALYSIS = os.path.join(BASE_DIR, "bag_analysis")

# ═══════════════════════════════════════════════════════════════════════
#  PARAMÈTRES DE DÉTECTION
# ═══════════════════════════════════════════════════════════════════════
DEFAULT_VX_THRESHOLD = 0.05   # m/s — seuil d'activité sur |vx|
DEFAULT_MIN_GAP = 4.0         # s — gap minimum entre deux essais
DEFAULT_MIN_DURATION = 3.0    # s — durée min d'un essai valide
DEFAULT_PADDING = 2.0         # s — marge ajoutée avant/après chaque essai
SMOOTHING_WINDOW = 1.0        # s — fenêtre de lissage du signal vx


def detect_trials(odom_df, vx_threshold=DEFAULT_VX_THRESHOLD,
                  min_gap=DEFAULT_MIN_GAP, min_duration=DEFAULT_MIN_DURATION,
                  padding=DEFAULT_PADDING):
    """
    Détecte les essais dans l'odométrie en analysant le signal |vx|.

    Retourne une liste de dicts :
        [{"trial": 1, "t_start": ..., "t_end": ..., "duration": ...}, ...]
    """
    if odom_df.empty or "vx" not in odom_df.columns:
        return []

    odom = odom_df.sort_values("t").reset_index(drop=True)
    t = odom["t"].to_numpy()
    vx = np.abs(odom["vx"].to_numpy())

    # Lissage par fenêtre glissante (médiane, robuste aux pics)
    dt_median = np.median(np.diff(t)) if len(t) > 1 else 0.05
    win_samples = max(1, int(SMOOTHING_WINDOW / dt_median))
    vx_smooth = pd.Series(vx).rolling(win_samples, center=True, min_periods=1).median().to_numpy()

    # Signal booléen "actif"
    active = vx_smooth > vx_threshold

    # Extraire les régions actives
    regions = []
    in_region = False
    start_idx = 0
    for i in range(len(active)):
        if active[i] and not in_region:
            in_region = True
            start_idx = i
        elif not active[i] and in_region:
            in_region = False
            regions.append((t[start_idx], t[i - 1]))
    if in_region:
        regions.append((t[start_idx], t[-1]))

    if not regions:
        return []

    # Fusionner les régions proches (gap < min_gap)
    merged = [regions[0]]
    for r_start, r_end in regions[1:]:
        prev_start, prev_end = merged[-1]
        if r_start - prev_end < min_gap:
            merged[-1] = (prev_start, r_end)
        else:
            merged.append((r_start, r_end))

    # Filtrer les essais trop courts
    filtered = [(s, e) for s, e in merged if (e - s) >= min_duration]

    if not filtered:
        return []

    # Ajouter le padding
    t_min, t_max = t.min(), t.max()
    trials = []
    for i, (s, e) in enumerate(filtered):
        ts = max(t_min, s - padding)
        te = min(t_max, e + padding)
        trials.append({
            "trial": i + 1,
            "t_start": round(ts, 3),
            "t_end": round(te, 3),
            "t_start_raw": round(s, 3),
            "t_end_raw": round(e, 3),
            "duration": round(te - ts, 3),
            "duration_active": round(e - s, 3),
        })

    return trials


def plot_split_preview(bag_name, odom_df, trials, out_path, extra_dfs=None):
    """
    Validation plot : robot activity + detected trials.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 2, 1]})
    fig.suptitle(f"Trial Detection -- {bag_name}", fontsize=14, fontweight="bold")

    odom = odom_df.sort_values("t").reset_index(drop=True)
    t = odom["t"].to_numpy()
    vx = odom["vx"].to_numpy()
    wz = odom["wz"].to_numpy()

    # --- Axis 1 : Odom Velocities ---
    ax1 = axes[0]
    ax1.plot(t, vx, linewidth=0.6, alpha=0.8, label="vx (m/s)", color="#2196F3")
    ax1.plot(t, wz, linewidth=0.4, alpha=0.5, label="ωz (rad/s)", color="#FF9800")
    ax1.axhline(DEFAULT_VX_THRESHOLD, color="red", linestyle="--", linewidth=0.8,
                alpha=0.5, label=f"vx threshold = {DEFAULT_VX_THRESHOLD}")
    ax1.axhline(-DEFAULT_VX_THRESHOLD, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("Velocity")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Shading of detected trials
    colors_trial = plt.cm.Set2(np.linspace(0, 1, max(len(trials), 1)))
    for trial in trials:
        for ax in axes:
            ax.axvspan(trial["t_start"], trial["t_end"],
                       alpha=0.15, color=colors_trial[trial["trial"] - 1])
        ax1.axvspan(trial["t_start_raw"], trial["t_end_raw"],
                    alpha=0.25, color=colors_trial[trial["trial"] - 1])
        # Top label
        mid_t = (trial["t_start"] + trial["t_end"]) / 2
        ax1.text(mid_t, ax1.get_ylim()[1] * 0.9,
                 f"Trial {trial['trial']}\n{trial['duration_active']:.1f}s",
                 ha="center", va="top", fontsize=8, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # --- Axis 2 : Avoidance Commands ---
    ax2 = axes[1]
    if extra_dfs and "avoidance_cmd_vel" in extra_dfs:
        avoid = extra_dfs["avoidance_cmd_vel"]
        ax2.scatter(avoid["t"], avoid["ang_z"], s=3, alpha=0.6,
                    color="#E91E63", label="avoid ang_z")
        ax2.scatter(avoid["t"], avoid["lin_x"], s=3, alpha=0.6,
                    color="#4CAF50", label="avoid lin_x")
        ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylabel("Avoidance command")
    ax2.grid(True, alpha=0.3)

    # --- Axis 3 : E-STOP ---
    ax3 = axes[2]
    if extra_dfs and "emergency_stop" in extra_dfs:
        estop = extra_dfs["emergency_stop"]
        estop_val = estop["value"].astype(float)
        ax3.fill_between(estop["t"], 0, estop_val, color="red", alpha=0.4,
                         step="post", label="E-STOP")
        ax3.legend(loc="upper right", fontsize=8)
    ax3.set_ylabel("E-STOP")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylim(-0.1, 1.3)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Preview saved : {out_path}")


def split_csv_by_trials(bag_folder, trials, out_base):
    """
    Découpe tous les CSV du bag_folder selon les fenêtres temporelles
    des essais et les sauve dans out_base/trial_XX/.
    """
    csv_files = glob.glob(os.path.join(bag_folder, "*.csv"))
    resource_files = [f for f in csv_files if os.path.basename(f).startswith("resource")]
    ros_csv_files = [f for f in csv_files if not os.path.basename(f).startswith("resource")]

    for trial in trials:
        trial_dir = os.path.join(out_base, f"trial_{trial['trial']:02d}")
        os.makedirs(trial_dir, exist_ok=True)

        t_start = trial["t_start"]
        t_end = trial["t_end"]

        for csv_path in ros_csv_files:
            fname = os.path.basename(csv_path)
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            if "t" not in df.columns:
                continue

            # Découpe temporelle
            mask = (df["t"] >= t_start) & (df["t"] <= t_end)
            sub = df[mask].copy()

            if sub.empty:
                continue

            # Recaler le temps à 0 pour chaque essai
            t_offset = sub["t"].min()
            sub["t"] = sub["t"] - t_offset
            sub["t_original"] = df.loc[mask, "t"]

            sub.to_csv(os.path.join(trial_dir, fname), index=False)

        # Copier les fichiers resource tels quels (timestamps absolus,
        # pas de colonne 't' relative au bag)
        for res_path in resource_files:
            fname = os.path.basename(res_path)
            try:
                df = pd.read_csv(res_path)
                df.to_csv(os.path.join(trial_dir, fname), index=False)
            except Exception:
                pass

        # Copier le summary.json avec les infos du trial
        summary_path = os.path.join(bag_folder, "summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path, "r") as f:
                summary = json.load(f)
            summary["_trial_number"] = trial["trial"]
            summary["_trial_t_start"] = trial["t_start"]
            summary["_trial_t_end"] = trial["t_end"]
            summary["_trial_duration"] = trial["duration"]
            summary["_trial_duration_active"] = trial["duration_active"]
            with open(os.path.join(trial_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  [OK] {len(trials)} essais découpés dans {out_base}/")


def recompute_trial_metrics(trial_dir):
    """
    Recalcule les métriques clés pour un essai individuel
    (d_min, ttc_min, estop_count, etc.) à partir des CSV découpés.
    """
    metrics = {}

    # --- d_min et TTC depuis flow_diagnostics ---
    fd_path = os.path.join(trial_dir, "flow_diagnostics.csv")
    if os.path.isfile(fd_path):
        fd = pd.read_csv(fd_path)
        if "min_dist_m" in fd.columns:
            vals = fd["min_dist_m"].dropna()
            metrics["d_min_m"] = float(vals.min()) if not vals.empty else None
        if "ttc" in fd.columns:
            vals = fd["ttc"].dropna()
            metrics["ttc_min_s"] = float(vals.min()) if not vals.empty else None

    # --- d_min depuis ground_detections (YOLO) ---
    gd_path = os.path.join(trial_dir, "ground_detections.csv")
    if os.path.isfile(gd_path) and "d_min_m" not in metrics:
        gd = pd.read_csv(gd_path)
        if "dist_m" in gd.columns:
            vals = gd["dist_m"].dropna()
            if not vals.empty:
                metrics["d_min_m"] = float(vals.min())

    # --- d_min_depth depuis depth ---
    depth_path = os.path.join(trial_dir, "camera_aligned_depth_to_color_image_raw.csv")
    if os.path.isfile(depth_path):
        depth = pd.read_csv(depth_path)
        if "d_min_depth_m" in depth.columns:
            vals = depth["d_min_depth_m"].dropna()
            metrics["d_min_depth_m"] = float(vals.min()) if not vals.empty else None

    # --- E-STOP ---
    estop_path = os.path.join(trial_dir, "emergency_stop.csv")
    if os.path.isfile(estop_path):
        estop = pd.read_csv(estop_path)
        if "value" in estop.columns:
            estop = estop.sort_values("t")
            # Compter les fronts montants
            bool_vals = estop["value"].astype(str).str.strip().map(
                {"True": True, "False": False, "1": True, "0": False}).fillna(False)
            count = 0
            prev = False
            for v in bool_vals:
                if v and not prev:
                    count += 1
                prev = v
            metrics["estop_count"] = count

    # --- Épisodes d'évitement ---
    avoid_path = os.path.join(trial_dir, "avoidance_cmd_vel.csv")
    if os.path.isfile(avoid_path):
        avoid = pd.read_csv(avoid_path)
        if not avoid.empty and "t" in avoid.columns:
            t = avoid["t"].sort_values().to_numpy()
            if len(t) > 0:
                gaps = np.diff(t)
                n_episodes = 1 + int((gaps > 0.5).sum())
                metrics["n_avoidance_episodes"] = n_episodes

    # --- Durée ---
    odom_path = os.path.join(trial_dir, "odometry_filtered.csv")
    if os.path.isfile(odom_path):
        odom = pd.read_csv(odom_path)
        if not odom.empty and "t" in odom.columns:
            metrics["trial_duration_s"] = round(float(odom["t"].max() - odom["t"].min()), 3)

    return metrics


def process_bag_folder(bag_folder, args):
    """Traite un dossier bag : détection, preview, découpe."""
    bag_name = os.path.basename(bag_folder)
    print(f"\n{'='*60}")
    print(f"  {bag_name}")
    print(f"{'='*60}")

    # Charger l'odométrie
    odom_path = os.path.join(bag_folder, "odometry_filtered.csv")
    if not os.path.isfile(odom_path):
        print(f"  [!] odometry_filtered.csv introuvable -> SKIP")
        return None

    odom = pd.read_csv(odom_path)
    print(f"  Durée du bag : {odom['t'].max():.1f}s ({len(odom)} points odom)")

    # Charger CSV auxiliaires pour la preview
    extra_dfs = {}
    for name in ["avoidance_cmd_vel", "emergency_stop"]:
        path = os.path.join(bag_folder, f"{name}.csv")
        if os.path.isfile(path):
            try:
                extra_dfs[name] = pd.read_csv(path)
            except Exception:
                pass

    # Détecter les essais
    trials = detect_trials(odom,
                           vx_threshold=args.vx_threshold,
                           min_gap=args.min_gap,
                           min_duration=args.min_duration,
                           padding=args.padding)

    print(f"  -> {len(trials)} essai(s) détecté(s) :")
    for trial in trials:
        print(f"    Essai {trial['trial']:2d} : "
              f"t=[{trial['t_start']:.1f}s -> {trial['t_end']:.1f}s] "
              f"({trial['duration_active']:.1f}s actif, "
              f"{trial['duration']:.1f}s total avec padding)")

    # Créer le dossier de sortie
    out_base = os.path.join(bag_folder, "trials")
    os.makedirs(out_base, exist_ok=True)

    # Preview graphique
    preview_path = os.path.join(out_base, "split_preview.png")
    plot_split_preview(bag_name, odom, trials, preview_path, extra_dfs)

    if args.preview_only:
        print(f"  [i] Mode preview -- pas de decoupe.")
        return trials

    # Mode interactif ou GUI direct
    if args.interactive or getattr(args, "gui_mode", False):
        mode = "gui" if getattr(args, "gui_mode", False) else "prompt"
        trials = interactive_prompt(bag_name, trials, odom, out_base, extra_dfs, default_mode=mode)
        if trials is None:
            return None

    # Découper les CSV
    split_csv_by_trials(bag_folder, trials, out_base)

    # Recalculer les métriques par essai
    trial_summaries = []
    for trial in trials:
        trial_dir = os.path.join(out_base, f"trial_{trial['trial']:02d}")
        if os.path.isdir(trial_dir):
            m = recompute_trial_metrics(trial_dir)
            m.update(trial)
            m["bag_name"] = bag_name
            trial_summaries.append(m)

            # Sauver dans le trial_dir
            with open(os.path.join(trial_dir, "trial_metrics.json"), "w") as f:
                json.dump(m, f, indent=2, ensure_ascii=False)

    # Sauver le résumé global
    if trial_summaries:
        summary_df = pd.DataFrame(trial_summaries)
        summary_path = os.path.join(out_base, "trial_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"  [OK] Résumé sauvé : {summary_path}")

    return trials


def parse_manual_ranges(text, t_max):
    """
    Parse des plages manuelles au format: t_start-t_end, t_start-t_end, ...
    Retourne une liste de dicts compatibles avec le format trials.
    """
    trials = []
    parts = [p.strip() for p in text.split(",") if p.strip()]
    for i, part in enumerate(parts, start=1):
        try:
            a, b = part.split("-", 1)
            t_start = float(a.strip())
            t_end = float(b.strip())
            if t_end <= t_start:
                print(f"  [!] Plage invalide ignoree : {part} (t_end <= t_start)")
                continue
            t_start = max(0, t_start)
            t_end = min(t_max, t_end)
            trials.append({
                "trial": i,
                "t_start": t_start,
                "t_end": t_end,
                "duration": t_end - t_start,
                "duration_active": t_end - t_start,
                "manual": True,
            })
        except ValueError:
            print(f"  [!] Format invalide ignore : '{part}' (attendu: t_start-t_end)")
    return trials


def interactive_gui_selection(bag_name, odom_df, extra_dfs=None, initial_trials=None):
    """
    Ouvre une fenetre graphique interactive avec les 3 sous-graphes :
      1. Vitesse odom (vx, wz, seuils)
      2. Commandes d'evitement (avoidance lin_x, ang_z)
      3. Arret d'urgence (E-STOP)

    Permet a l'utilisateur de cliquer-glisser a la souris pour definir
    visuellement les plages d'essais [t_start, t_end].
    Boutons : [Reset Selection], [Restore Auto-Detect], [Validate & Split].
    Raccourcis clavier : 'Enter' = valider, 'r' = reset, 'a' = auto, 'Esc' = annuler.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1]})
    fig.subplots_adjust(bottom=0.12, top=0.92, hspace=0.15)
    ax1, ax2, ax3 = axes

    odom = odom_df.sort_values("t").reset_index(drop=True)
    t = odom["t"].to_numpy()
    vx = odom["vx"].to_numpy()
    wz = odom["wz"].to_numpy()
    t_max = float(t.max()) if len(t) > 0 else 100.0

    # 1. Axe Vitesse
    ax1.plot(t, vx, linewidth=0.7, alpha=0.85, label="vx (m/s)", color="#2196F3")
    ax1.plot(t, wz, linewidth=0.5, alpha=0.6, label="ωz (rad/s)", color="#FF9800")
    ax1.axhline(DEFAULT_VX_THRESHOLD, color="red", linestyle="--", linewidth=0.8,
                alpha=0.5, label=f"vx threshold = {DEFAULT_VX_THRESHOLD}")
    ax1.axhline(-DEFAULT_VX_THRESHOLD, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("Velocity (m/s, rad/s)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Axe Avoidance
    if extra_dfs and "avoidance_cmd_vel" in extra_dfs:
        avoid = extra_dfs["avoidance_cmd_vel"]
        if not avoid.empty and "t" in avoid.columns:
            ax2.scatter(avoid["t"], avoid["ang_z"], s=4, alpha=0.6,
                        color="#E91E63", label="avoid ang_z")
            ax2.scatter(avoid["t"], avoid["lin_x"], s=4, alpha=0.6,
                        color="#4CAF50", label="avoid lin_x")
            ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylabel("Avoidance cmd")
    ax2.grid(True, alpha=0.3)

    # 3. Axe E-STOP
    if extra_dfs and "emergency_stop" in extra_dfs:
        estop = extra_dfs["emergency_stop"]
        if not estop.empty and "t" in estop.columns:
            estop_val = estop["value"].astype(float)
            ax3.fill_between(estop["t"], 0, estop_val, color="red", alpha=0.4,
                             step="post", label="E-STOP")
            ax3.legend(loc="upper right", fontsize=8)
    ax3.set_ylabel("E-STOP")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylim(-0.1, 1.3)
    ax3.grid(True, alpha=0.3)

    selected_ranges = []
    if initial_trials:
        for tr in initial_trials:
            selected_ranges.append((float(tr["t_start"]), float(tr["t_end"])))

    drawn_patches = []

    def redraw():
        for p in drawn_patches:
            try:
                p.remove()
            except Exception:
                pass
        drawn_patches.clear()

        selected_ranges.sort(key=lambda r: r[0])
        cmap = plt.colormaps.get_cmap("tab10")

        ylim1 = ax1.get_ylim()
        y_top = ylim1[1]

        for idx, (t1, t2) in enumerate(selected_ranges):
            col = cmap(idx % 10)
            for ax in axes:
                p = ax.axvspan(t1, t2, alpha=0.25, color=col, zorder=0)
                drawn_patches.append(p)

            mid_t = (t1 + t2) / 2.0
            txt = ax1.text(mid_t, y_top * 0.88,
                           f"Trial {idx+1}\n{(t2 - t1):.1f}s",
                           ha="center", va="top", fontsize=8, fontweight="bold",
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                     alpha=0.9, edgecolor=col))
            drawn_patches.append(txt)

        fig.suptitle(f"Manual Trial Selection -- {bag_name} [{len(selected_ranges)} trial(s) selected]\n"
                     f"Instructions: Click & drag on velocity plot to add a trial | [Reset] to clear | [Validate] or Enter to finish",
                     fontsize=11, fontweight="bold")
        fig.canvas.draw_idle()

    def on_select_span(xmin, xmax):
        t1 = max(0.0, float(min(xmin, xmax)))
        t2 = min(t_max, float(max(xmin, xmax)))
        if (t2 - t1) >= 1.0:
            selected_ranges.append((t1, t2))
            redraw()

    span_selector = SpanSelector(
        ax1, on_select_span, "horizontal",
        useblit=True, props=dict(alpha=0.35, facecolor="#FFEB3B"),
        interactive=False, drag_from_anywhere=True
    )

    # Boutons
    ax_reset = fig.add_axes([0.15, 0.02, 0.18, 0.05])
    b_reset = Button(ax_reset, "Reset Selection", color="#FFCDD2", hovercolor="#EF9A9A")

    def on_reset_click(event):
        selected_ranges.clear()
        redraw()

    b_reset.on_clicked(on_reset_click)

    ax_auto = fig.add_axes([0.38, 0.02, 0.22, 0.05])
    b_auto = Button(ax_auto, "Restore Auto-Detect", color="#FFF9C4", hovercolor="#FFF59D")

    def on_auto_click(event):
        selected_ranges.clear()
        if initial_trials:
            for tr in initial_trials:
                selected_ranges.append((float(tr["t_start"]), float(tr["t_end"])))
        redraw()

    b_auto.on_clicked(on_auto_click)

    ax_val = fig.add_axes([0.65, 0.02, 0.22, 0.05])
    b_val = Button(ax_val, "Validate & Split", color="#C8E6C9", hovercolor="#A5D6A7")

    def on_val_click(event):
        plt.close(fig)

    b_val.on_clicked(on_val_click)

    def on_key_press(event):
        if event.key in ("enter", "return"):
            plt.close(fig)
        elif event.key in ("r", "R"):
            on_reset_click(None)
        elif event.key in ("a", "A"):
            on_auto_click(None)
        elif event.key in ("escape",):
            selected_ranges.clear()
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key_press)

    redraw()
    plt.show()

    if not selected_ranges:
        return None

    selected_ranges.sort(key=lambda r: r[0])
    final_trials = []
    for idx, (t1, t2) in enumerate(selected_ranges, 1):
        final_trials.append({
            "trial": idx,
            "t_start": t1,
            "t_end": t2,
            "t_start_raw": t1,
            "t_end_raw": t2,
            "duration": t2 - t1,
            "duration_active": t2 - t1,
            "manual": True,
        })

    return final_trials


def parse_manual_ranges(text, t_max):
    """
    Parse des plages manuelles au format: t_start-t_end, t_start-t_end, ...
    Retourne une liste de dicts compatibles avec le format trials.
    """
    trials = []
    parts = [p.strip() for p in text.split(",") if p.strip()]
    for i, part in enumerate(parts, start=1):
        try:
            a, b = part.split("-", 1)
            t_start = float(a.strip())
            t_end = float(b.strip())
            if t_end <= t_start:
                print(f"  [!] Plage invalide ignoree : {part} (t_end <= t_start)")
                continue
            t_start = max(0, t_start)
            t_end = min(t_max, t_end)
            trials.append({
                "trial": i,
                "t_start": t_start,
                "t_end": t_end,
                "t_start_raw": t_start,
                "t_end_raw": t_end,
                "duration": t_end - t_start,
                "duration_active": t_end - t_start,
                "manual": True,
            })
        except ValueError:
            print(f"  [!] Format invalide ignore : '{part}' (attendu: t_start-t_end)")
    return trials


def interactive_prompt(bag_name, auto_trials, odom, out_base, extra_dfs, default_mode="prompt"):
    """
    Prompt interactif : accepter les splits auto, decouper a la souris sur GUI, taper au clavier, ou skip.
    """
    preview_path = os.path.join(out_base, "split_preview.png")

    if default_mode == "gui":
        choice = "m"
    else:
        print(f"")
        print(f"  Preview : {preview_path}")
        print(f"  Options :")
        print(f"    [Enter]  Accepter les {len(auto_trials)} splits detectes")
        print(f"    [m]      Decouper a la SOURIS sur le graphique interactif (GUI)")
        print(f"    [t]      Taper les plages au clavier (format: 70-120, 140-190)")
        print(f"    [s]      Skip ce bag")
        print(f"")

        try:
            choice = input("  Choix > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  [i] Interruption -- skip")
            return None

    if choice == "s":
        print("  [i] Skip.")
        return None
    elif choice in ("m", "g"):
        print("  [+] Ouverture de la fenetre interactive...")
        print("      -> Cliquez-glissez sur la courbe de vitesse pour surligner chaque essai.")
        print("      -> Boutons [Reset], [Restore Auto], [Validate & Split] ou touche Entree.")
        gui_trials = interactive_gui_selection(bag_name, odom, extra_dfs=extra_dfs, initial_trials=auto_trials)
        if not gui_trials:
            print("  [!] Aucun essai selectionne -- conservation de la detection auto")
            return auto_trials
        print(f"  -> {len(gui_trials)} essai(s) definis manuellement a la souris :")
        for trial in gui_trials:
            print(f"    Essai {trial['trial']:2d} : "
                  f"t=[{trial['t_start']:.1f}s -> {trial['t_end']:.1f}s] "
                  f"({trial['duration']:.1f}s)")
        plot_split_preview(bag_name, odom, gui_trials, preview_path, extra_dfs)
        print(f"  [OK] Preview mise a jour : {preview_path}")
        return gui_trials
    elif choice == "t":
        print(f"  Entrez les plages (ex: 70-120, 140-190, 200-220) :")
        try:
            raw = input("  Plages > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  [i] Interruption -- skip")
            return None
        t_max = odom["t"].max()
        manual_trials = parse_manual_ranges(raw, t_max)
        if not manual_trials:
            print("  [!] Aucune plage valide -- skip")
            return None
        print(f"  -> {len(manual_trials)} essai(s) manuels :")
        for trial in manual_trials:
            print(f"    Essai {trial['trial']:2d} : "
                  f"t=[{trial['t_start']:.1f}s -> {trial['t_end']:.1f}s] "
                  f"({trial['duration']:.1f}s)")
        plot_split_preview(bag_name, odom, manual_trials, preview_path, extra_dfs)
        print(f"  [OK] Preview mise a jour : {preview_path}")
        return manual_trials
    else:
        return auto_trials


def main():
    ap = argparse.ArgumentParser(
        description="Découpe les CSV d'un bag ROS en essais individuels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    ap.add_argument("bag_folder", nargs="?", default=None,
                    help="Chemin d'un dossier bag_analysis/<bag_name>/ à traiter")
    ap.add_argument("--all", action="store_true",
                    help="Traite tous les dossiers de bag_analysis/")
    ap.add_argument("--bag-analysis-dir", default=DEFAULT_BAG_ANALYSIS,
                    help=f"Chemin du dossier bag_analysis/ (défaut: {DEFAULT_BAG_ANALYSIS})")
    ap.add_argument("--vx-threshold", type=float, default=DEFAULT_VX_THRESHOLD,
                    help=f"Seuil |vx| pour détecter l'activité (défaut: {DEFAULT_VX_THRESHOLD})")
    ap.add_argument("--min-gap", type=float, default=DEFAULT_MIN_GAP,
                    help=f"Gap minimum entre essais en secondes (défaut: {DEFAULT_MIN_GAP})")
    ap.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION,
                    help=f"Durée minimum d'un essai en secondes (défaut: {DEFAULT_MIN_DURATION})")
    ap.add_argument("--padding", type=float, default=DEFAULT_PADDING,
                    help=f"Padding avant/apres chaque essai en secondes (defaut: {DEFAULT_PADDING})")
    ap.add_argument("--preview-only", action="store_true",
                    help="Genere seulement les graphiques de preview sans decouper")
    ap.add_argument("--interactive", action="store_true",
                    help="Mode interactif : choisir d'accepter, modifier ou "
                         "skipper les splits pour chaque bag")
    ap.add_argument("--gui", "--manual", dest="gui_mode", action="store_true",
                    help="Ouvre directement la fenetre interactive pour decouper a la souris")
    ap.add_argument("--force", action="store_true",
                    help="Ecrase les resultats existants")
    args = ap.parse_args()

    # Déterminer les dossiers à traiter
    if args.all:
        bag_dir = args.bag_analysis_dir
        if not os.path.isdir(bag_dir):
            print(f"[ERREUR] Dossier bag_analysis introuvable : {bag_dir}")
            sys.exit(1)
        folders = sorted([
            os.path.join(bag_dir, d) for d in os.listdir(bag_dir)
            if os.path.isdir(os.path.join(bag_dir, d))
        ])
    elif args.bag_folder:
        if not os.path.isdir(args.bag_folder):
            print(f"[ERREUR] Dossier introuvable : {args.bag_folder}")
            sys.exit(1)
        folders = [args.bag_folder]
    else:
        print("[ERREUR] Spécifiez un dossier bag ou utilisez --all")
        ap.print_help()
        sys.exit(1)

    print(f"[+] Paramètres de détection :")
    print(f"    seuil |vx|     = {args.vx_threshold} m/s")
    print(f"    gap min        = {args.min_gap} s")
    print(f"    durée min      = {args.min_duration} s")
    print(f"    padding        = {args.padding} s")
    print(f"    preview only   = {args.preview_only}")
    print(f"[+] {len(folders)} dossier(s) à traiter")

    all_trials = {}
    for folder in folders:
        # Vérifier si déjà traité
        trials_dir = os.path.join(folder, "trials")
        if os.path.isdir(trials_dir) and not args.force and not args.preview_only:
            existing = [d for d in os.listdir(trials_dir)
                        if d.startswith("trial_") and os.path.isdir(os.path.join(trials_dir, d))]
            if existing:
                print(f"\n[i] {os.path.basename(folder)} : déjà découpé "
                      f"({len(existing)} essais). --force pour refaire.")
                continue

        trials = process_bag_folder(folder, args)
        if trials:
            all_trials[os.path.basename(folder)] = trials

    # Résumé global
    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ GLOBAL")
    print(f"{'='*60}")
    total = sum(len(t) for t in all_trials.values())
    print(f"  {len(all_trials)} bags traités, {total} essais détectés au total")
    for bag_name, trials in all_trials.items():
        durations = [t["duration_active"] for t in trials]
        print(f"    {bag_name} : {len(trials)} essais "
              f"(durées actives : {', '.join(f'{d:.1f}s' for d in durations)})")


if __name__ == "__main__":
    main()
