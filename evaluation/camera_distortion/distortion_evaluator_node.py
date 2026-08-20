#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distortion_evaluator_node.py
================================================================================
BANC DE TEST DISTORSION OPTIQUE — TESTS A & B (INTEL REALSENSE D435i / D400)
================================================================================

Ce nœud prend le flux vidéo reçu en direct (ROS 2 ou mode hors-ligne) et permet
d'évaluer précisément la distorsion optique selon 2 tests fondamentaux :

  [1] TEST A — Re-projection Error (RMSE) Damier / ChArUco
      • Détection des coins de damier en direct avec raffinement sub-pixel.
      • Carte thermique de couverture spatiale (Heatmap FOV) pour vérifier
        que les 4 coins et les bordures du capteur sont correctement échantillonnés.
      • Calcul de K_calib, D_calib et du RMSE global et par vue.
      • Verdicts :
          - RMSE < 0.3 px      : 🟢 Excellente calibration
          - 0.3 <= RMSE <= 0.7 : 🟡 Standard acceptable (navigation robotique)
          - RMSE > 1.0 px      : 🔴 Calibration médiocre (flou, couverture insuffisante)

  [2] TEST B — Rectilinéarité des Lignes Droites (Arêtes Réelles)
      • Vue comparatrice : Image brute vs Image dédistordue cv2.undistort(img, K, D).
      • Sélection interactive à la souris (clic 2 points sur une arête de mur/porte/sol).
      • Tracé de la droite mathématique de référence rouge.
      • Calcul du profil d'écart de courbure résiduelle (Max & Moyenne en pixels).
      • Critère de conformité : Écart maximal <= 1.0 px (notamment aux 4 coins).
      • Visualisation de la grille de déformation et du champ vectoriel de distorsion.

TOUCHES CLAVIER (HUD INTERACTIF) :
  [1]           : Activer le Mode A (Damier & RMSE)
  [2]           : Activer le Mode B (Rectilinéarité Lignes Droites)
  [Espace]      : Capturer une frame de mire (Mode A) / Geler-Dégeler l'image (Mode B)
  [C]           : Lancer le calcul de calibration (Mode A) / Effacer la ligne (Mode B)
  [U]           : Basculer la vue Dédistordue (Undistort) ON / OFF
  [G]           : Basculer la Grille de déformation / Vecteurs de distorsion ON / OFF
  [S]           : Sauvegarder les résultats (distortion_test_results.json + PNG)
  [R]           : Réinitialiser les captures du mode actif
  [Q] / [Échap] : Quitter
"""

import sys
import os
import time
import json
import argparse
import math
import glob
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

# Compatibilité encodage console Windows
if sys.platform.startswith('win'):
    try:
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS OPTIONNELS ROS 2
# ══════════════════════════════════════════════════════════════════════════════
ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image as RosImage
    from sensor_msgs.msg import CameraInfo as RosCameraInfo
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    class Node: pass
    class RosImage: pass
    class RosCameraInfo: pass
    class CvBridge: pass


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION PAR DÉFAUT & HELPERS I/O
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CHESSBOARD_SIZE = (9, 6)   # Coins internes horizontaux et verticaux
DEFAULT_SQUARE_SIZE_MM  = 25.0     # Taille d'une case en millimètres

DEFAULT_K = np.array([
    [910.0,   0.0, 640.0],
    [  0.0, 910.0, 360.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

DEFAULT_D = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def imread_unicode(filepath: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Charge une image sans problème d'accents (Unicode) sous Windows."""
    try:
        with open(filepath, 'rb') as f:
            bytes_data = bytearray(f.read())
            arr = np.asarray(bytes_data, dtype=np.uint8)
            return cv2.imdecode(arr, flags)
    except Exception:
        return cv2.imread(filepath, flags)


def imwrite_unicode(filepath: str, img: np.ndarray) -> bool:
    """Sauvegarde une image sans problème d'accents (Unicode) sous Windows."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        ext = os.path.splitext(filepath)[1] or '.png'
        is_success, buffer = cv2.imencode(ext, img)
        if is_success:
            with open(filepath, 'wb') as f:
                f.write(buffer)
            return True
    except Exception:
        pass
    return bool(cv2.imwrite(filepath, img))


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSES UTILITAIRES & MOTEUR D'ÉVALUATION (TESTS A & B)
# ══════════════════════════════════════════════════════════════════════════════

class InteractiveLineSelector:
    """Gestionnaire de sélection interactive de ligne droite pour le Test B."""
    def __init__(self):
        self.points: List[Tuple[int, int]] = []

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) >= 2:
                self.points = [(x, y)]
            else:
                self.points.append((x, y))

    def reset(self):
        self.points = []


class DistortionEvaluatorAB:
    """Moteur de calcul pour les Tests A (Damier/RMSE) et B (Rectilinéarité)."""
    def __init__(self,
                 chessboard_size: Tuple[int, int] = DEFAULT_CHESSBOARD_SIZE,
                 square_size_mm: float = DEFAULT_SQUARE_SIZE_MM):
        self.chessboard_size = chessboard_size
        self.square_size_mm = square_size_mm

        self.K = DEFAULT_K.copy()      # "driver" K — from live /camera_info topic
        self.D = DEFAULT_D.copy()      # "driver" D — from live /camera_info topic (RealSense publishes D=0 for color)
        self.distortion_model = "plumb_bob"
        self.has_camera_info = False
        self.image_shape: Optional[Tuple[int, int]] = None

        # v_fix: BUG ROOT CAUSE — Test B previously called cv2.undistort() using
        # self.K/self.D (driver values), which are D=[0,0,0,0,0] for the RealSense
        # color stream. The Test A calibration below (self.calib_K/self.calib_D)
        # was computed but NEVER fed back into the undistort() calls, so "Test B"
        # always dedistorted with zero distortion — i.e. did nothing. This flag
        # makes the active source explicit and observable instead of assumed.
        self.use_calibrated_distortion: bool = False

        # ── Test A : Damier ──────────────────────────────────────────────────
        self.calib_obj_points: List[np.ndarray] = []
        self.calib_img_points: List[np.ndarray] = []
        self.calib_coverage_map: Optional[np.ndarray] = None
        self.calib_rmse: Optional[float] = None
        self.calib_per_view_rmse: List[float] = []
        self.calib_K: Optional[np.ndarray] = None
        self.calib_D: Optional[np.ndarray] = None
        self.calib_status_msg: str = "En attente de captures (Appuyez sur [Espace])"

        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= (self.square_size_mm / 1000.0)

        # ── Test B : Lignes Droites ──────────────────────────────────────────
        self.line_selector = InteractiveLineSelector()
        self.straight_line_results: Optional[Dict[str, Any]] = None

    def update_camera_info(self, K: np.ndarray, D: np.ndarray, model: str = "plumb_bob"):
        self.K = np.array(K, dtype=np.float64).reshape(3, 3)
        self.D = np.array(D, dtype=np.float64).flatten()
        self.distortion_model = model
        self.has_camera_info = True

    def init_coverage_map(self, h: int, w: int):
        if self.calib_coverage_map is None or self.calib_coverage_map.shape != (h, w):
            self.calib_coverage_map = np.zeros((h, w), dtype=np.float32)
            self.image_shape = (h, w)

    def get_active_KD(self) -> Tuple[np.ndarray, np.ndarray, str]:
        """v_fix: single source of truth for 'which K/D is actually applied'.

        Returns (K, D, source_label). source_label is 'calibrated_test_a' when
        the Test A chessboard calibration is loaded AND selected, otherwise
        'driver' (the D=0 RealSense CameraInfo values — i.e. no real correction).
        Every undistort()/undistortPoints() call and every HUD label MUST go
        through this method rather than reading self.K/self.D directly, so the
        active source can never silently diverge from what's displayed.
        """
        if self.use_calibrated_distortion and self.calib_K is not None and self.calib_D is not None:
            return self.calib_K, self.calib_D, 'calibrated_test_a'
        return self.K, self.D, 'driver'

    # ── Test A Logic ──────────────────────────────────────────────────────────
    def detect_chessboard(self, gray: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, self.chessboard_size, flags)
        if found and corners is not None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners_subpix
        return False, None

    def add_calibration_frame(self, gray: np.ndarray, corners: np.ndarray) -> bool:
        h, w = gray.shape[:2]
        self.init_coverage_map(h, w)
        self.calib_obj_points.append(self.objp.copy())
        self.calib_img_points.append(corners.copy())

        for pt in corners.reshape(-1, 2):
            px, py = int(round(pt[0])), int(round(pt[1]))
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(self.calib_coverage_map, (px, py), radius=35, color=(1.0,), thickness=-1)

        self.calib_status_msg = f"{len(self.calib_img_points)} captures enregistrées. [C] pour calculer."
        return True

    def compute_calibration(self) -> Optional[Dict[str, Any]]:
        if len(self.calib_img_points) < 3:
            self.calib_status_msg = "ERREUR : Minimum 3 captures requises (idéal 10-20)."
            return None

        if self.image_shape is None:
            return None

        h, w = self.image_shape
        ret_rmse, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.calib_obj_points,
            self.calib_img_points,
            (w, h),
            None,
            None
        )
        self.calib_rmse = float(ret_rmse)
        self.calib_K = mtx
        self.calib_D = dist.flatten()

        self.calib_per_view_rmse = []
        for i in range(len(self.calib_obj_points)):
            imgpoints2, _ = cv2.projectPoints(self.calib_obj_points[i], rvecs[i], tvecs[i], mtx, dist)
            err = cv2.norm(self.calib_img_points[i], imgpoints2, cv2.NORM_L2) / np.sqrt(len(imgpoints2))
            self.calib_per_view_rmse.append(float(err))

        if self.calib_rmse < 0.3:
            verdict = "EXCELLENT (< 0.3 px)"
            color_code = "GREEN"
        elif self.calib_rmse <= 0.7:
            verdict = "STANDARD / ACCEPTABLE (0.3 - 0.7 px)"
            color_code = "YELLOW"
        else:
            verdict = "MÉDIOCRE (> 1.0 px) — Flou ou couverture faible"
            color_code = "RED"

        coverage_pct = float(np.count_nonzero(self.calib_coverage_map > 0.1) / (w * h) * 100.0)

        results = {
            "rmse_total_px": self.calib_rmse,
            "verdict": verdict,
            "color_code": color_code,
            "num_views": len(self.calib_img_points),
            "coverage_pct": coverage_pct,
            "K_calib": self.calib_K.tolist(),
            "D_calib": self.calib_D.tolist(),
            "per_view_rmse": self.calib_per_view_rmse,
        }
        # v_fix: previously computing calibration had ZERO effect on Test B —
        # calib_K/calib_D were stored but never selected as the active source.
        # Now Test B automatically switches to the real calibrated distortion.
        # Press [K] to compare against the driver D=0 baseline if needed.
        self.use_calibrated_distortion = True

        self.calib_status_msg = f"RMSE : {self.calib_rmse:.3f} px | {verdict} | D calibrée ACTIVE pour Test B"
        return results

    def reset_calibration(self):
        self.calib_obj_points = []
        self.calib_img_points = []
        self.calib_coverage_map = None
        self.calib_rmse = None
        self.calib_per_view_rmse = []
        self.calib_K = None
        self.calib_D = None
        self.use_calibrated_distortion = False
        self.calib_status_msg = "Prêt pour nouvelles captures de damier."

    # ── Test B Logic ──────────────────────────────────────────────────────────
    def evaluate_straight_line(self,
                               raw_img: np.ndarray,
                               undist_img: np.ndarray,
                               p1: Tuple[int, int],
                               p2: Tuple[int, int]) -> Dict[str, Any]:
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        line_len = math.hypot(x2 - x1, y2 - y1)

        if line_len < 20.0:
            return {"valid": False, "error": "Ligne trop courte (< 20 px)."}

        num_samples = int(max(50, line_len))
        t_vals = np.linspace(0.0, 1.0, num_samples)
        dx, dy = (x2 - x1) / line_len, (y2 - y1) / line_len
        nx, ny = -dy, dx

        raw_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY) if len(raw_img.shape) == 3 else raw_img
        undist_gray = cv2.cvtColor(undist_img, cv2.COLOR_BGR2GRAY) if len(undist_img.shape) == 3 else undist_img

        raw_grad = cv2.magnitude(cv2.Sobel(raw_gray, cv2.CV_32F, 1, 0), cv2.Sobel(raw_gray, cv2.CV_32F, 0, 1))
        undist_grad = cv2.magnitude(cv2.Sobel(undist_gray, cv2.CV_32F, 1, 0), cv2.Sobel(undist_gray, cv2.CV_32F, 0, 1))

        search_radius = 20
        raw_residuals = []
        undist_residuals = []
        sample_coords_ideal = []
        sample_coords_undist = []

        h, w = raw_gray.shape[:2]

        for t in t_vals:
            cx = x1 + t * (x2 - x1)
            cy = y1 + t * (y2 - y1)
            sample_coords_ideal.append((cx, cy))

            # Image brute
            best_offset_raw = 0.0
            max_val_raw = -1.0
            for r in np.linspace(-search_radius, search_radius, 2 * search_radius + 1):
                sx = int(round(cx + r * nx))
                sy = int(round(cy + r * ny))
                if 0 <= sx < w and 0 <= sy < h:
                    v = raw_grad[sy, sx]
                    if v > max_val_raw:
                        max_val_raw = v
                        best_offset_raw = r
            raw_residuals.append(float(best_offset_raw))

            # Image dédistordue
            best_offset_undist = 0.0
            max_val_undist = -1.0
            for r in np.linspace(-search_radius, search_radius, 2 * search_radius + 1):
                sx = int(round(cx + r * nx))
                sy = int(round(cy + r * ny))
                if 0 <= sx < w and 0 <= sy < h:
                    v = undist_grad[sy, sx]
                    if v > max_val_undist:
                        max_val_undist = v
                        best_offset_undist = r
            undist_residuals.append(float(best_offset_undist))
            sample_coords_undist.append((cx + best_offset_undist * nx, cy + best_offset_undist * ny))

        raw_residuals = np.array(raw_residuals)
        undist_residuals = np.array(undist_residuals)

        max_raw_err = float(np.max(np.abs(raw_residuals)))
        mean_raw_err = float(np.mean(np.abs(raw_residuals)))
        max_undist_err = float(np.max(np.abs(undist_residuals)))
        mean_undist_err = float(np.mean(np.abs(undist_residuals)))
        is_compliant = (max_undist_err <= 1.0)

        # v_fix: radial distortion grows with distance from the principal point
        # (cx, cy) — a line drawn near the image center will show almost no
        # curvature regardless of whether D is correct, and can silently make
        # a real bug (D never applied) look identical to "no distortion here".
        _, _, active_source = self.get_active_KD()
        cx, cy = self.K[0, 2], self.K[1, 2]
        mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dist_to_center_px = float(math.hypot(mid_x - cx, mid_y - cy))
        img_diag_half = float(math.hypot(w, h)) / 2.0
        near_center_ratio = dist_to_center_px / max(img_diag_half, 1e-6)
        near_center_warning = near_center_ratio < 0.35  # line sits in the weakest-distortion third of the FOV

        results = {
            "valid": True,
            "p1": p1,
            "p2": p2,
            "line_length_px": line_len,
            "max_raw_deviation_px": max_raw_err,
            "mean_raw_deviation_px": mean_raw_err,
            "max_undist_deviation_px": max_undist_err,
            "mean_undist_deviation_px": mean_undist_err,
            "is_compliant": is_compliant,
            "curvature_reduction_pct": float(max(0.0, (1.0 - max_undist_err / max(max_raw_err, 1e-4)) * 100.0)),
            "raw_residuals": raw_residuals.tolist(),
            "undist_residuals": undist_residuals.tolist(),
            "ideal_coords": sample_coords_ideal,
            "undist_coords": sample_coords_undist,
            "distortion_source": active_source,
            "dist_to_center_px": dist_to_center_px,
            "near_center_warning": near_center_warning,
        }
        self.straight_line_results = results
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  HUD GRAPHIQUE TESTS A & B
# ══════════════════════════════════════════════════════════════════════════════

class DistortionHUD_AB:
    COLOR_BG_PANEL  = (22, 24, 28)
    COLOR_PRIMARY   = (255, 170, 0)
    COLOR_ACCENT    = (0, 220, 255)
    COLOR_GREEN     = (50, 220, 100)
    COLOR_YELLOW    = (0, 215, 255)
    COLOR_RED       = (60, 60, 255)
    COLOR_TEXT_DIM  = (160, 160, 170)
    COLOR_TEXT_MAIN = (240, 240, 245)

    def __init__(self, evaluator: DistortionEvaluatorAB):
        self.evaluator = evaluator
        self.active_mode = 2            # 1: Damier/RMSE, 2: Lignes Droites
        self.show_undistorted = False
        self.show_distortion_grid = True
        self.is_frozen = False
        self.frozen_frame: Optional[np.ndarray] = None
        self.toast_msg: Optional[str] = None
        self.toast_time: float = 0.0

    def show_toast(self, msg: str, duration: float = 3.0):
        self.toast_msg = msg
        self.toast_time = time.time() + duration

    def draw_top_nav(self, canvas: np.ndarray):
        h, w = canvas.shape[:2]
        bar_h = 42

        cv2.rectangle(canvas, (0, 0), (w, bar_h), (16, 18, 22), -1)
        cv2.line(canvas, (0, bar_h), (w, bar_h), (50, 55, 65), 1)

        modes = [
            (1, " [1] TEST A : Damier / RMSE "),
            (2, " [2] TEST B : Rectilinéarité Lignes Droites "),
        ]

        x_offset = 15
        for mode_id, title in modes:
            is_cur = (self.active_mode == mode_id)
            bg_col = (50, 120, 20) if is_cur else (30, 32, 38)
            txt_col = (255, 255, 255) if is_cur else self.COLOR_TEXT_DIM

            tw = len(title) * 8 + 10
            cv2.rectangle(canvas, (x_offset, 6), (x_offset + tw, bar_h - 6), bg_col, -1)
            cv2.rectangle(canvas, (x_offset, 6), (x_offset + tw, bar_h - 6), (80, 85, 95), 1)
            cv2.putText(canvas, title, (x_offset + 5, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, txt_col, 1, cv2.LINE_AA)
            x_offset += tw + 15

        # Résolution
        cv2.putText(canvas, f"Flux: {w}x{h}", (w - 480, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA)

        # Status UNDISTORT & GRID
        status_undist = "[U] UNDISTORT: ON" if self.show_undistorted else "[U] UNDISTORT: OFF"
        undist_col = self.COLOR_GREEN if self.show_undistorted else self.COLOR_TEXT_DIM
        cv2.putText(canvas, status_undist, (w - 340, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, undist_col, 1, cv2.LINE_AA)

        status_grid = "[G] GRILLE: ON" if self.show_distortion_grid else "[G] GRILLE: OFF"
        grid_col = self.COLOR_ACCENT if self.show_distortion_grid else self.COLOR_TEXT_DIM
        cv2.putText(canvas, status_grid, (w - 170, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, grid_col, 1, cv2.LINE_AA)

    def draw_bottom_bar(self, canvas: np.ndarray):
        h, w = canvas.shape[:2]
        bar_h = 32
        cv2.rectangle(canvas, (0, h - bar_h), (w, h), (16, 18, 22), -1)
        cv2.line(canvas, (0, h - bar_h), (w, h - bar_h), (50, 55, 65), 1)

        help_text = "[Espace]: Capturer/Geler  |  [C]: Calculer/Effacer  |  [S]: Sauvegarder Rapport  |  [R]: Reset  |  [Q]: Quitter"
        cv2.putText(canvas, help_text, (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        if self.toast_msg and time.time() < self.toast_time:
            tw = len(self.toast_msg) * 8 + 20
            cv2.rectangle(canvas, (w - tw - 20, h - bar_h - 40), (w - 20, h - bar_h - 8), (20, 100, 40), -1)
            cv2.rectangle(canvas, (w - tw - 20, h - bar_h - 40), (w - 20, h - bar_h - 8), self.COLOR_GREEN, 1)
            cv2.putText(canvas, self.toast_msg, (w - tw - 10, h - bar_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_distortion_vector_field(self, canvas: np.ndarray):
        if not self.show_distortion_grid:
            return
        h, w = canvas.shape[:2]
        step = 40
        grid_y, grid_x = np.mgrid[20:h - 20:step, 20:w - 20:step]
        pts_raw = np.vstack((grid_x.flatten(), grid_y.flatten())).T.astype(np.float64).reshape(-1, 1, 2)
        # v_fix: was hardcoded to self.evaluator.K/D (driver, D=0 → zero-length
        # vectors always). Now uses whichever K/D is actually active.
        K_active, D_active, _ = self.evaluator.get_active_KD()
        pts_undist = cv2.undistortPoints(pts_raw, K_active, D_active, P=K_active)

        for p_raw, p_und in zip(pts_raw.reshape(-1, 2), pts_undist.reshape(-1, 2)):
            rx, ry = int(round(p_raw[0])), int(round(p_raw[1]))
            ux, uy = int(round(p_und[0])), int(round(p_und[1]))
            cv2.circle(canvas, (rx, ry), 1, (100, 100, 120), -1)
            dist_px = math.hypot(ux - rx, uy - ry)
            if dist_px >= 0.5:
                cv2.line(canvas, (rx, ry), (ux, uy), (0, 180, 255), 1, cv2.LINE_AA)
                cv2.circle(canvas, (ux, uy), 2, (0, 255, 255), -1)

    def render_mode_a(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        canvas = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = self.evaluator.detect_chessboard(gray)
        if found and corners is not None:
            cv2.drawChessboardCorners(canvas, self.evaluator.chessboard_size, corners, found)
            cv2.putText(canvas, "MIRE DETECTEE ! Appuyez sur [Espace] pour enregistrer.",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_GREEN, 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, f"Recherche damier ({self.evaluator.chessboard_size[0]}x{self.evaluator.chessboard_size[1]})...",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_YELLOW, 1, cv2.LINE_AA)

        panel_w, panel_h = 320, 320
        px0, py0 = w - panel_w - 20, 60

        overlay = canvas.copy()
        cv2.rectangle(overlay, (px0, py0), (px0 + panel_w, py0 + panel_h), self.COLOR_BG_PANEL, -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (px0, py0), (px0 + panel_w, py0 + panel_h), (70, 75, 85), 1)

        cv2.putText(canvas, "TEST A : CALIBRATION DAMIER", (px0 + 15, py0 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Vues capturees : {len(self.evaluator.calib_img_points)}", (px0 + 15, py0 + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

        map_w, map_h = 160, 90
        mx0, my0 = px0 + 15, py0 + 75
        cv2.rectangle(canvas, (mx0, my0), (mx0 + map_w, my0 + map_h), (10, 10, 15), -1)
        cv2.rectangle(canvas, (mx0, my0), (mx0 + map_w, my0 + map_h), (80, 85, 95), 1)

        if self.evaluator.calib_coverage_map is not None:
            cov_small = cv2.resize(self.evaluator.calib_coverage_map, (map_w, map_h))
            cov_colored = cv2.applyColorMap((cov_small * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.addWeighted(cov_colored, 0.7, canvas[my0:my0 + map_h, mx0:mx0 + map_w], 0.3, 0,
                            canvas[my0:my0 + map_h, mx0:mx0 + map_w])

        cv2.putText(canvas, "Couverture FOV", (mx0 + map_w + 10, my0 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        if self.evaluator.calib_rmse is not None:
            rmse = self.evaluator.calib_rmse
            col = self.COLOR_GREEN if rmse < 0.3 else (self.COLOR_YELLOW if rmse <= 0.7 else self.COLOR_RED)
            cv2.putText(canvas, f"RMSE Total : {rmse:.3f} px", (px0 + 15, py0 + 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2, cv2.LINE_AA)

            txt_eval = "QUALITE : EXCELLENTE (<0.3 px)" if rmse < 0.3 else ("QUALITE : STANDARD" if rmse <= 0.7 else "QUALITE : MEDIOCRE")
            cv2.putText(canvas, txt_eval, (px0 + 15, py0 + 215), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

            if self.evaluator.calib_K is not None:
                fx, fy = self.evaluator.calib_K[0, 0], self.evaluator.calib_K[1, 1]
                cv2.putText(canvas, f"Focale fx={fx:.1f}, fy={fy:.1f}", (px0 + 15, py0 + 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
            if self.evaluator.calib_D is not None:
                k1, k2 = self.evaluator.calib_D[0], self.evaluator.calib_D[1]
                cv2.putText(canvas, f"Distorsion k1={k1:.4f}, k2={k2:.4f}", (px0 + 15, py0 + 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "RMSE : Non calcule", (px0 + 15, py0 + 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
            cv2.putText(canvas, "Pressez [C] apres 10+ vues.", (px0 + 15, py0 + 215),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_YELLOW, 1, cv2.LINE_AA)

        cv2.putText(canvas, self.evaluator.calib_status_msg, (px0 + 15, py0 + 295),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

        return canvas

    def render_mode_b(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        # v_fix: THE bug — this used to always call cv2.undistort(frame, self.evaluator.K,
        # self.evaluator.D), where self.evaluator.D is the RealSense driver's D=[0,0,0,0,0].
        # Test A's calibrated D (self.evaluator.calib_D) was computed but never reached
        # here, so "Test B" dedistorted with zero distortion — a no-op that produced
        # identical raw/undist residuals no matter what. Now goes through get_active_KD().
        K_active, D_active, distortion_source = self.evaluator.get_active_KD()
        undist_frame = cv2.undistort(frame, K_active, D_active)
        canvas = undist_frame.copy() if self.show_undistorted else frame.copy()

        self.draw_distortion_vector_field(canvas)

        pts = self.evaluator.line_selector.points
        for i, pt in enumerate(pts):
            cv2.circle(canvas, pt, 6, self.COLOR_RED, -1, cv2.LINE_AA)
            cv2.putText(canvas, f"P{i+1}", (pt[0] + 10, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_RED, 1, cv2.LINE_AA)

        if len(pts) == 2:
            p1, p2 = pts[0], pts[1]
            cv2.line(canvas, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)

            res = self.evaluator.evaluate_straight_line(frame, undist_frame, p1, p2)
            if res.get("valid"):
                undist_coords = res["undist_coords"]
                for i in range(len(undist_coords) - 1):
                    c1 = (int(round(undist_coords[i][0])), int(round(undist_coords[i][1])))
                    c2 = (int(round(undist_coords[i+1][0])), int(round(undist_coords[i+1][1])))
                    cv2.line(canvas, c1, c2, (0, 255, 0), 1, cv2.LINE_AA)

        panel_w, panel_h = 340, 320
        px0, py0 = w - panel_w - 20, 60

        overlay = canvas.copy()
        cv2.rectangle(overlay, (px0, py0), (px0 + panel_w, py0 + panel_h), self.COLOR_BG_PANEL, -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (px0, py0), (px0 + panel_w, py0 + panel_h), (70, 75, 85), 1)

        cv2.putText(canvas, "TEST B : RECTILINEARITE", (px0 + 15, py0 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_ACCENT, 1, cv2.LINE_AA)

        # v_fix: make the active D source an observable HUD fact, not an assumption
        src_col = self.COLOR_GREEN if distortion_source == 'calibrated_test_a' else self.COLOR_RED
        src_txt = ("Source D : CALIBREE (Test A)" if distortion_source == 'calibrated_test_a'
                   else "Source D : DRIVER (D=0, AUCUNE correction)")
        cv2.putText(canvas, src_txt, (px0 + 15, py0 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, src_col, 1, cv2.LINE_AA)

        if len(pts) < 2:
            cv2.putText(canvas, "Cliquez sur 2 extremites d'une arete", (px0 + 15, py0 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_YELLOW, 1, cv2.LINE_AA)
            cv2.putText(canvas, "(mur, cadre de porte, plinthe au sol)", (px0 + 15, py0 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        else:
            res = self.evaluator.straight_line_results
            if res and res.get("valid"):
                max_undist = res["max_undist_deviation_px"]
                mean_undist = res["mean_undist_deviation_px"]
                max_raw = res["max_raw_deviation_px"]
                is_comp = res["is_compliant"]

                col = self.COLOR_GREEN if is_comp else self.COLOR_RED
                status_txt = "CONFORME (Ecart <= 1.0 px)" if is_comp else "NON-CONFORME (> 1.0 px)"

                cv2.putText(canvas, f"Verdict : {status_txt}", (px0 + 15, py0 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
                cv2.putText(canvas, "Image Dedistordue (Undistort) :", (px0 + 15, py0 + 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
                cv2.putText(canvas, f"  * Ecart Max : {max_undist:.2f} px", (px0 + 15, py0 + 118),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
                cv2.putText(canvas, f"  * Ecart Moyen : {mean_undist:.2f} px", (px0 + 15, py0 + 138),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

                cv2.putText(canvas, "Image Brute (Sans Undistort) :", (px0 + 15, py0 + 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
                cv2.putText(canvas, f"  * Ecart Max brut : {max_raw:.2f} px", (px0 + 15, py0 + 192),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

                red_pct = res["curvature_reduction_pct"]
                cv2.putText(canvas, f"Gain rectilinearite : +{red_pct:.1f}%", (px0 + 15, py0 + 225),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_GREEN, 1, cv2.LINE_AA)

                if res.get("near_center_warning"):
                    cv2.putText(canvas, "[!] Ligne proche du centre optique :", (px0 + 15, py0 + 255),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.COLOR_YELLOW, 1, cv2.LINE_AA)
                    cv2.putText(canvas, "    distorsion faible ici par nature.", (px0 + 15, py0 + 273),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.COLOR_YELLOW, 1, cv2.LINE_AA)
                    cv2.putText(canvas, "    Testez une arete pres d'un coin.", (px0 + 15, py0 + 291),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.COLOR_YELLOW, 1, cv2.LINE_AA)

        return canvas

    def render(self, frame: np.ndarray) -> np.ndarray:
        if self.is_frozen and self.frozen_frame is not None:
            active_frame = self.frozen_frame
        else:
            active_frame = frame

        if self.active_mode == 1:
            out = self.render_mode_a(active_frame)
        else:
            out = self.render_mode_b(active_frame)

        self.draw_top_nav(out)
        self.draw_bottom_bar(out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  NŒUD ROS 2 & RUNNER
# ══════════════════════════════════════════════════════════════════════════════

class DistortionNodeAB(Node if ROS2_AVAILABLE else object):
    def __init__(self, args):
        if ROS2_AVAILABLE:
            super().__init__('distortion_evaluator_node')
            self.get_logger().info("Nœud distortion_evaluator_node initialisé.")

        self.args = args
        self.evaluator = DistortionEvaluatorAB(
            chessboard_size=(args.cb_cols, args.cb_rows),
            square_size_mm=args.square_size_mm
        )
        self.visualizer = DistortionHUD_AB(self.evaluator)
        self.cv_bridge = CvBridge() if ROS2_AVAILABLE else None
        self.latest_cv_image: Optional[np.ndarray] = None

        if ROS2_AVAILABLE and not args.offline and not args.images and not args.image and args.device is None:
            # QoS BEST_EFFORT identique a RealSense / homography_node / raft_avoidance_node
            qos_camera = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            self.sub_image = self.create_subscription(RosImage, args.image_topic, self.image_callback, qos_camera)
            self.sub_info = self.create_subscription(RosCameraInfo, args.info_topic, self.camera_info_callback, qos_camera)
            self.get_logger().info(f"Abonnement actif sur : {args.image_topic} (QoS Best Effort)")

    def camera_info_callback(self, msg: RosCameraInfo):
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        D = np.array(msg.d, dtype=np.float64)
        model = msg.distortion_model or "plumb_bob"
        self.evaluator.update_camera_info(K, D, model)

    def image_callback(self, msg: RosImage):
        try:
            self.latest_cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            if hasattr(self, 'get_logger'):
                self.get_logger().error(f"Erreur decodage image : {e}")

    def save_results(self, rendered_frame: np.ndarray):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.args.output_dir
        os.makedirs(out_dir, exist_ok=True)

        K_active, D_active, distortion_source = self.evaluator.get_active_KD()

        data = {
            "timestamp": timestamp,
            "camera_info": {
                "K": self.evaluator.K.tolist(),
                "D": self.evaluator.D.tolist(),
                "distortion_model": self.evaluator.distortion_model,
            },
            "test_a_rmse_damier": {
                "calib_rmse": self.evaluator.calib_rmse,
                "num_views": len(self.evaluator.calib_img_points),
                "K_calib": self.evaluator.calib_K.tolist() if self.evaluator.calib_K is not None else None,
                "D_calib": self.evaluator.calib_D.tolist() if self.evaluator.calib_D is not None else None,
                "per_view_rmse": self.evaluator.calib_per_view_rmse,
                # v_fix: resolution the calibration was computed at — required to
                # correctly rescale fx/fy/cx/cy if this D_calib is reused at a
                # different stream resolution (e.g. by homography_distortion_evaluator_node.py).
                "image_width": self.evaluator.image_shape[1] if self.evaluator.image_shape else None,
                "image_height": self.evaluator.image_shape[0] if self.evaluator.image_shape else None,
            },
            "test_b_rectilinearite": self.evaluator.straight_line_results,
            # v_fix: explicit traceability of what was actually applied for Test B —
            # 'calibrated_test_a' = real D from the chessboard calibration above,
            # 'driver' = D=[0,0,0,0,0] from CameraInfo, i.e. no correction at all.
            "distortion_source_used": distortion_source,
            "K_used": K_active.tolist(),
            "D_used": D_active.tolist(),
        }

        json_path = os.path.join(out_dir, "distortion_test_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        img_path = os.path.join(out_dir, f"distortion_test_snapshot_{timestamp}.png")
        imwrite_unicode(img_path, rendered_frame)

        self.visualizer.show_toast("Resultats exportes dans distortion_test_results.json")
        print(f"\n[INFO] Resultats Tests A & B sauvegardes :\n  JSON -> {json_path}\n  PNG  -> {img_path}")


def run_evaluator(args):
    win_name = "Intel RealSense - Evaluation Distorsion (Tests A & B)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    if ROS2_AVAILABLE and not args.offline and not args.images and not args.image and args.device is None:
        rclpy.init(args=None)
        node = DistortionNodeAB(args)
    else:
        node = DistortionNodeAB(args)

    cv2.setMouseCallback(win_name, node.evaluator.line_selector.on_mouse)

    cap = None
    image_list: List[str] = []
    current_img_idx = 0

    if args.image:
        image_list = [args.image]
    elif args.images:
        exts = ('*.png', '*.jpg', '*.jpeg')
        for ext in exts:
            image_list.extend(sorted(glob.glob(os.path.join(args.images, ext))))
    elif args.device is not None:
        cap = cv2.VideoCapture(int(args.device) if str(args.device).isdigit() else args.device)

    print("\n================================================================")
    print("  BANC DE TEST DISTORSION (TESTS A & B) DEMARRE")
    print("  • [1] Mode A (Damier/RMSE)      • [2] Mode B (Lignes droites)")
    print("  • [Espace] Capturer / Geler     • [S] Sauvegarder Resultats")
    print("  • [K] Source D: Calibree/Driver • [Q] Quitter")
    print("================================================================\n")

    rendered_view = None

    try:
        while True:
            # Traitement systematique des messages ROS 2 en attente
            if ROS2_AVAILABLE and hasattr(node, 'sub_image') and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.005)

            frame = None
            if hasattr(node, 'latest_cv_image') and node.latest_cv_image is not None:
                frame = node.latest_cv_image.copy()
            elif cap is not None and cap.isOpened():
                ret, cam_frame = cap.read()
                if ret: frame = cam_frame
            elif image_list:
                img_p = image_list[current_img_idx % len(image_list)]
                frame = imread_unicode(img_p)

            if frame is None:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame, f"En attente du flux camera ({args.image_topic})...",
                            (150, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2, cv2.LINE_AA)

            rendered_view = node.visualizer.render(frame)
            cv2.imshow(win_name, rendered_view)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('1'):
                node.visualizer.active_mode = 1
                node.visualizer.show_toast("Mode A activé : Damier & RMSE")
            elif key == ord('2'):
                node.visualizer.active_mode = 2
                node.visualizer.show_toast("Mode B activé : Rectilinéarité")
            elif key == ord('u') or key == ord('U'):
                node.visualizer.show_undistorted = not node.visualizer.show_undistorted
            elif key == ord('k') or key == ord('K'):
                if node.evaluator.calib_D is not None:
                    node.evaluator.use_calibrated_distortion = not node.evaluator.use_calibrated_distortion
                    src = "CALIBREE (Test A)" if node.evaluator.use_calibrated_distortion else "DRIVER (D=0)"
                    node.visualizer.show_toast(f"Source D -> {src}")
                else:
                    node.visualizer.show_toast("Aucune calibration Test A disponible ([1]+[Espace]+[C] d'abord)")
            elif key == ord('g') or key == ord('G'):
                node.visualizer.show_distortion_grid = not node.visualizer.show_distortion_grid
            elif key == 32:
                if node.visualizer.active_mode == 1:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    found, corners = node.evaluator.detect_chessboard(gray)
                    if found and corners is not None:
                        node.evaluator.add_calibration_frame(gray, corners)
                        node.visualizer.show_toast(f"Capture #{len(node.evaluator.calib_img_points)} enregistrée !")
                else:
                    node.visualizer.is_frozen = not node.visualizer.is_frozen
                    node.visualizer.frozen_frame = frame.copy() if node.visualizer.is_frozen else None
            elif key == ord('c') or key == ord('C'):
                if node.visualizer.active_mode == 1:
                    node.evaluator.compute_calibration()
                else:
                    node.evaluator.line_selector.reset()
                    node.evaluator.straight_line_results = None
            elif key == ord('s') or key == ord('S'):
                if rendered_view is not None:
                    node.save_results(rendered_view)
            elif key == ord('r') or key == ord('R'):
                if node.visualizer.active_mode == 1:
                    node.evaluator.reset_calibration()
                else:
                    node.evaluator.line_selector.reset()
                    node.evaluator.straight_line_results = None
            elif key == ord('n') or key == ord('N') or key == 83:
                if image_list:
                    current_img_idx = (current_img_idx + 1) % len(image_list)
            elif key == ord('p') or key == ord('P') or key == 81:
                if image_list:
                    current_img_idx = (current_img_idx - 1 + len(image_list)) % len(image_list)

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if ROS2_AVAILABLE and not args.offline and not args.images and not args.image and args.device is None and rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Évaluation de Distorsion Caméra (Tests A & B)")
    parser.add_argument('--image-topic', type=str, default='/camera/color/image_raw')
    parser.add_argument('--info-topic', type=str, default='/camera/color/camera_info')
    parser.add_argument('--qos-best-effort', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--cb-cols', type=int, default=DEFAULT_CHESSBOARD_SIZE[0])
    parser.add_argument('--cb-rows', type=int, default=DEFAULT_CHESSBOARD_SIZE[1])
    parser.add_argument('--square-size-mm', type=float, default=DEFAULT_SQUARE_SIZE_MM)
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))

    args = parser.parse_args()
    run_evaluator(args)


if __name__ == '__main__':
    main()