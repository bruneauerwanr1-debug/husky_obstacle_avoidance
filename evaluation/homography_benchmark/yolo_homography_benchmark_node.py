#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_homography_benchmark_node.py
================================================================================
BANC D'ÉVALUATION COMPARATIVE D'HOMOGRAPHIE (H_OLD VS H_NEW) AVEC YOLO & DEPTH GT
================================================================================

Ce nœud évalue l'impact et la précision de l'estimation de distance d'obstacles
détectés par YOLO (yolo26m.pt) en comparant en direct 3 sources de mesure :

  1. VÉRITÉ TERRAIN (GROUND TRUTH) — Caméra de Profondeur RealSense :
     • Mesure physique directe issue du capteur infrarouge / profondeur (/camera/aligned_depth_to_color/image_raw)
     • Filtrage médian spatial subpixel sur la zone de contact au sol de l'obstacle.

  2. ANCIENNE MATRICE H_OLD (homography_calibration.json) :
     • Calibrée sur 4 AprilTags (0.73m x 0.72m) en basse résolution.

  3. NOUVELLE MATRICE H_NEW (homography_calibrationv2.json) :
     • Calibrée sur 8 AprilTags en HD (1280x720) avec moindres carrés régularisés.

FONCTIONNALITÉS :
  • Détection YOLOv26m (ou fallback YOLOv8) en temps réel / sur ROS Bag / vidéo.
  • Projection sol simultanée du point de contact au sol P_sol = ((x1+x2)/2, y2).
  • Calcul en continu des erreurs métriques en centimètres :
      - Erreur H_old = |D_old - D_GT|
      - Erreur H_new = |D_new - D_GT|
      - Gain de précision Delta = Erreur_old - Erreur_new
  • Affichage double vue : Caméra avec badges colorés + Bird-Eye View 2D comparative.
  • Mode sondeur interactif au clic de souris sur n'importe quel pixel de l'image.
  • Export des statistiques et jeu de données CSV pour analyse et courbes de dispersion.

TOUCHES CLAVIER :
  [Espace]      : Geler / Dégeler le flux vidéo
  [B]           : Basculer la vue Bird-Eye View (BEV) agrandie
  [S]           : Sauvegarder les statistiques (CSV + JSON + PNG)
  [R]           : Réinitialiser l'accumulateur statistique
  [Q] / [Échap] : Quitter
"""

import sys
import os
import time
import json
import csv
import argparse
import math
import glob
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

# Encodage console Windows
if sys.platform.startswith('win'):
    try:
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS OPTIONNELS ROS 2 & YOLO
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

YOLO_AVAILABLE = False
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — IDENTIQUE AU NŒUD ORIGINAL (predictive_1.py)
# ══════════════════════════════════════════════════════════════════════════════

# ── YOLO Model ────────────────────────────────────────────────────────────────
MODEL_VERSION  = 'yolo26m.pt'
CONF_THRESHOLD = 0.5
DEVICE         = 0
INFER_SIZE     = (640, 480)

# ── Chemins des Fichiers d'Homographie (Dans le même dossier que ce script) ────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
H_OLD_PATH = os.path.join(SCRIPT_DIR, 'homography_calibration.json')
H_NEW_PATH = os.path.join(SCRIPT_DIR, 'homography_calibrationv2.json')

# ── Paramètres Géométriques & Bird-Eye View ───────────────────────────────────
CAMERA_TILT_DEG    = 73.8
CAMERA_HEIGHT_M    = 0.78
BIRD_EYE_SCALE     = 0.01      # 1 px = 1 cm = 0.01 m
BIRD_EYE_W         = 320
BIRD_EYE_H         = 320
ROBOT_CAM_OFFSET_X = 0.0
ROBOT_CAM_OFFSET_Y = 0.0

# Couleurs BGR
C_GT    = (255, 180, 0)     # Bleu / Cyan = Vérité terrain Depth
C_OLD   = (60, 60, 255)     # Rouge = Ancienne Homographie (4 tags)
C_NEW   = (50, 220, 100)    # Vert  = Nouvelle Homographie (8 tags HD)
C_WARN  = (0, 215, 255)     # Jaune = Avertissement / Écart
C_PANEL = (22, 24, 28)


def imwrite_unicode(filepath: str, img: np.ndarray) -> bool:
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
#  MOTEUR DE PROJECTION & GESTION DES HOMOGRAPHIES
# ══════════════════════════════════════════════════════════════════════════════

class DualHomographyEngine:
    """Charge et applique simultanément H_old et H_new avec compensation d'échelle."""
    def __init__(self, old_path: str = H_OLD_PATH, new_path: str = H_NEW_PATH):
        self.old_path = os.path.abspath(old_path)
        self.new_path = os.path.abspath(new_path)

        self.H_old: Optional[np.ndarray] = None
        self.H_new: Optional[np.ndarray] = None

        self.old_calib_w = 640
        self.old_calib_h = 480
        self.new_calib_w = 1280
        self.new_calib_h = 720

        self.K = np.array([
            [616.74,   0.0, 321.07],
            [  0.0, 616.79, 236.59],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float64)

        self.load_homographies()

    def load_homographies(self):
        # 1. Chargement H_old
        if os.path.isfile(self.old_path):
            try:
                with open(self.old_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.H_old = np.array(data['H'], dtype=np.float64).reshape(3, 3)
                meta = data.get('metadata', {})
                self.old_calib_w = meta.get('calibrated_image_w', 640)
                self.old_calib_h = meta.get('calibrated_image_h', 480)
                print(f"[INFO] H_old chargée : {self.old_path} (Calibrée en {self.old_calib_w}x{self.old_calib_h})")
            except Exception as e:
                print(f"[AVERTISSEMENT] Erreur chargement H_old ({self.old_path}) : {e}")
        else:
            print(f"[ERREUR] Fichier H_old introuvable : {self.old_path}")

        # 2. Chargement H_new
        if os.path.isfile(self.new_path):
            try:
                with open(self.new_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.H_new = np.array(data['H'], dtype=np.float64).reshape(3, 3)
                meta = data.get('metadata', {})
                self.new_calib_w = meta.get('calibrated_image_w', 1280)
                self.new_calib_h = meta.get('calibrated_image_h', 720)
                print(f"[INFO] H_new chargée : {self.new_path} (Calibrée en {self.new_calib_w}x{self.new_calib_h})")
            except Exception as e:
                print(f"[AVERTISSEMENT] Erreur chargement H_new ({self.new_path}) : {e}")
        else:
            print(f"[ERREUR] Fichier H_new introuvable : {self.new_path}")

    def project_pixel_to_ground(self, u: float, v: float, H: np.ndarray,
                                calib_w: int, calib_h: int,
                                curr_w: int, curr_h: int) -> Optional[Tuple[float, float, float]]:
        """
        Projette un pixel (u, v) vers le repère sol métrique du robot (X=avant, Y=droite)
        Retourne (X_sol_m, Y_sol_m, Distance_euclidienne_m).
        """
        if H is None:
            return None

        # Facteur d'échelle résolution
        scale_x = float(calib_w) / max(1, curr_w)
        scale_y = float(calib_h) / max(1, curr_h)

        u_scaled = u * scale_x
        v_scaled = v * scale_y

        p = np.array([u_scaled, v_scaled, 1.0], dtype=np.float64)
        q = H @ p
        if abs(q[2]) < 1e-8:
            return None
        q /= q[2]

        # Conversion vers repère métrique robot (X en avant, Y à droite)
        # Note : Convention Bird-Eye View calibrée
        x_m = (BIRD_EYE_H * BIRD_EYE_SCALE) - (q[1] * BIRD_EYE_SCALE) + ROBOT_CAM_OFFSET_X
        y_m = (q[0] * BIRD_EYE_SCALE) - (BIRD_EYE_W * BIRD_EYE_SCALE / 2.0) + ROBOT_CAM_OFFSET_Y

        dist_m = math.hypot(x_m, y_m)
        return float(x_m), float(y_m), float(dist_m)

    def compute_depth_ground_truth(self, u: float, v: float,
                                   depth_img: np.ndarray,
                                   box: Optional[Tuple[int, int, int, int]] = None,
                                   rgb_shape: Optional[Tuple[int, int]] = None) -> Optional[Tuple[float, float, float]]:
        """
        Extrait la vérité terrain métrique (X_gt, Y_gt, Dist_gt) depuis l'image de profondeur RealSense.
        Adapte automatiquement la résolution si l'image RGB et Depth ont des dimensions différentes.
        """
        if depth_img is None or depth_img.size == 0:
            return None

        h_d, w_d = depth_img.shape[:2]

        # Facteur d'échelle si RGB et Depth ont des résolutions différentes
        scale_x = float(w_d) / max(1, rgb_shape[1]) if rgb_shape else 1.0
        scale_y = float(h_d) / max(1, rgb_shape[0]) if rgb_shape else 1.0

        ix = int(round(u * scale_x))
        iy = int(round(v * scale_y))

        if not (0 <= ix < w_d and 0 <= iy < h_d):
            return None

        # Échantillonnage médian sur patch 5x5 ou zone basse de la boîte
        vals = []
        if box is not None:
            bx1, by1, bx2, by2 = int(box[0] * scale_x), int(box[1] * scale_y), int(box[2] * scale_x), int(box[3] * scale_y)
            # Échantillonner dans les 25% inférieurs de la boîte englobante
            roi_y1 = max(0, int(by2 - (by2 - by1) * 0.25))
            roi_y2 = min(h_d, by2)
            roi_x1 = max(0, int(bx1 + (bx2 - bx1) * 0.2))
            roi_x2 = min(w_d, int(bx2 - (bx2 - bx1) * 0.2))
            if roi_y2 > roi_y1 and roi_x2 > roi_x1:
                roi = depth_img[roi_y1:roi_y2, roi_x1:roi_x2]
                valid = roi[(roi > 200) & (roi < 10000)]
                if len(valid) > 5:
                    vals = valid
        if len(vals) == 0:
            patch = depth_img[max(0, iy - 3):min(h_d, iy + 4), max(0, ix - 3):min(w_d, ix + 4)]
            valid = patch[(patch > 200) & (patch < 10000)]
            if len(valid) > 0:
                vals = valid

        if len(vals) == 0:
            return None

        raw_depth_mm = float(np.median(vals))
        z_cam_m = raw_depth_mm * 0.001

        # Reconstruction 3D dans le repère caméra
        fx = self.K[0, 0] * (float(w_d) / 640.0)
        fy = self.K[1, 1] * (float(h_d) / 480.0)
        cx = self.K[0, 2] * (float(w_d) / 640.0)
        cy = self.K[1, 2] * (float(h_d) / 480.0)

        x_cam = ((ix - cx) / fx) * z_cam_m
        y_cam = ((iy - cy) / fy) * z_cam_m

        # Projection dans le repère robot avec inclinaison caméra (0° = nadir vers le bas, 90° = horizontal vers l'avant)
        tilt_rad = math.radians(CAMERA_TILT_DEG)
        x_robot = z_cam_m * math.sin(tilt_rad) - y_cam * math.cos(tilt_rad) + ROBOT_CAM_OFFSET_X
        y_robot = x_cam + ROBOT_CAM_OFFSET_Y

        dist_gt = math.hypot(x_robot, y_robot)
        return float(x_robot), float(y_robot), float(dist_gt)


# ══════════════════════════════════════════════════════════════════════════════
#  ACCUMULATEUR STATISTIQUE & EXPORT
# ══════════════════════════════════════════════════════════════════════════════

class BenchmarkStatsTracker:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_record(self, obj_class: str, conf: float,
                   d_gt: Optional[float], d_old: float, d_new: float,
                   x_gt: Optional[float], x_old: float, x_new: float,
                   y_gt: Optional[float], y_old: float, y_new: float):
        err_old = abs(d_old - d_gt) if d_gt is not None else None
        err_new = abs(d_new - d_gt) if d_gt is not None else None
        err_old_cm = (abs(d_old - d_gt) * 100.0) if d_gt is not None else None
        err_new_cm = (abs(d_new - d_gt) * 100.0) if d_gt is not None else None
        delta_old_new_cm = abs(d_new - d_old) * 100.0

        rec = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "obstacle_class": obj_class,
            "yolo_confidence": round(conf, 3),
            # ── 3 Distances Fondamentales ─────────────────────────────────────
            "distance_depth_gt_m": round(d_gt, 3) if d_gt is not None else None,
            "distance_homography_old_m": round(d_old, 3),
            "distance_homography_new_m": round(d_new, 3),
            # ── Erreurs & Écarts en cm ────────────────────────────────────────
            "error_h_old_vs_depth_cm": round(err_old_cm, 1) if err_old_cm is not None else None,
            "error_h_new_vs_depth_cm": round(err_new_cm, 1) if err_new_cm is not None else None,
            "delta_old_vs_new_cm": round(delta_old_new_cm, 1),
            # ── Coordonnées Sol Métriques (X=avant, Y=droite) ─────────────────
            "x_depth_gt_m": round(x_gt, 3) if x_gt is not None else None,
            "y_depth_gt_m": round(y_gt, 3) if y_gt is not None else None,
            "x_h_old_m": round(x_old, 3),
            "y_h_old_m": round(y_old, 3),
            "x_h_new_m": round(x_new, 3),
            "y_h_new_m": round(y_new, 3),
        }
        self.records.append(rec)

    def get_summary(self) -> Dict[str, Any]:
        if not self.records:
            return {"count": 0}

        valid_gt = [r for r in self.records if r["distance_depth_gt_m"] is not None]
        n_gt = len(valid_gt)

        # Distances moyennes mesurées par chaque source
        d_gt_list = [r["distance_depth_gt_m"] for r in valid_gt]
        d_old_list = [r["distance_homography_old_m"] for r in self.records]
        d_new_list = [r["distance_homography_new_m"] for r in self.records]

        summary: Dict[str, Any] = {
            "total_detections": len(self.records),
            "paired_with_depth_gt": n_gt,
            "mean_distance_depth_gt_m": float(np.mean(d_gt_list)) if d_gt_list else None,
            "mean_distance_h_old_m": float(np.mean(d_old_list)),
            "mean_distance_h_new_m": float(np.mean(d_new_list)),
        }

        # Écarts H_old vs H_new globaux
        deltas = [r["delta_old_vs_new_cm"] for r in self.records]
        summary["median_delta_old_vs_new_cm"] = float(np.median(deltas))
        summary["max_delta_old_vs_new_cm"] = float(np.max(deltas))

        if n_gt > 0:
            errs_old = [r["error_h_old_vs_depth_cm"] for r in valid_gt]
            errs_new = [r["error_h_new_vs_depth_cm"] for r in valid_gt]

            summary["mae_old_cm"] = float(np.mean(errs_old))
            summary["median_err_old_cm"] = float(np.median(errs_old))
            summary["rmse_old_cm"] = float(np.sqrt(np.mean(np.array(errs_old)**2)))

            summary["mae_new_cm"] = float(np.mean(errs_new))
            summary["median_err_new_cm"] = float(np.median(errs_new))
            summary["rmse_new_cm"] = float(np.sqrt(np.mean(np.array(errs_new)**2)))

            improvement_cm = summary["median_err_old_cm"] - summary["median_err_new_cm"]
            improvement_pct = max(0.0, (1.0 - summary["median_err_new_cm"] / max(summary["median_err_old_cm"], 1e-4)) * 100.0)
            summary["improvement_cm"] = float(improvement_cm)
            summary["improvement_pct"] = float(improvement_pct)

            # Analyse par tranches de distance (<1m, 1-2m, >2m)
            ranges = [
                ("near_under_1m", lambda d: d <= 1.0),
                ("mid_1m_to_2m", lambda d: 1.0 < d <= 2.0),
                ("far_over_2m", lambda d: d > 2.0),
            ]
            summary["distance_brackets"] = {}
            for name, cond in ranges:
                sub = [r for r in valid_gt if cond(r["distance_depth_gt_m"])]
                if sub:
                    e_old = [r["error_h_old_vs_depth_cm"] for r in sub]
                    e_new = [r["error_h_new_vs_depth_cm"] for r in sub]
                    summary["distance_brackets"][name] = {
                        "count": len(sub),
                        "median_err_old_cm": float(np.median(e_old)),
                        "median_err_new_cm": float(np.median(e_new)),
                        "gain_pct": float(max(0.0, (1.0 - np.median(e_new) / max(np.median(e_old), 1e-4)) * 100.0))
                    }

        return summary

    def export_csv(self, filepath: str):
        if not self.records:
            return
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        keys = list(self.records[0].keys())
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)


# ══════════════════════════════════════════════════════════════════════════════
#  HUD GRAPHIQUE COMPARATIF & BIRD-EYE VIEW
# ══════════════════════════════════════════════════════════════════════════════

class YoloHomographyVisualizer:
    def __init__(self, engine: DualHomographyEngine, stats: BenchmarkStatsTracker):
        self.engine = engine
        self.stats = stats
        self.is_frozen = False
        self.frozen_rgb: Optional[np.ndarray] = None
        self.frozen_depth: Optional[np.ndarray] = None
        self.mouse_probe_pt: Optional[Tuple[int, int]] = None
        self.pending_click_save: bool = False
        self.toast_msg: Optional[str] = None
        self.toast_time: float = 0.0

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_probe_pt = (x, y)
            self.pending_click_save = True

    def show_toast(self, msg: str, duration: float = 3.0):
        self.toast_msg = msg
        self.toast_time = time.time() + duration

    def draw_bev_map(self, detections_bev: List[Dict[str, Any]]) -> np.ndarray:
        """Génère la carte Bird-Eye View 2D comparative avec footprint du Husky."""
        bev = np.full((BIRD_EYE_H, BIRD_EYE_W, 3), (25, 28, 32), dtype=np.uint8)

        # Grille métrique (lignes tous les 0.5m et 1.0m)
        cx_px = BIRD_EYE_W // 2
        robot_y_px = BIRD_EYE_H - 40

        for dist_m in np.arange(0.5, 3.5, 0.5):
            py = int(robot_y_px - (dist_m / BIRD_EYE_SCALE))
            if 0 <= py < BIRD_EYE_H:
                col = (60, 65, 75) if int(dist_m * 10) % 10 == 0 else (40, 45, 50)
                cv2.line(bev, (20, py), (BIRD_EYE_W - 20, py), col, 1)
                cv2.putText(bev, f"{dist_m:.1f}m", (BIRD_EYE_W - 48, py - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 145, 155), 1, cv2.LINE_AA)

        # Tracé des corridors latéraux
        for lateral_m in [-0.5, 0.0, 0.5]:
            px = int(cx_px + (lateral_m / BIRD_EYE_SCALE))
            cv2.line(bev, (px, 20), (px, robot_y_px + 20), (50, 55, 65), 1)

        # Dessin du Husky A200 (99cm x 67cm)
        hw_px = int(0.67 / BIRD_EYE_SCALE / 2)
        hl_px = int(0.99 / BIRD_EYE_SCALE)
        cv2.rectangle(bev, (cx_px - hw_px, robot_y_px - hl_px // 2), (cx_px + hw_px, robot_y_px + hl_px // 2), (70, 80, 95), -1)
        cv2.rectangle(bev, (cx_px - hw_px, robot_y_px - hl_px // 2), (cx_px + hw_px, robot_y_px + hl_px // 2), (180, 190, 205), 1)
        cv2.putText(bev, "HUSKY", (cx_px - 18, robot_y_px + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1)

        # Dessin des obstacles
        for d in detections_bev:
            # 1. Point Vérité Terrain (Depth)
            if d.get("gt_m"):
                xg, yg = d["gt_m"][0], d["gt_m"][1]
                px_g = int(cx_px + (yg / BIRD_EYE_SCALE))
                py_g = int(robot_y_px - (xg / BIRD_EYE_SCALE))
                if 0 <= px_g < BIRD_EYE_W and 0 <= py_g < BIRD_EYE_H:
                    cv2.circle(bev, (px_g, py_g), 5, C_GT, -1)

            # 2. Point H_old
            if d.get("old_m"):
                xo, yo = d["old_m"][0], d["old_m"][1]
                px_o = int(cx_px + (yo / BIRD_EYE_SCALE))
                py_o = int(robot_y_px - (xo / BIRD_EYE_SCALE))
                if 0 <= px_o < BIRD_EYE_W and 0 <= py_o < BIRD_EYE_H:
                    cv2.circle(bev, (px_o, py_o), 4, C_OLD, -1)

            # 3. Point H_new
            if d.get("new_m"):
                xn, yn = d["new_m"][0], d["new_m"][1]
                px_n = int(cx_px + (yn / BIRD_EYE_SCALE))
                py_n = int(robot_y_px - (xn / BIRD_EYE_SCALE))
                if 0 <= px_n < BIRD_EYE_W and 0 <= py_n < BIRD_EYE_H:
                    cv2.circle(bev, (px_n, py_n), 4, C_NEW, -1)

            # Trait de divergence entre H_old et H_new
            if d.get("old_m") and d.get("new_m"):
                xo, yo = d["old_m"][0], d["old_m"][1]
                xn, yn = d["new_m"][0], d["new_m"][1]
                p1 = (int(cx_px + (yo / BIRD_EYE_SCALE)), int(robot_y_px - (xo / BIRD_EYE_SCALE)))
                p2 = (int(cx_px + (yn / BIRD_EYE_SCALE)), int(robot_y_px - (xn / BIRD_EYE_SCALE)))
                cv2.line(bev, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)

        return bev

    def render(self, rgb: np.ndarray, depth: Optional[np.ndarray], yolo_results) -> np.ndarray:
        h, w = rgb.shape[:2]
        canvas = rgb.copy()
        detections_bev: List[Dict[str, Any]] = []

        # ── 1. Inférence & Traitement des Détections YOLO ─────────────────────
        if yolo_results is not None:
            boxes = yolo_results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = yolo_results[0].names[cls_id] if hasattr(yolo_results[0], 'names') else f"cls_{cls_id}"

                bx1, by1, bx2, by2 = xyxy
                u_foot = float((bx1 + bx2) / 2.0)
                v_foot = float(by2)

                # Projection H_old
                res_old = self.engine.project_pixel_to_ground(
                    u_foot, v_foot, self.engine.H_old,
                    self.engine.old_calib_w, self.engine.old_calib_h, w, h
                )
                # Projection H_new
                res_new = self.engine.project_pixel_to_ground(
                    u_foot, v_foot, self.engine.H_new,
                    self.engine.new_calib_w, self.engine.new_calib_h, w, h
                )
                # Vérité terrain Depth
                res_gt = self.engine.compute_depth_ground_truth(u_foot, v_foot, depth, box=(bx1, by1, bx2, by2), rgb_shape=(h, w))

                if res_old and res_new:
                    x_o, y_o, d_o = res_old
                    x_n, y_n, d_n = res_new
                    x_g, y_g, d_g = res_gt if res_gt else (None, None, None)

                    self.stats.add_record(cls_name, conf, d_g, d_o, d_n, x_g, x_o, x_n, y_g, y_o, y_n)

                    detections_bev.append({
                        "class": cls_name,
                        "old_m": (x_o, y_o, d_o),
                        "new_m": (x_n, y_n, d_n),
                        "gt_m": (x_g, y_g, d_g) if res_gt else None
                    })

                    # Rendu visuel de la boîte englobante
                    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (240, 240, 245), 2)
                    cv2.circle(canvas, (int(u_foot), int(v_foot)), 5, (0, 255, 255), -1)

                    # Badges de distance au-dessus de l'objet
                    label_y = max(20, by1 - 55)
                    label_w = 190
                    cv2.rectangle(canvas, (bx1, label_y), (bx1 + label_w, by1), (18, 20, 25), -1)
                    cv2.rectangle(canvas, (bx1, label_y), (bx1 + label_w, by1), (60, 65, 75), 1)

                    txt_gt = f"GT (Depth): {d_g:.2f}m" if d_g is not None else "GT (Depth): N/A"
                    cv2.putText(canvas, txt_gt, (bx1 + 5, label_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_GT, 1)

                    err_old_txt = f"(Err: {abs(d_o - d_g)*100:.0f}cm)" if d_g is not None else ""
                    cv2.putText(canvas, f"Old H: {d_o:.2f}m {err_old_txt}", (bx1 + 5, label_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_OLD, 1)

                    err_new_txt = f"(Err: {abs(d_n - d_g)*100:.0f}cm)" if d_g is not None else ""
                    cv2.putText(canvas, f"New H: {d_n:.2f}m {err_new_txt}", (bx1 + 5, label_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_NEW, 1)

        # ── 2. Mode Sondeur Clic Souris Interactif ────────────────────────────
        if self.mouse_probe_pt is not None:
            mx, my = self.mouse_probe_pt
            if 0 <= mx < w and 0 <= my < h:
                cv2.drawMarker(canvas, (mx, my), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
                p_old = self.engine.project_pixel_to_ground(mx, my, self.engine.H_old, self.engine.old_calib_w, self.engine.old_calib_h, w, h)
                p_new = self.engine.project_pixel_to_ground(mx, my, self.engine.H_new, self.engine.new_calib_w, self.engine.new_calib_h, w, h)
                p_gt  = self.engine.compute_depth_ground_truth(mx, my, depth, rgb_shape=(h, w))

                # Enregistrement immédiat dans le CSV si nouvellement cliqué
                if self.pending_click_save and p_old and p_new:
                    d_g = p_gt[2] if p_gt else None
                    x_g = p_gt[0] if p_gt else None
                    y_g = p_gt[1] if p_gt else None
                    self.stats.add_record(
                        obj_class="manual_click",
                        conf=1.0,
                        d_gt=d_g,
                        d_old=p_old[2],
                        d_new=p_new[2],
                        x_gt=x_g,
                        x_old=p_old[0],
                        x_new=p_new[0],
                        y_gt=y_g,
                        y_old=p_old[1],
                        y_new=p_new[1]
                    )
                    txt_dgt = f"{d_g:.2f}m" if d_g else "N/A"
                    self.show_toast(f"Clic ({mx},{my}) enregistre ! Old:{p_old[2]:.2f}m | New:{p_new[2]:.2f}m | GT:{txt_dgt}")
                    self.pending_click_save = False

                px0, py0 = min(w - 220, mx + 15), max(20, my - 40)
                cv2.rectangle(canvas, (px0, py0), (px0 + 200, py0 + 65), (15, 18, 22), -1)
                cv2.rectangle(canvas, (px0, py0), (px0 + 200, py0 + 65), (0, 220, 255), 1)
                cv2.putText(canvas, f"SONDE PIXEL ({mx}, {my})", (px0 + 5, py0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 255), 1)
                if p_gt: cv2.putText(canvas, f"GT Depth : {p_gt[2]:.2f}m (X={p_gt[0]:.2f}m)", (px0 + 5, py0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_GT, 1)
                if p_old: cv2.putText(canvas, f"Old H    : {p_old[2]:.2f}m (X={p_old[0]:.2f}m)", (px0 + 5, py0 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_OLD, 1)
                if p_new: cv2.putText(canvas, f"New H    : {p_new[2]:.2f}m (X={p_new[0]:.2f}m)", (px0 + 5, py0 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_NEW, 1)

        # ── 3. Bandeau Supérieur & HUD Global ─────────────────────────────────
        bar_h = 42
        cv2.rectangle(canvas, (0, 0), (w, bar_h), (16, 18, 22), -1)
        cv2.line(canvas, (0, bar_h), (w, bar_h), (50, 55, 65), 1)

        title = " BENCHMARK ESTIMATION DISTANCE : H_OLD (4 TAGS) VS H_NEW (8 TAGS) "
        cv2.putText(canvas, title, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Légende
        cv2.putText(canvas, "Depth GT", (w - 300, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_GT, 1)
        cv2.putText(canvas, "Old H", (w - 200, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_OLD, 1)
        cv2.putText(canvas, "New H", (w - 110, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_NEW, 1)

        # ── 4. Composition avec Bird-Eye View à droite ────────────────────────
        bev_img = self.draw_bev_map(detections_bev)

        # Panneau statistique sous la BEV
        stat_panel = np.full((h - BIRD_EYE_H, BIRD_EYE_W, 3), (18, 20, 24), dtype=np.uint8)
        cv2.rectangle(stat_panel, (0, 0), (BIRD_EYE_W, h - BIRD_EYE_H), (40, 45, 55), 1)
        cv2.putText(stat_panel, "STATISTIQUES COMPARATIVES", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1)

        summary = self.stats.get_summary()
        tot = summary.get("total_detections", 0)
        n_gt = summary.get("paired_with_depth_gt", 0)

        cv2.putText(stat_panel, f"Objets detectes : {tot} (Apparies GT: {n_gt})", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 225), 1)

        if n_gt > 0:
            e_old = summary.get("median_err_old_cm", 0.0)
            e_new = summary.get("median_err_new_cm", 0.0)
            gain = summary.get("improvement_pct", 0.0)

            cv2.putText(stat_panel, f"Erreur Mediane H_old : {e_old:.1f} cm", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_OLD, 1)
            cv2.putText(stat_panel, f"Erreur Mediane H_new : {e_new:.1f} cm", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_NEW, 1)
            cv2.putText(stat_panel, f"Gain de precision   : +{gain:.1f}%", (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 255, 120), 1)

            # Analyse par tranche
            brackets = summary.get("distance_brackets", {})
            y_b = 165
            if "near_under_1m" in brackets:
                b = brackets["near_under_1m"]
                cv2.putText(stat_panel, f"<1.0m : Old {b['median_err_old_cm']:.1f}cm -> New {b['median_err_new_cm']:.1f}cm", (15, y_b), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 185, 195), 1)
                y_b += 22
            if "mid_1m_to_2m" in brackets:
                b = brackets["mid_1m_to_2m"]
                cv2.putText(stat_panel, f"1-2m  : Old {b['median_err_old_cm']:.1f}cm -> New {b['median_err_new_cm']:.1f}cm", (15, y_b), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 185, 195), 1)
                y_b += 22
            if "far_over_2m" in brackets:
                b = brackets["far_over_2m"]
                cv2.putText(stat_panel, f">2.0m : Old {b['median_err_old_cm']:.1f}cm -> New {b['median_err_new_cm']:.1f}cm", (15, y_b), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 185, 195), 1)
        else:
            cv2.putText(stat_panel, "En attente de flux Depth...", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_WARN, 1)

        side_col = np.vstack((bev_img, stat_panel))
        final_layout = np.hstack((canvas, side_col))

        # Barre inférieure
        fh, fw = final_layout.shape[:2]
        cv2.rectangle(final_layout, (0, fh - 32), (fw, fh), (16, 18, 22), -1)
        cv2.line(final_layout, (0, fh - 32), (fw, fh - 32), (50, 55, 65), 1)
        cv2.putText(final_layout, "[Espace]: Geler/Degeler  |  [Clic Gauche]: Sonde Pixel  |  [S]: Sauvegarder Stats  |  [R]: Reset  |  [Q]: Quitter",
                    (20, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 170), 1, cv2.LINE_AA)

        if self.toast_msg and time.time() < self.toast_time:
            tw = len(self.toast_msg) * 8 + 20
            cv2.rectangle(final_layout, (fw - tw - 20, fh - 72), (fw - 20, fh - 40), (20, 100, 40), -1)
            cv2.rectangle(final_layout, (fw - tw - 20, fh - 72), (fw - 20, fh - 40), C_NEW, 1)
            cv2.putText(final_layout, self.toast_msg, (fw - tw - 10, fh - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return final_layout


# ══════════════════════════════════════════════════════════════════════════════
#  NŒUD ROS 2 & RUNNER
# ══════════════════════════════════════════════════════════════════════════════

class YoloHomographyBenchmarkNode(Node if ROS2_AVAILABLE else object):
    def __init__(self, args):
        if ROS2_AVAILABLE:
            super().__init__('yolo_homography_benchmark_node')
            self.get_logger().info("Nœud yolo_homography_benchmark_node initialisé.")

        self.args = args
        self.engine = DualHomographyEngine(args.h_old, args.h_new)
        self.stats = BenchmarkStatsTracker()
        self.visualizer = YoloHomographyVisualizer(self.engine, self.stats)

        self.cv_bridge = CvBridge() if ROS2_AVAILABLE else None
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None

        # Chargement YOLO (Identique à predictive_1.py)
        self.model = None
        if YOLO_AVAILABLE:
            model_target = args.model if args.model else MODEL_VERSION
            try:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    device_target = f'cuda:{DEVICE}'
                    print(f"[INFO] GPU détecté : {torch.cuda.get_device_name(DEVICE)}")
                else:
                    device_target = 'cpu'
                    print("[INFO] CUDA non disponible — Exécution sur CPU")

                print(f"[INFO] Chargement du modèle {model_target} sur {device_target}...")
                self.model = YOLO(model_target)
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    self.model.to(device_target)
                # Warmup
                self.model(np.zeros((INFER_SIZE[1], INFER_SIZE[0], 3), dtype=np.uint8), verbose=False)
                print("[INFO] YOLO initialisé et prêt ✓")
            except Exception as e:
                print(f"[AVERTISSEMENT] Erreur chargement modèle {model_target} : {e}")

        # Souscriptions ROS 2
        if ROS2_AVAILABLE and not args.offline and not args.images and not args.video:
            qos_sensor = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            self.sub_rgb = self.create_subscription(RosImage, args.rgb_topic, self.rgb_callback, qos_sensor)
            self.sub_depth = self.create_subscription(RosImage, args.depth_topic, self.depth_callback, qos_sensor)
            self.sub_info = self.create_subscription(RosCameraInfo, args.info_topic, self.info_callback, qos_sensor)
            self.get_logger().info(f"Souscriptions : RGB -> {args.rgb_topic} | Depth -> {args.depth_topic}")

    def info_callback(self, msg: RosCameraInfo):
        self.engine.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def rgb_callback(self, msg: RosImage):
        try:
            self.latest_rgb = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            pass

    def depth_callback(self, msg: RosImage):
        try:
            # Conversion depth 16UC1 (millimètres) ou 32FC1 (mètres)
            if msg.encoding == '16UC1':
                self.latest_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            elif msg.encoding == '32FC1':
                depth_m = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
                self.latest_depth = (depth_m * 1000.0).astype(np.uint16)
            else:
                self.latest_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception:
            pass

    def save_benchmark_data(self, frame: np.ndarray):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.args.output_dir
        os.makedirs(out_dir, exist_ok=True)

        # 1. Export CSV des mesures
        csv_path = os.path.join(out_dir, f"yolo_homography_benchmark_{timestamp}.csv")
        self.stats.export_csv(csv_path)

        # 2. Export Résumé JSON
        summary = self.stats.get_summary()
        json_path = os.path.join(out_dir, f"yolo_homography_summary_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 3. Snapshot PNG
        img_path = os.path.join(out_dir, f"yolo_homography_snapshot_{timestamp}.png")
        imwrite_unicode(img_path, frame)

        self.visualizer.show_toast(f"Export réussi ({len(self.stats.records)} détections)")
        print(f"\n[INFO] Données du Benchmark enregistrées :")
        print(f"  CSV  -> {csv_path}")
        print(f"  JSON -> {json_path}")
        print(f"  PNG  -> {img_path}")


def run_benchmark(args):
    win_name = "Benchmark Homographie YOLO - H_old vs H_new & Depth GT"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1440, 720)

    if ROS2_AVAILABLE and not args.offline and not args.images and not args.video:
        rclpy.init(args=None)
        node = YoloHomographyBenchmarkNode(args)
    else:
        node = YoloHomographyBenchmarkNode(args)

    cv2.setMouseCallback(win_name, node.visualizer.on_mouse)

    cap_video = cv2.VideoCapture(args.video) if args.video else None

    print("\n================================================================")
    print("  BANC COMPARATIF HOMOGRAPHIE (H_OLD VS H_NEW & DEPTH GT)")
    print("  • [Espace] Geler / Degeler       • [Clic Souris] Sonde Pixel")
    print("  • [S] Exporter Statistiques      • [R] Reset Accumulateur")
    print("  • [Q] Quitter")
    print("================================================================\n")

    rendered_layout = None

    try:
        while True:
            if ROS2_AVAILABLE and hasattr(node, 'sub_rgb') and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.005)

            rgb_frame = None
            depth_frame = None

            if hasattr(node, 'latest_rgb') and node.latest_rgb is not None:
                rgb_frame = node.latest_rgb.copy()
                if node.latest_depth is not None:
                    depth_frame = node.latest_depth.copy()
            elif cap_video is not None and cap_video.isOpened():
                ret, v_frame = cap_video.read()
                if ret: rgb_frame = v_frame

            if rgb_frame is None:
                rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(rgb_frame, f"En attente de flux ({args.rgb_topic})...",
                            (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)

            # Inférence YOLO
            yolo_res = None
            if node.model is not None:
                try:
                    yolo_res = node.model(rgb_frame, conf=args.conf, verbose=False)
                except Exception:
                    pass

            rendered_layout = node.visualizer.render(rgb_frame, depth_frame, yolo_res)
            cv2.imshow(win_name, rendered_layout)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == 32:
                node.visualizer.is_frozen = not node.visualizer.is_frozen
            elif key == ord('s') or key == ord('S'):
                if rendered_layout is not None:
                    node.save_benchmark_data(rendered_layout)
            elif key == ord('r') or key == ord('R'):
                node.stats.records = []
                node.visualizer.show_toast("Accumulateur statistique reinitialise.")

    finally:
        # Sauvegarde automatique des données accumulées en quittant
        if hasattr(node, 'stats') and len(node.stats.records) > 0:
            print("\n[INFO] Enregistrement automatique des donnees du benchmark...")
            node.save_benchmark_data(rendered_layout if rendered_layout is not None else np.zeros((100, 100, 3), dtype=np.uint8))

            # Affichage de la synthèse dans le terminal
            summary = node.stats.get_summary()
            print("\n" + "=" * 65)
            print("  SYNTHESE BENCHMARK HOMOGRAPHIE (H_OLD VS H_NEW & DEPTH GT)")
            print("=" * 65)
            print(f"  • Total detections enregistrees : {summary.get('total_detections', 0)}")
            print(f"  • Paires validees avec Depth GT : {summary.get('paired_with_depth_gt', 0)}")
            if summary.get('paired_with_depth_gt', 0) > 0:
                print(f"  • Erreur Mediane H_old (4 tags) : {summary.get('median_err_old_cm', 0.0):.2f} cm")
                print(f"  • Erreur Mediane H_new (8 tags) : {summary.get('median_err_new_cm', 0.0):.2f} cm")
                print(f"  • Gain de precision global      : +{summary.get('improvement_pct', 0.0):.1f}%")
            print("=" * 65 + "\n")

        if cap_video is not None:
            cap_video.release()
        cv2.destroyAllWindows()
        if ROS2_AVAILABLE and not args.offline and not args.images and not args.video and rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Homographie H_old vs H_new avec YOLO & Depth GT")
    parser.add_argument('--rgb-topic', type=str, default='/camera/color/image_raw')
    parser.add_argument('--depth-topic', type=str, default='/camera/aligned_depth_to_color/image_raw')
    parser.add_argument('--info-topic', type=str, default='/camera/color/camera_info')
    parser.add_argument('--model', type=str, default=MODEL_VERSION)
    parser.add_argument('--h-old', type=str, default=H_OLD_PATH)
    parser.add_argument('--h-new', type=str, default=H_NEW_PATH)
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--images', type=str, default=None)
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == '__main__':
    main()
