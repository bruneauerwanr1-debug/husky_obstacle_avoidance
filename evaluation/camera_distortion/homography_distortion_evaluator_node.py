#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
homography_distortion_evaluator_node.py
================================================================================
ÉVALUATION D'HOMOGRAPHIE & IMPACT DE LA DISTORSION AU SOL — TEST D
================================================================================

Ce nœud prend le flux vidéo reçu en direct (recommandé en 1280x720x15 comme la calibration)
et évalue l'impact de la distorsion optique sur l'homographie et le repère sol :

  [1] DÉTECTION DES APRILTAGS AU SOL (DICT_APRILTAG_36h11) :
      • Détection automatique multi-passes des marqueurs sol configurés.
      • Synchronisation avec votre configuration ARUCO_CONFIG (ou fichier JSON).

  [2] COMPARAISON SIMULTANÉE DES HOMOGRAPHIES :
      • H_raw : Homographie brute calculée directement sur les pixels sans correction.
      • H_undist : Homographie calculée après dédistorsion rigoureuse avec K et D.

  [3] ANALYSE DES ERREURS MÉTRIQUES RÉELLES AU SOL (en cm) :
      • Erreur médiane Centre (X < 1.5 m / proximité)
      • Erreur médiane Périphérie (FOV > 25° / angles rasants lointains)
      • Gain de précision apporté par la dédistorsion (en cm et %).

TOUCHES CLAVIER (HUD INTERACTIF) :
  [Espace]      : Geler / Dégeler l'image pour analyse fixe
  [U]           : Basculer la vue Dédistordue (Undistort) ON / OFF
  [S]           : Sauvegarder les résultats (homography_test_results.json + PNG)
  [R]           : Réinitialiser les mesures
  [Q] / [Échap] : Quitter
"""

import sys
import os
import time
import json
import argparse
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
#  CONFIGURATION APRILTAGS & INTRINSÈQUES PAR DÉFAUT
# ══════════════════════════════════════════════════════════════════════════════

APRILTAG_DICT_ID = cv2.aruco.DICT_APRILTAG_36h11

# Positions sol réelles en mètres (X = avant, Y = droite) - synchronisé avec homography_calibrationv2.json
DEFAULT_ARUCO_CONFIG: Dict[int, Tuple[float, float]] = {
    41: (1.43, -0.69),
    26: (0.70, -0.70),
    40: (1.41,  0.00),
    25: (0.70,  0.00),
    10: (0.00,  0.00),
    24: (0.70,  0.70),
    39: (1.41,  0.70),
    38: (1.41,  1.41),
}

DEFAULT_K = np.array([
    [910.0,   0.0, 640.0],
    [  0.0, 910.0, 360.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

DEFAULT_D = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def imread_unicode(filepath: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    try:
        with open(filepath, 'rb') as f:
            bytes_data = bytearray(f.read())
            arr = np.asarray(bytes_data, dtype=np.uint8)
            return cv2.imdecode(arr, flags)
    except Exception:
        return cv2.imread(filepath, flags)


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
#  MOTEUR D'ÉVALUATION HOMOGRAPHIE & DISTORSION
# ══════════════════════════════════════════════════════════════════════════════

class HomographyDistortionEvaluator:
    def __init__(self, aruco_config: Dict[int, Tuple[float, float]] = None):
        self.aruco_config = aruco_config or DEFAULT_ARUCO_CONFIG
        self.K = DEFAULT_K.copy()      # "driver" K — updated from live /camera_info every frame
        self.D = DEFAULT_D.copy()      # "driver" D — always [0,0,0,0,0] for the RealSense color stream
        self.distortion_model = "plumb_bob"
        self.has_camera_info = False

        # v_fix: BUG ROOT CAUSE — evaluate_homography() called cv2.undistortPoints
        # with self.K/self.D. Those were overwritten EVERY frame by
        # camera_info_callback() with the live driver values (D=0), so any
        # calibration loaded via load_calibration_file()/load_distortion_calibration()
        # was silently discarded on the very next frame. H_raw and H_undist were
        # therefore always computed from identical inputs — hence the 0.0% gain.
        self.calib_K: Optional[np.ndarray] = None      # real chessboard-calibrated K (Test A)
        self.calib_D: Optional[np.ndarray] = None      # real chessboard-calibrated D (Test A)
        self.calib_source_wh: Optional[Tuple[int, int]] = None  # (w, h) the calibration was computed at
        self.use_calibrated_distortion: bool = False

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(APRILTAG_DICT_ID)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.minMarkerPerimeterRate = 0.003
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 33
        self.aruco_params.adaptiveThreshWinSizeStep = 5
        self.aruco_params.adaptiveThreshConstant = 3
        self.aruco_params.polygonalApproxAccuracyRate = 0.06
        self.aruco_params.errorCorrectionRate = 0.6
        try:
            self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        except AttributeError:
            self.aruco_detector = None

        self.last_results: Optional[Dict[str, Any]] = None

    def update_camera_info(self, K: np.ndarray, D: np.ndarray, model: str = "plumb_bob"):
        self.K = np.array(K, dtype=np.float64).reshape(3, 3)
        self.D = np.array(D, dtype=np.float64).flatten()
        self.distortion_model = model
        self.has_camera_info = True

    def load_calibration_file(self, json_path: str) -> bool:
        """Loads the AprilTag ground config (homography_calibrationv2.json).

        v_fix: previously wrote any top-level 'K'/'D' straight into self.K/self.D,
        which camera_info_callback() then overwrote on the very next ROS message
        (driver D=0). Even when this file DID contain real K/D, the load had no
        lasting effect. Now stored as calib_K/calib_D, which camera_info_callback
        never touches.
        """
        if not os.path.isfile(json_path):
            return False
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'metadata' in data and 'aruco_config' in data['metadata']:
                cfg = data['metadata']['aruco_config']
                self.aruco_config = {int(k): (float(v[0]), float(v[1])) for k, v in cfg.items()}
                print(f"[INFO] Configuration AprilTags chargée ({len(self.aruco_config)} tags)")
            if 'K' in data and 'D' in data:
                self.calib_K = np.array(data['K'], dtype=np.float64).reshape(3, 3)
                self.calib_D = np.array(data['D'], dtype=np.float64).flatten()
                if 'image_width' in data and 'image_height' in data:
                    self.calib_source_wh = (int(data['image_width']), int(data['image_height']))
                self.use_calibrated_distortion = True
                print(f"[INFO] K/D calibrés chargés depuis {os.path.basename(json_path)} — actifs pour Test D.")
            return True
        except Exception as e:
            print(f"[AVERTISSEMENT] Erreur lecture {json_path} : {e}")
            return False

    def load_distortion_calibration(self, json_path: str) -> bool:
        """Loads Test A's chessboard calibration output (distortion_test_results.json).

        v_fix: this is the file that actually contains a NON-ZERO D (from
        cv2.calibrateCamera), nested under test_a_rmse_damier.K_calib/D_calib —
        a structure load_calibration_file() never knew how to read. Without this
        method, Test D had no path at all to the real distortion coefficients,
        regardless of what was passed via --calib-json.
        """
        if not os.path.isfile(json_path):
            print(f"[AVERTISSEMENT] Fichier introuvable : {json_path}")
            return False
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            test_a = data.get('test_a_rmse_damier', {})
            K_calib = test_a.get('K_calib')
            D_calib = test_a.get('D_calib')
            if K_calib is None or D_calib is None:
                print(f"[AVERTISSEMENT] Pas de K_calib/D_calib dans {json_path} "
                      f"(lancez Test A + [C] dans distortion_evaluator_node.py d'abord).")
                return False
            self.calib_K = np.array(K_calib, dtype=np.float64).reshape(3, 3)
            self.calib_D = np.array(D_calib, dtype=np.float64).flatten()
            w = test_a.get('image_width')
            h = test_a.get('image_height')
            self.calib_source_wh = (int(w), int(h)) if (w and h) else None
            self.use_calibrated_distortion = True
            rmse = test_a.get('calib_rmse')
            rmse_str = f"{rmse:.3f}px" if rmse is not None else "?"
            print(f"[INFO] Distorsion calibrée chargée depuis {os.path.basename(json_path)} "
                  f"(RMSE={rmse_str}, résolution source={self.calib_source_wh}) — active pour Test D.")
            return True
        except Exception as e:
            print(f"[AVERTISSEMENT] Erreur lecture {json_path} : {e}")
            return False

    def get_active_KD(self, target_w: int, target_h: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """v_fix: single source of truth for 'which K/D is actually applied'.

        If a calibration is loaded/selected, rescales fx/fy/cx/cy to the CURRENT
        stream resolution when it differs from the resolution the calibration
        was computed at (D itself is resolution-independent for the plumb_bob
        model, so it is reused as-is). Falls back to the driver K/D (D=0,
        i.e. no real correction) only when nothing calibrated is active.
        """
        if self.use_calibrated_distortion and self.calib_K is not None and self.calib_D is not None:
            K = self.calib_K.copy()
            if self.calib_source_wh is not None:
                src_w, src_h = self.calib_source_wh
                if src_w != target_w or src_h != target_h:
                    sx = float(target_w) / float(src_w)
                    sy = float(target_h) / float(src_h)
                    K[0, 0] *= sx  # fx
                    K[0, 2] *= sx  # cx
                    K[1, 1] *= sy  # fy
                    K[1, 2] *= sy  # cy
            return K, self.calib_D, 'calibrated_test_a'
        return self.K, self.D, 'driver'

    def detect_apriltags(self, gray: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """Détection multi-passes (Native + Upscale x2 + ROI Lointaine)."""
        h_img, w_img = gray.shape[:2]
        detected_dict: Dict[int, np.ndarray] = {}

        def _detect_pass(img_gray):
            if self.aruco_detector is not None:
                c, i, _ = self.aruco_detector.detectMarkers(img_gray)
            else:
                c, i, _ = cv2.aruco.detectMarkers(img_gray, self.aruco_dict, parameters=self.aruco_params)
            return c, i

        # 1. Native
        c_nat, i_nat = _detect_pass(gray)
        if i_nat is not None:
            for c_pt, aid in zip(c_nat, i_nat.flatten()):
                detected_dict[int(aid)] = c_pt

        # 2. Upscale x2.0
        gray_up = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        c_up, i_up = _detect_pass(gray_up)
        if i_up is not None:
            for c_pt, aid in zip(c_up, i_up.flatten()):
                aid = int(aid)
                if aid not in detected_dict:
                    detected_dict[aid] = c_pt / 2.0

        # 3. ROI lointaine
        y1, y2 = int(h_img * 0.15), int(h_img * 0.75)
        roi = gray[y1:y2, 0:w_img]
        if roi.size > 0:
            roi_up = cv2.resize(roi, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            c_roi, i_roi = _detect_pass(roi_up)
            if i_roi is not None:
                for c_pt, aid in zip(c_roi, i_roi.flatten()):
                    aid = int(aid)
                    if aid not in detected_dict:
                        corner_native = (c_pt / 2.0) + np.array([[[0, y1]]], dtype=np.float32)
                        detected_dict[aid] = corner_native

        if not detected_dict:
            return [], []

        flat_ids = list(detected_dict.keys())
        corners_list = [detected_dict[aid] for aid in flat_ids]
        return corners_list, flat_ids

    def evaluate_homography(self, gray: np.ndarray) -> Dict[str, Any]:
        """Calcule H_brute vs H_undist et quantifie les résidus métriques au sol (en cm)."""
        corners, ids = self.detect_apriltags(gray)

        if len(ids) < 4:
            res = {
                "valid": False,
                "detected_count": len(ids),
                "error": f"Seulement {len(ids)} tags détectés (min 4 requis pour H)."
            }
            self.last_results = res
            return res

        img_pts_raw = []
        obj_pts_metric = []
        matched_ids = []

        for corner, tag_id in zip(corners, ids):
            if tag_id in self.aruco_config:
                center_px = np.mean(corner[0], axis=0)
                metric_xy = self.aruco_config[tag_id]
                img_pts_raw.append(center_px)
                obj_pts_metric.append(metric_xy)
                matched_ids.append(tag_id)

        if len(matched_ids) < 4:
            res = {
                "valid": False,
                "detected_count": len(ids),
                "matched_count": len(matched_ids),
                "error": f"Seulement {len(matched_ids)} tags configurés (min 4 requis pour H)."
            }
            self.last_results = res
            return res

        pts_raw = np.array(img_pts_raw, dtype=np.float64).reshape(-1, 1, 2)
        pts_obj = np.array(obj_pts_metric, dtype=np.float64).reshape(-1, 1, 2)

        # 1. H Brute
        H_raw, _ = cv2.findHomography(pts_raw, pts_obj, cv2.RANSAC, 5.0)

        # 2. H Dédistordue
        # v_fix: THE bug — this used to call cv2.undistortPoints(pts_raw, self.K,
        # self.D, ...) where self.K/self.D are silently overwritten every frame
        # by camera_info_callback() with the driver's D=[0,0,0,0,0]. Any
        # calibration loaded via load_calibration_file()/load_distortion_calibration()
        # never reached this line, so H_undist == H_raw bit-for-bit and
        # improvement_pct was always 0.0%. Now uses get_active_KD(), which is
        # never touched by the live CameraInfo callback.
        h_img, w_img = gray.shape[:2]
        K_active, D_active, distortion_source = self.get_active_KD(w_img, h_img)
        pts_undist = cv2.undistortPoints(pts_raw, K_active, D_active, P=K_active)
        H_undist, _ = cv2.findHomography(pts_undist, pts_obj, cv2.RANSAC, 5.0)

        if H_raw is None or H_undist is None:
            res = {"valid": False, "error": "Échec du calcul matriciel d'homographie."}
            self.last_results = res
            return res

        # 3. Erreurs métriques en cm
        proj_raw = cv2.perspectiveTransform(pts_raw, H_raw)
        proj_undist = cv2.perspectiveTransform(pts_undist, H_undist)

        errors_raw_cm = np.linalg.norm(proj_raw - pts_obj, axis=2).flatten() * 100.0
        errors_undist_cm = np.linalg.norm(proj_undist - pts_obj, axis=2).flatten() * 100.0

        c_raw_errs, c_und_errs = [], []
        p_raw_errs, p_und_errs = [], []

        for i, (mx, my) in enumerate(obj_pts_metric):
            is_center = (mx < 1.5 and abs(my) <= 0.5)
            if is_center:
                c_raw_errs.append(errors_raw_cm[i])
                c_und_errs.append(errors_undist_cm[i])
            else:
                p_raw_errs.append(errors_raw_cm[i])
                p_und_errs.append(errors_undist_cm[i])

        med_c_raw = float(np.median(c_raw_errs)) if c_raw_errs else float(np.median(errors_raw_cm))
        med_c_und = float(np.median(c_und_errs)) if c_und_errs else float(np.median(errors_raw_cm))
        med_p_raw = float(np.median(p_raw_errs)) if p_raw_errs else med_c_raw
        med_p_und = float(np.median(p_und_errs)) if p_und_errs else med_c_und

        global_raw_med = float(np.median(errors_raw_cm))
        global_und_med = float(np.median(errors_undist_cm))
        gain_pct = float(max(0.0, (1.0 - global_und_med / max(global_raw_med, 1e-4)) * 100.0))

        res = {
            "valid": True,
            "matched_tag_ids": matched_ids,
            "num_tags": len(matched_ids),
            "median_center_raw_cm": med_c_raw,
            "median_center_undist_cm": med_c_und,
            "median_periph_raw_cm": med_p_raw,
            "median_periph_undist_cm": med_p_und,
            "global_raw_median_cm": global_raw_med,
            "global_undist_median_cm": global_und_med,
            "improvement_pct": gain_pct,
            "H_raw": H_raw.tolist(),
            "H_undist": H_undist.tolist(),
            "distortion_source": distortion_source,
        }
        self.last_results = res
        return res


# ══════════════════════════════════════════════════════════════════════════════
#  HUD GRAPHIQUE TEST D
# ══════════════════════════════════════════════════════════════════════════════

class HomographyHUD:
    COLOR_BG_PANEL  = (22, 24, 28)
    COLOR_ACCENT    = (0, 220, 255)
    COLOR_GREEN     = (50, 220, 100)
    COLOR_YELLOW    = (0, 215, 255)
    COLOR_RED       = (60, 60, 255)
    COLOR_TEXT_DIM  = (160, 160, 170)
    COLOR_TEXT_MAIN = (240, 240, 245)

    def __init__(self, evaluator: HomographyDistortionEvaluator):
        self.evaluator = evaluator
        self.show_undistorted = False
        self.is_frozen = False
        self.frozen_frame: Optional[np.ndarray] = None
        self.toast_msg: Optional[str] = None
        self.toast_time: float = 0.0

    def show_toast(self, msg: str, duration: float = 3.0):
        self.toast_msg = msg
        self.toast_time = time.time() + duration

    def render(self, frame: np.ndarray) -> np.ndarray:
        if self.is_frozen and self.frozen_frame is not None:
            active_frame = self.frozen_frame
        else:
            active_frame = frame

        h, w = active_frame.shape[:2]
        canvas = active_frame.copy()
        gray = cv2.cvtColor(active_frame, cv2.COLOR_BGR2GRAY)

        # Détection et dessin des tags
        corners, ids = self.evaluator.detect_apriltags(gray)
        if ids:
            for corner, tag_id in zip(corners, ids):
                pts = corner[0].astype(np.int32)
                is_calib = (tag_id in self.evaluator.aruco_config)
                color = self.COLOR_GREEN if is_calib else (255, 0, 255)
                cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
                cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                cv2.circle(canvas, (cx, cy), 4, color, -1)
                cv2.putText(canvas, f"ID:{tag_id}", (cx - 15, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Calcul d'homographie
        res = self.evaluator.evaluate_homography(gray)

        # Barre supérieure
        bar_h = 42
        cv2.rectangle(canvas, (0, 0), (w, bar_h), (16, 18, 22), -1)
        cv2.line(canvas, (0, bar_h), (w, bar_h), (50, 55, 65), 1)

        title = " TEST D : EVALUATION HOMOGRAPHIE & DISTORSION SOL "
        cv2.rectangle(canvas, (15, 6), (15 + len(title) * 8 + 10, bar_h - 6), (50, 120, 20), -1)
        cv2.putText(canvas, title, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"Flux: {w}x{h}", (w - 380, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA)

        status_undist = "[U] UNDISTORT: ON" if self.show_undistorted else "[U] UNDISTORT: OFF"
        undist_col = self.COLOR_GREEN if self.show_undistorted else self.COLOR_TEXT_DIM
        cv2.putText(canvas, status_undist, (w - 240, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, undist_col, 1, cv2.LINE_AA)

        # Panneau HUD latéral
        panel_w, panel_h = 400, 320
        px0, py0 = w - panel_w - 20, 60

        overlay = canvas.copy()
        cv2.rectangle(overlay, (px0, py0), (px0 + panel_w, py0 + panel_h), self.COLOR_BG_PANEL, -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (px0, py0), (px0 + panel_w, py0 + panel_h), (70, 75, 85), 1)

        cv2.putText(canvas, "EVALUATION HOMOGRAPHIE (H BRUTE VS UNDISTORT)", (px0 + 15, py0 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, self.COLOR_ACCENT, 1, cv2.LINE_AA)

        if not res or not res.get("valid"):
            err_msg = res.get("error", "Recherche AprilTags...") if res else "En attente..."
            cv2.putText(canvas, err_msg, (px0 + 15, py0 + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_YELLOW, 1, cv2.LINE_AA)
        else:
            num_t = res["num_tags"]
            cv2.putText(canvas, f"AprilTags actifs : {num_t} reconnus au sol", (px0 + 15, py0 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

            # v_fix: make the active D source an observable HUD fact — a 0.0%
            # gain with source=driver means undistortion was never applied,
            # which is very different from a 0.0% gain with a real calibrated D.
            distortion_source = res.get("distortion_source", "driver")
            src_col = self.COLOR_GREEN if distortion_source == 'calibrated_test_a' else self.COLOR_RED
            src_txt = ("Source D : CALIBREE (Test A)" if distortion_source == 'calibrated_test_a'
                       else "Source D : DRIVER (D=0, AUCUNE correction)")
            cv2.putText(canvas, src_txt, (px0 + 15, py0 + 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, src_col, 1, cv2.LINE_AA)

            t_y = py0 + 95
            cv2.putText(canvas, "Methode", (px0 + 15, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_DIM, 1)
            cv2.putText(canvas, "Centre (<1.5m)", (px0 + 145, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_DIM, 1)
            cv2.putText(canvas, "Peripherie (>25 deg)", (px0 + 260, t_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_DIM, 1)
            cv2.line(canvas, (px0 + 15, t_y + 8), (px0 + panel_w - 15, t_y + 8), (80, 85, 95), 1)

            c_raw, p_raw = res["median_center_raw_cm"], res["median_periph_raw_cm"]
            cv2.putText(canvas, "H brute", (px0 + 15, t_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_MAIN, 1)
            cv2.putText(canvas, f"~{c_raw:.1f} cm", (px0 + 150, t_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 215, 255), 1)
            cv2.putText(canvas, f"~{p_raw:.1f} cm", (px0 + 275, t_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_RED, 1)

            c_und, p_und = res["median_center_undist_cm"], res["median_periph_undist_cm"]
            cv2.putText(canvas, "Undistort + H", (px0 + 15, t_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_GREEN, 1)
            cv2.putText(canvas, f"~{c_und:.1f} cm", (px0 + 150, t_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_GREEN, 1)
            cv2.putText(canvas, f"~{p_und:.1f} cm", (px0 + 275, t_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_GREEN, 1)

            cv2.line(canvas, (px0 + 15, t_y + 70), (px0 + panel_w - 15, t_y + 70), (80, 85, 95), 1)

            gain = res["improvement_pct"]
            cv2.putText(canvas, f"Gain de precision globale : +{gain:.1f}%", (px0 + 15, t_y + 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_GREEN, 1, cv2.LINE_AA)

            if distortion_source != 'calibrated_test_a':
                diag = "-> Aucune distorsion calibree active : [K] ou --distortion-json"
                diag_col = self.COLOR_YELLOW
            elif p_raw > 10.0 and p_und <= 5.0:
                diag = "-> L'undistort elimine la distorsion en peripherie !"
                diag_col = self.COLOR_TEXT_MAIN
            else:
                diag = "-> Homographie sol precise et uniforme."
                diag_col = self.COLOR_TEXT_MAIN
            cv2.putText(canvas, diag, (px0 + 15, t_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.35, diag_col, 1)

        # Barre inférieure
        cv2.rectangle(canvas, (0, h - 32), (w, h), (16, 18, 22), -1)
        cv2.line(canvas, (0, h - 32), (w, h - 32), (50, 55, 65), 1)
        cv2.putText(canvas, "[Espace]: Geler/Degeler  |  [S]: Sauvegarder Resultats  |  [Q]: Quitter",
                    (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        if self.toast_msg and time.time() < self.toast_time:
            tw = len(self.toast_msg) * 8 + 20
            cv2.rectangle(canvas, (w - tw - 20, h - 72), (w - 20, h - 40), (20, 100, 40), -1)
            cv2.rectangle(canvas, (w - tw - 20, h - 72), (w - 20, h - 40), self.COLOR_GREEN, 1)
            cv2.putText(canvas, self.toast_msg, (w - tw - 10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return canvas


# ══════════════════════════════════════════════════════════════════════════════
#  NŒUD ROS 2 & RUNNER
# ══════════════════════════════════════════════════════════════════════════════

class HomographyDistortionNode(Node if ROS2_AVAILABLE else object):
    def __init__(self, args):
        if ROS2_AVAILABLE:
            super().__init__('homography_distortion_evaluator_node')
            self.get_logger().info("Nœud homography_distortion_evaluator_node initialisé.")

        self.args = args
        self.evaluator = HomographyDistortionEvaluator()
        if args.calib_json:
            self.evaluator.load_calibration_file(args.calib_json)
        else:
            default_calib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "homography", "homography_calibrationv2.json")
            if os.path.isfile(default_calib):
                self.evaluator.load_calibration_file(default_calib)

        # v_fix: this is the load that was MISSING entirely — without it, Test D
        # had no path at all to a real (non-zero) D, regardless of --calib-json,
        # because that file is the ground-tag config, not a camera calibration.
        if args.distortion_json:
            self.evaluator.load_distortion_calibration(args.distortion_json)
        else:
            default_dist = os.path.join(args.output_dir, "distortion_test_results.json")
            if os.path.isfile(default_dist):
                self.evaluator.load_distortion_calibration(default_dist)
            else:
                print("[AVERTISSEMENT] Pas de distortion_test_results.json trouvé — "
                      "Test D tournera avec D=0 (aucune correction) jusqu'à ce que vous "
                      "lanciez distortion_evaluator_node.py (Test A) ou passiez --distortion-json.")

        self.visualizer = HomographyHUD(self.evaluator)
        self.cv_bridge = CvBridge() if ROS2_AVAILABLE else None
        self.latest_cv_image: Optional[np.ndarray] = None

        if ROS2_AVAILABLE and not args.offline and not args.images and not args.image and args.device is None:
            # QoS BEST_EFFORT identique a RealSense / homography_node
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

        data = {
            "timestamp": timestamp,
            "camera_info": {
                "K": self.evaluator.K.tolist(),
                "D": self.evaluator.D.tolist(),
                "distortion_model": self.evaluator.distortion_model,
            },
            "test_d_homographie_sol": self.evaluator.last_results,
            # v_fix: explicit traceability — 'calibrated_test_a' means H_undist
            # above really used a non-zero D; 'driver' means it was a no-op.
            "distortion_source_used": (self.evaluator.last_results.get("distortion_source")
                                       if self.evaluator.last_results else "unknown"),
            "calib_source_resolution": list(self.evaluator.calib_source_wh) if self.evaluator.calib_source_wh else None,
        }

        json_path = os.path.join(out_dir, "homography_test_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        img_path = os.path.join(out_dir, f"homography_test_snapshot_{timestamp}.png")
        imwrite_unicode(img_path, rendered_frame)

        self.visualizer.show_toast("Resultats exportes dans homography_test_results.json")
        print(f"\n[INFO] Resultats Test D sauvegardes :\n  JSON -> {json_path}\n  PNG  -> {img_path}")


def run_evaluator(args):
    win_name = "Intel RealSense - Evaluation Homographie & Distorsion Sol"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    if ROS2_AVAILABLE and not args.offline and not args.images and not args.image and args.device is None:
        rclpy.init(args=None)
        node = HomographyDistortionNode(args)
    else:
        node = HomographyDistortionNode(args)

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
    print("  BANC DE TEST HOMOGRAPHIE SOL (TEST D) DEMARRE")
    print("  • [Espace] Geler / Degeler     • [S] Sauvegarder Resultats")
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
            elif key == ord('u') or key == ord('U'):
                node.visualizer.show_undistorted = not node.visualizer.show_undistorted
            elif key == ord('k') or key == ord('K'):
                if node.evaluator.calib_D is not None:
                    node.evaluator.use_calibrated_distortion = not node.evaluator.use_calibrated_distortion
                    src = "CALIBREE (Test A)" if node.evaluator.use_calibrated_distortion else "DRIVER (D=0)"
                    node.visualizer.show_toast(f"Source D -> {src}")
                else:
                    node.visualizer.show_toast("Aucune calibration chargee (--distortion-json)")
            elif key == 32:
                node.visualizer.is_frozen = not node.visualizer.is_frozen
                node.visualizer.frozen_frame = frame.copy() if node.visualizer.is_frozen else None
            elif key == ord('s') or key == ord('S'):
                if rendered_view is not None:
                    node.save_results(rendered_view)
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
    parser = argparse.ArgumentParser(description="Évaluation Homographie & Distorsion Sol (Test D)")
    parser.add_argument('--image-topic', type=str, default='/camera/color/image_raw')
    parser.add_argument('--info-topic', type=str, default='/camera/color/camera_info')
    parser.add_argument('--qos-best-effort', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--calib-json', type=str, default=None)
    parser.add_argument('--distortion-json', type=str, default=None,
                        help="Chemin vers distortion_test_results.json (Test A) "
                             "contenant le K_calib/D_calib réel — sans ça, Test D "
                             "utilise D=0 (aucune correction).")
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))

    args = parser.parse_args()
    run_evaluator(args)


if __name__ == '__main__':
    main()
