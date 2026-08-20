"""
homography_node.py — Ground-Plane Homography Calibration with AprilTags
========================================================================
HOMOGRAPHY PERSISTENCE:
  - At startup, the node searches for H_SAVE_PATH (JSON).
  - If the file exists and is valid -> H is loaded immediately, no visual
    calibration phase needed.
  - If missing or invalid -> interactive visual calibration (all 4 AprilTags
    must be visible).
  - As soon as a new calibration succeeds -> JSON file is updated.

SERVICES:
  /force_recalibrate  (std_srvs/Trigger) — ignores saved file, forces a new
                                            visual calibration and overwrites file.
  /set_robot_ref_pixel (std_srvs/Trigger) — recalculates the robot ground reference
                                             pixel from current frame.

QUICK CONFIGURATION:
  1. ARUCO_CONFIG      : Verify tag IDs match physical corners in metric space.
  2. ROBOT_CAM_OFFSET  : Camera-to-robot base offset (meters).
  3. CAMERA_HEIGHT_M   : Camera height above ground (meters).
  4. CAMERA_TILT_DEG   : Camera tilt angle from vertical (0° = nadir, 90° = horizontal).
  5. H_SAVE_PATH       : JSON persistence file path.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import numpy as np
import os
import json
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from geometry_msgs.msg import Point
from std_srvs.srv import Trigger
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — APRILTAG MARKERS
# ══════════════════════════════════════════════════════════════════════════════

ARUCO_DICT_ID = cv2.aruco.DICT_APRILTAG_36h11

ARUCO_CONFIG = {
    # Convention: X = Forward depth (meters), Y = Lateral right (meters)
    41: (0.73,  0.0),   # Top-left (far ahead, left)
    26: (0.0,   0.0),   # Bottom-left (close, left) -> ORIGIN
    40: (0.73,  0.72),  # Top-right (far ahead, right)
    25: (0.0,   0.72),  # Bottom-right (close, right)
}

MIN_TAGS_REQUIRED = 4

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — ROBOT / CAMERA POSE
# ══════════════════════════════════════════════════════════════════════════════

ROBOT_CAM_OFFSET = (0.0, 0.0)   # (x=forward, y=left), in meters
CAMERA_HEIGHT_M  = 0.78         # in meters
CAMERA_TILT_DEG  = 73.8         # in degrees (0°=nadir, 90°=horizontal)

# ══════════════════════════════════════════════════════════════════════════════
#  PERSISTENCE — HOMOGRAPHY SAVE PATH
# ══════════════════════════════════════════════════════════════════════════════

H_SAVE_PATH = 'config/homography_calibration.json'

# ── YOLO ──────────────────────────────────────────────────────────────────────
MODEL_VERSION  = 'yolo26m.pt'
CONF_THRESHOLD = 0.5
DEVICE         = 0

# ── Bird-Eye View ─────────────────────────────────────────────────────────────
BIRD_EYE_SCALE = 0.01
BIRD_EYE_W     = 300
BIRD_EYE_H     = 250

# ── Debug ──────────────────────────────────────────────────────────────────────
ENABLE_DISPLAY = os.environ.get('DISPLAY') is not None
DEBUG_SCALE    = 0.7

TOPIC_COLOR = '/camera/color/image_raw'

MARKER_COLORS = [
    (0,   255,   0),
    (0,   165, 255),
    (255,   0, 255),
    (0,   255, 255),
]

COLOR_CALIB_TAG    = (0, 255, 0)      # Vert vif pour les tags configurés
COLOR_UNCONFIG_TAG = (255, 0, 255)   # Violet / Magenta vif pour les tags hors-calibration

# ── Configuration Haute Résolution & Détection Multi-Échelles ─────────────────
ENABLE_GLOBAL_UPSCALE   = True        # Active l'upscale global de l'image (ex: 640x480 -> 1280x960)
GLOBAL_UPSCALE_FACTOR   = 2.0         # Facteur d'agrandissement de l'image entière pour détection fine
ENABLE_FAR_ROI_UPSCALE  = True        # Scan supplémentaire sur la zone lointaine
FAR_ROI_Y_MIN_RATIO     = 0.15        # Début de la zone lointaine (15% du haut de l'image)
FAR_ROI_Y_MAX_RATIO     = 0.70        # Fin de la zone lointaine (70% de l'image)
FAR_ROI_UPSCALE_FACTOR  = 2.0         # Facteur d'agrandissement de la ROI





# ══════════════════════════════════════════════════════════════════════════════
#  PERSISTENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_homography(H: np.ndarray, path: str, metadata: dict = None):
    """Saves 3x3 homography matrix H to JSON file."""
    data = {
        'H': H.flatten().tolist(),
        'shape': [3, 3],
        'metadata': metadata or {},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_homography(path: str):
    """Loads 3x3 homography matrix H from JSON file."""
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        H = np.array(data['H'], dtype=np.float64).reshape(3, 3)
        if H.shape != (3, 3):
            return None, None
        np.linalg.inv(H)
        return H, data.get('metadata', {})
    except Exception:
        return None, None


class HomographyNode(Node):
    def __init__(self):
        super().__init__('homography_node')

        self.aruco_ids    = list(ARUCO_CONFIG.keys())
        self.aruco_colors = {aid: MARKER_COLORS[i % len(MARKER_COLORS)]
                             for i, aid in enumerate(self.aruco_ids)}
        self.ground_pos   = {aid: np.array(pos, dtype=np.float32)
                             for aid, pos in ARUCO_CONFIG.items()}
        self.bird_eye_dst = self._ground_to_bird_pixels()

        self.bridge        = CvBridge()
        self.latest_frame  = None
        self.H             = None
        self.H_inv         = None
        self.calibrated    = False

        self._robot_ref_pixel  = None
        self._robot_ground_pos = None
        self._robot_bird_px    = None

        self.aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.minMarkerPerimeterRate = 0.003         # Détecte les tags petits et très éloignés (défaut: 0.03)
        self.aruco_params.maxMarkerPerimeterRate = 4.0           # Détecte les tags très proches
        self.aruco_params.adaptiveThreshWinSizeMin = 3           # Fenêtre de seuillage adaptatif min
        self.aruco_params.adaptiveThreshWinSizeMax = 33          # Fenêtre de seuillage adaptatif max
        self.aruco_params.adaptiveThreshWinSizeStep = 3          # Échantillonnage dense pour les contrastes faibles (défaut: 10)
        self.aruco_params.adaptiveThreshConstant = 3             # Seuil optimisé pour zones d'ombres et reflets rasants
        self.aruco_params.polygonalApproxAccuracyRate = 0.06     # Tolérance optimale pour contours aplatis à 73.8°
        self.aruco_params.perspectiveRemovePixelPerCell = 8      # Résolution d'extraction de la grille de bits
        self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        self.aruco_params.errorCorrectionRate = 0.6              # Tolérance d'erreur de décodage des bits
        try:
            self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        except AttributeError:
            pass


        self.get_logger().info(f'Loading {MODEL_VERSION}...')
        self.model = YOLO(MODEL_VERSION)
        try:
            self.model.to(f'cuda:{DEVICE}')
            self.model(np.zeros((240, 320, 3), dtype=np.uint8), verbose=False, device=DEVICE)
        except Exception:
            self.get_logger().warn('CUDA unavailable for YOLO homography node, falling back to CPU.')
            self.model.to('cpu')
            self.model(np.zeros((240, 320, 3), dtype=np.uint8), verbose=False, device='cpu')
        self.get_logger().info('YOLO ready ✓')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )

        self.create_subscription(Image, TOPIC_COLOR, self.cb_image, qos)

        self.pub_H         = self.create_publisher(Float64MultiArray, '/homography_matrix', 10)
        self.pub_bird      = self.create_publisher(Image, '/bird_eye_view', 10)
        self.pub_debug     = self.create_publisher(Image, '/debug_view', 10)
        self.pub_robot_pos = self.create_publisher(Point, '/robot_ground_position', 10)

        self.create_service(Trigger, '/force_recalibrate',   self.srv_force_recalibrate)
        self.create_service(Trigger, '/set_robot_ref_pixel', self.srv_set_robot_ref_pixel)
        self.create_timer(0.1, self.run)

        self._try_load_saved_H()

        self.get_logger().info(
            f'\nHomography node started.'
            f'\n  Persistence file   : {H_SAVE_PATH}'
            f'\n  Configured tags ({len(self.aruco_ids)}) :'
        )
        for aid, (x, y) in ARUCO_CONFIG.items():
            self.get_logger().info(f'    ID {aid} → ({x}m, {y}m)')
        self.get_logger().info(
            f'\n  Robot camera offset : {ROBOT_CAM_OFFSET[0]:.2f}m, {ROBOT_CAM_OFFSET[1]:.2f}m'
            f'\n  Camera height       : {CAMERA_HEIGHT_M:.2f}m'
            f'\n  Camera tilt         : {CAMERA_TILT_DEG:.1f}°'
        )

    def _try_load_saved_H(self):
        H, meta = load_homography(H_SAVE_PATH)
        if H is None:
            self.get_logger().info(
                f'No saved calibration found ({H_SAVE_PATH}).\n'
                '  → Visual calibration required (show all 4 AprilTags).'
            )
            return

        self.H          = H
        self.H_inv      = np.linalg.inv(H)
        self.calibrated = True
        self._H_calib_w = meta.get('calibrated_image_w', 1280) if meta else 1280
        self._H_calib_h = meta.get('calibrated_image_h', 720) if meta else 720

        err_str  = f"{meta.get('reprojection_error_px', '?'):.2f}px" \
                   if isinstance(meta.get('reprojection_error_px'), float) \
                   else str(meta.get('reprojection_error_px', '?'))
        date_str = meta.get('calibrated_at', 'unknown')

        self.get_logger().info(
            '\n  ╔══════════════════════════════════════════╗\n'
            '  ║  H LOADED FROM PERSISTENCE FILE          ║\n'
            f'  ║  Reprojection error  : {err_str:<14}  ║\n'
            f'  ║  Calibrated at       : {date_str[:19]:<14}  ║\n'
            f'  ║  File                : {H_SAVE_PATH[-25:]:<25}  ║\n'
            '  ║  Use /force_recalibrate to redo          ║\n'
            '  ╚══════════════════════════════════════════╝'
        )

    def _save_H(self, err: float):
        import datetime
        h_cam, w_cam = (self.latest_frame.shape[:2] if self.latest_frame is not None else (720, 1280))
        meta = {
            'reprojection_error_px': round(err, 4),
            'calibrated_at':         datetime.datetime.now().isoformat(),
            'calibrated_image_w':    int(w_cam),
            'calibrated_image_h':    int(h_cam),
            'aruco_config':          {str(k): list(v) for k, v in ARUCO_CONFIG.items()},
            'bird_eye_scale':        BIRD_EYE_SCALE,
            'bird_eye_w':            BIRD_EYE_W,
            'bird_eye_h':            BIRD_EYE_H,
            'camera_height_m':       CAMERA_HEIGHT_M,
            'camera_tilt_deg':       CAMERA_TILT_DEG,
            'robot_cam_offset':      list(ROBOT_CAM_OFFSET),
        }
        try:
            save_homography(self.H, H_SAVE_PATH, meta)
            self.get_logger().info(f'H saved (resolution {w_cam}x{h_cam}) → {H_SAVE_PATH}')
        except Exception as e:
            self.get_logger().error(f'Unable to save H: {e}')

    def srv_force_recalibrate(self, request, response):
        self.calibrated        = False
        self.H                 = None
        self.H_inv             = None
        self._robot_ref_pixel  = None
        self._robot_ground_pos = None
        self._robot_bird_px    = None
        self.get_logger().info(
            'Force recalibration requested — searching for AprilTags...\n'
            f'  File {H_SAVE_PATH} will be updated on next successful calibration.'
        )
        response.success = True
        response.message = 'Force recalibration started — show all 4 AprilTags.'
        return response

    def srv_set_robot_ref_pixel(self, request, response):
        if self.latest_frame is None:
            response.success = False
            response.message = 'No image frame available.'
            return response
        h, w = self.latest_frame.shape[:2]
        self._robot_ref_pixel = self._estimate_robot_ref_pixel(w, h)
        self.get_logger().info(
            f'Robot reference pixel updated: '
            f'({self._robot_ref_pixel[0]:.1f}, {self._robot_ref_pixel[1]:.1f})'
        )
        response.success = True
        response.message = (
            f'Robot ref pixel set to '
            f'({self._robot_ref_pixel[0]:.1f}, {self._robot_ref_pixel[1]:.1f})'
        )
        return response

    def cb_image(self, msg: Image):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _detect_all_tags(self, gray, h_img, w_img):
        """Détecte tous les AprilTags : Passe 1 (Global Haute Résolution x2) + Passe 2 (ROI Lointaine x2)."""
        raw_detections = {}

        def _detect_in_image(img_gray):
            try:
                detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
                return detector.detectMarkers(img_gray)
            except AttributeError:
                c, i, _ = cv2.aruco.detectMarkers(img_gray, self.aruco_dict, parameters=self.aruco_params)
                return c, i, None

        # ── Passe 1 : Image globale (Haute Résolution x2.0 si activée) ───
        if ENABLE_GLOBAL_UPSCALE and GLOBAL_UPSCALE_FACTOR > 1.0:
            scale_g = float(GLOBAL_UPSCALE_FACTOR)
            gray_up = cv2.resize(gray, (0, 0), fx=scale_g, fy=scale_g, interpolation=cv2.INTER_LINEAR)
            corners_full, ids_full, _ = _detect_in_image(gray_up)
            if ids_full is not None:
                for i, aid in enumerate(ids_full.flatten()):
                    # Division stricte par scale_g pour retour exact au repère natif (640x480)
                    corner_pts = corners_full[i][0] / scale_g
                    centre     = corner_pts.mean(axis=0).astype(int)
                    raw_detections[int(aid)] = (corner_pts, centre)
        else:
            corners_full, ids_full, _ = _detect_in_image(gray)
            if ids_full is not None:
                for i, aid in enumerate(ids_full.flatten()):
                    corner_pts = corners_full[i][0]
                    centre     = corner_pts.mean(axis=0).astype(int)
                    raw_detections[int(aid)] = (corner_pts, centre)

        # ── Passe 2 : ROI lointaine ultra-fine upscalée (bande 15%-70%) ──
        if ENABLE_FAR_ROI_UPSCALE:
            y1 = int(h_img * FAR_ROI_Y_MIN_RATIO)
            y2 = int(h_img * FAR_ROI_Y_MAX_RATIO)
            x1 = 0
            x2 = w_img

            roi = gray[y1:y2, x1:x2]
            if roi.size > 0:
                scale_roi = float(FAR_ROI_UPSCALE_FACTOR)
                roi_up = cv2.resize(roi, (0, 0), fx=scale_roi, fy=scale_roi, interpolation=cv2.INTER_LINEAR)
                corners_roi, ids_roi, _ = _detect_in_image(roi_up)

                if ids_roi is not None:
                    for i, aid in enumerate(ids_roi.flatten()):
                        aid = int(aid)
                        corner_pts_roi = corners_roi[i][0]
                        # Remise à l'échelle sur le repère natif (640x480)
                        corner_global  = (corner_pts_roi / scale_roi) + np.array([x1, y1], dtype=np.float32)
                        centre_global  = corner_global.mean(axis=0).astype(int)

                        if aid not in raw_detections:
                            raw_detections[aid] = (corner_global, centre_global)

        # ── Séparation Tags Calibration vs Hors-Calibration ───────────────
        detected_calib_tags = {}
        detected_other_tags = {}
        for aid, (corner_pts, centre) in raw_detections.items():
            if aid in self.aruco_ids:
                detected_calib_tags[aid] = (corner_pts, centre)
            else:
                detected_other_tags[aid] = (corner_pts, centre)

        return detected_calib_tags, detected_other_tags

    def run(self):
        if self.latest_frame is None:
            return

        frame        = self.latest_frame.copy()
        h_img, w_img = frame.shape[:2]

        if self._robot_ref_pixel is None:
            self._robot_ref_pixel = self._estimate_robot_ref_pixel(w_img, h_img)

        small   = cv2.resize(frame, (320, 240))
        results = self.model(small, conf=CONF_THRESHOLD, verbose=False, device=DEVICE)
        sx, sy  = w_img / 320, h_img / 240
        yolo_boxes = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                if float(box.conf[0]) >= CONF_THRESHOLD:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    yolo_boxes.append((
                        int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy),
                        self.model.names[int(box.cls[0])], float(box.conf[0])
                    ))

        # ── Détection AprilTags multi-passes (Global + ROI Upscale lointaine) ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_calib_tags, detected_other_tags = self._detect_all_tags(gray, h_img, w_img)

        if detected_other_tags:
            other_ids_sorted = sorted(list(detected_other_tags.keys()))
            self.get_logger().info(
                f'[Scanner AprilTag] Tags hors-calibration détectés (VIOLET) : IDs {other_ids_sorted}',
                throttle_duration_sec=3.0,
            )

        if not self.calibrated:
            self._try_calibrate(detected_calib_tags)

        if self.calibrated:
            self._update_robot_position()

        cam_view  = self._draw_camera_view(frame, detected_calib_tags, detected_other_tags, yolo_boxes, w_img, h_img)
        bird_view = self._draw_bird_eye(yolo_boxes, w_img, h_img)
        debug     = self._compose_debug(cam_view, bird_view)

        if self.calibrated:
            self._publish_H()
            self._publish_robot_pos()

        self.pub_bird.publish(self.bridge.cv2_to_imgmsg(bird_view, encoding='bgr8'))
        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(debug,    encoding='bgr8'))

        if ENABLE_DISPLAY:
            h_d, w_d = debug.shape[:2]
            small_debug = cv2.resize(debug, (int(w_d * DEBUG_SCALE), int(h_d * DEBUG_SCALE)))
            cv2.imshow('Homography Debug', small_debug)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rclpy.shutdown()
            elif key in [ord('s'), ord('S'), ord('c'), ord('C')]:
                ts = time.strftime('%Y%m%d_%H%M%S')
                raw_path = os.path.join('homography', f'raw_frame_{ts}.png')
                os.makedirs('homography', exist_ok=True)
                cv2.imwrite(raw_path, self.latest_frame)
                self.get_logger().info(f'📸 Frame brute 640x480 sauvegardée pour benchmark : {raw_path}')

    def _estimate_robot_ref_pixel(self, w: int, h: int):
        tilt_rad   = np.radians(CAMERA_TILT_DEG)
        v_fraction = 0.5 + 0.5 * np.cos(tilt_rad)
        return (w / 2.0, h * v_fraction)

    def _update_robot_position(self):
        if self.H is None or self._robot_ref_pixel is None:
            self._robot_ground_pos = None
            self._robot_bird_px    = None
            return

        u, v = self._robot_ref_pixel
        if self.latest_frame is not None and getattr(self, '_H_calib_w', None) and getattr(self, '_H_calib_h', None):
            h_curr, w_curr = self.latest_frame.shape[:2]
            scale_x = float(self._H_calib_w) / max(1, w_curr)
            scale_y = float(self._H_calib_h) / max(1, h_curr)
            u = u * scale_x
            v = v * scale_y

        p = np.array([u, v, 1.0])
        q = self.H @ p
        if abs(q[2]) < 1e-8:
            return
        q /= q[2]

        x_m = BIRD_EYE_H * BIRD_EYE_SCALE - q[1] * BIRD_EYE_SCALE
        y_m = q[0] * BIRD_EYE_SCALE - BIRD_EYE_W * BIRD_EYE_SCALE / 2.0

        x_m += ROBOT_CAM_OFFSET[0]
        y_m += ROBOT_CAM_OFFSET[1]

        by_corr = (BIRD_EYE_H * BIRD_EYE_SCALE - x_m) / BIRD_EYE_SCALE
        bx_corr = (y_m + BIRD_EYE_W * BIRD_EYE_SCALE / 2.0) / BIRD_EYE_SCALE

        self._robot_ground_pos = (x_m, y_m)
        self._robot_bird_px    = (int(bx_corr), int(by_corr))

    def _try_calibrate(self, detected_calib_tags):
        valid_detected = {aid: detected_calib_tags[aid] for aid in self.aruco_ids if aid in detected_calib_tags}
        if len(valid_detected) < MIN_TAGS_REQUIRED:
            missing = [aid for aid in self.aruco_ids if aid not in valid_detected]
            self.get_logger().warn(
                f'{len(valid_detected)}/{len(self.aruco_ids)} AprilTags de calib trouvés '
                f'(min requis: {MIN_TAGS_REQUIRED}). Manquants: {missing}',
                throttle_duration_sec=3.0,
            )
            return

        src_pts, dst_pts, tag_ids = [], [], []
        for aid, (_, centre) in valid_detected.items():
            src_pts.append(centre.astype(np.float32))
            dst_pts.append(self.bird_eye_dst[aid])
            tag_ids.append(aid)

        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        if H is None:
            self.get_logger().error('findHomography failed.')
            return

        err = self._reprojection_error(src_pts, dst_pts, H, tag_ids)
        if err > 8.0:
            self.get_logger().warn(
                f'Reprojection error too high: {err:.1f}px (max 8px).',
                throttle_duration_sec=3.0,
            )
            return

        self.H          = H
        self.H_inv      = np.linalg.inv(H)
        self.calibrated = True

        self._save_H(err)

        self.get_logger().info(
            '\n  ╔══════════════════════════════════════╗\n'
            '  ║  HOMOGRAPHY CALIBRATION SUCCESSFUL   ║\n'
            f'  ║  Reprojection error : {err:.2f} px       ║\n'
            '  ║  H matrix saved to disk              ║\n'
            '  ╚══════════════════════════════════════╝'
        )

    def _draw_camera_view(self, frame, detected_calib_tags, detected_other_tags, yolo_boxes, w, h):
        out = frame.copy()

        for (x1, y1, x2, y2, cls, conf) in yolo_boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f'{cls} {conf:.2f}', (x1, max(y1-8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # ── 1. Dessin des AprilTags utilisés pour la calibration (VERT) ──
        for aid, (corners, centre) in detected_calib_tags.items():
            color = COLOR_CALIB_TAG
            pts   = corners.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(out, [pts], True, color, 3)
            cv2.circle(out, tuple(centre), 6, color, -1)
            cv2.putText(out, f'ID: {aid} (CALIB)', (centre[0] + 8, centre[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            pos = self.ground_pos[aid]
            cv2.putText(out, f'({pos[0]:.1f}m, {pos[1]:.1f}m)',
                        (centre[0] + 8, centre[1] + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # ── 2. Dessin des AprilTags hors-calibration / nouveaux (VIOLET) ──
        for aid, (corners, centre) in detected_other_tags.items():
            color = COLOR_UNCONFIG_TAG
            pts   = corners.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(out, [pts], True, color, 3)
            cv2.circle(out, tuple(centre), 6, color, -1)
            cv2.putText(out, f'ID: {aid}', (centre[0] + 8, centre[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
            cv2.putText(out, f'ID: {aid}', (centre[0] + 8, centre[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(out, '[NON CALIB]', (centre[0] + 8, centre[1] + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        missing = [aid for aid in self.aruco_ids if aid not in detected_calib_tags]
        for i, aid in enumerate(missing):
            color = (0, 165, 255)  # Orange pour les manquants
            cv2.putText(out, f'MISSING ID: {aid}', (10, h - 20 - i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if self._robot_ref_pixel is not None:
            rx, ry = int(self._robot_ref_pixel[0]), int(self._robot_ref_pixel[1])
            cv2.drawMarker(out, (rx, ry), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(out, 'ROBOT REF', (rx + 10, ry - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if self._robot_ground_pos is not None:
            x_m, y_m = self._robot_ground_pos
            cv2.putText(out, f'Robot: ({x_m:.2f}m, {y_m:.2f}m)',
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        h_file_exists = os.path.isfile(H_SAVE_PATH)
        file_color    = (0, 200, 0) if h_file_exists else (0, 100, 255)
        file_txt      = f'H file OK ({os.path.basename(H_SAVE_PATH)})' \
                        if h_file_exists else 'H: No file saved'
        cv2.putText(out, file_txt, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, file_color, 1)

        status_color = (0, 200, 0) if self.calibrated else (0, 100, 255)
        status_txt   = (f'CALIBRATED | H active ({len(detected_calib_tags)}/{len(self.aruco_ids)} Tags)' if self.calibrated
                        else f'CALIBRATING... {len(detected_calib_tags)}/{len(self.aruco_ids)} Tags')
        if detected_other_tags:
            status_txt += f' | HORS-CALIB: IDs {sorted(list(detected_other_tags.keys()))}'

        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 38), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
        cv2.putText(out, status_txt, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        return out

    def _draw_bird_eye(self, yolo_boxes, w_cam, h_cam):
        bird = np.zeros((BIRD_EYE_H, BIRD_EYE_W, 3), dtype=np.uint8)
        bird[:] = (30, 30, 30)

        step = int(0.5 / BIRD_EYE_SCALE)
        for x in range(0, BIRD_EYE_W, step):
            cv2.line(bird, (x, 0), (x, BIRD_EYE_H), (60, 60, 60), 1)
        for y in range(0, BIRD_EYE_H, step):
            cv2.line(bird, (0, y), (BIRD_EYE_W, y), (60, 60, 60), 1)
        step_m = int(1.0 / BIRD_EYE_SCALE)
        for x in range(0, BIRD_EYE_W, step_m):
            cv2.line(bird, (x, 0), (x, BIRD_EYE_H), (80, 80, 80), 1)
        for y in range(0, BIRD_EYE_H, step_m):
            cv2.line(bird, (0, y), (BIRD_EYE_W, y), (80, 80, 80), 1)

        for aid, px_pos in self.bird_eye_dst.items():
            color = self.aruco_colors[aid]
            pt    = tuple(px_pos.astype(int))
            cv2.drawMarker(bird, pt, color, cv2.MARKER_SQUARE, 12, 2)
            gx, gy = ARUCO_CONFIG[aid]
            cv2.putText(bird, f'{aid} ({gx:.1f},{gy:.1f})', (pt[0]+6, pt[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        if (self._robot_bird_px is not None and
                0 <= self._robot_bird_px[0] < BIRD_EYE_W and
                0 <= self._robot_bird_px[1] < BIRD_EYE_H):
            robot_pt = self._robot_bird_px
            x_m, y_m = self._robot_ground_pos
            cv2.drawMarker(bird, robot_pt, (0, 255, 255), cv2.MARKER_TRIANGLE_UP, 18, 2)
            cv2.putText(bird, f'({x_m:.2f}m, {y_m:.2f}m)',
                        (robot_pt[0] + 10, robot_pt[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
        else:
            robot_pt = (BIRD_EYE_W // 2, BIRD_EYE_H - 8)
            cv2.drawMarker(bird, robot_pt, (0, 80, 80), cv2.MARKER_TRIANGLE_UP, 14, 1)
            cv2.putText(bird, 'out of view', (robot_pt[0] + 8, robot_pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 80, 80), 1)

        if self.H_inv is not None:
            for (x1, y1, x2, y2, cls, conf) in yolo_boxes:
                # Point de contact au sol : centre horizontal, bas de la boîte (y2)
                cx_px, cy_px = (x1 + x2) / 2.0, float(y2)
                p = np.array([cx_px, cy_px, 1.0])
                q = self.H_inv @ p
                q /= q[2]
                bx, by = int(q[0]), int(q[1])
                if 0 <= bx < BIRD_EYE_W and 0 <= by < BIRD_EYE_H:
                    cv2.circle(bird, (bx, by), 6, (0, 255, 0), -1)
                    cv2.putText(bird, cls, (bx+7, by+4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                    if self._robot_bird_px is not None:
                        cv2.line(bird, self._robot_bird_px, (bx, by),
                                 (0, 255, 0), 1, cv2.LINE_AA)

        calib_src = 'file' if os.path.isfile(H_SAVE_PATH) and self.calibrated else 'visual'
        calib_txt = f'BIRD-EYE (H active — src: {calib_src})' \
                    if self.calibrated else 'BIRD-EYE (waiting H)'
        cv2.putText(bird, calib_txt, (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 200, 0) if self.calibrated else (0, 100, 255), 1)
        return bird

    def _compose_debug(self, cam_view, bird_view):
        h_cam   = cam_view.shape[0]
        scale   = h_cam / BIRD_EYE_H
        new_w   = int(BIRD_EYE_W * scale)
        bird_rs = cv2.resize(bird_view, (new_w, h_cam), interpolation=cv2.INTER_NEAREST)
        sep     = np.full((h_cam, 3, 3), (100, 100, 100), dtype=np.uint8)
        return np.hstack([cam_view, sep, bird_rs])

    def _ground_to_bird_pixels(self):
        pts = {}
        for aid, (x_m, y_m) in ARUCO_CONFIG.items():
            px = int((y_m + BIRD_EYE_W * BIRD_EYE_SCALE / 2) / BIRD_EYE_SCALE)
            py = int((BIRD_EYE_H * BIRD_EYE_SCALE - x_m) / BIRD_EYE_SCALE)
            pts[aid] = np.array([px, py], dtype=np.float32)
        return pts

    def _reprojection_error(self, src_pts, dst_pts, H, tag_ids=None):
        errors = []
        details = []
        for i in range(len(src_pts)):
            p  = np.array([src_pts[i][0], src_pts[i][1], 1.0])
            p_ = H @ p
            p_ /= p_[2]
            err_px = float(np.linalg.norm(p_[:2] - dst_pts[i]))
            err_cm = err_px * BIRD_EYE_SCALE * 100.0
            errors.append(err_px)
            tid_str = f"Tag {tag_ids[i]}" if tag_ids and i < len(tag_ids) else f"Tag #{i+1}"
            details.append(f"{tid_str}: {err_px:.2f}px ({err_cm:.1f}cm)")

        avg_err_px = sum(errors) / len(errors)
        avg_err_cm = avg_err_px * BIRD_EYE_SCALE * 100.0
        self.get_logger().info(
            f"🎯 Reprojection Error: Average={avg_err_px:.2f}px ({avg_err_cm:.1f}cm) | "
            f"Breakdown: [{' | '.join(details)}]"
        )
        return avg_err_px

    def _publish_H(self):
        msg = Float64MultiArray()
        msg.data = self.H.flatten().tolist()
        d0 = MultiArrayDimension(); d0.label='rows'; d0.size=3; d0.stride=9
        d1 = MultiArrayDimension(); d1.label='cols'; d1.size=3; d1.stride=3
        msg.layout.dim = [d0, d1]
        self.pub_H.publish(msg)

    def _publish_robot_pos(self):
        if self._robot_ground_pos is None:
            return
        msg = Point()
        msg.x = float(self._robot_ground_pos[0])
        msg.y = float(self._robot_ground_pos[1])
        msg.z = 0.0
        self.pub_robot_pos.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HomographyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if ENABLE_DISPLAY:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
