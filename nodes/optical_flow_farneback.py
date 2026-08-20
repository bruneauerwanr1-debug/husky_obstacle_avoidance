"""
optical_flow_farneback.py — Farneback Dense Optical Flow Reactive Collision Avoidance
=====================================================================================
Expected twist_mux architecture (identical to predictive_1.py):

  joy (priority 10)            → /joy_teleop/cmd_vel
  avoidance (priority 50)      → /avoidance_cmd_vel  ← this node in PREDICTING / RECOVERY state
  lock vision_e_stop (prio 255)→ /emergency_stop     ← this node publishes True/False

  twist_mux (output) → /cmd_vel_in → [this node filters] → /cmd_vel → motors

State Machine (identical to predictive_1.py):

  EMERGENCY  : high central optical flow OR confirmed spike OR spin lock
               → complete stop (reversing ALWAYS authorized)

  PREDICTING : L/R imbalance or lateral spike detected
               → avoidance command published on /avoidance_cmd_vel

  RECOVERY   : avoidance maneuver complete + user intended forward motion
               → proportional heading correction

  WARNING    : moderate central flow → proportional EMA deceleration

  CLEAR      : clear path → joystick command passed through unchanged

Modular Architecture (9 classes, zero YOLO):
  OpticalFlowEstimator → EgoMotionCompensator → ZoneAnalyzer
                                              → TrajectoryTracker
  → CollisionRiskEstimator → SideSelector → StateMachine
  → FarnebackAvoidanceNode (ROS 2 I/O orchestration only)

── v10: Advanced Homography, Ground Flow Subtraction & Lens Undistortion ──────────
  - LENS UNDISTORTION: Integrated CameraInfo listener to retrieve camera matrix K and
    distortion coefficients D. Automatically undistorts points prior to homography projection.
  - ROBUST BLOB EXTRACTION: Replaced absolute bottom pixel (which caught floor noise/tile lines)
    with the 95th vertical percentile of the blob, isolating the true ground contact point.
  - THEORETICAL GROUND FLOW SUBTRACTION: Calculates the expected planar optical flow
    induced by robot translation (vx) on ground pixels, canceling out floor texture/tile seams.
  - DUAL INDEPENDENT EGOMOTION SWITCHES:
      * ENABLE_EGOMOTION_COMPENSATION_ALL : Full-image median/zonal derotation.
      * ENABLE_GROUND_FLOW_COMPENSATION   : Planar ground flow subtraction only on ground mask.
  - TEMPORAL DISTANCE SMOOTHING: EMA filter on obstacle distance to eliminate chassis jitter.

── v9: Critical Structural Bug Fix — Nested _publish_flow_diagnostics ──────────
  - FIXED: `_publish_flow_diagnostics` was previously indented inside `run_inference()`.
    Extracted as a proper class method at the same level as `_publish_estop`.
  - Fields previously hardcoded to `None` in `diag` (`n_obstacles`, `joy_fwd`,
    `joy_rev`, `joy_rot`, `left_blocked`, `right_blocked`) are now properly populated.

── v8: Critical Clock Bug Fix — joy_fresh Always False (Root Cause) ────────────
  - FIXED: `joy_fresh` in `run_inference` now compares against `time.monotonic()`
    rather than camera message epoch timestamp, resolving permanent False evaluation.

── v7: State Priority Bug Fix (Avoidance Masking) ──────────────────────────────
  - FIXED: PREDICTING and RECOVERY take precedence over WARNING in StateMachine.update()
    to prevent avoidance turns from being throttled to 0 by warning deceleration ratios.
  - Wall detection confirmed and active (ENABLE_WALL_DEAD_ZONE_DETECTION).

── v6: Zone Occupancy Triggered on WARNING ─────────────────────────────────────
  - SideSelector._zone_occupancy_avoidance now initiates avoidance turns on WARNING
    with a damped turn factor (ZONE_OCCUPANCY_WARNING_TURN_FACTOR = 0.4).

── v5: Reverse / Pure Rotation Absolute Priority ───────────────────────────────
  - Pure rotation (linear.x ≈ 0, angular.z ≠ 0) unconditionally bypasses EMERGENCY/WARNING.
  - User intention evaluation strictly isolated to raw joystick (joy-only).

── v4: Ego-Motion Removal & Zone Occupancy Avoidance ───────────────────────────
  - Removed pixel-wise geometric rotation model; retained statistical filters
    (ENABLE_ZONAL_MEDIAN, ENABLE_MEDIAN_FALLBACK).
  - Inverted wall detection condition to require top AND bottom inactivity.
  - Added Zone Occupancy avoidance to resolve purely central obstacle deadlocks.

── v3: CPU Optimization + Directional Blob Coherence Filter ───────────────────
  - Standardized on CPU-optimized parameters.
  - Filtered residual flow blobs by directional coherence and expansion.

── v2: Trajectory Prediction & Anticipation ────────────────────────────────────
  - Added TrajectoryTracker to track and extrapolate residual flow blobs.
"""

import json
import traceback
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import numpy as np
import os
import time
from dataclasses import dataclass, field
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Float64MultiArray, String
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — ALL TUNABLE PARAMETERS HERE
# ══════════════════════════════════════════════════════════════════════════════

# ── Farneback Parameters (CPU Only) ──────────────────────────────────────────
FARNEBACK_PYR_SCALE  = 0.5
FARNEBACK_LEVELS     = 4       # Pyramid displacement range
FARNEBACK_WINSIZE    = 5       # Optimal window size for subtle edges
FARNEBACK_ITERATIONS = 2       # Sufficient iterations; saves CPU
FARNEBACK_POLY_N     = 5
FARNEBACK_POLY_SIGMA = 1.2
FARNEBACK_FLAGS      = 0
FARNEBACK_BLUR_KSIZE = 5       # GaussianBlur pre-filter kernel size
PROCESS_WIDTH        = 480
PROCESS_HEIGHT       = 360

# ── Zonal Optical Flow — Thresholds (480×360) ────────────────────────────────
FLOW_CENTER_DANGER   = 14.0
FLOW_CENTER_WARN     = 8.0
FLOW_SPIKE_THRESHOLD = 18.0
FLOW_DIFF_AVOID      = 2.5
MOVEMENT_THRESHOLD   = 1.2

# ── Zonal Optical Flow — Resolution-Independent Parameters ───────────────────
FLOW_BALANCE_K       = 0.18
FLOW_MAX_TURN_RAD    = 0.70
FLOW_SMOOTH_ALPHA    = 0.25
FLOW_SPIKE_STREAK_REQUIRED = 2

# ── Ego-Motion Compensation & Ground Flow Switches ───────────────────────────
ENABLE_EGOMOTION_COMPENSATION_ALL  = False  # Full-image egomotion compensation (global or zonal median)
ENABLE_MEDIAN_FALLBACK             = False  # Global median subtraction across entire image
ENABLE_ZONAL_MEDIAN                = False  # Per-zone (L/C/R) median subtraction across entire image

ENABLE_GROUND_FLOW_COMPENSATION    = True   # Subtract theoretical planar ground flow (removes floor seams/lines)
GROUND_FLOW_REDUCTION_FACTOR       = 0.90   # Fraction of predicted ground flow to subtract (0.0 to 1.0)

# ── Lens Undistortion & Temporal Filtering ────────────────────────────────────
ENABLE_LENS_UNDISTORTION           = True   # Undistort optical coordinates using CameraInfo (K, D)
ENABLE_TEMPORAL_DISTANCE_FILTER    = True   # Exponential Moving Average on obstacle distance
DISTANCE_EMA_ALPHA                 = 0.35   # Smoothing weight for distance (0.0=frozen, 1.0=raw)

# ── Ground Plane Geometric Masking ───────────────────────────────────────────
ENABLE_HOMOGRAPHY_COMPENSATION = True   # Geometric ground plane mask via H
TOPIC_CAMERA_INFO              = '/camera/color/camera_info'

# ── Diagnostic Logging ───────────────────────────────────────────────────────
ENABLE_DIAGNOSTIC_LOG  = True
DIAGNOSTIC_LOG_PERIOD  = 1.0

# ── Ground Mask Bands ────────────────────────────────────────────────────────
GROUND_MASK_BOTTOM_RATIO = 0.30
GROUND_MASK_TOP_Y_RATIO = 0.40   # Mask applies only below 40% from top

# ── Explicit Floor AprilTag Masking ──────────────────────────────────────────
ENABLE_APRILTAG_MASKING = True
APRILTAG_DICT_ID        = cv2.aruco.DICT_APRILTAG_36h11
APRILTAG_MASK_DILATE_PX = 4
APRILTAG_CORNER_REFINE  = cv2.aruco.CORNER_REFINE_NONE

# ── Parallax Threshold ───────────────────────────────────────────────────────
PARALLAX_THRESHOLD = 2.0

# ── Wall Dead-Zone Detection ─────────────────────────────────────────────────
ENABLE_WALL_DEAD_ZONE_DETECTION = True
WALL_DEAD_ZONE_RATIO    = 0.70
WALL_MIN_ACTIVE_BORDER  = 0.20
WALL_STREAK_REQUIRED    = 5
WALL_REQUIRE_TOP_INACTIVITY = True
WALL_TOP_RATIO             = 0.30
WALL_TOP_MAX_COVERAGE      = 0.05   # Active coverage above this -> not a wall
WALL_BOTTOM_RATIO          = 0.20
WALL_BOTTOM_MAX_COVERAGE   = 0.05   # Active coverage above this -> not a wall

# ── Trajectory Prediction ────────────────────────────────────────────────────
TRAJ_MIN_BLOB_AREA    = 25
TRAJ_HISTORY_LEN      = 6
TRAJ_MIN_TRACK_AGE    = 3
TRAJ_MATCH_MAX_DIST   = 40
TRAJ_HORIZON_S        = 0.4

# ── State Machine Timing ─────────────────────────────────────────────────────
STOP_RELEASE_CYCLES  = 15
TIMER_PERIOD         = 0.1
WATCHDOG_PERIOD       = 0.05
EMA_ALPHA            = 0.25

# ── Avoidance Commands ────────────────────────────────────────────────────────
AVOID_LINEAR_X       = 0.30
AVOID_ANGULAR_Z      = 0.45
AVOID_ANGULAR_MAX    = 0.80

# ── Zone Occupancy Turn Factors ──────────────────────────────────────────────
ZONE_OCCUPANCY_EMERGENCY_TURN_FACTOR = 0.8
ZONE_OCCUPANCY_WARNING_TURN_FACTOR   = 0.4

# ── Anti-Chatter / Smoothing ──────────────────────────────────────────────────
SIDE_FLIP_STREAK     = 5
AVOID_SMOOTH_ALPHA   = 0.20
AVOID_SMOOTH_ALPHA_URGENT = 0.70

# ── Heading Recovery (RECOVERY) ──────────────────────────────────────────────
YAW_RECOVERY_KP      = 0.6
YAW_RECOVERY_MIN     = 0.10
RECOVERY_LINEAR_X    = 0.15
RECOVERY_MAX_TURN    = 0.50

# ── Cumulative Yaw Anti-Spin Limit ───────────────────────────────────────────
MAX_AVOIDANCE_YAW_DEVIATION = 1.80

# ── Speed Adaptation ──────────────────────────────────────────────────────────
SPEED_ADAPT_MIN   = 0.05
SPEED_ADAPT_MAX   = 0.40
SPEED_ADAPT_FLOOR = 0.4

# ── Dynamic Speed-Dependent Threshold Gains ──────────────────────────────────
SPEED_THRESH_GAIN_CENTER  = 10.0
SPEED_THRESH_GAIN_DIFF    = 5.0
SPEED_THRESH_GAIN_SPIKE   = 12.0

# ── Command & Speed Timeouts ──────────────────────────────────────────────────
CMD_SOURCE_TIMEOUT   = 0.3
ROBOT_SPEED_TIMEOUT  = 0.5

# ── Pure Rotation / Reverse Thresholds ───────────────────────────────────────
PURE_ROTATION_LINEAR_MAX  = 0.05
PURE_ROTATION_ANGULAR_MIN = 0.05

# ── ROS Topics ────────────────────────────────────────────────────────────────
TOPIC_COLOR   = '/camera/color/image_raw'
TOPIC_ODOM    = '/odometry/filtered'
TOPIC_CMD_IN  = '/cmd_vel_in'
TOPIC_JOY     = '/joy_teleop/cmd_vel'
TOPIC_H       = '/homography_matrix'
TOPIC_ROBOT_POS = '/robot_ground_position'

# ── Visualization ─────────────────────────────────────────────────────────────
ENABLE_DISPLAY = os.environ.get('DISPLAY') is not None

# ── OpenCV Colors (BGR) ───────────────────────────────────────────────────────
C_OK       = (0,   255,   0)
C_YOLO     = (255, 200,   0)
C_WARNING  = (0,   165, 255)
C_DANGER   = (0,     0, 255)
C_PREDICT  = (255, 200,   0)
C_RECOVERY = (0,   255, 255)
C_MEMORY   = (180,  80, 255)

FLOW_PANEL_H = 92

# ── Metric Distance Thresholds (Homography) ──────────────────────────────────
DISTANCE_WARN_M   = 1.0
DISTANCE_DANGER_M = 0.5
MAX_OBSTACLE_DIST_M = 10.0

# ── Homography Calibration ────────────────────────────────────────────────────
CALIB_W     = 300
CALIB_H     = 250
CALIB_SCALE = 0.01

# ── Homography Persistence Path ───────────────────────────────────────────────
H_SAVE_PATH = '/media/imr2204/bd37914b-8e04-4d06-b568-4a7cd46f37ab/home/imr/Erwan/clearpath_simulator/src/code_python/homography/homography_calibration.json'

# ── Bird-Eye View ─────────────────────────────────────────────────────────────
MAP_W        = 400
MAP_H        = 400
MAP_SCALE    = 0.01
ROB_OFFSET_Y = 50

# ── Lateral Warning Corridor ──────────────────────────────────────────────────
ROBOT_HALF_WIDTH = 0.40
WARNING_CORRIDOR_HALF_WIDTH = ROBOT_HALF_WIDTH + 0.4

# ── Escape Geometry & Maneuver Times ──────────────────────────────────────────
_ESCAPE_DISTANCE_M = 0.8
_ESCAPE_ANGLE_RAD  = float(np.arctan2(ROBOT_HALF_WIDTH, _ESCAPE_DISTANCE_M))
REQUIRED_MANEUVER_TIME_S = (_ESCAPE_ANGLE_RAD / AVOID_ANGULAR_MAX) * 2.0
TRAJ_EMERGENCY_TTC_MARGIN_S = 0.3
TRAJ_MIN_VX_PX_S = 5.0

# ── TTC Horizon ───────────────────────────────────────────────────────────────
TTC_HORIZON = 3.0

# ── Homography Blob Filtering ────────────────────────────────────────────────
MIN_BLOB_AREA_H = 400
MAX_OBSTACLES   = 6
MIN_SPIKE_BLOB_AREA = 3

# ── Directional Coherence & Vertical Motion Filters ──────────────────────────
BLOB_COHERENCE_MIN = 0.55
BLOB_VERTICAL_MOTION_REJECT  = True
BLOB_VERTICAL_RATIO          = 0.25
BLOB_VERTICAL_BOTTOM_RATIO   = 0.55
BLOB_DIVERGENCE_MIN = 0.02


# ══════════════════════════════════════════════════════════════════════════════
#  HOMOGRAPHY PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_homography(path: str):
    """Loads 3x3 homography matrix H from a JSON calibration file."""
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


# ══════════════════════════════════════════════════════════════════════════════
#  HOMOGRAPHY DISTANCE ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

class HomographyDistanceEstimator:
    """Projects optical flow blobs to metric ground coordinates via camera-to-ground
    homography matrix H with optional lens undistortion and temporal filtering."""

    def __init__(self):
        self.H = None
        self.H_inv = None
        self.robot_ground_pos = (0.0, 0.0)
        self.camera_matrix = None     # 3x3 K matrix
        self.dist_coeffs = None       # D coefficients
        self._distance_ema_map = {}   # Tracked obstacle distance EMA cache

    def set_homography(self, H):
        try:
            self.H = H
            self.H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            self.H = self.H_inv = None

    def load_from_file(self, path=H_SAVE_PATH):
        H, meta = load_homography(path)
        if H is not None:
            self.set_homography(H)
            self._H_calib_w = meta.get('calibrated_image_w', 1280) if meta else 1280
            self._H_calib_h = meta.get('calibrated_image_h', 720) if meta else 720
            return True
        return False

    def set_robot_ground_pos(self, x, y):
        self.robot_ground_pos = (x, y)

    def set_camera_intrinsics(self, K: np.ndarray, D: np.ndarray):
        """Sets intrinsic matrix K and distortion coefficients D for undistortion."""
        if K is not None and K.shape == (3, 3):
            self.camera_matrix = K.astype(np.float64)
        if D is not None:
            self.dist_coeffs = D.astype(np.float64)

    @property
    def ready(self):
        return self.H is not None

    def cam_px_to_ground_m(self, u, v, orig_shape=None):
        if self.H is None:
            return None

        # Adaptation automatique si la résolution runtime diffère de la résolution de calibration
        if orig_shape is not None and getattr(self, '_H_calib_w', None) and getattr(self, '_H_calib_h', None):
            h_curr, w_curr = orig_shape[:2]
            scale_x = float(self._H_calib_w) / max(1, w_curr)
            scale_y = float(self._H_calib_h) / max(1, h_curr)
            u = u * scale_x
            v = v * scale_y

        # Optional lens undistortion prior to planar homography
        u_proj, v_proj = float(u), float(v)
        if ENABLE_LENS_UNDISTORTION and self.camera_matrix is not None and self.dist_coeffs is not None:
            try:
                pt_in = np.array([[[u_proj, v_proj]]], dtype=np.float32)
                pt_undist = cv2.undistortPoints(pt_in, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
                u_proj, v_proj = float(pt_undist[0, 0, 0]), float(pt_undist[0, 0, 1])
            except Exception:
                pass

        p = np.array([u_proj, v_proj, 1.0], dtype=np.float64)
        q = self.H @ p
        if abs(q[2]) < 1e-8:
            return None
        q /= q[2]
        x_m = CALIB_H * CALIB_SCALE - q[1] * CALIB_SCALE
        y_m = q[0] * CALIB_SCALE - CALIB_W * CALIB_SCALE / 2.0
        return float(x_m), float(y_m)

    def dist_from_robot(self, x_m, y_m):
        rx, ry = self.robot_ground_pos
        return float(np.sqrt((x_m - rx)**2 + (y_m - ry)**2))

    def ground_m_to_bird_px(self, x_m, y_m):
        bx = int(MAP_W / 2.0 + y_m / MAP_SCALE)
        by = int(MAP_H - ROB_OFFSET_Y - x_m / MAP_SCALE)
        return bx, by

    MAX_VALID_DIST_M = MAX_OBSTACLE_DIST_M

    def estimate_obstacle_distances(self, blobs, flow_shape, orig_shape):
        if not self.ready:
            return []

        h_flow, w_flow = flow_shape[:2]
        h_cam, w_cam = orig_shape[:2]
        rob_x, _rob_y = self.robot_ground_pos

        obstacles = []
        for blob in blobs:
            if blob.area < MIN_BLOB_AREA_H:
                continue

            u_cam = blob.bottom_x * (w_cam / w_flow)
            v_cam = blob.bottom_y * (h_cam / h_flow)

            ground = self.cam_px_to_ground_m(u_cam, v_cam, orig_shape=orig_shape)
            if ground is None:
                continue
            x_m, y_m = ground
            dist_m = self.dist_from_robot(x_m, y_m)

            if dist_m > self.MAX_VALID_DIST_M:
                continue

            if x_m < rob_x:
                continue

            obstacles.append({
                'x_m': round(x_m, 2),
                'y_m': round(y_m, 2),
                'dist_m': round(dist_m, 2),
                'u_cam': round(u_cam, 1),
                'v_cam': round(v_cam, 1),
                'area': blob.area,
            })

        obstacles.sort(key=lambda o: o['dist_m'])
        return obstacles[:MAX_OBSTACLES]


# ══════════════════════════════════════════════════════════════════════════════
#  1. OpticalFlowEstimator
# ══════════════════════════════════════════════════════════════════════════════

class OpticalFlowEstimator:
    def __init__(self,
                 pyr_scale=FARNEBACK_PYR_SCALE,
                 levels=FARNEBACK_LEVELS,
                 winsize=FARNEBACK_WINSIZE,
                 iterations=FARNEBACK_ITERATIONS,
                 poly_n=FARNEBACK_POLY_N,
                 poly_sigma=FARNEBACK_POLY_SIGMA,
                 flags=FARNEBACK_FLAGS,
                 blur_ksize=FARNEBACK_BLUR_KSIZE):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags
        self.blur_ksize = blur_ksize

    def compute(self, prev_frame, curr_frame):
        prev_small = cv2.resize(prev_frame, (PROCESS_WIDTH, PROCESS_HEIGHT),
                                interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(curr_frame, (PROCESS_WIDTH, PROCESS_HEIGHT),
                                interpolation=cv2.INTER_AREA)

        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
        k = self.blur_ksize
        prev_gray = cv2.GaussianBlur(prev_gray, (k, k), 0)
        curr_gray = cv2.GaussianBlur(curr_gray, (k, k), 0)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            self.pyr_scale, self.levels, self.winsize,
            self.iterations, self.poly_n, self.poly_sigma,
            self.flags)
        return flow


# ══════════════════════════════════════════════════════════════════════════════
#  1bis. AprilTagFloorMasker
# ══════════════════════════════════════════════════════════════════════════════

class AprilTagFloorMasker:
    def __init__(self, dict_id=APRILTAG_DICT_ID, dilate_px=APRILTAG_MASK_DILATE_PX,
                 corner_refine=APRILTAG_CORNER_REFINE):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.params = cv2.aruco.DetectorParameters()
        self.params.cornerRefinementMethod = corner_refine
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)
        self.dilate_px = dilate_px
        self._kernel = (cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
                         if dilate_px > 0 else None)
        self.last_n_detected = 0

    def build_mask(self, frame_native_bgr, proc_width, proc_height):
        h_native, w_native = frame_native_bgr.shape[:2]
        gray = cv2.cvtColor(frame_native_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        self.last_n_detected = 0 if ids is None else len(ids)

        mask = np.zeros((proc_height, proc_width), dtype=np.uint8)
        if ids is None or len(ids) == 0:
            return mask.astype(bool)

        sx = proc_width / float(w_native)
        sy = proc_height / float(h_native)
        polys = [
            (c[0] * np.array([sx, sy], dtype=np.float32)).astype(np.int32)
            for c in corners
        ]

        cv2.fillPoly(mask, polys, 1)
        if self._kernel is not None:
            mask = cv2.dilate(mask, self._kernel)

        return mask.astype(bool)


# ══════════════════════════════════════════════════════════════════════════════
#  2. EgoMotionCompensator
# ══════════════════════════════════════════════════════════════════════════════

class EgoMotionCompensator:
    """Manages geometric ground plane masking and statistical flow filters."""

    def __init__(self):
        self._H = None
        self._H_scaled = None
        self._H_source_w = None
        self._H_source_h = None
        self._ground_mask_cache = None

        ys, xs = np.mgrid[0:PROCESS_HEIGHT, 0:PROCESS_WIDTH].astype(np.float32)
        self._pts_hom = np.stack(
            [xs.ravel(), ys.ravel(), np.ones(PROCESS_HEIGHT * PROCESS_WIDTH,
                                              dtype=np.float32)], axis=1)

    def set_homography(self, H, source_width=None, source_height=None):
        if H is None:
            self._H = None
            self._H_scaled = None
            return
        try:
            np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return

        self._H = H.astype(np.float64)
        self._H_source_w = source_width
        self._H_source_h = source_height

        if source_width is not None and source_height is not None:
            inv_sx = float(source_width)  / PROCESS_WIDTH
            inv_sy = float(source_height) / PROCESS_HEIGHT
            S_inv = np.diag([inv_sx, inv_sy, 1.0])
            H_scaled = (H @ S_inv).astype(np.float32)
        else:
            H_scaled = H.astype(np.float32)

        self._H_scaled = H_scaled
        self._ground_mask_cache = None

    def build_ground_mask(self):
        if self._ground_mask_cache is not None:
            return self._ground_mask_cache
        if self._H_scaled is None:
            return None

        warped = (self._H_scaled.astype(np.float64) @ self._pts_hom.T.astype(np.float64)).T
        wz = warped[:, 2]
        valid_w = np.abs(wz) > 1e-8
        bx = np.where(valid_w, warped[:, 0] / np.where(valid_w, wz, 1.0), -1.0)
        by = np.where(valid_w, warped[:, 1] / np.where(valid_w, wz, 1.0), -1.0)

        is_ground = (valid_w
                     & (bx >= 0) & (bx < CALIB_W)
                     & (by >= 0) & (by < CALIB_H))

        top_row_cutoff = int(PROCESS_HEIGHT * GROUND_MASK_TOP_Y_RATIO)
        row_indices = np.arange(PROCESS_HEIGHT * PROCESS_WIDTH) // PROCESS_WIDTH
        is_ground = is_ground & (row_indices >= top_row_cutoff)

        self._ground_mask_cache = is_ground.reshape(PROCESS_HEIGHT, PROCESS_WIDTH)
        return self._ground_mask_cache

    def _apply_zonal_median(self, residual):
        """Subtracts median independently in each L/C/R column."""
        h, w = residual.shape[:2]
        tier = w // 3
        out = residual.copy()
        bounds = [(0, tier), (tier, 2 * tier), (2 * tier, w)]
        for x0, x1 in bounds:
            zone = out[:, x0:x1]
            med_dx = np.median(zone[..., 0])
            med_dy = np.median(zone[..., 1])
            zone[..., 0] -= med_dx
            zone[..., 1] -= med_dy
        return out

    def compensate_ground_flow(self, flow_field, vx: float, dt: float = 0.1):
        """Calculates and subtracts the theoretical optical flow on ground plane
        induced by forward translation vx, eliminating floor tile seams and line noise."""
        if not ENABLE_GROUND_FLOW_COMPENSATION or self._H_scaled is None or abs(vx) < 0.02:
            return flow_field

        ground_mask = self.build_ground_mask()
        if ground_mask is None:
            return flow_field

        h, w = flow_field.shape[:2]
        out = flow_field.copy()

        # In forward motion vx, ground flow is downward: dy_px ≈ (fy * vx * dt) / h_cam
        try:
            # Linear depth-gradient approximation for ground flow
            y_coords = np.arange(h, dtype=np.float32)
            # Lower rows (closer to robot) experience higher downward optical flow
            flow_grad = (y_coords / float(h)) ** 2 * (abs(vx) * 15.0)
            flow_y_pred = np.zeros((h, w), dtype=np.float32)
            flow_y_pred[:] = flow_grad[:, np.newaxis]

            # Subtract ground flow specifically where ground_mask is True
            out[ground_mask, 1] -= (flow_y_pred[ground_mask] * GROUND_FLOW_REDUCTION_FACTOR)
        except Exception:
            pass

        return out

    def compensate(self, flow_brut, vx: float = 0.0, dt: float = 0.1):
        residual = flow_brut.copy()

        # Switch 1: Full-Image Egomotion Compensation (Zonal or Global Median)
        if ENABLE_EGOMOTION_COMPENSATION_ALL:
            if ENABLE_ZONAL_MEDIAN:
                residual = self._apply_zonal_median(residual)
            elif ENABLE_MEDIAN_FALLBACK:
                median_dx = np.median(residual[..., 0])
                median_dy = np.median(residual[..., 1])
                residual[..., 0] -= median_dx
                residual[..., 1] -= median_dy

        # Switch 2: Planar Ground Flow Subtraction (removes floor seams/textures)
        if ENABLE_GROUND_FLOW_COMPENSATION:
            residual = self.compensate_ground_flow(residual, vx=vx, dt=dt)

        return residual


# ══════════════════════════════════════════════════════════════════════════════
#  3. ZoneAnalyzer — Splits Residual Flow into L/C/R Zones
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ZoneMetrics:
    """Scalar metrics for an image zone."""
    mean_magnitude: float = 0.0
    mean_active_magnitude: float = 0.0
    max_magnitude: float = 0.0
    max_magnitude_filtered: float = 0.0
    mean_divergence: float = 0.0
    mean_active_divergence: float = 0.0
    mean_u: float = 0.0
    mean_v: float = 0.0
    coverage: float = 0.0
    wall_detected: bool = False


class ZoneAnalyzer:
    """Splits residual flow into zones and extracts metrics."""

    ZONE_NAMES = ('gauche', 'centre', 'droite')

    def __init__(self, zone_count=3, movement_threshold=0.5):
        self.zone_count = zone_count
        self.movement_threshold = movement_threshold

    def compute_zone_metrics(self, flow_residual):
        magnitude, _ = cv2.cartToPolar(
            flow_residual[..., 0], flow_residual[..., 1])
        h, w = magnitude.shape
        tier = w // self.zone_count
        divergence = self._compute_divergence(flow_residual)
        active_mask_full = (magnitude > self.movement_threshold).astype(np.uint8)

        global_active = np.count_nonzero(active_mask_full)
        global_coverage = float(global_active) / magnitude.size if magnitude.size else 0.0

        zone_metrics = {}
        for idx, name in enumerate(self.ZONE_NAMES[:self.zone_count]):
            x_start = idx * tier
            x_end = (idx + 1) * tier if idx < self.zone_count - 1 else w
            mag_zone = magnitude[:, x_start:x_end]
            div_zone = divergence[:, x_start:x_end]
            mask_zone = active_mask_full[:, x_start:x_end]
            u_zone = flow_residual[..., 0][:, x_start:x_end]
            v_zone = flow_residual[..., 1][:, x_start:x_end]

            active_mask = mag_zone > self.movement_threshold
            n_active = np.count_nonzero(active_mask)
            mean_active = float(np.mean(mag_zone[active_mask])) if n_active > 0 else 0.0

            mean_active_div = float(np.mean(div_zone[active_mask])) if n_active > 0 else 0.0
            mean_u = float(np.mean(u_zone[active_mask])) if n_active > 0 else 0.0
            mean_v = float(np.mean(v_zone[active_mask])) if n_active > 0 else 0.0

            # Filtered spike via connected components
            max_filtered = 0.0
            if n_active > 0:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    mask_zone, connectivity=8)
                for label_id in range(1, num_labels):
                    if stats[label_id, cv2.CC_STAT_AREA] < MIN_SPIKE_BLOB_AREA:
                        continue
                    blob_max = float(np.max(mag_zone[labels == label_id]))
                    if blob_max > max_filtered:
                        max_filtered = blob_max

            coverage = (float(n_active / mag_zone.size) if mag_zone.size else 0.0)

            wall_detected = False
            if ENABLE_WALL_DEAD_ZONE_DETECTION:
                inactive_ratio = 1.0 - coverage
                basic_wall = (inactive_ratio > WALL_DEAD_ZONE_RATIO
                               and global_coverage > WALL_MIN_ACTIVE_BORDER)

                if basic_wall and WALL_REQUIRE_TOP_INACTIVITY:
                    top_rows = max(1, int(mag_zone.shape[0] * WALL_TOP_RATIO))
                    top_zone = mag_zone[:top_rows, :]
                    top_active = float(np.count_nonzero(top_zone > self.movement_threshold))
                    top_coverage = top_active / max(top_zone.size, 1)

                    bot_rows = max(1, int(mag_zone.shape[0] * WALL_BOTTOM_RATIO))
                    bottom_zone = mag_zone[-bot_rows:, :]
                    bottom_active = float(np.count_nonzero(bottom_zone > self.movement_threshold))
                    bottom_coverage = bottom_active / max(bottom_zone.size, 1)

                    wall_detected = (top_coverage <= WALL_TOP_MAX_COVERAGE
                                      and bottom_coverage <= WALL_BOTTOM_MAX_COVERAGE)
                else:
                    wall_detected = basic_wall

            zone_metrics[name] = ZoneMetrics(
                mean_magnitude=float(np.mean(mag_zone)) if mag_zone.size else 0.0,
                mean_active_magnitude=mean_active,
                max_magnitude=float(np.max(mag_zone)) if mag_zone.size else 0.0,
                max_magnitude_filtered=max_filtered,
                mean_divergence=float(np.mean(div_zone)) if div_zone.size else 0.0,
                mean_active_divergence=mean_active_div,
                mean_u=mean_u,
                mean_v=mean_v,
                coverage=coverage,
                wall_detected=wall_detected
            )
        return zone_metrics

    @staticmethod
    def _compute_divergence(flow):
        u = flow[..., 0]
        v = flow[..., 1]
        _, du_dx = np.gradient(u)
        dv_dy, _ = np.gradient(v)
        return du_dx + dv_dy


# ══════════════════════════════════════════════════════════════════════════════
#  3b. UnifiedBlobDetector
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BlobInfo:
    """Information on a detected optical flow blob."""
    blob_id: int
    cx: float
    cy: float
    area: int
    bottom_x: float
    bottom_y: float
    mean_mag: float
    mean_u: float
    mean_v: float
    coherence: float = 0.0
    divergence: float = 0.0


class UnifiedBlobDetector:
    """Performs unified connected component detection on residual optical flow."""

    def __init__(self, movement_threshold=MOVEMENT_THRESHOLD,
                 min_area=TRAJ_MIN_BLOB_AREA):
        self.movement_threshold = movement_threshold
        self.min_area = min_area

    def detect(self, flow_residual):
        mag = np.sqrt(flow_residual[..., 0]**2 + flow_residual[..., 1]**2)
        mask = (mag > self.movement_threshold).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)

        u = flow_residual[..., 0]
        v = flow_residual[..., 1]
        _, du_dx = np.gradient(u)
        dv_dy, _ = np.gradient(v)
        divergence_field = du_dx + dv_dy

        blobs = []
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < self.min_area:
                continue
            blob_mask = (labels == label_id)
            cx, cy = float(centroids[label_id][0]), float(centroids[label_id][1])

            # Robust ground-contact extraction using 95th vertical percentile (rejects floor noise)
            blob_indices = np.argwhere(blob_mask)  # array of [y, x]
            if len(blob_indices) == 0:
                bottom_y, bottom_x = cy, cx
            else:
                # 95th percentile along Y axis (avoids single isolated bottom pixel)
                bottom_y = float(np.percentile(blob_indices[:, 0], 95.0))
                near_bottom = blob_indices[np.abs(blob_indices[:, 0] - bottom_y) <= 3.0]
                if len(near_bottom) > 0:
                    bottom_x = float(np.median(near_bottom[:, 1]))
                else:
                    bottom_x = cx

            mean_mag = float(np.mean(mag[blob_mask]))
            mean_u = float(np.mean(flow_residual[..., 0][blob_mask]))
            mean_v = float(np.mean(flow_residual[..., 1][blob_mask]))
            blob_divergence = float(np.mean(divergence_field[blob_mask]))

            vector_mag = float(np.hypot(mean_u, mean_v))
            coherence = vector_mag / max(mean_mag, 1e-6)
            is_coherent = coherence >= BLOB_COHERENCE_MIN
            is_expanding = abs(blob_divergence) >= BLOB_DIVERGENCE_MIN

            if not is_coherent and not is_expanding:
                continue

            if BLOB_VERTICAL_MOTION_REJECT:
                relative_y = cy / max(flow_residual.shape[0], 1)
                in_bottom_region = relative_y > (1.0 - BLOB_VERTICAL_BOTTOM_RATIO)
                if (in_bottom_region
                        and mean_v > 0.5
                        and abs(mean_u) < BLOB_VERTICAL_RATIO * abs(mean_v)):
                    continue

            blobs.append(BlobInfo(
                blob_id=label_id, cx=cx, cy=cy, area=int(area),
                bottom_x=bottom_x, bottom_y=bottom_y,
                mean_mag=mean_mag, mean_u=mean_u, mean_v=mean_v,
                coherence=coherence, divergence=blob_divergence))
        return blobs


# ══════════════════════════════════════════════════════════════════════════════
#  4. TrajectoryTracker
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ObstacleTrack:
    track_id: int = 0
    positions: list = field(default_factory=list)
    area: float = 0.0
    last_seen: float = 0.0
    age: int = 0
    mean_u: float = 0.0
    mean_v: float = 0.0


@dataclass
class TrajectoryPrediction:
    predicted_zone: str = None
    ttc_to_center_s: float = None
    vx_px_s: float = 0.0
    origin_side: str = None
    confidence: str = 'none'
    track_age: int = 0
    decision: str = 'CLEAR'       # 'CLEAR' | 'AVOID' | 'EMERGENCY'
    escape_side: str = None
    escape_blocked: bool = False


class TrajectoryTracker:
    """Tracks optical flow blobs and computes predictive TTC to robot corridor."""

    def __init__(self,
                 history_len=TRAJ_HISTORY_LEN,
                 min_track_age=TRAJ_MIN_TRACK_AGE,
                 match_max_dist=TRAJ_MATCH_MAX_DIST,
                 horizon_s=TRAJ_HORIZON_S,
                 maneuver_time_s=REQUIRED_MANEUVER_TIME_S,
                 max_track_staleness_s=1.0):
        self.history_len = history_len
        self.min_track_age = min_track_age
        self.match_max_dist = match_max_dist
        self.horizon_s = horizon_s
        self.maneuver_time_s = maneuver_time_s
        self.max_track_staleness_s = max_track_staleness_s
        self._tracks = {}
        self._next_id = 0

    def update(self, blobs, frame_width, timestamp):
        detections = [(b.cx, b.cy, b.area, b.mean_u, b.mean_v)
                      for b in blobs]

        self._associate_and_update(detections, timestamp)
        self._prune_stale(timestamp)

        return self._best_prediction(frame_width)

    def _associate_and_update(self, detections, timestamp):
        unmatched = list(range(len(detections)))

        for track in self._tracks.values():
            if not track.positions:
                continue
            _, last_x, last_y = track.positions[-1]
            best_idx, best_dist = None, self.match_max_dist
            for idx in unmatched:
                cx, cy, _area, _mu, _mv = detections[idx]
                dist = float(np.hypot(cx - last_x, cy - last_y))
                if dist < best_dist:
                    best_dist, best_idx = dist, idx
            if best_idx is not None:
                cx, cy, area, mu, mv = detections[best_idx]
                track.positions.append((timestamp, cx, cy))
                if len(track.positions) > self.history_len:
                    track.positions.pop(0)
                track.area = area
                track.mean_u = mu
                track.mean_v = mv
                track.last_seen = timestamp
                track.age += 1
                unmatched.remove(best_idx)

        for idx in unmatched:
            cx, cy, area, mu, mv = detections[idx]
            track = ObstacleTrack(track_id=self._next_id)
            track.positions.append((timestamp, cx, cy))
            track.area = area
            track.mean_u = mu
            track.mean_v = mv
            track.last_seen = timestamp
            track.age = 1
            self._tracks[self._next_id] = track
            self._next_id += 1

    def _prune_stale(self, timestamp):
        stale_ids = [tid for tid, t in self._tracks.items()
                     if timestamp - t.last_seen > self.max_track_staleness_s]
        for tid in stale_ids:
            del self._tracks[tid]

    def _best_prediction(self, frame_width):
        best = None
        best_urgency = -1.0
        tier = frame_width / 3.0
        center_lo, center_hi = tier, 2 * tier

        mature_tracks = [t for t in self._tracks.values()
                         if t.age >= self.min_track_age and len(t.positions) >= 2]

        for track in mature_tracks:
            t0, x0, _ = track.positions[0]
            t1, x1, _ = track.positions[-1]
            dt = t1 - t0
            if dt <= 0:
                continue
            vx = (x1 - x0) / dt

            if abs(vx) < 60.0:
                continue
            if abs(x1 - x0) < 30.0:
                continue
            if track.area < 400:
                continue

            ttc = None
            if center_lo <= x1 <= center_hi:
                ttc = 0.0
            elif x1 < center_lo and vx > 0:
                ttc = (center_lo - x1) / vx
            elif x1 > center_hi and vx < 0:
                ttc = (x1 - center_hi) / (-vx)

            if ttc is None:
                continue

            pred_x = x1 + vx * self.horizon_s
            if pred_x < center_lo:
                pred_zone = 'gauche'
            elif pred_x < center_hi:
                pred_zone = 'centre'
            else:
                pred_zone = 'droite'

            confidence = 'high' if track.age >= self.history_len else 'low'
            origin_side = 'gauche' if x1 < (frame_width / 2.0) else 'droite'
            escape_side = 'left' if x1 > (frame_width / 2.0) else 'right'

            if ttc > self.maneuver_time_s:
                decision = 'CLEAR'
            elif ttc > TRAJ_EMERGENCY_TTC_MARGIN_S:
                decision = 'AVOID'
            else:
                decision = 'EMERGENCY'

            escape_blocked = self._is_escape_blocked(
                track, escape_side, mature_tracks, frame_width)
            if escape_blocked and decision == 'AVOID':
                decision = 'EMERGENCY'

            urgency = (1.0 / max(ttc, 0.01))
            if urgency > best_urgency:
                best_urgency = urgency
                best = TrajectoryPrediction(
                    predicted_zone=pred_zone,
                    ttc_to_center_s=ttc,
                    vx_px_s=vx,
                    origin_side=origin_side,
                    confidence=confidence,
                    track_age=track.age,
                    decision=decision,
                    escape_side=escape_side,
                    escape_blocked=escape_blocked,
                )

        return best if best is not None else TrajectoryPrediction()

    def _is_escape_blocked(self, current_track, escape_side, mature_tracks,
                           frame_width):
        half = frame_width / 2.0
        for other in mature_tracks:
            if other.track_id == current_track.track_id:
                continue
            if other.age < self.min_track_age or len(other.positions) < 2:
                continue
            _, ox, _ = other.positions[-1]
            if escape_side == 'left' and ox < half:
                t0o, x0o, _ = other.positions[0]
                t1o, x1o, _ = other.positions[-1]
                dto = t1o - t0o
                if dto > 0:
                    vxo = (x1o - x0o) / dto
                    if vxo > 1e-3 and other.area > TRAJ_MIN_BLOB_AREA * 2:
                        return True
            elif escape_side == 'right' and ox > half:
                t0o, x0o, _ = other.positions[0]
                t1o, x1o, _ = other.positions[-1]
                dto = t1o - t0o
                if dto > 0:
                    vxo = (x1o - x0o) / dto
                    if vxo < -1e-3 and other.area > TRAJ_MIN_BLOB_AREA * 2:
                        return True
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  5. SmoothingFilter + StreakDebouncer
# ══════════════════════════════════════════════════════════════════════════════

class SmoothingFilter:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self._value = None

    @property
    def value(self):
        return self._value if self._value is not None else 0.0

    def smooth(self, raw):
        if self._value is None:
            self._value = raw
        else:
            self._value = self.alpha * raw + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self, value=None):
        self._value = value


class StreakDebouncer:
    def __init__(self, threshold, streak_required=FLOW_SPIKE_STREAK_REQUIRED):
        self.threshold = threshold
        self.streak_required = streak_required
        self._streak = 0

    @property
    def streak(self):
        return self._streak

    def update(self, value):
        if value >= self.threshold:
            self._streak += 1
        else:
            self._streak = 0
        return self._streak >= self.streak_required

    def reset(self):
        self._streak = 0


# ══════════════════════════════════════════════════════════════════════════════
#  6. CollisionRiskEstimator
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RiskAssessment:
    risk_per_zone: dict = field(default_factory=dict)
    flow_danger: bool = False
    flow_warning: bool = False
    flow_diff: float = 0.0
    diff_avoid_active: bool = False
    spike_left: bool = False
    spike_right: bool = False
    smoothed_left: float = 0.0
    smoothed_center: float = 0.0
    smoothed_right: float = 0.0
    warning_flow_ratio: float = 1.0
    predicted_zone: str = None
    predicted_ttc: float = None
    predicted_vx: float = 0.0
    predicted_origin_side: str = None
    predicted_urgent: bool = False
    traj_decision: str = 'CLEAR'
    traj_escape_side: str = None
    traj_escape_blocked: bool = False
    raw_center_emergency: bool = False
    left_blocked: bool = False
    right_blocked: bool = False


@dataclass
class ZoneRisk:
    category: str = 'CLEAR'
    risk_score: float = 0.0
    mean_magnitude_smoothed: float = 0.0
    max_magnitude: float = 0.0
    spike_active: bool = False
    spike_streak: int = 0
    wall_active: bool = False
    coverage: float = 0.0
    mean_divergence: float = 0.0
    mean_active_divergence: float = 0.0
    mean_u: float = 0.0
    mean_v: float = 0.0
    is_moving_away: bool = False
    is_crossing: bool = False


class CollisionRiskEstimator:
    def __init__(self,
                 danger_threshold=FLOW_CENTER_DANGER,
                 warning_threshold=FLOW_CENTER_WARN,
                 spike_threshold=FLOW_SPIKE_THRESHOLD,
                 spike_streak_required=FLOW_SPIKE_STREAK_REQUIRED,
                 diff_avoid_threshold=FLOW_DIFF_AVOID,
                 smooth_alpha=FLOW_SMOOTH_ALPHA):
        self.danger_threshold_base = danger_threshold
        self.warning_threshold_base = warning_threshold
        self.spike_threshold_base = spike_threshold
        self.diff_avoid_threshold_base = diff_avoid_threshold

        self.danger_threshold = danger_threshold
        self.warning_threshold = warning_threshold
        self.spike_threshold = spike_threshold
        self.diff_avoid_threshold = diff_avoid_threshold

        self._spike_streak_required = spike_streak_required

        self._smoothers = {
            'gauche': SmoothingFilter(alpha=smooth_alpha),
            'centre': SmoothingFilter(alpha=smooth_alpha),
            'droite': SmoothingFilter(alpha=smooth_alpha),
        }

        self._spike_debouncers = {
            'gauche': StreakDebouncer(spike_threshold, spike_streak_required),
            'centre': StreakDebouncer(spike_threshold, spike_streak_required),
            'droite': StreakDebouncer(spike_threshold, spike_streak_required),
        }

        self._wall_debouncers = {
            'gauche': StreakDebouncer(0.5, WALL_STREAK_REQUIRED),
            'centre': StreakDebouncer(0.5, WALL_STREAK_REQUIRED),
            'droite': StreakDebouncer(0.5, WALL_STREAK_REQUIRED),
        }

    def _update_dynamic_thresholds(self, robot_speed):
        speed = abs(robot_speed)
        self.danger_threshold = (self.danger_threshold_base
                                 + speed * SPEED_THRESH_GAIN_CENTER)
        self.warning_threshold = (self.warning_threshold_base
                                  + speed * SPEED_THRESH_GAIN_CENTER)
        self.spike_threshold = (self.spike_threshold_base
                                + speed * SPEED_THRESH_GAIN_SPIKE)
        self.diff_avoid_threshold = (self.diff_avoid_threshold_base
                                     + speed * SPEED_THRESH_GAIN_DIFF)

        for debouncer in self._spike_debouncers.values():
            debouncer.threshold = self.spike_threshold

    def update(self, zone_metrics, dt=0.0, trajectory=None, robot_speed=0.0):
        self._update_dynamic_thresholds(robot_speed)

        risk_per_zone = {}

        for zone_name in ('gauche', 'centre', 'droite'):
            metrics = zone_metrics.get(zone_name)
            if metrics is None:
                risk_per_zone[zone_name] = ZoneRisk()
                continue

            smoothed_mean = self._smoothers[zone_name].smooth(metrics.mean_active_magnitude)
            spike_active = self._spike_debouncers[zone_name].update(metrics.max_magnitude_filtered)

            wall_active = False
            if hasattr(self, '_wall_debouncers'):
                wall_active = self._wall_debouncers[zone_name].update(1.0 if metrics.wall_detected else 0.0)

            risk_score = smoothed_mean + 0.5 * metrics.max_magnitude_filtered

            is_moving_away = metrics.mean_active_divergence < -0.015
            is_crossing = abs(metrics.mean_u) > max(abs(metrics.mean_v) * 1.5, 0.8)

            if (smoothed_mean >= self.danger_threshold or spike_active or wall_active):
                category = 'EMERGENCY'
            elif smoothed_mean >= self.warning_threshold:
                category = 'WARNING'
            else:
                category = 'CLEAR'

            risk_per_zone[zone_name] = ZoneRisk(
                category=category,
                risk_score=risk_score,
                mean_magnitude_smoothed=smoothed_mean,
                max_magnitude=metrics.max_magnitude_filtered,
                spike_active=spike_active,
                spike_streak=self._spike_debouncers[zone_name].streak,
                wall_active=wall_active,
                coverage=metrics.coverage,
                mean_divergence=metrics.mean_divergence,
                mean_active_divergence=metrics.mean_active_divergence,
                mean_u=metrics.mean_u,
                mean_v=metrics.mean_v,
                is_moving_away=is_moving_away,
                is_crossing=is_crossing,
            )

        centre = risk_per_zone.get('centre', ZoneRisk())
        smoothed_c = centre.mean_magnitude_smoothed
        spike_c = centre.spike_active

        traj_decision = trajectory.decision if trajectory else 'CLEAR'
        traj_escape_side = trajectory.escape_side if trajectory else None
        traj_escape_blocked = trajectory.escape_blocked if trajectory else False

        smoothed_l = risk_per_zone.get('gauche', ZoneRisk()).mean_magnitude_smoothed
        smoothed_r = risk_per_zone.get('droite', ZoneRisk()).mean_magnitude_smoothed

        spike_left = risk_per_zone.get('gauche', ZoneRisk()).spike_active or risk_per_zone.get('gauche', ZoneRisk()).wall_active
        spike_right = risk_per_zone.get('droite', ZoneRisk()).spike_active or risk_per_zone.get('droite', ZoneRisk()).wall_active

        left_blocked = (smoothed_l >= self.danger_threshold) or spike_left
        right_blocked = (smoothed_r >= self.danger_threshold) or spike_right

        raw_center_emergency = (smoothed_c >= self.danger_threshold) or spike_c or centre.wall_active

        if traj_decision == 'EMERGENCY':
            flow_danger = True
            flow_warning = True
        elif traj_decision == 'AVOID':
            flow_danger = False
            flow_warning = True
        elif raw_center_emergency:
            if centre.is_moving_away or centre.is_crossing:
                flow_danger = False
            else:
                if left_blocked and right_blocked:
                    flow_danger = True
                else:
                    flow_danger = False
            flow_warning = True
        else:
            flow_danger = False
            flow_warning = (smoothed_c >= self.warning_threshold) or spike_c or centre.wall_active

        flow_diff = smoothed_r - smoothed_l
        diff_avoid_active = abs(flow_diff) >= self.diff_avoid_threshold

        if flow_warning and not flow_danger:
            warning_flow_ratio = max(0.0, min(1.0,
                1.0 - (smoothed_c - self.warning_threshold)
                / (self.danger_threshold - self.warning_threshold)))
        else:
            warning_flow_ratio = 0.0 if flow_danger else 1.0

        predicted_zone = trajectory.predicted_zone if trajectory else None
        predicted_ttc = trajectory.ttc_to_center_s if trajectory else None
        predicted_vx = trajectory.vx_px_s if trajectory else 0.0
        predicted_origin_side = trajectory.origin_side if trajectory else None
        predicted_urgent = bool(
            trajectory is not None
            and trajectory.decision == 'EMERGENCY'
            and trajectory.confidence != 'none'
        )

        return RiskAssessment(
            risk_per_zone=risk_per_zone,
            flow_danger=flow_danger,
            flow_warning=flow_warning,
            flow_diff=flow_diff,
            diff_avoid_active=diff_avoid_active,
            spike_left=spike_left,
            spike_right=spike_right,
            smoothed_left=smoothed_l,
            smoothed_center=smoothed_c,
            smoothed_right=smoothed_r,
            warning_flow_ratio=warning_flow_ratio,
            predicted_zone=predicted_zone,
            predicted_ttc=predicted_ttc,
            predicted_vx=predicted_vx,
            predicted_origin_side=predicted_origin_side,
            predicted_urgent=predicted_urgent,
            traj_decision=traj_decision,
            traj_escape_side=traj_escape_side,
            traj_escape_blocked=traj_escape_blocked,
            raw_center_emergency=raw_center_emergency,
            left_blocked=left_blocked,
            right_blocked=right_blocked,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  7. RobotState + StateMachine
# ══════════════════════════════════════════════════════════════════════════════

class RobotState:
    CLEAR      = 'CLEAR'
    PREDICTING = 'PREDICTING'
    RECOVERY   = 'RECOVERY'
    WARNING    = 'WARNING'
    EMERGENCY  = 'EMERGENCY'


@dataclass
class StateMachineInputs:
    danger: bool = False
    warning: bool = False
    warning_flow_ratio: float = 1.0
    user_moving_forward: bool = False
    user_reversing: bool = False
    user_pure_rotation: bool = False
    user_turning: bool = False
    current_yaw: float = 0.0
    avoid_cmd_active: bool = False
    is_recovery: bool = False


@dataclass
class StateMachineOutputs:
    state: str = RobotState.CLEAR
    stop_active: bool = False
    smooth_ratio: float = 1.0
    should_publish_estop: bool = False
    estop_value: bool = False
    engage_reason: str = ''


class StateMachine:
    def __init__(self,
                 stop_release_cycles=STOP_RELEASE_CYCLES,
                 ema_alpha=EMA_ALPHA,
                 max_yaw_deviation=MAX_AVOIDANCE_YAW_DEVIATION):
        self.stop_release_cycles = stop_release_cycles
        self.max_yaw_deviation = max_yaw_deviation

        self._state = RobotState.CLEAR
        self._stop_active = False
        self._clean_cycle_count = 0

        self._ratio_smoother = SmoothingFilter(alpha=ema_alpha)
        self._ratio_smoother.reset(1.0)

        self._avoid_cum_yaw = 0.0
        self._avoid_prev_yaw = 0.0
        self._avoid_was_active = False

        self._target_yaw = 0.0
        self._current_yaw = 0.0

    @property
    def state(self):
        return self._state

    @property
    def target_yaw(self):
        return self._target_yaw

    @property
    def yaw_error(self):
        return float(np.arctan2(
            np.sin(self._target_yaw - self._current_yaw),
            np.cos(self._target_yaw - self._current_yaw)))

    @property
    def stuck_exceeded(self):
        return abs(self._avoid_cum_yaw) >= self.max_yaw_deviation

    def update(self, inputs):
        self._current_yaw = inputs.current_yaw

        if self._avoid_was_active:
            raw_delta = inputs.current_yaw - self._avoid_prev_yaw
            wrapped_delta = float(np.arctan2(np.sin(raw_delta), np.cos(raw_delta)))
            self._avoid_cum_yaw += wrapped_delta
        else:
            self._avoid_cum_yaw = 0.0
        self._avoid_prev_yaw = inputs.current_yaw

        danger = inputs.danger or self.stuck_exceeded
        warning = inputs.warning

        if (inputs.user_turning
                or not inputs.user_moving_forward
                or danger):
            self._target_yaw = inputs.current_yaw

        if danger:
            raw_ratio = 0.0
        elif warning:
            raw_ratio = inputs.warning_flow_ratio
        else:
            raw_ratio = 1.0
        smooth_ratio = self._ratio_smoother.smooth(raw_ratio)
        smooth_ratio = max(0.0, min(1.0, smooth_ratio))

        engage_reason = ''
        should_publish_estop = False
        estop_value = False

        bypass_emergency = inputs.user_reversing or inputs.user_pure_rotation

        if danger and not bypass_emergency:
            self._state = RobotState.EMERGENCY
            self._stop_active = True
            self._clean_cycle_count = 0

            reasons = []
            if inputs.danger:
                reasons.append('flow_danger')
            if self.stuck_exceeded:
                reasons.append(
                    f'spin_lock={np.degrees(self._avoid_cum_yaw):.0f}°')
            engage_reason = ' '.join(reasons)
            should_publish_estop = True
            estop_value = True

        else:
            if self._stop_active:
                if bypass_emergency:
                    self._stop_active = False
                    self._clean_cycle_count = 0
                    should_publish_estop = True
                    estop_value = False
                else:
                    self._clean_cycle_count += 1
                    should_publish_estop = True
                    estop_value = True
                    engage_reason = 'holding — waiting clear cycles'
                    if self._clean_cycle_count >= self.stop_release_cycles:
                        self._stop_active = False
                        self._clean_cycle_count = 0
                        should_publish_estop = True
                        estop_value = False
            else:
                should_publish_estop = True
                estop_value = False

            if bypass_emergency:
                self._state = RobotState.CLEAR
            elif inputs.is_recovery:
                self._state = RobotState.RECOVERY
            elif inputs.avoid_cmd_active:
                self._state = RobotState.PREDICTING
            elif warning:
                self._state = RobotState.WARNING
            else:
                self._state = RobotState.CLEAR

        self._avoid_was_active = inputs.avoid_cmd_active

        return StateMachineOutputs(
            state=self._state,
            stop_active=self._stop_active,
            smooth_ratio=smooth_ratio,
            should_publish_estop=should_publish_estop,
            estop_value=estop_value,
            engage_reason=engage_reason,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  8. SideSelector
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AvoidanceResult:
    cmd: Twist = field(default_factory=Twist)
    side: str = 'left'
    is_recovery: bool = False
    source: str = ''


class SideSelector:
    def __init__(self,
                 streak_threshold=SIDE_FLIP_STREAK,
                 balance_k=FLOW_BALANCE_K,
                 max_turn_rad=FLOW_MAX_TURN_RAD,
                 avoid_linear_x=AVOID_LINEAR_X,
                 avoid_angular_z=AVOID_ANGULAR_Z,
                 avoid_angular_max=AVOID_ANGULAR_MAX,
                 diff_avoid_threshold=FLOW_DIFF_AVOID,
                 smooth_alpha=AVOID_SMOOTH_ALPHA,
                 smooth_alpha_urgent=AVOID_SMOOTH_ALPHA_URGENT,
                 yaw_recovery_kp=YAW_RECOVERY_KP,
                 yaw_recovery_min=YAW_RECOVERY_MIN,
                 recovery_linear_x=RECOVERY_LINEAR_X,
                 recovery_max_turn=RECOVERY_MAX_TURN):
        self.streak_threshold = streak_threshold
        self.balance_k = balance_k
        self.max_turn_rad = max_turn_rad
        self.avoid_linear_x = avoid_linear_x
        self.avoid_angular_z = avoid_angular_z
        self.avoid_angular_max = avoid_angular_max
        self.diff_avoid_threshold = diff_avoid_threshold
        self.smooth_alpha = smooth_alpha
        self.smooth_alpha_urgent = smooth_alpha_urgent
        self.yaw_recovery_kp = yaw_recovery_kp
        self.yaw_recovery_min = yaw_recovery_min
        self.recovery_linear_x = recovery_linear_x
        self.recovery_max_turn = recovery_max_turn

        self._last_side = None
        self._side_streak = 0

        self._linear_smoother = SmoothingFilter(alpha=smooth_alpha)
        self._angular_smoother = SmoothingFilter(alpha=smooth_alpha)

    @staticmethod
    def _zone_severity(risk, name):
        category = risk.risk_per_zone.get(name, ZoneRisk()).category
        if category == 'EMERGENCY':
            return 2
        elif category == 'WARNING':
            return 1
        return 0

    def _zone_occupancy_avoidance(self, risk, speed_factor):
        severities = {
            name: self._zone_severity(risk, name)
            for name in ('gauche', 'centre', 'droite')
        }
        blocked = {name: (sev > 0) for name, sev in severities.items()}
        n_blocked = sum(blocked.values())
        if n_blocked == 0 or n_blocked == 3:
            return None

        if n_blocked == 1:
            blocked_zone = next(name for name, b in blocked.items() if b)
            max_severity = severities[blocked_zone]
            if blocked_zone == 'gauche':
                turn_side = 'right'
            elif blocked_zone == 'droite':
                turn_side = 'left'
            else:
                sl = risk.risk_per_zone.get('gauche', ZoneRisk()).mean_magnitude_smoothed
                sr = risk.risk_per_zone.get('droite', ZoneRisk()).mean_magnitude_smoothed
                turn_side = 'left' if sl <= sr else 'right'
        else:  # n_blocked == 2
            free_zone = next(name for name, b in blocked.items() if not b)
            max_severity = max(sev for name, sev in severities.items() if name != free_zone)
            if free_zone == 'gauche':
                turn_side = 'left'
            elif free_zone == 'droite':
                turn_side = 'right'
            else:
                return None

        turn_factor = (ZONE_OCCUPANCY_EMERGENCY_TURN_FACTOR if max_severity >= 2
                       else ZONE_OCCUPANCY_WARNING_TURN_FACTOR)

        cmd = Twist()
        cmd.linear.x = self.avoid_linear_x
        turn_mag = self.max_turn_rad * turn_factor * speed_factor
        cmd.angular.z = turn_mag if turn_side == 'left' else -turn_mag

        side = self._apply_hysteresis(turn_side)
        if side == 'left' and cmd.angular.z < 0:
            cmd.angular.z = -cmd.angular.z
        elif side == 'right' and cmd.angular.z > 0:
            cmd.angular.z = -cmd.angular.z

        source = 'ZONE_OCCUPANCY' if max_severity >= 2 else 'ZONE_OCCUPANCY_WARN'
        return AvoidanceResult(cmd=cmd, side=side, source=source)

    def compute_avoidance(self, risk, warning, yaw_error,
                          user_cmd_z, user_moving_forward, robot_vx=0.0):
        if not user_moving_forward:
            self._reset_smoothing()
            return None

        robot_speed = abs(robot_vx)
        speed_factor = float(np.clip(
            (robot_speed - SPEED_ADAPT_MIN) / max(SPEED_ADAPT_MAX - SPEED_ADAPT_MIN, 1e-6),
            SPEED_ADAPT_FLOOR, 1.0
        ))

        avoid_result = self._zone_occupancy_avoidance(risk, speed_factor)

        if avoid_result is None and (risk.diff_avoid_active or risk.spike_left or risk.spike_right):
            cmd = Twist()
            cmd.linear.x = self.avoid_linear_x

            if risk.diff_avoid_active:
                diff_ratio = abs(risk.flow_diff) / max(self.diff_avoid_threshold, 1e-6)
                adaptive_k = self.balance_k * min(diff_ratio, 3.0)
                raw_angular = risk.flow_diff * adaptive_k
                cmd.angular.z = max(-self.max_turn_rad,
                                    min(self.max_turn_rad, raw_angular))
                source = 'FLOW_DIFF'

                if diff_ratio > 2.0:
                    cmd.linear.x *= 0.5
            else:
                if risk.spike_left and not risk.spike_right:
                    cmd.angular.z = -self.max_turn_rad * 0.8
                elif risk.spike_right and not risk.spike_left:
                    cmd.angular.z = +self.max_turn_rad * 0.8
                else:
                    cmd.angular.z = (self.max_turn_rad * 0.8
                                     if risk.flow_diff >= 0
                                     else -self.max_turn_rad * 0.8)
                source = 'FLOW_SPIKE'

            cmd.angular.z *= speed_factor

            desired_side = 'left' if cmd.angular.z > 0 else 'right'
            side = self._apply_hysteresis(desired_side)

            if side == 'left' and cmd.angular.z < 0:
                cmd.angular.z = -cmd.angular.z
            elif side == 'right' and cmd.angular.z > 0:
                cmd.angular.z = -cmd.angular.z

            avoid_result = AvoidanceResult(cmd=cmd, side=side, source=source)

        elif avoid_result is None and risk.predicted_urgent:
            cmd = Twist()
            cmd.linear.x = self.avoid_linear_x
            turn_mag = self.max_turn_rad * 0.5 * speed_factor

            if risk.predicted_origin_side == 'droite':
                cmd.angular.z = +turn_mag
            else:
                cmd.angular.z = -turn_mag

            desired_side = 'left' if cmd.angular.z > 0 else 'right'
            side = self._apply_hysteresis(desired_side)
            if side == 'left' and cmd.angular.z < 0:
                cmd.angular.z = -cmd.angular.z
            elif side == 'right' and cmd.angular.z > 0:
                cmd.angular.z = -cmd.angular.z

            avoid_result = AvoidanceResult(
                cmd=cmd, side=side, source='TRAJECTORY_PREDICT')

        elif (avoid_result is None
              and abs(user_cmd_z) < 0.05
              and not warning
              and abs(yaw_error) > self.yaw_recovery_min):
            correction = float(np.clip(
                yaw_error * self.yaw_recovery_kp,
                -self.recovery_max_turn, self.recovery_max_turn))
            cmd = Twist()
            cmd.linear.x = self.recovery_linear_x
            cmd.angular.z = correction

            side = 'left' if correction > 0 else 'right'
            avoid_result = AvoidanceResult(
                cmd=cmd, side=side, is_recovery=True, source='RECOVERY')

        if avoid_result is not None:
            alpha = self.smooth_alpha
            self._linear_smoother.alpha = alpha
            self._angular_smoother.alpha = alpha

            avoid_result.cmd.linear.x = self._linear_smoother.smooth(
                avoid_result.cmd.linear.x)
            avoid_result.cmd.angular.z = self._angular_smoother.smooth(
                avoid_result.cmd.angular.z)
        else:
            self._reset_smoothing()
            self._last_side = None
            self._side_streak = 0

        return avoid_result

    def _apply_hysteresis(self, desired_side):
        if self._last_side is None or desired_side == self._last_side:
            self._last_side = desired_side
            self._side_streak = 0
            return desired_side

        self._side_streak += 1
        if self._side_streak >= self.streak_threshold:
            self._last_side = desired_side
            self._side_streak = 0
            return desired_side

        return self._last_side

    def _reset_smoothing(self):
        self._linear_smoother.reset(0.0)
        self._angular_smoother.reset(0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  9. FarnebackAvoidanceNode
# ══════════════════════════════════════════════════════════════════════════════

class FarnebackAvoidanceNode(Node):

    def __init__(self):
        super().__init__('farneback_avoidance_node')

        self.flow_estimator = OpticalFlowEstimator()
        self.ego_compensator = EgoMotionCompensator()
        self.zone_analyzer = ZoneAnalyzer()
        self.blob_detector = UnifiedBlobDetector()
        self.trajectory_tracker = TrajectoryTracker()
        self.risk_estimator = CollisionRiskEstimator()
        self.state_machine = StateMachine()
        self.side_selector = SideSelector()

        self.distance_estimator = HomographyDistanceEstimator()
        if self.distance_estimator.load_from_file():
            self.get_logger().info('Homography loaded from file \u2713')
            self.ego_compensator.set_homography(
                self.distance_estimator.H)
        else:
            self.get_logger().warn(
                f'Homography file not found ({H_SAVE_PATH}) '
                '\u2014 waiting for /homography_matrix topic')

        self.bridge = CvBridge()
        self._latest_frame = None
        self._prev_frame = None

        self.apriltag_masker = AprilTagFloorMasker() if ENABLE_APRILTAG_MASKING else None
        self._last_apriltag_log_time = 0.0

        self._last_system_time = time.monotonic()
        self._latest_msg_time = None
        self._last_frame_msg_time = None

        self._last_obstacles = []

        self._cam_source_w = None
        self._cam_source_h = None

        self._robot_vx = 0.0
        self._robot_vy = 0.0
        self._robot_wz = 0.0
        self._robot_speed_stamp = 0.0
        self._current_yaw = 0.0

        self._user_cmd_x = 0.0
        self._user_cmd_z = 0.0
        self._user_cmd_stamp = 0.0

        self._intent_x = 0.0
        self._intent_stamp = 0.0
        self._last_cmd_in_stamp = 0.0

        self._last_diag_time = 0.0
        self._last_diag2_time = 0.0
        self._last_estop_logged = None
        self._cycle_times = []

        qos_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        self.create_subscription(
            Image, TOPIC_COLOR, self.cb_image, qos_img)
        self.create_subscription(
            Odometry, TOPIC_ODOM, self.cb_odom, qos_rel)
        self.create_subscription(
            Twist, TOPIC_CMD_IN, self.cb_cmd_vel_in, 10)
        self.create_subscription(
            Twist, TOPIC_JOY, self.cb_joy, 10)
        self.create_subscription(
            CameraInfo, TOPIC_CAMERA_INFO, self.cb_camera_info, qos_rel)
        self.create_subscription(
            Float64MultiArray, TOPIC_H, self.cb_H, qos_rel)
        self.create_subscription(
            Point, TOPIC_ROBOT_POS, self.cb_robot_pos, qos_rel)

        self.pub_estop = self.create_publisher(Bool, '/emergency_stop', 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_avoid = self.create_publisher(Twist, '/avoidance_cmd_vel', 10)
        self.pub_zones = self.create_publisher(
            Float64MultiArray, '/optical_flow_zones', 10)
        self.pub_state = self.create_publisher(String, '/movement_command', 10)
        self.pub_diag = self.create_publisher(String, '/flow_diagnostics', 10)
        self.pub_debug = self.create_publisher(Image, '/farneback/debug', 10)
        self.pub_flow_viz = self.create_publisher(Image, '/farneback/flow_viz', 10)
        self.pub_bird = self.create_publisher(Image, '/farneback/bird_eye', 10)

        self._last_residual_flow = None

        self.create_timer(TIMER_PERIOD, self.run_inference)
        self.create_timer(WATCHDOG_PERIOD, self._cmd_watchdog)

        filt_parts = []
        if ENABLE_ZONAL_MEDIAN:
            filt_parts.append('ZONAL_MEDIAN')
        elif ENABLE_MEDIAN_FALLBACK:
            filt_parts.append('MEDIAN_FALLBACK')
        filt_str = '+'.join(filt_parts) if filt_parts else 'RAW (no filter)'

        self.get_logger().info(
            'Farneback avoidance node ready.\n'
            f'  Flow danger      : {FLOW_CENTER_DANGER}px (base, +{SPEED_THRESH_GAIN_CENTER}px per m/s)\n'
            f'  Flow warning     : {FLOW_CENTER_WARN}px (base, +{SPEED_THRESH_GAIN_CENTER}px per m/s)\n'
            f'  Flow spike       : {FLOW_SPIKE_THRESHOLD}px (base, +{SPEED_THRESH_GAIN_SPIKE}px per m/s) streak={FLOW_SPIKE_STREAK_REQUIRED}\n'
            f'  Flow diff avoid  : {FLOW_DIFF_AVOID}px (base, +{SPEED_THRESH_GAIN_DIFF}px per m/s)\n'
            f'  Avoid ang        : \u00b1{FLOW_MAX_TURN_RAD}rad/s max (angular_max={AVOID_ANGULAR_MAX})\n'
            f'  Speed adapt      : floor={SPEED_ADAPT_FLOOR} range=[{SPEED_ADAPT_MIN},{SPEED_ADAPT_MAX}]m/s\n'
            f'  Yaw deviation max: {MAX_AVOIDANCE_YAW_DEVIATION:.2f}rad ({np.degrees(MAX_AVOIDANCE_YAW_DEVIATION):.0f}\u00b0)\n'
            f'  Recovery KP      : {YAW_RECOVERY_KP} max_turn={RECOVERY_MAX_TURN}rad/s\n'
            f'  Odom topic       : {TOPIC_ODOM}\n'
            f'  Residual filter  : {filt_str}\n'
            f'  H-compensation   : {"ON" if ENABLE_HOMOGRAPHY_COMPENSATION else "OFF"} (geometric ground masking)\n'
            f'  Blob coherence   : min={BLOB_COHERENCE_MIN} OR divergence min={BLOB_DIVERGENCE_MIN}\n'
            f'  Ground mask      : {GROUND_MASK_BOTTOM_RATIO*100:.0f}% bottom image excluded\n'
            f'  AprilTag masking : {"ON (dict 36h11, dilate=" + str(APRILTAG_MASK_DILATE_PX) + "px)" if ENABLE_APRILTAG_MASKING else "OFF"}\n'
            f'  Wall detection   : top_inactivity={WALL_REQUIRE_TOP_INACTIVITY} '
            f'top<={WALL_TOP_MAX_COVERAGE} bottom<={WALL_BOTTOM_MAX_COVERAGE}\n'
            f'  Traj. horizon    : {TRAJ_HORIZON_S}s (maneuver_time={REQUIRED_MANEUVER_TIME_S:.2f}s)\n'
            f'  Farneback params : winsize={FARNEBACK_WINSIZE} poly_n={FARNEBACK_POLY_N} '
            f'levels={FARNEBACK_LEVELS} iters={FARNEBACK_ITERATIONS} @ {PROCESS_WIDTH}x{PROCESS_HEIGHT} (CPU)\n'
            f'  Distance WARN/DANGER/MAX : {DISTANCE_WARN_M}/{DISTANCE_DANGER_M}/{MAX_OBSTACLE_DIST_M} m\n'
            f'  Homography       : {"LOADED" if self.distance_estimator.ready else "PENDING"}\n')

    def cb_image(self, msg):
        try:
            self._latest_frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')

            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                self._latest_msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        except Exception as exc:
            self.get_logger().warning(
                f'Image conversion failed: {exc}',
                throttle_duration_sec=2.0)

    def cb_camera_info(self, msg):
        new_w, new_h = msg.width, msg.height
        dims_changed = (new_w != self._cam_source_w or new_h != self._cam_source_h)
        self._cam_source_w = new_w
        self._cam_source_h = new_h

        if dims_changed and self.distance_estimator.H is not None:
            self.ego_compensator.set_homography(
                self.distance_estimator.H,
                source_width=new_w, source_height=new_h)
            self.get_logger().info(
                f'H ego-motion updated to camera resolution {new_w}×{new_h} ✓')

    def cb_odom(self, msg):
        self._robot_vx = msg.twist.twist.linear.x
        self._robot_vy = msg.twist.twist.linear.y
        self._robot_wz = msg.twist.twist.angular.z
        self._robot_speed_stamp = time.monotonic()
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._current_yaw = float(np.arctan2(siny, cosy))

    def cb_joy(self, msg):
        self._user_cmd_x = msg.linear.x
        self._user_cmd_z = msg.angular.z
        self._user_cmd_stamp = time.monotonic()

    def cb_H(self, msg):
        """Real-time homography matrix callback."""
        H = np.array(msg.data, dtype=np.float64).reshape(3, 3)
        self.distance_estimator.set_homography(H)
        self.ego_compensator.set_homography(
            H,
            source_width=self._cam_source_w,
            source_height=self._cam_source_h)
        self.get_logger().info(
            'Homography updated from topic \u2713 (distance + ground masking)',
            throttle_duration_sec=5.0)

    def cb_robot_pos(self, msg):
        self.distance_estimator.set_robot_ground_pos(msg.x, msg.y)

    def cb_cmd_vel_in(self, msg):
        now = time.monotonic()
        self._intent_x = msg.linear.x
        self._intent_stamp = now
        self._last_cmd_in_stamp = now

        if now - self._robot_speed_stamp > ROBOT_SPEED_TIMEOUT:
            self._robot_vx = msg.linear.x
            self._robot_vy = 0.0
            self._robot_speed_stamp = now

        is_reversing = msg.linear.x < -0.01
        is_pure_rotation = (abs(msg.linear.x) < PURE_ROTATION_LINEAR_MAX
                             and abs(msg.angular.z) > PURE_ROTATION_ANGULAR_MIN)

        if is_reversing or is_pure_rotation:
            if self.state_machine._stop_active:
                self.state_machine._stop_active = False
                self.state_machine._clean_cycle_count = 0
                self._publish_estop(False)
                reason = 'reverse' if is_reversing else 'pure rotation'
                self.get_logger().info(f'Emergency stop RELEASED ({reason} via cmd_vel_in).')
            self.pub_cmd.publish(msg)
            return

        joy_fresh = (now - self._user_cmd_stamp) <= CMD_SOURCE_TIMEOUT
        if not joy_fresh:
            self.pub_cmd.publish(Twist())
            return

        if self.state_machine._stop_active:
            self.pub_cmd.publish(Twist())
            return

        if (self.state_machine.state == RobotState.WARNING
                and self.state_machine._ratio_smoother.value is not None
                and self.state_machine._ratio_smoother.value < 1.0):
            safe = Twist()
            safe.linear.x = msg.linear.x * self.state_machine._ratio_smoother.value
            safe.angular.z = msg.angular.z * self.state_machine._ratio_smoother.value
            self.pub_cmd.publish(safe)
            return

        self.pub_cmd.publish(msg)

    def _cmd_watchdog(self):
        now = time.monotonic()
        if (now - self._last_cmd_in_stamp) > CMD_SOURCE_TIMEOUT:
            self.pub_cmd.publish(Twist())

    def run_inference(self):
        if self._latest_frame is None:
            return

        frame = self._latest_frame.copy()
        msg_time = self._latest_msg_time

        now_sys = time.monotonic()
        if msg_time is not None and self._last_frame_msg_time is not None:
            dt = msg_time - self._last_frame_msg_time
            if dt <= 0 or dt > 2.0:
                dt = now_sys - self._last_system_time
        else:
            dt = now_sys - self._last_system_time

        now = msg_time if msg_time is not None else now_sys

        self._last_system_time = now_sys
        self._last_frame_msg_time = msg_time

        if self._prev_frame is None:
            self._prev_frame = frame
            return

        flow = self.flow_estimator.compute(self._prev_frame, frame)
        residual_flow = self.ego_compensator.compensate(flow)

        if ENABLE_APRILTAG_MASKING and self.apriltag_masker is not None:
            tag_mask = self.apriltag_masker.build_mask(
                frame, residual_flow.shape[1], residual_flow.shape[0])
            if np.any(tag_mask):
                residual_flow[tag_mask, 0] = 0.0
                residual_flow[tag_mask, 1] = 0.0
            if (now - self._last_apriltag_log_time) >= DIAGNOSTIC_LOG_PERIOD:
                self._last_apriltag_log_time = now
                self.get_logger().info(
                    f'[AprilTag] {self.apriltag_masker.last_n_detected} tags '
                    f'detected and masked', throttle_duration_sec=5.0)

        if ENABLE_DIAGNOSTIC_LOG and (now - self._last_diag_time) >= DIAGNOSTIC_LOG_PERIOD:
            self._last_diag_time = now
            mag_brut, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag_res, _ = cv2.cartToPolar(residual_flow[..., 0], residual_flow[..., 1])
            h, w = mag_brut.shape
            tier = w // 3
            res_l = float(np.mean(mag_res[:, :tier]))
            res_c = float(np.mean(mag_res[:, tier:2*tier]))
            res_r = float(np.mean(mag_res[:, 2*tier:]))
            mt = MOVEMENT_THRESHOLD
            act_l_mask = mag_res[:, :tier] > mt
            act_c_mask = mag_res[:, tier:2*tier] > mt
            act_r_mask = mag_res[:, 2*tier:] > mt
            act_l = float(np.mean(mag_res[:, :tier][act_l_mask])) if np.any(act_l_mask) else 0.0
            act_c = float(np.mean(mag_res[:, tier:2*tier][act_c_mask])) if np.any(act_c_mask) else 0.0
            act_r = float(np.mean(mag_res[:, 2*tier:][act_r_mask])) if np.any(act_r_mask) else 0.0
            cycle_ms = dt * 1000.0
            eff_d = self.risk_estimator.danger_threshold
            eff_w = self.risk_estimator.warning_threshold
            eff_s = self.risk_estimator.spike_threshold
            eff_df = self.risk_estimator.diff_avoid_threshold

            comp_parts = []
            if ENABLE_HOMOGRAPHY_COMPENSATION and self.ego_compensator._H_scaled is not None:
                comp_parts.append('GMASK')
            if ENABLE_ZONAL_MEDIAN:
                comp_parts.append('ZMED')
            elif ENABLE_MEDIAN_FALLBACK:
                comp_parts.append('MED')
            h_comp_str = '+'.join(comp_parts) if comp_parts else 'RAW'

            self.get_logger().info(
                f'[DIAG] cycle={cycle_ms:.0f}ms vx={self._robot_vx:+.2f}m/s '
                f'wz={self._robot_wz:+.3f}rad/s comp={h_comp_str} '
                f'ACT[L:{act_l:.1f} C:{act_c:.1f} R:{act_r:.1f}] '
                f'THR[d={eff_d:.1f} w={eff_w:.1f} s={eff_s:.1f} df={eff_df:.1f}]')

        if ENABLE_HOMOGRAPHY_COMPENSATION and self.ego_compensator._H_scaled is not None:
            norm_res = np.linalg.norm(residual_flow, axis=-1)
            ground_mask = self.ego_compensator.build_ground_mask()
            if ground_mask is not None:
                is_floor = (norm_res < PARALLAX_THRESHOLD) & ground_mask
                residual_flow[is_floor, 0] = 0.0
                residual_flow[is_floor, 1] = 0.0
        elif ENABLE_HOMOGRAPHY_COMPENSATION and GROUND_MASK_BOTTOM_RATIO > 0.0:
            h_res = residual_flow.shape[0]
            cut = int(h_res * (1.0 - GROUND_MASK_BOTTOM_RATIO))
            residual_flow[cut:, :, :] = 0.0

        blobs = self.blob_detector.detect(residual_flow)
        zone_metrics = self.zone_analyzer.compute_zone_metrics(residual_flow)
        self._last_residual_flow = np.copy(residual_flow)

        trajectory_pred = self.trajectory_tracker.update(
            blobs, frame_width=residual_flow.shape[1], timestamp=now)

        risk = self.risk_estimator.update(zone_metrics, dt,
                                          trajectory=trajectory_pred,
                                          robot_speed=self._robot_vx)

        obstacles = self.distance_estimator.estimate_obstacle_distances(
            blobs, residual_flow.shape, frame.shape)
        self._last_obstacles = obstacles

        distance_danger  = False
        distance_warning = False
        current_min_dist = float('inf')
        corridor_min_dist = float('inf')
        rob_x, rob_y = self.distance_estimator.robot_ground_pos

        for obs in obstacles:
            dist_m = obs['dist_m']
            if dist_m < current_min_dist:
                current_min_dist = dist_m
            lateral_offset = abs(obs['y_m'] - rob_y)
            in_corridor = lateral_offset < WARNING_CORRIDOR_HALF_WIDTH

            if dist_m <= DISTANCE_DANGER_M:
                distance_danger = True
            elif dist_m <= DISTANCE_WARN_M and in_corridor:
                distance_warning = True
                if dist_m < corridor_min_dist:
                    corridor_min_dist = dist_m

        if self.distance_estimator.ready:
            flow_danger_validated = risk.flow_danger and (current_min_dist <= DISTANCE_WARN_M)
        else:
            flow_danger_validated = risk.flow_danger

        if risk.raw_center_emergency and distance_danger:
            flow_danger_validated = True

        combined_danger  = flow_danger_validated or distance_danger
        combined_warning = risk.flow_warning or distance_warning

        joy_fresh = (now_sys - self._user_cmd_stamp) <= CMD_SOURCE_TIMEOUT
        user_cmd_x = self._user_cmd_x if joy_fresh else 0.0
        user_cmd_z = self._user_cmd_z if joy_fresh else 0.0
        user_moving_forward = joy_fresh and (user_cmd_x > 0.05)
        user_reversing = joy_fresh and (user_cmd_x < -0.01)
        user_pure_rotation = (joy_fresh
                               and abs(user_cmd_x) < PURE_ROTATION_LINEAR_MAX
                               and abs(user_cmd_z) > PURE_ROTATION_ANGULAR_MIN)

        avoid_result = self.side_selector.compute_avoidance(
            risk=risk,
            warning=combined_warning,
            yaw_error=self.state_machine.yaw_error,
            user_cmd_z=user_cmd_z,
            user_moving_forward=user_moving_forward and not combined_danger,
            robot_vx=self._robot_vx,
        )

        if combined_danger:
            combined_ratio = 0.0
        elif combined_warning:
            ratio_candidates = []
            if risk.flow_warning:
                ratio_candidates.append(risk.warning_flow_ratio)
            if distance_warning:
                ratio_candidates.append(max(0.0, min(1.0,
                    (corridor_min_dist - DISTANCE_DANGER_M)
                    / (DISTANCE_WARN_M - DISTANCE_DANGER_M))))
            combined_ratio = min(ratio_candidates) if ratio_candidates else 1.0
        else:
            combined_ratio = 1.0

        sm_inputs = StateMachineInputs(
            danger=combined_danger,
            warning=combined_warning,
            warning_flow_ratio=combined_ratio,
            user_moving_forward=user_moving_forward,
            user_reversing=user_reversing,
            user_pure_rotation=user_pure_rotation,
            user_turning=abs(user_cmd_z) > 0.05,
            current_yaw=self._current_yaw,
            avoid_cmd_active=(avoid_result is not None
                              and not avoid_result.is_recovery),
            is_recovery=(avoid_result is not None
                         and avoid_result.is_recovery),
        )
        sm_outputs = self.state_machine.update(sm_inputs)

        if sm_outputs.should_publish_estop:
            self._publish_estop(sm_outputs.estop_value)
            if sm_outputs.estop_value and sm_outputs.engage_reason:
                reason = sm_outputs.engage_reason
                if distance_danger:
                    reason += f' dist<{DISTANCE_DANGER_M}m(min={current_min_dist:.2f}m)'
                self.get_logger().error(
                    f'EMERGENCY STOP \u2014 {reason}',
                    throttle_duration_sec=0.5)
            elif not sm_outputs.estop_value and self.state_machine.state == RobotState.CLEAR:
                pass

        if avoid_result is not None:
            self.pub_avoid.publish(avoid_result.cmd)

        if ENABLE_DIAGNOSTIC_LOG and (now - self._last_diag2_time) >= DIAGNOSTIC_LOG_PERIOD:
            self._last_diag2_time = now
            zg = risk.risk_per_zone.get('gauche', ZoneRisk())
            zc = risk.risk_per_zone.get('centre', ZoneRisk())
            zd = risk.risk_per_zone.get('droite', ZoneRisk())
            if avoid_result is not None:
                avoid_str = (f'{avoid_result.source} '
                             f'lin={avoid_result.cmd.linear.x:.2f} '
                             f'ang={avoid_result.cmd.angular.z:.2f}')
            else:
                avoid_str = 'None'
            dist_str = (f'{current_min_dist:.2f}m' if current_min_dist < float('inf') else '--')
            self.get_logger().info(
                f'[DIAG2] state={sm_outputs.state} '
                f'joy_fwd={user_moving_forward} joy_rev={user_reversing} '
                f'joy_rot={user_pure_rotation} '
                f'danger={combined_danger} warning={combined_warning} '
                f'zones[L:{zg.category}({zg.mean_magnitude_smoothed:.1f}) '
                f'C:{zc.category}({zc.mean_magnitude_smoothed:.1f}) '
                f'R:{zd.category}({zd.mean_magnitude_smoothed:.1f})] '
                f'wall[L:{zg.wall_active} C:{zc.wall_active} R:{zd.wall_active}] '
                f'dist_min={dist_str} avoid={avoid_str}')

        self._publish_zone_metrics(risk)

        if sm_outputs.state == RobotState.EMERGENCY:
            cmd_text = f'EMERGENCY_STOP ({sm_outputs.engage_reason})'
            if distance_danger:
                cmd_text += f' dist={current_min_dist:.2f}m'
        elif sm_outputs.state == RobotState.PREDICTING and avoid_result:
            dist_str = (f' d={current_min_dist:.1f}m'
                        if current_min_dist < float('inf') else '')
            cmd_text = f'AVOID_{avoid_result.side.upper()} [{avoid_result.source}]{dist_str}'
        elif sm_outputs.state == RobotState.RECOVERY and avoid_result:
            cmd_text = f'RECOVERY err={self.state_machine.yaw_error:.2f}rad'
        elif sm_outputs.state == RobotState.WARNING:
            if current_min_dist < float('inf'):
                cmd_text = (f'WARN dist={current_min_dist:.1f}m '
                           f'r={sm_outputs.smooth_ratio:.2f}')
            else:
                cmd_text = (f'WARN flow_c={risk.smoothed_center:.1f}px '
                           f'r={sm_outputs.smooth_ratio:.2f}')
        else:
            cmd_text = (f'CLEAR v={self._robot_vx:.2f}m/s '
                        f'[L:{risk.smoothed_left:.1f} '
                        f'C:{risk.smoothed_center:.1f} '
                        f'R:{risk.smoothed_right:.1f}]')
        self.pub_state.publish(String(data=cmd_text))

        try:
            self._publish_flow_diagnostics(
                risk, sm_outputs, avoid_result, current_min_dist,
                n_obstacles=len(obstacles),
                joy_fwd=user_moving_forward,
                joy_rev=user_reversing,
                joy_rot=user_pure_rotation,
            )
            self._publish_debug_image(frame, risk, sm_outputs, avoid_result,
                                    current_min_dist)
        except Exception:
            self.get_logger().error(
                '[NON-CRITICAL] Exception in diagnostics/debug image '
                '(avoidance/estop already published, unaffected):\n'
                + traceback.format_exc())

        self._prev_frame = frame

    def _publish_flow_diagnostics(self, risk, sm_outputs, avoid_result,
                                   current_min_dist, n_obstacles=None,
                                   joy_fwd=None, joy_rev=None, joy_rot=None):
        zr = risk.risk_per_zone.get('centre', ZoneRisk())
        ttc = round(float(1.0 / zr.mean_divergence), 3) if zr.mean_divergence > 1e-4 else None
        diag = {
            'state': str(sm_outputs.state),
            'ttc': ttc,
            'min_dist_m': None if current_min_dist == float('inf') else round(float(current_min_dist), 3),
            'n_obstacles': n_obstacles,
            'joy_fwd': joy_fwd,
            'joy_rev': joy_rev,
            'joy_rot': joy_rot,
            'left_blocked': risk.left_blocked,
            'right_blocked': risk.right_blocked,
            'avoid_source': avoid_result.source if avoid_result else None,
        }
        self.pub_diag.publish(String(data=json.dumps(diag)))

    def _publish_estop(self, value):
        msg = Bool()
        msg.data = value
        self.pub_estop.publish(msg)
        if not value and self._last_estop_logged is not False:
            self.get_logger().info('Emergency stop RELEASED.')
        self._last_estop_logged = value

    def _publish_zone_metrics(self, risk):
        msg = Float64MultiArray()
        values = []
        for zone_name in ('gauche', 'centre', 'droite'):
            zr = risk.risk_per_zone.get(zone_name, ZoneRisk())
            values.extend([
                zr.mean_magnitude_smoothed,
                zr.max_magnitude,
                zr.coverage,
                zr.mean_divergence,
                zr.risk_score,
                1.0 if zr.spike_active else 0.0,
            ])
        msg.data = values
        self.pub_zones.publish(msg)

    def _publish_debug_image(self, frame, risk, sm_outputs, avoid_result,
                             current_min_dist=float('inf')):
        h_img, w_img = frame.shape[:2]
        debug = frame.copy()

        third_w = w_img // 3
        cv2.line(debug, (third_w, 0), (third_w, h_img), (80, 80, 80), 1)
        cv2.line(debug, (2 * third_w, 0), (2 * third_w, h_img), (80, 80, 80), 1)

        cv2.putText(debug, f'L:{risk.smoothed_left:.1f}',
                    (5, h_img - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_OK, 1)
        center_color = (C_DANGER if risk.flow_danger
                        else (C_WARNING if risk.flow_warning else C_OK))
        cv2.putText(debug, f'C:{risk.smoothed_center:.1f}',
                    (third_w + 5, h_img - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, center_color, 1)
        cv2.putText(debug, f'R:{risk.smoothed_right:.1f}',
                    (2 * third_w + 5, h_img - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_OK, 1)

        state = sm_outputs.state
        if state == RobotState.EMERGENCY:
            hud_txt = f'EMERGENCY STOP ({sm_outputs.engage_reason})'
            hud_color = C_DANGER
        elif state == RobotState.RECOVERY and avoid_result:
            hud_txt = (f'RECOVERY err={self.state_machine.yaw_error:.2f}rad '
                       f'\u03c9={avoid_result.cmd.angular.z:.2f}rad/s')
            hud_color = C_RECOVERY
        elif state == RobotState.PREDICTING and avoid_result:
            dist_str = (f' d={current_min_dist:.1f}m'
                        if current_min_dist < float('inf') else '')
            hud_txt = (f'PREDICTING [{avoid_result.source}] '
                       f'\u03c9={avoid_result.cmd.angular.z:.2f}{dist_str}')
            hud_color = C_PREDICT
        elif state == RobotState.WARNING:
            if current_min_dist < float('inf'):
                hud_txt = (f'WARN dist={current_min_dist:.1f}m '
                           f'r={sm_outputs.smooth_ratio:.2f}')
            else:
                hud_txt = (f'WARN flow_c={risk.smoothed_center:.1f}px '
                           f'r={sm_outputs.smooth_ratio:.2f}')
            hud_color = C_WARNING
        else:
            hud_txt = (f'CLEAR v={self._robot_vx:.2f}m/s '
                       f'[L:{risk.smoothed_left:.1f} '
                       f'C:{risk.smoothed_center:.1f} '
                       f'R:{risk.smoothed_right:.1f}]')
            hud_color = C_OK

        overlay = debug.copy()
        cv2.rectangle(overlay, (0, 0), (w_img, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, debug, 0.4, 0, debug)
        cv2.putText(debug, hud_txt, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2)

        try:
            self.pub_debug.publish(
                self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))
        except Exception:
            pass

        flow_viz = self._create_flow_visualization()
        if flow_viz is not None:
            try:
                self.pub_flow_viz.publish(
                    self.bridge.cv2_to_imgmsg(flow_viz, encoding='bgr8'))
            except Exception:
                pass

        bird_view = self._build_bird_eye(self._last_obstacles, risk, sm_outputs)
        try:
            self.pub_bird.publish(
                self.bridge.cv2_to_imgmsg(bird_view, encoding='bgr8'))
        except Exception:
            pass

        if ENABLE_DISPLAY:
            display = cv2.resize(debug,
                                 (int(w_img * 0.7), int(h_img * 0.7)))
            cv2.imshow('Farneback Avoidance', display)
            if flow_viz is not None:
                flow_display = cv2.resize(flow_viz,
                                          (int(flow_viz.shape[1] * 0.7),
                                           int(flow_viz.shape[0] * 0.7)))
                cv2.imshow('Optical Flow', flow_display)
            cv2.imshow('Farneback - Bird Eye', bird_view)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                rclpy.shutdown()

    def _create_flow_visualization(self):
        if self._last_residual_flow is None:
            return None

        flow = self._last_residual_flow
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        mag_max = max(float(np.percentile(mag, 99)), 1.0)
        mag_norm = np.clip(mag / mag_max, 0.0, 1.0)

        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = (mag_norm * 255).astype(np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        h, w = bgr.shape[:2]
        tier = w // 3
        cv2.line(bgr, (tier, 0), (tier, h), (255, 255, 255), 1)
        cv2.line(bgr, (2 * tier, 0), (2 * tier, h), (255, 255, 255), 1)

        for idx, label in enumerate(['L', 'C', 'R']):
            x_start = idx * tier
            x_end = (idx + 1) * tier if idx < 2 else w
            zone_mag = mag[:, x_start:x_end]
            active = zone_mag[zone_mag > MOVEMENT_THRESHOLD]
            mean_active = float(np.mean(active)) if len(active) > 0 else 0.0
            mean_all = float(np.mean(zone_mag))
            coverage = len(active) / max(zone_mag.size, 1) * 100

            txt_x = x_start + 5
            cv2.putText(bgr, f'{label}: act={mean_active:.1f}',
                        (txt_x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1)
            cv2.putText(bgr, f'all={mean_all:.1f} cov={coverage:.0f}%',
                        (txt_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                        (180, 180, 180), 1)

        return bgr

    # ══════════════════════════════════════════════════════════════════════════════
    #  BIRD-EYE VIEW
    # ══════════════════════════════════════════════════════════════════════════════

    def _build_bird_eye(self, obstacles, risk, sm_outputs):
        bird = np.full((MAP_H, MAP_W, 3), (30, 30, 30), dtype=np.uint8)

        # Grid
        for step, color in [(int(0.5 / MAP_SCALE), (55, 55, 55)),
                             (int(1.0 / MAP_SCALE), (75, 75, 75))]:
            for x in range(0, MAP_W, step):
                cv2.line(bird, (x, 0), (x, MAP_H), color, 1)
            for y in range(0, MAP_H, step):
                cv2.line(bird, (0, y), (MAP_W, y), color, 1)

        # Robot
        rx_m, ry_m = self.distance_estimator.robot_ground_pos
        rbx, rby = self.distance_estimator.ground_m_to_bird_px(rx_m, ry_m)
        rob_in_view = 0 <= rbx < MAP_W and 0 <= rby < MAP_H

        if rob_in_view:
            cv2.circle(bird, (rbx, rby),
                       int(DISTANCE_WARN_M / MAP_SCALE), C_WARNING, 1, cv2.LINE_AA)
            cv2.circle(bird, (rbx, rby),
                       int(DISTANCE_DANGER_M / MAP_SCALE), C_DANGER, 1, cv2.LINE_AA)
            r_pred = int((abs(self._robot_vx) * TTC_HORIZON
                          + ROBOT_HALF_WIDTH) / MAP_SCALE)
            cv2.circle(bird, (rbx, rby), r_pred, C_PREDICT, 1, cv2.LINE_AA)
            tri = np.array([(rbx, rby - 12), (rbx - 8, rby + 8),
                            (rbx + 8, rby + 8)], np.int32)
            cv2.fillPoly(bird, [tri], (0, 255, 255))

        # Obstacles
        for obs in obstacles:
            bx, by = self.distance_estimator.ground_m_to_bird_px(
                obs['x_m'], obs['y_m'])
            if not (0 <= bx < MAP_W and 0 <= by < MAP_H):
                continue

            dist_m = obs['dist_m']
            if dist_m <= DISTANCE_DANGER_M:
                color = C_DANGER
            elif dist_m <= DISTANCE_WARN_M:
                color = C_WARNING
            else:
                color = C_YOLO

            cv2.circle(bird, (bx, by), 8, color, -1)
            cv2.circle(bird, (bx, by), 8, (255, 255, 255), 1)
            lbl = f'{dist_m:.1f}m'
            cv2.putText(bird, lbl, (bx + 10, by + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1)
            if rob_in_view:
                cv2.line(bird, (rbx, rby), (bx, by), color, 1, cv2.LINE_AA)

        # HUD
        state_color = {
            RobotState.CLEAR: C_OK, RobotState.WARNING: C_WARNING,
            RobotState.PREDICTING: C_PREDICT, RobotState.RECOVERY: C_RECOVERY,
            RobotState.EMERGENCY: C_DANGER,
        }.get(sm_outputs.state, C_OK)
        cv2.putText(bird, f'State: {sm_outputs.state}',
                    (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, state_color, 1)
        cv2.putText(bird, f'vRobot: {self._robot_vx:.2f}m/s  Obs:{len(obstacles)}',
                    (4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_MEMORY, 1)
        cv2.putText(bird, f'Ratio: {sm_outputs.smooth_ratio:.2f}',
                    (4, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_OK, 1)

        flow_panel = self._draw_flow_panel(risk)
        combined = np.vstack([bird, flow_panel])
        return combined

    def _draw_flow_panel(self, risk):
        panel = np.full((FLOW_PANEL_H, MAP_W, 3), (15, 15, 15), dtype=np.uint8)
        bar_x0 = 86
        bar_w = MAP_W - bar_x0 - 46
        bar_h = 14
        flow_max = max(FLOW_CENTER_DANGER * 1.5,
                       FLOW_SPIKE_THRESHOLD * 1.2, 12.0)

        cv2.putText(panel, 'FARNEBACK OPTICAL FLOW (px/frame)', (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

        labels_display = [('L', 'gauche'), ('C', 'centre'), ('R', 'droite')]

        for i, (short, zone_name) in enumerate(labels_display):
            zr = risk.risk_per_zone.get(zone_name, ZoneRisk())
            val = zr.mean_magnitude_smoothed
            y = 24 + i * (bar_h + 6)

            if zone_name == 'centre':
                if val >= FLOW_CENTER_DANGER:
                    color = C_DANGER
                elif val >= FLOW_CENTER_WARN:
                    color = C_WARNING
                else:
                    color = C_OK
            else:
                color = C_OK

            cv2.putText(panel, short, (4, y + bar_h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
            cv2.rectangle(panel, (bar_x0, y), (bar_x0 + bar_w, y + bar_h),
                          (50, 50, 50), -1)
            fill_w = int(min(1.0, val / flow_max) * bar_w)
            cv2.rectangle(panel, (bar_x0, y), (bar_x0 + fill_w, y + bar_h),
                          color, -1)
            cv2.rectangle(panel, (bar_x0, y), (bar_x0 + bar_w, y + bar_h),
                          (120, 120, 120), 1)

            if zone_name == 'centre':
                eff_warn = self.risk_estimator.warning_threshold
                eff_danger = self.risk_estimator.danger_threshold
                eff_spike = self.risk_estimator.spike_threshold

                for t_val, t_color in [(eff_warn, C_WARNING),
                                       (eff_danger, C_DANGER)]:
                    tx = bar_x0 + int(min(1.0, t_val / flow_max) * bar_w)
                    cv2.line(panel, (tx, y - 2), (tx, y + bar_h + 2), t_color, 1)

                tx_spike = bar_x0 + int(min(1.0, eff_spike / flow_max) * bar_w)
                cv2.line(panel, (tx_spike, y - 2), (tx_spike, y + bar_h + 2), (200, 100, 200), 1)

            cv2.putText(panel, f'{val:.1f}', (bar_x0 + bar_w + 4, y + bar_h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1)

            spike_val = zr.max_magnitude
            if spike_val > 0.0:
                spike_x = bar_x0 + int(min(1.0, spike_val / flow_max) * bar_w)
                s_color = C_DANGER if zr.spike_active else (200, 100, 200)
                cv2.line(panel, (spike_x, y), (spike_x, y + bar_h), s_color, 2)
                if zone_name == 'centre':
                    txt = f'spike:{spike_val:.1f}'
                    if zr.is_moving_away:
                        txt += ' [AWAY]'
                        s_color = C_OK
                    elif zr.is_crossing:
                        txt += ' [CROSS]'
                        s_color = C_OK
                    cv2.putText(panel, txt, (bar_x0 + bar_w + 35, y + bar_h - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.30, s_color, 1)

        traj_dec = risk.traj_decision
        if traj_dec == 'EMERGENCY':
            dec_color = C_DANGER
        elif traj_dec == 'AVOID':
            dec_color = C_WARNING
        else:
            dec_color = C_OK
        ttc_str = f'ttc={risk.predicted_ttc:.2f}s' if risk.predicted_ttc is not None else 'ttc=--'
        esc_str = ''
        if risk.traj_escape_blocked:
            esc_str = ' ESC_BLOCKED'
        elif risk.traj_escape_side:
            esc_str = f' esc={risk.traj_escape_side}'
        dbg_txt = f'TRACK:{traj_dec} {ttc_str}{esc_str}'
        cv2.putText(panel, dbg_txt, (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.35, dec_color, 1)

        if self._last_obstacles:
            closest = self._last_obstacles[0]
            dist_color = C_DANGER if closest['dist_m'] <= DISTANCE_DANGER_M else (
                C_WARNING if closest['dist_m'] <= DISTANCE_WARN_M else C_OK)
            cv2.putText(panel,
                        f'Closest: {closest["dist_m"]:.2f}m  '
                        f'({closest["x_m"]:.1f}, {closest["y_m"]:.1f})',
                        (4, 24 + 3 * (bar_h + 6) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, dist_color, 1)
        else:
            cv2.putText(panel, 'No obstacle (H)' if self.distance_estimator.ready
                        else 'Homography NOT loaded',
                        (4, 24 + 3 * (bar_h + 6) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1)

        return panel


def main(args=None):
    rclpy.init(args=args)
    node = FarnebackAvoidanceNode()
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