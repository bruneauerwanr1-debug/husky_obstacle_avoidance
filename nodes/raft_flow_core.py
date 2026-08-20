# -*- coding: utf-8 -*-
"""
raft_flow_core.py — RAFT Algorithmic Core (NO ROS Dependencies)
===================================================================
Contains all decision logic for dense optical flow obstacle avoidance
using RAFT: flow estimation, ego-motion compensation, zonal analysis,
risk estimation, dodge side selection, command smoothing, and anti-lock guard.

Intentionally WITHOUT rclpy / geometry_msgs imports: each class must be
testable in isolation on frames extracted from a rosbag (see extract_bag.py),
matching the Farneback pipeline structure so that comparative analysis
can share the exact same evaluation infrastructure (analyze_bag_csv.py).

The ROS node (raft_avoidance_node.py) imports this module and only handles
plumbing: topic subscriptions, Twist message creation, and publication.

Side Selection Convention (image plane): left/right selection is based on the
SPATIAL POSITION of zones (image split into thirds), NEVER on the sign of a
flow velocity vector — this follows lessons learned from the FOE (Focus of
Expansion) decorrelation issue on the Farneback predictive branch (the velocity
sign of a blob can decorrelate from the true obstacle side due to radial
expansion around the focus of expansion). Here, this issue is avoided since
zones are defined purely by spatial position (left/center/right).
"""

import json
import os

import cv2
import numpy as np
import torch


# =============================================================================
#  CONFIGURATION
# =============================================================================

# ---- RAFT -------------------------------------------------------------------
RAFT_VARIANT = 'small'   # 'small' (fast, default choice for embedded hardware
                         # such as Jetson Orin/Xavier/TX2 vs x86+GPU) or 'large'
                         # (accurate, requires high-end GPU).
# Inference resolution — MUST be a multiple of 8 (RAFT architecture constraint).
# Real-time resolution depends on embedded compute platform.
RAFT_INFER_SIZE = (480, 640)   # (H, W), multiples of 8
RAFT_ITERS = 12                 # Refinement iterations (12 = high quality)

# ---- Zones ------------------------------------------------------------------
# Image split into left / center / right thirds, matching YOLO/LK and Farneback
# systems to remain directly comparable across the evaluation framework.
N_ZONES = 3
ZONE_LABELS = ['left', 'center', 'right']

# ---- Risk Thresholds --------------------------------------------------------
# Average residual magnitude per zone (px/frame, at RAFT_INFER_SIZE resolution,
# NOT native camera resolution).
FLOW_CENTER_WARN = 2.0          # Lowered (was 6.0) — RAFT small produces smaller
                                # magnitudes than Farneback for identical real motion
FLOW_CENTER_DANGER = 5.0        # Lowered (was 12.0)
# SPIKE threshold (raw zone maximum before smoothing), capturing thin/low-texture
# obstacles that produce low average signal — matching FLOW_SPIKE_THRESHOLD logic.
FLOW_SPIKE_THRESHOLD = 10.0     # Lowered (was 18.0)
SPIKE_STREAK_REQUIRED = 3       # Consecutive frames required to confirm spike

# Divergence (radial expansion) — signature of an approaching object,
# contrasting with uniform translation flow (divergence near 0).
# Thresholds expressed in Time-To-Collision (TTC in seconds).
TTC_DANGER_S = 1.2   # s before extrapolated "collision"
TTC_WARN_S   = 2.5   # s
DIVERGENCE_DANGER = 1.0 / TTC_DANGER_S
DIVERGENCE_WARN   = 1.0 / TTC_WARN_S

# Left/right flow imbalance required to trigger avoidance turn
FLOW_DIFF_AVOID = 1.2           # Lowered (was 2.5)

# ---- Smoothing / Debounce ---------------------------------------------------
# Matching Farneback baseline values for comparative benchmarking.
FLOW_SMOOTH_ALPHA = 0.30
SIDE_FLIP_STREAK = 3

# ---- Ego-Motion Compensation ------------------------------------------------
ROT_COMPENSATION_SIGN = -1.0
# Approximate focal length in pixels AT INFERENCE RESOLUTION.
# Derived from horizontal FOV (~69 deg for Intel RealSense D435 color camera).
CAMERA_FOV_H_DEG = 69.0
CAMERA_FOCAL_PX = RAFT_INFER_SIZE[1] / (2.0 * np.tan(np.radians(CAMERA_FOV_H_DEG / 2.0)))

# ---- Avoidance Commands -----------------------------------------------------
AVOID_LINEAR_X = 0.20
AVOID_ANGULAR_BASE = 0.35
AVOID_ANGULAR_MAX = 0.60
AVOID_SMOOTH_ALPHA = 0.35
AVOID_SMOOTH_ALPHA_URGENT = 0.70

# ---- Recovery ---------------------------------------------------------------
YAW_RECOVERY_KP = 0.35
YAW_RECOVERY_MIN = 0.10
RECOVERY_LINEAR_X = 0.15
RECOVERY_MAX_TURN = 0.20

# ---- Anti-Spin Guard (Cumulative Yaw Deviation) -----------------------------
MAX_AVOIDANCE_YAW_DEVIATION = 1.30   # rad ~ 75 deg

# ---- Wall Dead-Zone Detection (Ported from Farneback v4) ---------------------
# A dark or low-texture wall generates almost zero optical flow (aperture problem /
# lack of local contrast). Total silence (both top and bottom zone inactive)
# distinguishes a wall from open floor (which exhibits moving ground flow).
ENABLE_WALL_DEAD_ZONE_DETECTION = True
WALL_DEAD_ZONE_RATIO    = 0.70   # Inactivity ratio required in zone to qualify
WALL_MIN_ACTIVE_BORDER  = 0.20   # Global frame coverage required (excludes stationary robot)
WALL_STREAK_REQUIRED    = 5      # Consecutive frames to confirm (transient filter)
WALL_TOP_RATIO           = 0.30  # Top band ratio of analyzed zone
WALL_TOP_MAX_COVERAGE    = 0.05  # Above this -> not a wall (visible background/horizon)
WALL_BOTTOM_RATIO        = 0.20  # Bottom band (ground) ratio of analyzed zone
WALL_BOTTOM_MAX_COVERAGE = 0.05  # Above this -> not a wall (visible ground flow)

# ---- TTC --------------------------------------------------------------------
TTC_HORIZON = 3.0

# ---- Metric Distance Thresholds (Homography) --------------------------------
DISTANCE_WARN_M   = 1.5    # m -> Deceleration (warning)
DISTANCE_DANGER_M = 0.8    # m -> Immediate stop (emergency)

# ---- Homography Calibration (Identical to predictive_1.py) ------------------
CALIB_W     = 300
CALIB_H     = 250
CALIB_SCALE = 0.01

# ---- Homography Persistence -------------------------------------------------
H_SAVE_PATH = '/home/imr2204/Desktop/Erwan/clearpath_simulator/src/code_python/homography/homography_calibration.json'

# ---- Bird-Eye View ----------------------------------------------------------
MAP_W        = 400
MAP_H        = 400
MAP_SCALE    = 0.01
ROB_OFFSET_Y = 50

# ---- Movement Threshold for Blob Extraction ---------------------------------
MOVEMENT_THRESHOLD = 1.5    # px/frame, lowered for RAFT small sensitivity
MIN_BLOB_AREA      = 100    # px, smaller blobs ignored as noise
MAX_OBSTACLES      = 5      # Max number of obstacles returned (closest first)

# ---- WARNING Corridor (Lateral Filter, identical to predictive_1.py) --------
ROBOT_HALF_WIDTH = 0.40
WARNING_CORRIDOR_HALF_WIDTH = ROBOT_HALF_WIDTH + 0.4   # m

# =============================================================================
#  FIXES PORTED FROM optical_flow_farneback.py (v4/v5/v7/v8)
# =============================================================================

# ---- Reverse / Pure Rotation Bypass (Ported from Farneback v5) --------------
# Pure rotation (in-place turning without linear translation) bypasses EMERGENCY
# to prevent getting trapped when attempting to turn away from obstacles.
PURE_ROTATION_LINEAR_MAX  = 0.05   # |linear.x| below this -> no significant translation
PURE_ROTATION_ANGULAR_MIN = 0.05   # |angular.z| above this -> intentional rotation

# ---- Ego-Motion Compensation — Active Filters (Ported from Farneback v4) ----
# Rotational flow compensation model (Longuet-Higgins). Disabled by default
# as a precaution until bench validation on the robot.
ENABLE_ROTATION_COMPENSATION = False
# ---- Statistical Filters (Fallback) -----------------------------------------
ENABLE_MEDIAN_FALLBACK = True    # Global median subtraction
ENABLE_ZONAL_MEDIAN    = False   # Per-zone (L/C/R) median (takes precedence if True)

# ---- Ground Plane Masking via Homography (Ported from Farneback v4) ---------
ENABLE_GROUND_MASK       = False
GROUND_MASK_BOTTOM_RATIO = 0.0    # Bottom band fallback ratio
GROUND_MASK_TOP_Y_RATIO  = 0.40   # Used if ENABLE_GROUND_MASK is True
PARALLAX_THRESHOLD       = 2.0

# ---- Explicit AprilTag Floor Masking (Ported from Farneback) ----------------
ENABLE_APRILTAG_MASKING = True
APRILTAG_DICT_ID        = cv2.aruco.DICT_APRILTAG_36h11
APRILTAG_MASK_DILATE_PX = 4
APRILTAG_CORNER_REFINE  = cv2.aruco.CORNER_REFINE_NONE

# ---- Max Obstacle Distance / Behind-Robot Rejection (Ported from v4) --------
MAX_OBSTACLE_DIST_M = 10.0

# ---- Zone Occupancy Turn Factor Based on Severity (Ported from v4/v6) -------
ZONE_OCCUPANCY_EMERGENCY_TURN_FACTOR = 0.8
ZONE_OCCUPANCY_WARNING_TURN_FACTOR   = 0.4


# =============================================================================
#  DENSE OPTICAL FLOW ESTIMATION — RAFT
# =============================================================================

class RaftFlowEstimator:
    """
    Computes dense optical flow using RAFT (torchvision) between two consecutive frames.
    Self-contained: replace this class (same compute(frame_prev, frame_curr) -> flow (H,W,2)
    interface) to benchmark against Farneback or other flow backends.

    Requires torchvision >= 0.12. Pretrained weights are cached locally upon first download.
    """

    def __init__(self, device=None, variant=RAFT_VARIANT, infer_size=RAFT_INFER_SIZE):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.infer_size = infer_size   # (H, W)
        self._model = None
        self._transforms = None
        # Half precision (float16) for ~2x GPU inference acceleration
        self._use_half = (self.device != 'cpu' and torch.cuda.is_available())
        self._load_model(variant)

    def _load_model(self, variant):
        from torchvision.models.optical_flow import (
            raft_small, raft_large, Raft_Small_Weights, Raft_Large_Weights)
        if variant == 'small':
            weights = Raft_Small_Weights.DEFAULT
            self._model = raft_small(weights=weights, progress=False)
        else:
            weights = Raft_Large_Weights.DEFAULT
            self._model = raft_large(weights=weights, progress=False)
        self._transforms = weights.transforms()
        self._model = self._model.to(self.device).eval()

        # Convert model to float16 if GPU is available
        if self._use_half:
            self._model = self._model.half()

        # Warmup pass to avoid CUDA/cuDNN initialization lag on first live frame
        h, w = self.infer_size
        dummy = torch.zeros(1, 3, h, w, dtype=torch.uint8, device=self.device)
        with torch.no_grad():
            if self._use_half:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    self._model(*self._transforms(dummy, dummy), num_flow_updates=RAFT_ITERS)
            else:
                self._model(*self._transforms(dummy, dummy), num_flow_updates=RAFT_ITERS)

    def _preprocess(self, frame_bgr):
        h, w = self.infer_size
        resized = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # uint8 tensor (1,3,H,W) — RAFT transforms handle pixel normalization/scaling
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
        return tensor.to(self.device)

    def compute(self, frame_prev_bgr, frame_curr_bgr):
        """
        Returns dense optical flow (H_orig, W_orig, 2) in px/frame, UPSCALED to the
        native resolution of the input frame (not RAFT inference resolution).
        Flow vectors are rescaled proportionally to match native resolution.
        """
        orig_h, orig_w = frame_curr_bgr.shape[:2]
        img1 = self._preprocess(frame_prev_bgr)
        img2 = self._preprocess(frame_curr_bgr)
        img1, img2 = self._transforms(img1, img2)
        with torch.no_grad():
            if self._use_half:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    flow_predictions = self._model(img1, img2, num_flow_updates=RAFT_ITERS)
            else:
                flow_predictions = self._model(img1, img2, num_flow_updates=RAFT_ITERS)
        flow = flow_predictions[-1][0]                      # (2, H, W) — final iteration
        flow = flow.float().permute(1, 2, 0).cpu().numpy()  # (H, W, 2)

        # Upscale flow to native camera resolution
        infer_h, infer_w = self.infer_size
        if (orig_h, orig_w) != (infer_h, infer_w):
            scale_x = orig_w / infer_w
            scale_y = orig_h / infer_h
            flow_up = cv2.resize(flow, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            flow_up[..., 0] *= scale_x   # Vector u scale correction
            flow_up[..., 1] *= scale_y   # Vector v scale correction
            return flow_up
        return flow


# =============================================================================
#  HOMOGRAPHY PERSISTENCE
# =============================================================================

def load_homography(path: str):
    """Loads homography matrix H from a JSON file (matching predictive_1.py format)."""
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        H = np.array(data['H'], dtype=np.float64).reshape(3, 3)
        if H.shape != (3, 3):
            return None, None
        np.linalg.inv(H)   # Verify invertibility
        return H, data.get('metadata', {})
    except Exception:
        return None, None


# =============================================================================
#  HOMOGRAPHY DISTANCE ESTIMATION
# =============================================================================

class HomographyDistanceEstimator:
    """Projects optical flow blobs to metric ground coordinates via camera-to-ground
    homography matrix H.

    ROS-independent: ROS node calls set_homography() and set_robot_ground_pos() from callbacks.
    """

    def __init__(self):
        self.H = None
        self.H_inv = None
        self.robot_ground_pos = (0.0, 0.0)

    def set_homography(self, H):
        """Receives 3x3 matrix H and precomputes its inverse."""
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

    @property
    def ready(self):
        return self.H is not None

    def cam_px_to_ground_m(self, u, v, orig_shape=None):
        """Projects pixel coordinate (u,v) to metric ground coordinate (x_m, y_m)."""
        if self.H is None:
            return None

        # Adaptation automatique si la résolution runtime diffère de la résolution de calibration
        if orig_shape is not None and getattr(self, '_H_calib_w', None) and getattr(self, '_H_calib_h', None):
            h_curr, w_curr = orig_shape[:2]
            scale_x = float(self._H_calib_w) / max(1, w_curr)
            scale_y = float(self._H_calib_h) / max(1, h_curr)
            u = u * scale_x
            v = v * scale_y

        p = np.array([u, v, 1.0], dtype=np.float64)
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
        """Converts metric ground coordinates to Bird-Eye View pixel coordinates."""
        bx = int(MAP_W / 2.0 + y_m / MAP_SCALE)
        by = int(MAP_H - ROB_OFFSET_Y - x_m / MAP_SCALE)
        return bx, by

    def estimate_obstacle_distances(self, flow_residual, orig_shape):
        """Identifies significant residual flow blobs and estimates their metric distance
        from the robot via homography.

        Parameters:
            flow_residual : (H_flow, W_flow, 2) — Residual flow after ego-motion compensation
            orig_shape    : (H_cam, W_cam) — Native camera image shape for rescaling

        Returns: list of dicts {'x_m', 'y_m', 'dist_m', 'u_cam', 'v_cam', 'area'}
        """
        if not self.ready:
            return []

        h_flow, w_flow = flow_residual.shape[:2]
        h_cam, w_cam = orig_shape[:2]
        rob_x, _rob_y = self.robot_ground_pos

        # Magnitude -> Binary mask -> Connected components
        mag = np.sqrt(flow_residual[..., 0]**2 + flow_residual[..., 1]**2)
        mask = (mag > MOVEMENT_THRESHOLD).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)

        obstacles = []
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < MIN_BLOB_AREA:
                continue

            cx, cy = centroids[label_id]
            # Lowest point of blob (closest ground contact in camera view)
            blob_mask = (labels == label_id)
            ys_blob = np.where(blob_mask.any(axis=1))[0]
            if len(ys_blob) == 0:
                continue
            bottom_y = ys_blob[-1]
            xs_at_bottom = np.where(blob_mask[bottom_y])[0]
            bottom_x = float(np.mean(xs_at_bottom))

            # Rescale to native camera resolution
            u_cam = bottom_x * (w_cam / w_flow)
            v_cam = bottom_y * (h_cam / h_flow)

            ground = self.cam_px_to_ground_m(u_cam, v_cam, orig_shape=orig_shape)
            if ground is None:
                continue
            x_m, y_m = ground
            dist_m = self.dist_from_robot(x_m, y_m)

            # Max range limit (extrapolation artifact rejection)
            if dist_m > MAX_OBSTACLE_DIST_M:
                continue

            # Reject points projected behind the robot
            if x_m < rob_x:
                continue

            obstacles.append({
                'x_m': round(x_m, 2),
                'y_m': round(y_m, 2),
                'dist_m': round(dist_m, 2),
                'u_cam': round(u_cam, 1),
                'v_cam': round(v_cam, 1),
                'area': area,
            })

        # Sort by distance (closest first) and limit to max obstacles
        obstacles.sort(key=lambda o: o['dist_m'])
        return obstacles[:MAX_OBSTACLES]


# =============================================================================
#  EXPLICIT FLOOR APRILTAG MASKING
# =============================================================================

class AprilTagFloorMasker:
    """Detects floor AprilTags and generates a dilated binary mask to zero out
    their contribution to residual flow BEFORE zonal analysis."""

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


# =============================================================================
#  EGO-MOTION COMPENSATION (Yaw Rotation)
# =============================================================================

class EgoMotionCompensator:
    """
    Estimates expected flow caused by robot yaw rotation and subtracts it
    from raw RAFT flow to isolate residual flow caused by approaching obstacles.

    Model: Longuet-Higgins & Prazdny formulation for pure yaw rotation:
        u_rot ~= SIGN * omega_z * dt * f * (1 + xn^2)
        v_rot ~= SIGN * omega_z * dt * f * (xn * yn)
    """

    def __init__(self, focal_px=CAMERA_FOCAL_PX, sign=ROT_COMPENSATION_SIGN):
        self.focal_px = focal_px
        self.sign = sign
        self._xn = None
        self._yn = None
        self._shape = None
        # Ground plane masking
        self._H = None
        self._ground_mask_cache = None
        self._ground_mask_shape = None

    def _grid(self, h, w):
        if self._shape != (h, w):
            ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            self._xn = (xs - w / 2.0) / self.focal_px
            self._yn = (ys - h / 2.0) / self.focal_px
            self._shape = (h, w)
        return self._xn, self._yn

    def expected_rotational_flow(self, omega_z, dt, h, w):
        xn, yn = self._grid(h, w)
        u_rot = self.sign * omega_z * dt * self.focal_px * (1.0 + xn ** 2)
        v_rot = self.sign * omega_z * dt * self.focal_px * (xn * yn)
        return np.stack([u_rot, v_rot], axis=-1)

    def set_homography(self, H):
        """Receives 3x3 camera->ground homography matrix H. Invalidates mask cache."""
        if H is None:
            self._H = None
            self._ground_mask_cache = None
            self._ground_mask_shape = None
            return
        try:
            np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return
        self._H = H.astype(np.float64)
        self._ground_mask_cache = None
        self._ground_mask_shape = None

    def build_ground_mask(self, h, w):
        """Static geometric mask of ground plane at resolution (h, w)."""
        if (self._ground_mask_cache is not None
                and self._ground_mask_shape == (h, w)):
            return self._ground_mask_cache
        if self._H is None:
            return None

        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        pts = np.stack([xs.ravel(), ys.ravel(), np.ones(h * w, dtype=np.float64)], axis=1)
        warped = (self._H @ pts.T).T
        wz = warped[:, 2]
        valid_w = np.abs(wz) > 1e-8
        bx = np.where(valid_w, warped[:, 0] / np.where(valid_w, wz, 1.0), -1.0)
        by = np.where(valid_w, warped[:, 1] / np.where(valid_w, wz, 1.0), -1.0)

        is_ground = (valid_w
                     & (bx >= 0) & (bx < CALIB_W)
                     & (by >= 0) & (by < CALIB_H))

        top_cutoff = int(h * GROUND_MASK_TOP_Y_RATIO)
        row_idx = np.arange(h * w) // w
        is_ground = is_ground & (row_idx >= top_cutoff)

        self._ground_mask_cache = is_ground.reshape(h, w)
        self._ground_mask_shape = (h, w)
        return self._ground_mask_cache

    def _apply_ground_mask(self, residual):
        h, w = residual.shape[:2]
        if self._H is not None:
            mask = self.build_ground_mask(h, w)
            if mask is not None:
                norm_res = np.linalg.norm(residual, axis=-1)
                is_floor = (norm_res < PARALLAX_THRESHOLD) & mask
                residual[is_floor, 0] = 0.0
                residual[is_floor, 1] = 0.0
                return residual
        # Fallback bottom-band blanking if H is not yet available
        if GROUND_MASK_BOTTOM_RATIO > 0.0:
            cut = int(h * (1.0 - GROUND_MASK_BOTTOM_RATIO))
            residual[cut:, :, :] = 0.0
        return residual

    def _apply_zonal_median(self, residual):
        """Subtracts median independently in each of the 3 L/C/R columns."""
        h, w = residual.shape[:2]
        tier = w // 3
        out = residual.copy()
        bounds = [(0, tier), (tier, 2 * tier), (2 * tier, w)]
        for x0, x1 in bounds:
            zone = out[:, x0:x1]
            zone[..., 0] -= np.median(zone[..., 0])
            zone[..., 1] -= np.median(zone[..., 1])
        return out

    def compensate(self, flow_raw, omega_z, dt):
        h, w = flow_raw.shape[:2]
        residual = flow_raw.copy()

        # Geometric rotation compensation
        if ENABLE_ROTATION_COMPENSATION and abs(omega_z) >= 1e-4:
            flow_expected = self.expected_rotational_flow(omega_z, dt, h, w)
            residual = residual - flow_expected

        # Statistical median filtering
        if ENABLE_ZONAL_MEDIAN:
            residual = self._apply_zonal_median(residual)
        elif ENABLE_MEDIAN_FALLBACK:
            residual[..., 0] -= np.median(residual[..., 0])
            residual[..., 1] -= np.median(residual[..., 1])

        # Ground plane mask
        if ENABLE_GROUND_MASK:
            residual = self._apply_ground_mask(residual)

        # Post-RAFT Gaussian blur to reduce interpolation noise
        residual[..., 0] = cv2.GaussianBlur(residual[..., 0], (5, 5), 0)
        residual[..., 1] = cv2.GaussianBlur(residual[..., 1], (5, 5), 0)

        return residual


# =============================================================================
#  ZONAL ANALYSIS
# =============================================================================

class ZoneAnalyzer:
    """
    Splits residual flow into N_ZONES vertical columns (left/center/right)
    and summarizes each zone with magnitude, divergence, and wall detection metrics.
    """

    def __init__(self, n_zones=N_ZONES):
        self.n_zones = n_zones

    def compute_zone_metrics(self, flow_residual):
        h, w = flow_residual.shape[:2]
        u = flow_residual[..., 0]
        v = flow_residual[..., 1]
        mag = np.sqrt(u ** 2 + v ** 2)

        # Local field divergence (du/dx + dv/dy) — signature of radial expansion
        du_dx = np.gradient(u, axis=1)
        dv_dy = np.gradient(v, axis=0)
        divergence = du_dx + dv_dy

        # Global frame coverage (distinguishes inactive wall zone from stationary robot)
        active_mask_full = mag > MOVEMENT_THRESHOLD
        global_coverage = (float(np.count_nonzero(active_mask_full)) / mag.size
                            if mag.size else 0.0)

        zone_w = w // self.n_zones
        metrics = {}
        labels = ZONE_LABELS if self.n_zones == 3 else list(range(self.n_zones))
        for i, label in enumerate(labels):
            x0 = i * zone_w
            x1 = (i + 1) * zone_w if i < self.n_zones - 1 else w
            zone_mag = mag[:, x0:x1]
            zone_div = divergence[:, x0:x1]

            wall_raw = False
            if ENABLE_WALL_DEAD_ZONE_DETECTION:
                zone_active = zone_mag > MOVEMENT_THRESHOLD
                n_active = int(np.count_nonzero(zone_active))
                coverage = float(n_active) / zone_active.size if zone_active.size else 0.0
                inactive_ratio = 1.0 - coverage
                basic_wall = (inactive_ratio > WALL_DEAD_ZONE_RATIO
                               and global_coverage > WALL_MIN_ACTIVE_BORDER)

                if basic_wall:
                    # Low-texture wall produces virtually no flow top or bottom
                    top_rows = max(1, int(zone_mag.shape[0] * WALL_TOP_RATIO))
                    top_zone = zone_mag[:top_rows, :]
                    top_active = float(np.count_nonzero(top_zone > MOVEMENT_THRESHOLD))
                    top_coverage = top_active / max(top_zone.size, 1)

                    bot_rows = max(1, int(zone_mag.shape[0] * WALL_BOTTOM_RATIO))
                    bottom_zone = zone_mag[-bot_rows:, :]
                    bottom_active = float(np.count_nonzero(bottom_zone > MOVEMENT_THRESHOLD))
                    bottom_coverage = bottom_active / max(bottom_zone.size, 1)

                    wall_raw = (top_coverage <= WALL_TOP_MAX_COVERAGE
                                and bottom_coverage <= WALL_BOTTOM_MAX_COVERAGE)

            metrics[label] = {
                'mean_mag': float(np.mean(zone_mag)),
                'max_mag': float(np.max(zone_mag)),
                # Only positive expansion represents an approaching obstacle
                'divergence': float(np.mean(np.clip(zone_div, 0.0, None))),
                'wall_raw': wall_raw,
            }
        return metrics


# =============================================================================
#  RISK ESTIMATION — Temporal Smoothing + Spike Debounce
# =============================================================================

class RiskEstimator:
    """
    Smooths zone metrics over time (EMA on mean + streak debounce on spikes)
    and produces discrete risk evaluations per zone.
    """

    def __init__(self):
        self._smoothed = {}
        self._spike_streak = {}
        self._wall_streak = {}

    def update(self, zone_metrics):
        risk = {}
        for label, m in zone_metrics.items():
            prev = self._smoothed.get(label, {'mag': 0.0, 'div': 0.0})
            mag = FLOW_SMOOTH_ALPHA * m['mean_mag'] + (1.0 - FLOW_SMOOTH_ALPHA) * prev['mag']
            div = FLOW_SMOOTH_ALPHA * m['divergence'] + (1.0 - FLOW_SMOOTH_ALPHA) * prev['div']
            self._smoothed[label] = {'mag': mag, 'div': div}

            streak = self._spike_streak.get(label, 0)
            streak = streak + 1 if m['max_mag'] >= FLOW_SPIKE_THRESHOLD else 0
            self._spike_streak[label] = streak
            confirmed_spike = streak >= SPIKE_STREAK_REQUIRED

            # Wall detection debounce
            wall_streak = self._wall_streak.get(label, 0)
            wall_streak = wall_streak + 1 if m.get('wall_raw', False) else 0
            self._wall_streak[label] = wall_streak
            wall_active = (ENABLE_WALL_DEAD_ZONE_DETECTION
                            and wall_streak >= WALL_STREAK_REQUIRED)

            danger = (mag >= FLOW_CENTER_DANGER) or (div >= DIVERGENCE_DANGER) or confirmed_spike or wall_active
            warning = (mag >= FLOW_CENTER_WARN) or (div >= DIVERGENCE_WARN) or wall_active

            # Approximate TTC from divergence: TTC ~= 1 / expansion_rate
            ttc = (1.0 / div) if div > 1e-4 else float('inf')

            risk[label] = {
                'mag': mag, 'div': div,
                'danger': danger, 'warning': warning,
                'spike': confirmed_spike, 'ttc': ttc,
                'wall': wall_active,
            }
        return risk


# =============================================================================
#  RISK COMBINATION + ZONE OCCUPANCY
# =============================================================================

def combine_zone_risk(risk):
    """Combines zonal risk flags into system-level danger/warning states.
    A purely central danger with open sides initiates avoidance rather than
    an immediate hard emergency stop, unless both escape sides are blocked.

    Returns (danger, warning, left_blocked, right_blocked).
    """
    left_blocked = risk['left']['danger']
    right_blocked = risk['right']['danger']
    center_danger = risk['center']['danger']

    if center_danger:
        danger = left_blocked and right_blocked   # No escape route -> emergency stop
    else:
        # Pure lateral danger does not block forward progress directly
        danger = False

    warning = danger or any(r['warning'] for r in risk.values())
    return danger, warning, left_blocked, right_blocked


def _zone_severity(zr):
    """0 = CLEAR, 1 = WARNING, 2 = EMERGENCY."""
    if zr['danger']:
        return 2
    elif zr['warning']:
        return 1
    return 0


def zone_occupancy_side(risk):
    """Determines steering side based on occupancy patterns across the 3 zones:
      - 1 zone occupied  -> Steer away (center occupied -> steer toward less loaded side)
      - 2 zones occupied -> Steer toward the single open zone
      - 0 or 3 occupied  -> None (3 handled by deceleration/stop)

    Returns (turn_side, max_severity) or (None, 0).
    """
    sev = {name: _zone_severity(risk[name]) for name in ('left', 'center', 'right')}
    blocked = {name: (s > 0) for name, s in sev.items()}
    n_blocked = sum(blocked.values())
    if n_blocked == 0 or n_blocked == 3:
        return None, 0

    if n_blocked == 1:
        zone = next(name for name, b in blocked.items() if b)
        max_sev = sev[zone]
        if zone == 'left':
            return 'right', max_sev
        elif zone == 'right':
            return 'left', max_sev
        else:  # Center occupied -> steer toward side with lower flow magnitude
            side = 'left' if risk['left']['mag'] <= risk['right']['mag'] else 'right'
            return side, max_sev
    else:  # n_blocked == 2
        free = next(name for name, b in blocked.items() if not b)
        max_sev = max(s for name, s in sev.items() if name != free)
        if free == 'left':
            return 'left', max_sev
        elif free == 'right':
            return 'right', max_sev
        else:
            return None, 0   # Center free, L+R blocked -> straight ahead


# =============================================================================
#  SIDE SELECTION — Hysteresis
# =============================================================================

class SideSelector:
    """
    Side hysteresis filter: requires streak_required consecutive cycles
    before flipping dodge direction, preventing high-frequency chattering.
    """

    def __init__(self, streak_required=SIDE_FLIP_STREAK):
        self.streak_required = streak_required
        self._last_side = None
        self._streak = 0

    def resolve(self, candidate):
        """Generic hysteresis resolving candidate side ('left', 'right', or None)."""
        if candidate is None:
            self._last_side = None
            self._streak = 0
            return None

        if self._last_side is None or candidate == self._last_side:
            self._last_side = candidate
            self._streak = 0
            return candidate

        self._streak += 1
        if self._streak >= self.streak_required:
            self._last_side = candidate
            self._streak = 0
            return candidate
        return self._last_side

    def select(self, risk_left, risk_right):
        if abs(risk_right - risk_left) < FLOW_DIFF_AVOID:
            candidate = None
        else:
            candidate = 'left' if risk_right > risk_left else 'right'
        return self.resolve(candidate)


# =============================================================================
#  COMMAND SMOOTHING
# =============================================================================

class CommandSmoother:
    """
    Adaptive EMA smoothing for published avoidance velocity commands.
    """

    def __init__(self):
        self.linear = 0.0
        self.angular = 0.0

    def update(self, target_linear, target_angular, urgency=0.0):
        alpha = AVOID_SMOOTH_ALPHA + (AVOID_SMOOTH_ALPHA_URGENT - AVOID_SMOOTH_ALPHA) * urgency
        self.linear = alpha * target_linear + (1.0 - alpha) * self.linear
        self.angular = alpha * target_angular + (1.0 - alpha) * self.angular
        return self.linear, self.angular

    def reset(self):
        self.linear = 0.0
        self.angular = 0.0


# =============================================================================
#  ANTI-SPIN GUARD — Unwrapped Cumulative Yaw Deviation
# =============================================================================

class YawSpinGuard:
    """
    Detects circular spin entrapment via unwrapped cumulative yaw integration.
    """

    def __init__(self, max_deviation=MAX_AVOIDANCE_YAW_DEVIATION):
        self.max_deviation = max_deviation
        self._cum_yaw = 0.0
        self._prev_yaw = None
        self._was_active_prev_cycle = False

    def update(self, current_yaw, was_avoid_active_last_cycle):
        if self._was_active_prev_cycle and self._prev_yaw is not None:
            raw_delta = current_yaw - self._prev_yaw
            wrapped = float(np.arctan2(np.sin(raw_delta), np.cos(raw_delta)))
            self._cum_yaw += wrapped
        else:
            self._cum_yaw = 0.0
        self._prev_yaw = current_yaw
        self._was_active_prev_cycle = was_avoid_active_last_cycle
        return abs(self._cum_yaw) >= self.max_deviation

    @property
    def cumulative_yaw(self):
        return self._cum_yaw


# =============================================================================
#  ROBOT STATES
# =============================================================================

class RobotState:
    CLEAR = 'CLEAR'
    PREDICTING = 'PREDICTING'
    RECOVERY = 'RECOVERY'
    WARNING = 'WARNING'
    EMERGENCY = 'EMERGENCY'


# =============================================================================
#  COMMAND HELPERS
# =============================================================================

def generate_avoidance_cmd(side, angular_mag=AVOID_ANGULAR_BASE):
    """angular_z > 0 = left (standard ROS REP-103 convention)."""
    return AVOID_LINEAR_X, (+angular_mag if side == 'left' else -angular_mag)
