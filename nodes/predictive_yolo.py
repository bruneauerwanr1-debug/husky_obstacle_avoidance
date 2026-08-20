"""
emergency_stop_node.py — Emergency Stop + Predictive Avoidance + Trajectory Recovery
=====================================================================================
Expected twist_mux architecture:

  joy (priority 10)            → /joy_teleop/cmd_vel
  avoidance (priority 50)      → /avoidance_cmd_vel  ← this node in PREDICTING / RECOVERY state
  lock vision_e_stop (prio 255)→ /emergency_stop     ← this node publishes True/False

  twist_mux (output) → /cmd_vel_in → [this node filters] → /cmd_vel → motors

State Machine (decreasing priority):

  EMERGENCY  : obstacle < DISTANCE_DANGER_M  OR  high central optical flow
               → complete stop (reversing ALWAYS allowed)

  PREDICTING : predicted TTC in ]0, TTC_HORIZON] seconds  OR  static obstacle in corridor
               → avoidance command published on /avoidance_cmd_vel (priority 50 twist_mux)
               → ALSO triggered by optical flow alone if no YOLO detection

  RECOVERY   : avoidance maneuver complete + user commanded forward motion
               → proportional heading correction (turn + forward motion)

  WARNING    : obstacle in [DISTANCE_DANGER_M, DISTANCE_WARN_M]
               → proportional EMA deceleration

  CLEAR      : clear path → joystick command passed through unchanged

Robot Speed:
  Primary source  : /odometry/filtered (ACTUAL speed from encoder + EKF yaw)
  Fallback source : /cmd_vel_in if odom unavailable (timeout 0.5s)

Optical Flow Avoidance (YOLO-independent):
  - High central zone (>= FLOW_CENTER_DANGER) → EMERGENCY
  - Moderate central zone (>= FLOW_CENTER_WARN) → WARNING EMA
  - L/R Imbalance (|flowR - flowL| > FLOW_DIFF_AVOID) → Reflex PREDICTING
    Robot turns toward the side with LOWER flow (clear side), then returns
    to original trajectory via RECOVERY.
"""

import json
import os
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64MultiArray, String
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — ALL TUNABLE PARAMETERS HERE
# ══════════════════════════════════════════════════════════════════════════════

# ── YOLO Model ────────────────────────────────────────────────────────────────
MODEL_VERSION  = 'yolo26m.pt'
CONF_THRESHOLD = 0.5
DEVICE         = 0
INFER_SIZE     = (640, 480)

# ── Distance Thresholds ───────────────────────────────────────────────────────
DISTANCE_WARN_M   = 1.5   # m → Deceleration (warning)
DISTANCE_DANGER_M = 0.8   # m → Immediate stop (danger)

# ── Zonal Optical Flow (Bio-inspired) ─────────────────────────────────────────
# Central zone = middle 1/3 of image width
FLOW_CENTER_DANGER = 8.0   # px/frame → EMERGENCY (non-YOLO frontal obstacle)
FLOW_CENTER_WARN   = 4.0   # px/frame → WARNING EMA
# SPIKE threshold (max per zone), capturing thin/low-texture objects (e.g. poles)
FLOW_SPIKE_THRESHOLD = 14.0   # px/frame on MAX (not average) of a zone
# L/R Imbalance: if |flowR - flowL| > threshold → reflex avoidance
FLOW_DIFF_AVOID    = 1.5   # px/frame
# Reflex angular gain (rad/s per px imbalance)
FLOW_BALANCE_K     = 0.06
FLOW_MAX_TURN_RAD  = 0.45  # Saturation for reflex turn (rad/s)
# EMA smoothing of raw flow BEFORE thresholding
FLOW_SMOOTH_ALPHA  = 0.30
# Debounce streak for SPIKE signal (flow_*_max)
FLOW_SPIKE_STREAK_REQUIRED = 2   # ≈0.2s at 10Hz

# ── Lucas-Kanade ──────────────────────────────────────────────────────────────
FLOW_MAX_CORNERS = 150
FLOW_QUALITY     = 0.01
FLOW_MIN_DIST    = 10
FLOW_WIN_SIZE    = (21, 21)
FLOW_MAX_LEVEL   = 3

# ── State Machine Timing ──────────────────────────────────────────────────────
STOP_RELEASE_CYCLES = 15    # Clear cycles before releasing stop (≈1.5s at 10Hz)
TIMER_PERIOD        = 0.1   # s → 10 Hz (vision / decision)
WATCHDOG_PERIOD      = 0.05  # s → 20 Hz (motor safety watchdog)
EMA_ALPHA           = 0.15  # Deceleration smoothing (≈0.6s time constant)

# ── TTC / Velocity Obstacle ───────────────────────────────────────────────────
TTC_HORIZON      = 3.0    # s: if TTC < horizon → PREDICTING
TTC_MIN_SPEED    = 0.05   # m/s: valid movement threshold
ROBOT_HALF_WIDTH = 0.40   # m: Husky half-width + safety margin
HUMAN_SAFETY_MARGIN = 0.25  # m: Additional margin for 'person'

# ── Deceleration Corridor ─────────────────────────────────────────────────────
# Lateral bounding corridor for WARNING state
WARNING_CORRIDOR_HALF_WIDTH = ROBOT_HALF_WIDTH + 0.4   # m

# ── YOLO Avoidance Commands ───────────────────────────────────────────────────
AVOID_LINEAR_X  = 0.20   # m/s forward during avoidance
AVOID_ANGULAR_Z = 0.35   # rad/s base YOLO turn rate
AVOID_ANGULAR_MAX = 0.60   # rad/s max turn rate for frontal/urgent obstacles

# ── Anti-Chatter / Smoothing ──────────────────────────────────────────────────
SIDE_FLIP_STREAK = 3      # Consecutive cycles required to flip dodge side
AVOID_SMOOTH_ALPHA = 0.35 # Standard EMA smoothing
AVOID_SMOOTH_ALPHA_URGENT = 0.70 # Fast EMA smoothing for urgent encounters

# ── Heading Recovery (RECOVERY) ───────────────────────────────────────────────
YAW_RECOVERY_KP  = 0.6    # Proportional gain for returning to initial heading
YAW_RECOVERY_MIN = 0.10   # Angular error threshold before recovery triggers (rad)
RECOVERY_LINEAR_X = 0.15  # m/s forward speed during recovery
RECOVERY_MAX_TURN = 0.35  # rad/s turn rate saturation

# ── Cumulative Yaw Deviation Limit ────────────────────────────────────────────
MAX_AVOIDANCE_YAW_DEVIATION = 1.30   # rad ≈ 75° (prevents infinite orbital circling)

# ── Robot Speed Timeout ───────────────────────────────────────────────────────
ROBOT_SPEED_TIMEOUT = 0.5  # s: if no odom update → speed = 0

# ── Command Source Timeout ────────────────────────────────────────────────────
CMD_SOURCE_TIMEOUT = 0.3   # s

# ── Homography Calibration ────────────────────────────────────────────────────
CALIB_W     = 300
CALIB_H     = 250
CALIB_SCALE = 0.01

# ── Bird-Eye View ─────────────────────────────────────────────────────────────
MAP_W        = 400
MAP_H        = 400
FLOW_PANEL_H = 92     # Height of flow panel stacked below bird-eye view
MAP_SCALE    = 0.01
ROB_OFFSET_Y = 50

# ── Obstacle Memory ───────────────────────────────────────────────────────────
DYNAMIC_CLASSES   = {'person'}
OBSTACLE_MEMORY_S = 15.0   # s: Retention duration for static obstacles

# ── Velocity Tracker Grace Period ────────────────────────────────────────────
VELOCITY_GRACE_S = 0.3     # s: ≈3 cycles at 10Hz

# ── Track Re-Association ──────────────────────────────────────────────────────
REASSOC_WINDOW_S   = 0.5   # s: Window to preserve recently lost track
REASSOC_MAX_DIST_M = 0.6   # m: Max distance to match lost track with new ID

# ── Homography Persistence Path ───────────────────────────────────────────────
H_SAVE_PATH = '/home/imr2204/Desktop/Erwan/clearpath_simulator/src/homography_calibration.json'

# ── ROS Topics ────────────────────────────────────────────────────────────────
TOPIC_COLOR     = '/camera/color/image_raw'
TOPIC_H         = '/homography_matrix'
TOPIC_ROBOT_POS = '/robot_ground_position'
TOPIC_CMD_IN    = '/cmd_vel_in'
TOPIC_ODOM      = '/odometry/filtered'
TOPIC_JOY       = '/joy_teleop/cmd_vel'

# ── OpenCV Colors (BGR) ───────────────────────────────────────────────────────
C_OK       = (0,   255,   0)
C_YOLO     = (255, 200,   0)
C_WARNING  = (0,   165, 255)
C_DANGER   = (0,     0, 255)
C_MEMORY   = (180,  80, 255)
C_PREDICT  = (255, 200,   0)
C_RECOVERY = (0,   255, 255)   # Cyan → Heading recovery

ENABLE_DISPLAY = os.environ.get('DISPLAY') is not None


# ══════════════════════════════════════════════════════════════════════════════
#  HOMOGRAPHY PERSISTENCE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def load_homography(path: str):
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
#  TTC AND VO FUNCTIONS — Pure
# ══════════════════════════════════════════════════════════════════════════════

def compute_ttc(obs_x, obs_y, obs_vx, obs_vy,
                rob_x, rob_y, rob_vx, rob_vy,
                safety_radius=ROBOT_HALF_WIDTH):
    """
    Time-To-Collision computed via projection of relative velocity onto robot-obstacle vector.
      dist <= safety_radius → 0.0 (already in collision)
      v_approach <= TTC_MIN_SPEED → inf (moving away or stationary)
    """
    dx   = obs_x - rob_x
    dy   = obs_y - rob_y
    dist = float(np.sqrt(dx**2 + dy**2))

    if dist <= safety_radius:
        return 0.0

    rel_vx = obs_vx - rob_vx
    rel_vy = obs_vy - rob_vy
    v_approach = -(dx * rel_vx + dy * rel_vy) / dist

    if v_approach < TTC_MIN_SPEED:
        return float('inf')

    return max(0.0, (dist - safety_radius) / v_approach)


def compute_vo_side(obs_x, obs_y, obs_vx, obs_vy,
                    rob_x, rob_y, rob_vx=0.0, rob_vy=0.0):
    """
    Dodge side determination via cross product (Simplified Velocity Obstacle).
    Convention: y_m > 0 = Robot Right, y_m < 0 = Robot Left (camera homography frame).
    Returns 'left' or 'right'.
    """
    dx = obs_x - rob_x
    dy = obs_y - rob_y

    obs_speed = float(np.sqrt(obs_vx**2 + obs_vy**2))
    if obs_speed > TTC_MIN_SPEED:
        cross = dx * obs_vy - dy * obs_vx
        if cross > 0.05:
            return 'left'
        elif cross < -0.05:
            return 'right'

    rob_speed = float(np.sqrt(rob_vx**2 + rob_vy**2))
    if rob_speed > TTC_MIN_SPEED:
        cross_static = rob_vx * dy - rob_vy * dx
        if cross_static > 0.05:
            return 'left'
        elif cross_static < -0.05:
            return 'right'

    return 'left' if dy > 0 else 'right'


def generate_avoidance_cmd(side, angular_z=AVOID_ANGULAR_Z):
    """Generates avoidance Twist message. angular.z > 0 = left (ROS standard)."""
    cmd = Twist()
    cmd.linear.x  = AVOID_LINEAR_X
    cmd.angular.z = +angular_z if side == 'left' else -angular_z
    return cmd


def project_to_current_frame(x_m, y_m, cap_rx, cap_ry, cap_yaw,
                              rx_now, ry_now, yaw_now):
    """
    Recomputes relative position (x_m, y_m) of a MEMORIZED obstacle to account
    for robot motion since the last live detection.
    """
    # 1. World position at capture time
    world_x = cap_rx + np.cos(cap_yaw) * x_m + np.sin(cap_yaw) * y_m
    world_y = cap_ry + np.sin(cap_yaw) * x_m - np.cos(cap_yaw) * y_m

    # 2. Reprojection into CURRENT robot frame
    dx_world = world_x - rx_now
    dy_world = world_y - ry_now
    x_m_now  =  np.cos(yaw_now) * dx_world + np.sin(yaw_now) * dy_world
    y_m_now  =  np.sin(yaw_now) * dx_world - np.cos(yaw_now) * dy_world

    return float(x_m_now), float(y_m_now)


# ══════════════════════════════════════════════════════════════════════════════
#  MEMORIZED OBSTACLE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class MemorizedObstacle:
    """
    Track structure for tracked obstacles across frames.
    Estimates velocity by filtered differentiation (clamped at 3 m/s).
    """
    __slots__ = ('track_id', 'cls', 'x_m', 'y_m', 'last_seen', 'conf',
                 'vx', 'vy', '_prev_x_m', '_prev_y_m', '_prev_time',
                 'cap_rx', 'cap_ry', 'cap_yaw')

    def __init__(self, track_id, cls, x_m, y_m, conf,
                 cap_rx=0.0, cap_ry=0.0, cap_yaw=0.0):
        self.track_id  = track_id
        self.cls       = cls
        self.x_m       = x_m
        self.y_m       = y_m
        self.last_seen = time.monotonic()
        self.conf      = conf
        self.vx        = 0.0
        self.vy        = 0.0
        self._prev_x_m  = x_m
        self._prev_y_m  = y_m
        self._prev_time = time.monotonic()
        self.cap_rx  = cap_rx
        self.cap_ry  = cap_ry
        self.cap_yaw = cap_yaw

    def update(self, x_m, y_m, conf, cap_rx=None, cap_ry=None, cap_yaw=None):
        now = time.monotonic()
        dt  = now - self._prev_time
        if dt > 0.05:
            raw_vx = (x_m - self._prev_x_m) / dt
            raw_vy = (y_m - self._prev_y_m) / dt
            speed  = float(np.sqrt(raw_vx**2 + raw_vy**2))
            if speed > 3.0:
                scale   = 3.0 / speed
                self.vx = raw_vx * scale
                self.vy = raw_vy * scale
            else:
                self.vx = raw_vx
                self.vy = raw_vy
            self._prev_x_m  = x_m
            self._prev_y_m  = y_m
            self._prev_time = now
        self.x_m       = x_m
        self.y_m       = y_m
        self.conf      = conf
        self.last_seen = now
        if cap_rx is not None:
            self.cap_rx  = cap_rx
            self.cap_ry  = cap_ry
            self.cap_yaw = cap_yaw

    def age(self):
        return time.monotonic() - self.last_seen

    def is_expired(self):
        return self.age() > OBSTACLE_MEMORY_S


# ══════════════════════════════════════════════════════════════════════════════
#  ROBOT STATES
# ══════════════════════════════════════════════════════════════════════════════

class RobotState:
    CLEAR      = 'CLEAR'
    PREDICTING = 'PREDICTING'   # Active avoidance (YOLO or optical flow)
    RECOVERY   = 'RECOVERY'     # Heading recovery back to original path
    WARNING    = 'WARNING'      # EMA Deceleration
    EMERGENCY  = 'EMERGENCY'    # Complete emergency stop


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ROS 2 NODE
# ══════════════════════════════════════════════════════════════════════════════

class EmergencyStopNode(Node):

    def __init__(self):
        super().__init__('emergency_stop_node')

        # ── GPU + YOLO ─────────────────────────────────────────────────
        if not torch.cuda.is_available():
            self.get_logger().error('CUDA not available — running on CPU!')
        else:
            self.get_logger().info(f'GPU: {torch.cuda.get_device_name(DEVICE)}')

        self.get_logger().info(f'Loading {MODEL_VERSION}...')
        self.model = YOLO(MODEL_VERSION)
        self.model.to(f'cuda:{DEVICE}')
        self.model(np.zeros((INFER_SIZE[1], INFER_SIZE[0], 3), dtype=np.uint8),
                   verbose=False, device=DEVICE)
        self.get_logger().info('YOLO ready ✓')

        # ── Camera / Homography ────────────────────────────────────────
        self.bridge        = CvBridge()
        self.latest_frame  = None
        self.H             = None
        self.H_inv         = None
        self._H_calib_w    = 1280
        self._H_calib_h    = 720


        # ── Ground Position ────────────────────────────────────────────
        self.robot_ground_pos = None

        # ── Optical Flow (Lucas-Kanade) ────────────────────────────────
        self.prev_gray = None
        self.prev_pts  = None
        self._flow_l   = 0.0
        self._flow_c   = 0.0
        self._flow_r   = 0.0
        self._flow_l_max = 0.0
        self._flow_c_max = 0.0
        self._flow_r_max = 0.0
        self._flow_c_spike_streak = 0
        self._flow_l_spike_streak = 0
        self._flow_r_spike_streak = 0

        # ── State Machine ──────────────────────────────────────────────
        self.stop_active       = False
        self.clean_cycle_count = 0
        self.current_min_dist  = float('inf')
        self._current_state    = RobotState.CLEAR
        self._smooth_ratio     = 1.0

        # ── Obstacle Tracking Buffers ──────────────────────────────────
        self._obstacle_memory: dict[int, MemorizedObstacle] = {}
        self._velocity_tracker: dict[int, MemorizedObstacle] = {}
        self._recently_lost: list[dict] = []

        # ── Robot Odometry / Velocity ──────────────────────────────────
        self._robot_vx          = 0.0
        self._robot_vy          = 0.0
        self._robot_speed_stamp = 0.0

        # ── Absolute Pose + Yaw ────────────────────────────────────────
        self._robot_pose_x = 0.0
        self._robot_pose_y = 0.0
        self._current_yaw  = 0.0
        self._target_yaw   = 0.0

        # ── User Command Intent (Raw Joystick) ─────────────────────────
        self._user_cmd_x        = 0.0
        self._user_cmd_z        = 0.0
        self._user_cmd_stamp    = 0.0
        self._intent_x          = 0.0
        self._intent_stamp      = 0.0

        # ── TTC / Avoidance ────────────────────────────────────────────
        self._min_ttc       = float('inf')
        self._avoidance_cmd = None

        # ── Anti-Chatter / Smoothing ───────────────────────────────────
        self._last_avoid_side      = None
        self._side_streak          = 0
        self._avoid_linear_smooth  = 0.0
        self._avoid_angular_smooth = 0.0

        # ── Cumulative Yaw Anti-Spin Guard ────────────────────────────
        self._avoid_cum_yaw          = 0.0
        self._avoid_prev_yaw_for_cum = 0.0
        self._avoid_was_active       = False

        # ── Watchdog Stamp ────────────────────────────────────────────
        self._last_cmd_in_stamp = 0.0

        self._try_load_H_from_file()

        # ── QoS Profiles ──────────────────────────────────────────────
        qos_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        # ── Subscriptions ──────────────────────────────────────────────
        self.create_subscription(Image,             TOPIC_COLOR,     self.cb_image,      qos_img)
        self.create_subscription(Float64MultiArray, TOPIC_H,         self.cb_H,          qos_rel)
        self.create_subscription(Point,             TOPIC_ROBOT_POS, self.cb_robot_pos,  qos_rel)
        self.create_subscription(Odometry,          TOPIC_ODOM,      self.cb_odom,       qos_rel)
        self.create_subscription(Twist,             TOPIC_CMD_IN,    self.cb_cmd_vel_in, 10)
        self.create_subscription(Twist,             TOPIC_JOY,       self.cb_joy,        10)

        # ── Publishers ─────────────────────────────────────────────────
        self.pub_estop  = self.create_publisher(Bool,   '/emergency_stop',    10)
        self.pub_cmd    = self.create_publisher(Twist,  '/cmd_vel',           10)
        self.pub_avoid  = self.create_publisher(Twist,  '/avoidance_cmd_vel', 10)
        self.pub_debug  = self.create_publisher(Image,  '/yolo/image',        10)
        self.pub_bird   = self.create_publisher(Image,  '/bird_eye_estop',    10)
        self.pub_ground = self.create_publisher(String, '/ground_detections', 10)
        self.pub_diag   = self.create_publisher(String, '/flow_diagnostics', 10)

        self.create_timer(TIMER_PERIOD, self.run_inference)
        self.create_timer(WATCHDOG_PERIOD, self._cmd_watchdog)

        self.get_logger().info(
            'Emergency stop node ready.\n'
            f'  TTC horizon      : {TTC_HORIZON}s\n'
            f'  Dist danger      : {DISTANCE_DANGER_M}m\n'
            f'  Dist warning     : {DISTANCE_WARN_M}m\n'
            f'  Avoid ang (YOLO) : ±{AVOID_ANGULAR_Z}rad/s\n'
            f'  Avoid ang (flow) : ±{FLOW_MAX_TURN_RAD}rad/s max\n'
            f'  Recovery KP      : {YAW_RECOVERY_KP}\n'
            f'  Odom topic       : {TOPIC_ODOM}\n'
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  CALLBACKS
    # ══════════════════════════════════════════════════════════════════════════

    def _try_load_H_from_file(self):
        H, meta = load_homography(H_SAVE_PATH)
        if H is None:
            return
        self.H     = H
        self.H_inv = np.linalg.inv(H)
        self._H_calib_w = meta.get('calibrated_image_w', 1280) if meta else 1280
        self._H_calib_h = meta.get('calibrated_image_h', 720) if meta else 720
        self.get_logger().info(f'Homography loaded from file ✓ (calibrated for {self._H_calib_w}x{self._H_calib_h})')

    def cb_image(self, msg: Image):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def cb_H(self, msg: Float64MultiArray):
        H = np.array(msg.data, dtype=np.float64).reshape(3, 3)
        try:
            self.H     = H
            self.H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            self.H = self.H_inv = None

    def cb_robot_pos(self, msg: Point):
        self.robot_ground_pos = (msg.x, msg.y)

    def cb_odom(self, msg: Odometry):
        self._robot_vx          = msg.twist.twist.linear.x
        self._robot_vy          = msg.twist.twist.linear.y
        self._robot_speed_stamp = time.monotonic()
        self._robot_pose_x      = msg.pose.pose.position.x
        self._robot_pose_y      = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._current_yaw = float(np.arctan2(siny, cosy))

    def cb_joy(self, msg: Twist):
        self._user_cmd_x     = msg.linear.x
        self._user_cmd_z     = msg.angular.z
        self._user_cmd_stamp = time.monotonic()

    def cb_cmd_vel_in(self, msg: Twist):
        self._intent_x          = msg.linear.x
        self._intent_stamp      = time.monotonic()
        self._last_cmd_in_stamp = time.monotonic()

        if time.monotonic() - self._robot_speed_stamp > ROBOT_SPEED_TIMEOUT:
            self._robot_vx          = msg.linear.x
            self._robot_vy          = 0.0
            self._robot_speed_stamp = time.monotonic()

        # Reversing commands always pass through and release E-stop
        if msg.linear.x < -0.01:
            if self.stop_active:
                self.stop_active       = False
                self.clean_cycle_count = 0
                self._release_stop()
            self.pub_cmd.publish(msg)
            return

        if self.stop_active:
            self.pub_cmd.publish(Twist())
            return

        if self._current_state == RobotState.WARNING and self._smooth_ratio < 1.0:
            safe = Twist()
            safe.linear.x  = msg.linear.x  * self._smooth_ratio
            safe.angular.z = msg.angular.z * self._smooth_ratio
            self.pub_cmd.publish(safe)
            return

        self.pub_cmd.publish(msg)

    # ══════════════════════════════════════════════════════════════════════════
    #  OBSTACLE TRACKING & MEMORY
    # ══════════════════════════════════════════════════════════════════════════

    def _update_track_velocity(self, track_id, cls, x_m, y_m, conf):
        if track_id == -1:
            return
        cap_rx, cap_ry, cap_yaw = self._robot_pose_x, self._robot_pose_y, self._current_yaw
        if track_id in self._velocity_tracker:
            self._velocity_tracker[track_id].update(
                x_m, y_m, conf, cap_rx=cap_rx, cap_ry=cap_ry, cap_yaw=cap_yaw)
            return

        # Attempt re-association with recently lost track
        inherited_vx, inherited_vy = 0.0, 0.0
        best_dist = REASSOC_MAX_DIST_M
        best_idx  = None
        for i, lost in enumerate(self._recently_lost):
            if lost['cls'] != cls:
                continue
            d = float(np.hypot(x_m - lost['x_m'], y_m - lost['y_m']))
            if d <= best_dist:
                best_dist = d
                best_idx  = i

        if best_idx is not None:
            lost = self._recently_lost.pop(best_idx)
            inherited_vx, inherited_vy = lost['vx'], lost['vy']

        new_obs = MemorizedObstacle(track_id, cls, x_m, y_m, conf,
                                    cap_rx=cap_rx, cap_ry=cap_ry, cap_yaw=cap_yaw)
        new_obs.vx, new_obs.vy = inherited_vx, inherited_vy
        self._velocity_tracker[track_id] = new_obs

    def _get_grace_detections(self, active_ids):
        rx_now, ry_now, yaw_now = self._robot_pose_x, self._robot_pose_y, self._current_yaw
        out = []
        for tid, obs in self._velocity_tracker.items():
            if tid in active_ids or obs.cls not in DYNAMIC_CLASSES:
                continue
            x_proj, y_proj = project_to_current_frame(
                obs.x_m, obs.y_m, obs.cap_rx, obs.cap_ry, obs.cap_yaw,
                rx_now, ry_now, yaw_now)
            dist_m = self._dist_from_robot(x_proj, y_proj)
            out.append({
                'id': tid, 'class': obs.cls, 'conf': obs.conf,
                'x_m': round(x_proj, 2), 'y_m': round(y_proj, 2),
                'dist_m': round(dist_m, 2),
                'vx': obs.vx, 'vy': obs.vy,
                'memorized': True, 'age_s': round(obs.age(), 1),
            })
        return out

    def _purge_velocity_tracker(self, active_ids):
        now   = time.monotonic()
        stale = [tid for tid, obs in self._velocity_tracker.items()
                 if tid not in active_ids and obs.age() > VELOCITY_GRACE_S]
        for tid in stale:
            obs = self._velocity_tracker.pop(tid)
            self._recently_lost.append({
                'cls': obs.cls, 'x_m': obs.x_m, 'y_m': obs.y_m,
                'vx': obs.vx, 'vy': obs.vy, 'lost_time': now,
            })

        self._recently_lost = [
            l for l in self._recently_lost
            if (now - l['lost_time']) <= REASSOC_WINDOW_S
        ]

    def _update_memory(self, track_id, cls, x_m, y_m, conf):
        if cls in DYNAMIC_CLASSES or track_id == -1:
            return
        cap_rx, cap_ry, cap_yaw = self._robot_pose_x, self._robot_pose_y, self._current_yaw
        if track_id in self._obstacle_memory:
            self._obstacle_memory[track_id].update(
                x_m, y_m, conf, cap_rx=cap_rx, cap_ry=cap_ry, cap_yaw=cap_yaw)
        else:
            self._obstacle_memory[track_id] = MemorizedObstacle(
                track_id, cls, x_m, y_m, conf,
                cap_rx=cap_rx, cap_ry=cap_ry, cap_yaw=cap_yaw)

    def _purge_memory(self):
        expired = [tid for tid, o in self._obstacle_memory.items() if o.is_expired()]
        for tid in expired:
            self._obstacle_memory.pop(tid)

    def _get_memory_detections(self, active_ids):
        rx_now   = self._robot_pose_x
        ry_now   = self._robot_pose_y
        yaw_now  = self._current_yaw

        out = []
        for tid, obs in self._obstacle_memory.items():
            if tid in active_ids:
                continue
            x_proj, y_proj = project_to_current_frame(
                obs.x_m, obs.y_m, obs.cap_rx, obs.cap_ry, obs.cap_yaw,
                rx_now, ry_now, yaw_now)
            dist_m = self._dist_from_robot(x_proj, y_proj)
            out.append({
                'id': tid, 'class': obs.cls, 'conf': obs.conf,
                'x_m': round(x_proj, 2), 'y_m': round(y_proj, 2),
                'dist_m': round(dist_m, 2),
                'vx': obs.vx, 'vy': obs.vy,
                'memorized': True, 'age_s': round(obs.age(), 1),
            })
        return out

    # ══════════════════════════════════════════════════════════════════════════
    #  ZONAL OPTICAL FLOW (Bio-inspired)
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_zonal_optical_flow(self, gray_curr):
        if self.prev_gray is None:
            self.prev_gray = gray_curr.copy()
            self.prev_pts  = cv2.goodFeaturesToTrack(
                gray_curr, maxCorners=FLOW_MAX_CORNERS,
                qualityLevel=FLOW_QUALITY, minDistance=FLOW_MIN_DIST)
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, []

        if self.prev_pts is None or len(self.prev_pts) < 20:
            self.prev_pts = cv2.goodFeaturesToTrack(
                self.prev_gray, maxCorners=FLOW_MAX_CORNERS,
                qualityLevel=FLOW_QUALITY, minDistance=FLOW_MIN_DIST)
            if self.prev_pts is None:
                self.prev_gray = gray_curr.copy()
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, []

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray_curr, self.prev_pts, None,
            winSize=FLOW_WIN_SIZE, maxLevel=FLOW_MAX_LEVEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        if curr_pts is None or status is None:
            self.prev_gray = gray_curr.copy()
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, []

        ok        = status.flatten() == 1
        good_prev = self.prev_pts[ok].reshape(-1, 2)
        good_curr = curr_pts[ok].reshape(-1, 2)

        if len(good_prev) == 0:
            self.prev_gray = gray_curr.copy()
            self.prev_pts  = None
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, []

        dx = good_curr[:, 0] - good_prev[:, 0]
        dy = good_curr[:, 1] - good_prev[:, 1]
        dx_rel = dx - np.median(dx)
        dy_rel = dy - np.median(dy)
        mags   = np.sqrt(dx_rel**2 + dy_rel**2)

        w       = gray_curr.shape[1]
        pts_x   = good_curr[:, 0]
        mask_l  = pts_x < (w / 3.0)
        mask_r  = pts_x > (2.0 * w / 3.0)
        mask_c  = ~(mask_l | mask_r)

        flow_l = float(np.mean(mags[mask_l])) if np.any(mask_l) else 0.0
        flow_c = float(np.mean(mags[mask_c])) if np.any(mask_c) else 0.0
        flow_r = float(np.mean(mags[mask_r])) if np.any(mask_r) else 0.0

        flow_l_max = float(np.max(mags[mask_l])) if np.any(mask_l) else 0.0
        flow_c_max = float(np.max(mags[mask_c])) if np.any(mask_c) else 0.0
        flow_r_max = float(np.max(mags[mask_r])) if np.any(mask_r) else 0.0

        vectors = [(tuple(good_prev[i].astype(int)),
                    tuple(good_curr[i].astype(int)),
                    mags[i]) for i in range(len(good_prev))]

        self.prev_gray = gray_curr.copy()
        self.prev_pts  = good_curr.reshape(-1, 1, 2)
        return flow_l, flow_c, flow_r, flow_l_max, flow_c_max, flow_r_max, vectors

    # ══════════════════════════════════════════════════════════════════════════
    #  GROUND GEOMETRY / PROJECTIONS
    # ══════════════════════════════════════════════════════════════════════════

    def _cam_px_to_ground_m(self, u, v):
        if self.H is None:
            return None

        # Adaptation automatique si la résolution runtime diffère de la résolution de calibration
        if self.latest_frame is not None and getattr(self, '_H_calib_w', None) and getattr(self, '_H_calib_h', None):
            h_curr, w_curr = self.latest_frame.shape[:2]
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

    def _ground_m_to_bird_px(self, x_m, y_m):
        bx = int(MAP_W / 2.0 + y_m / MAP_SCALE)
        by = int(MAP_H - ROB_OFFSET_Y - x_m / MAP_SCALE)
        return bx, by

    def _dist_from_robot(self, x_m, y_m):
        rx, ry = self.robot_ground_pos if self.robot_ground_pos else (0.0, 0.0)
        return float(np.sqrt((x_m - rx)**2 + (y_m - ry)**2))

    # ══════════════════════════════════════════════════════════════════════════
    #  BIRD-EYE VIEW
    # ══════════════════════════════════════════════════════════════════════════

    def _build_bird_eye(self, ground_dets):
        bird = np.full((MAP_H, MAP_W, 3), (30, 30, 30), dtype=np.uint8)

        for step, color in [(int(0.5 / MAP_SCALE), (55, 55, 55)),
                             (int(1.0 / MAP_SCALE), (75, 75, 75))]:
            for x in range(0, MAP_W, step):
                cv2.line(bird, (x, 0), (x, MAP_H), color, 1)
            for y in range(0, MAP_H, step):
                cv2.line(bird, (0, y), (MAP_W, y), color, 1)

        rx_m, ry_m = self.robot_ground_pos if self.robot_ground_pos else (0.0, 0.0)
        rbx, rby   = self._ground_m_to_bird_px(rx_m, ry_m)
        rob_in_view = 0 <= rbx < MAP_W and 0 <= rby < MAP_H

        if rob_in_view:
            cv2.circle(bird, (rbx, rby), int(DISTANCE_WARN_M   / MAP_SCALE), C_WARNING, 1, cv2.LINE_AA)
            cv2.circle(bird, (rbx, rby), int(DISTANCE_DANGER_M / MAP_SCALE), C_DANGER,  1, cv2.LINE_AA)
            r_pred = int((abs(self._robot_vx) * TTC_HORIZON + ROBOT_HALF_WIDTH) / MAP_SCALE)
            cv2.circle(bird, (rbx, rby), r_pred, C_PREDICT, 1, cv2.LINE_AA)
            tri = np.array([(rbx, rby - 12), (rbx - 8, rby + 8), (rbx + 8, rby + 8)], np.int32)
            cv2.fillPoly(bird, [tri], (0, 255, 255))

        rob_x, rob_y = self.robot_ground_pos if self.robot_ground_pos else (0.0, 0.0)
        for det in ground_dets:
            x_m, y_m = det['x_m'], det['y_m']
            dist_m   = det['dist_m']
            vx, vy   = det.get('vx', 0.0), det.get('vy', 0.0)
            memorized = det.get('memorized', False)
            t_id     = det.get('id', -1)
            age_s    = det.get('age_s', 0.0)

            ttc_val = compute_ttc(x_m, y_m, vx, vy,
                                  rob_x, rob_y, self._robot_vx, self._robot_vy)
            bx, by  = self._ground_m_to_bird_px(x_m, y_m)
            if not (0 <= bx < MAP_W and 0 <= by < MAP_H):
                continue

            if dist_m <= DISTANCE_DANGER_M:
                color = C_DANGER
            elif dist_m <= DISTANCE_WARN_M:
                color = C_WARNING
            elif ttc_val <= TTC_HORIZON:
                color = C_PREDICT
            else:
                color = C_MEMORY if memorized else C_YOLO

            if memorized:
                cv2.circle(bird, (bx, by), 8, color, 1)
                cv2.circle(bird, (bx, by), 4, color, 1)
                ttc_str = f' T:{ttc_val:.1f}s' if ttc_val < TTC_HORIZON else ''
                lbl = f'#{t_id} {det["class"]} {dist_m:.1f}m [{OBSTACLE_MEMORY_S - age_s:.1f}s]{ttc_str}'
            else:
                cv2.circle(bird, (bx, by), 8, color, -1)
                cv2.circle(bird, (bx, by), 8, (255, 255, 255), 1)
                ttc_str = f' T:{ttc_val:.1f}s' if ttc_val < TTC_HORIZON else ''
                lbl = f'#{t_id} {det["class"]} {dist_m:.1f}m{ttc_str}'

            cv2.putText(bird, lbl, (bx + 10, by + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1)

            spd = float(np.sqrt(vx**2 + vy**2))
            if spd > TTC_MIN_SPEED:
                cv2.arrowedLine(bird, (bx, by),
                                (int(bx + vy * 20.0), int(by - vx * 20.0)),
                                color, 2, tipLength=0.3)
            if rob_in_view:
                cv2.line(bird, (rbx, rby), (bx, by), color, 1, cv2.LINE_AA)

        # HUD Text
        state_color = {
            RobotState.CLEAR:      C_OK,
            RobotState.PREDICTING: C_PREDICT,
            RobotState.RECOVERY:   C_RECOVERY,
            RobotState.WARNING:    C_WARNING,
            RobotState.EMERGENCY:  C_DANGER,
        }.get(self._current_state, C_OK)

        cv2.putText(bird, f'State: {self._current_state}',
                    (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, state_color, 1)
        cv2.putText(bird, f'vRobot: {self._robot_vx:.2f}m/s  Mem:{len(self._obstacle_memory)}',
                    (4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_MEMORY, 1)
        cv2.putText(bird, f'Ratio: {self._smooth_ratio:.2f}',
                    (4, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_OK, 1)
        if self._min_ttc < TTC_HORIZON:
            cv2.putText(bird, f'TTC: {self._min_ttc:.2f}s',
                        (4, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_PREDICT, 1)

        flow_panel = self._draw_flow_panel()
        combined   = np.vstack([bird, flow_panel])
        return combined

    def _draw_flow_panel(self):
        panel    = np.full((FLOW_PANEL_H, MAP_W, 3), (15, 15, 15), dtype=np.uint8)
        bar_x0   = 86
        bar_w    = MAP_W - bar_x0 - 46
        bar_h    = 14
        flow_max = max(FLOW_CENTER_DANGER * 1.5, FLOW_SPIKE_THRESHOLD * 1.2, 12.0)

        cv2.putText(panel, 'OPTICAL FLOW (px/frame)', (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

        rows = [
            ('L', self._flow_l, self._flow_l_max, self._flow_l_spike_streak, None),
            ('C', self._flow_c, self._flow_c_max, self._flow_c_spike_streak,
                  (FLOW_CENTER_WARN, FLOW_CENTER_DANGER)),
            ('R', self._flow_r, self._flow_r_max, self._flow_r_spike_streak, None),
        ]

        for i, (label, val, val_max, spike_streak, thresholds) in enumerate(rows):
            y = 24 + i * (bar_h + 6)

            if thresholds is not None:
                warn_t, danger_t = thresholds
                if val >= danger_t:
                    color = C_DANGER
                elif val >= warn_t:
                    color = C_WARNING
                else:
                    color = C_OK
            else:
                color = C_OK

            cv2.putText(panel, label, (4, y + bar_h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
            cv2.rectangle(panel, (bar_x0, y), (bar_x0 + bar_w, y + bar_h),
                         (50, 50, 50), -1)

            fill_w = int(min(1.0, val / flow_max) * bar_w)
            cv2.rectangle(panel, (bar_x0, y), (bar_x0 + fill_w, y + bar_h),
                         color, -1)
            cv2.rectangle(panel, (bar_x0, y), (bar_x0 + bar_w, y + bar_h),
                         (120, 120, 120), 1)

            if thresholds is not None:
                for t_val, t_color in [(warn_t, C_WARNING), (danger_t, C_DANGER)]:
                    tx = bar_x0 + int(min(1.0, t_val / flow_max) * bar_w)
                    cv2.line(panel, (tx, y - 2), (tx, y + bar_h + 2), t_color, 1)

            max_color = (255, 0, 255) if spike_streak >= FLOW_SPIKE_STREAK_REQUIRED else (160, 100, 160)
            mx = bar_x0 + int(min(1.0, val_max / flow_max) * bar_w)
            tri = np.array([(mx, y - 3), (mx - 4, y - 8), (mx + 4, y - 8)], np.int32)
            cv2.fillPoly(panel, [tri], max_color)

            cv2.putText(panel, f'{val:.1f} (spike {val_max:.1f})', (bar_x0 + bar_w + 4, y + bar_h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1)

        c_row_y  = 24 + 1 * (bar_h + 6)
        spike_tx = bar_x0 + int(min(1.0, FLOW_SPIKE_THRESHOLD / flow_max) * bar_w)
        cv2.line(panel, (spike_tx, c_row_y - 2), (spike_tx, c_row_y + bar_h + 2), (255, 0, 255), 1)

        diff        = self._flow_r - self._flow_l
        diff_active = abs(diff) >= FLOW_DIFF_AVOID
        diff_color  = C_PREDICT if diff_active else (140, 140, 140)
        cv2.putText(panel,
                    f'R-L={diff:+.1f}  (thresh {FLOW_DIFF_AVOID:.1f})'
                    + ('  -> REFLEX ACTIVE' if diff_active else ''),
                    (4, 24 + 3 * (bar_h + 6) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, diff_color, 1)

        return panel

    # ══════════════════════════════════════════════════════════════════════════
    #  STATE MACHINE TRANSITIONS
    # ══════════════════════════════════════════════════════════════════════════

    def _engage_stop(self, reason):
        self.pub_cmd.publish(Twist())
        msg      = Bool()
        msg.data = True
        self.pub_estop.publish(msg)
        self.get_logger().error(f'EMERGENCY STOP — {reason}',
                                throttle_duration_sec=0.5)

    def _release_stop(self):
        msg      = Bool()
        msg.data = False
        self.pub_estop.publish(msg)
        self.get_logger().info('Emergency stop RELEASED.')

    def _publish_clear(self):
        msg      = Bool()
        msg.data = False
        self.pub_estop.publish(msg)

    def _cmd_watchdog(self):
        now = time.monotonic()
        if (now - self._last_cmd_in_stamp) > CMD_SOURCE_TIMEOUT:
            self.pub_cmd.publish(Twist())

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN INFERENCE LOOP — 10 Hz
    # ══════════════════════════════════════════════════════════════════════════

    def run_inference(self):
        if self.latest_frame is None:
            return

        frame         = self.latest_frame.copy()
        h_img, w_img  = frame.shape[:2]
        gray          = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rob_x, rob_y = self.robot_ground_pos if self.robot_ground_pos else (0.0, 0.0)

        # ── 1. Zonal Optical Flow ──────────────────────────────────────
        flow_l_raw, flow_c_raw, flow_r_raw, flow_l_max, flow_c_max, flow_r_max, flow_vectors = \
            self._compute_zonal_optical_flow(gray)
        self._flow_l = FLOW_SMOOTH_ALPHA * flow_l_raw + (1.0 - FLOW_SMOOTH_ALPHA) * self._flow_l
        self._flow_c = FLOW_SMOOTH_ALPHA * flow_c_raw + (1.0 - FLOW_SMOOTH_ALPHA) * self._flow_c
        self._flow_r = FLOW_SMOOTH_ALPHA * flow_r_raw + (1.0 - FLOW_SMOOTH_ALPHA) * self._flow_r
        flow_l, flow_c, flow_r = self._flow_l, self._flow_c, self._flow_r

        self._flow_c_spike_streak = (self._flow_c_spike_streak + 1
            if flow_c_max >= FLOW_SPIKE_THRESHOLD else 0)
        self._flow_l_spike_streak = (self._flow_l_spike_streak + 1
            if flow_l_max >= FLOW_SPIKE_THRESHOLD else 0)
        self._flow_r_spike_streak = (self._flow_r_spike_streak + 1
            if flow_r_max >= FLOW_SPIKE_THRESHOLD else 0)

        flow_c_spike = self._flow_c_spike_streak >= FLOW_SPIKE_STREAK_REQUIRED
        flow_l_spike = self._flow_l_spike_streak >= FLOW_SPIKE_STREAK_REQUIRED
        flow_r_spike = self._flow_r_spike_streak >= FLOW_SPIKE_STREAK_REQUIRED

        self._flow_l_max = flow_l_max
        self._flow_c_max = flow_c_max
        self._flow_r_max = flow_r_max

        flow_danger  = (flow_c >= FLOW_CENTER_DANGER) or flow_c_spike
        flow_warning = (flow_c >= FLOW_CENTER_WARN)   or flow_c_spike

        # ── 2. Purge Expired Memory ────────────────────────────────────
        self._purge_memory()

        # ── 3. YOLO Tracking ──────────────────────────────────────────
        small   = cv2.resize(frame, INFER_SIZE)
        results = self.model.track(small, conf=CONF_THRESHOLD, persist=True,
                                   tracker='botsort.yaml', verbose=False, device=DEVICE)
        sx = w_img / INFER_SIZE[0]
        sy = h_img / INFER_SIZE[1]

        ground_dets        = []
        active_track_ids   = set()
        current_min_dist   = float('inf')
        corridor_min_dist  = float('inf')
        distance_danger  = False
        distance_warning = False

        if results and len(results[0].boxes) > 0:
            boxes     = results[0].boxes
            track_ids = (boxes.id.int().cpu().tolist()
                         if boxes.id is not None else [-1] * len(boxes))

            for box, track_id in zip(boxes, track_ids):
                if float(box.conf[0]) < CONF_THRESHOLD:
                    continue

                cls_name = self.model.names[int(box.cls[0])]
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1 = int(x1 * sx); y1 = int(y1 * sy)
                x2 = int(x2 * sx); y2 = int(y2 * sy)

                # Point de contact au sol de l'obstacle : centre horizontal, bas de la boîte (y2)
                ground_pos = self._cam_px_to_ground_m((x1 + x2) / 2.0, float(y2))
                if ground_pos is None:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), C_YOLO, 2)
                    continue

                x_m, y_m = ground_pos
                dist_m   = self._dist_from_robot(x_m, y_m)
                if dist_m < current_min_dist:
                    current_min_dist = dist_m

                lateral_offset = abs(y_m - rob_y)
                in_corridor    = lateral_offset < WARNING_CORRIDOR_HALF_WIDTH

                if dist_m <= DISTANCE_DANGER_M:
                    distance_danger  = True
                    box_color = C_DANGER
                elif dist_m <= DISTANCE_WARN_M and in_corridor:
                    distance_warning = True
                    if dist_m < corridor_min_dist:
                        corridor_min_dist = dist_m
                    box_color = C_WARNING
                else:
                    box_color = C_YOLO

                if track_id != -1:
                    active_track_ids.add(track_id)
                    self._update_track_velocity(track_id, cls_name, x_m, y_m, conf)
                    self._update_memory(track_id, cls_name, x_m, y_m, conf)

                obs_vx = obs_vy = 0.0
                if track_id in self._velocity_tracker:
                    obs_vx = self._velocity_tracker[track_id].vx
                    obs_vy = self._velocity_tracker[track_id].vy

                ground_dets.append({
                    'id': track_id, 'class': cls_name, 'conf': round(conf, 2),
                    'x_m': round(x_m, 2), 'y_m': round(y_m, 2),
                    'dist_m': round(dist_m, 2),
                    'vx': obs_vx, 'vy': obs_vy,
                    'memorized': False,
                })

                lbl = (f'#{track_id} {cls_name} {conf:.2f} {dist_m:.2f}m'
                       if track_id != -1 else f'{cls_name} {conf:.2f} {dist_m:.2f}m')
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, lbl, (x1, max(y1 - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # ── 4. Grace Detections + Memory + Purge ────────────────────────
        grace_dets = self._get_grace_detections(active_track_ids)
        self._purge_velocity_tracker(active_track_ids)
        memory_dets = self._get_memory_detections(active_track_ids)
        for det in grace_dets + memory_dets:
            d = det['dist_m']
            if d < current_min_dist:
                current_min_dist = d
            lateral_offset = abs(det['y_m'] - rob_y)
            in_corridor    = lateral_offset < WARNING_CORRIDOR_HALF_WIDTH
            if d <= DISTANCE_DANGER_M:
                distance_danger  = True
            elif d <= DISTANCE_WARN_M and in_corridor:
                distance_warning = True
                if d < corridor_min_dist:
                    corridor_min_dist = d

        all_dets              = ground_dets + grace_dets + memory_dets
        self.current_min_dist = current_min_dist

        try:
            ground_msg      = String()
            ground_msg.data = json.dumps([
                {k: (float(v) if isinstance(v, (int, float)) else v)
                 for k, v in det.items()}
                for det in all_dets
            ])
            self.pub_ground.publish(ground_msg)
        except Exception as e:
            self.get_logger().warning(f'/ground_detections publish failed: {e}',
                                      throttle_duration_sec=5.0)

        # ── 5. User Intent & Anti-Spin Accumulation ────────────────────
        now = time.monotonic()
        joy_fresh = (now - self._user_cmd_stamp) <= CMD_SOURCE_TIMEOUT

        user_cmd_x_safe = self._user_cmd_x if joy_fresh else 0.0
        user_cmd_z_safe = self._user_cmd_z if joy_fresh else 0.0

        user_moving_forward = joy_fresh and (user_cmd_x_safe > 0.05)
        user_reversing       = joy_fresh and (user_cmd_x_safe < -0.01)

        if self._avoid_was_active:
            raw_delta     = self._current_yaw - self._avoid_prev_yaw_for_cum
            wrapped_delta = float(np.arctan2(np.sin(raw_delta), np.cos(raw_delta)))
            self._avoid_cum_yaw += wrapped_delta
        else:
            self._avoid_cum_yaw = 0.0
        self._avoid_prev_yaw_for_cum = self._current_yaw

        stuck_exceeded = abs(self._avoid_cum_yaw) >= MAX_AVOIDANCE_YAW_DEVIATION

        danger  = flow_danger  or distance_danger or stuck_exceeded
        warning = flow_warning or distance_warning

        # ── 6. Target Heading Memory ───────────────────────────────────
        if (abs(user_cmd_z_safe) > 0.05
                or not user_moving_forward
                or danger):
            self._target_yaw = self._current_yaw

        yaw_error = float(np.arctan2(
            np.sin(self._target_yaw - self._current_yaw),
            np.cos(self._target_yaw - self._current_yaw)))

        # ── 7. Avoidance Command Computation ──────────────────────────
        avoid_cmd        = None
        is_recovery      = False
        min_ttc          = float('inf')
        best_turn_factor = 0.0
        is_yolo_avoid    = False

        if user_moving_forward and not danger:

            rob_x, rob_y = self.robot_ground_pos if self.robot_ground_pos else (0.0, 0.0)
            rob_vx       = self._robot_vx
            rob_vy       = self._robot_vy
            best_side     = None
            best_priority = (2, 0.0)

            # ── A. Cognitive YOLO Avoidance ────────────────────────────
            for det in all_dets:
                ovx, ovy = det.get('vx', 0.0), det.get('vy', 0.0)
                is_human = det.get('class') in DYNAMIC_CLASSES
                safety_r = ROBOT_HALF_WIDTH + (HUMAN_SAFETY_MARGIN if is_human else 0.0)
                ttc = compute_ttc(det['x_m'], det['y_m'], ovx, ovy,
                                  rob_x, rob_y, rob_vx, rob_vy,
                                  safety_radius=safety_r)
                if ttc < min_ttc:
                    min_ttc = ttc

                is_static = float(np.sqrt(ovx**2 + ovy**2)) < TTC_MIN_SPEED
                path_half = ROBOT_HALF_WIDTH + 0.3 + (HUMAN_SAFETY_MARGIN if is_human else 0.0)
                lateral_offset = abs(det['y_m'] - rob_y)
                in_path   = lateral_offset < path_half
                in_orange = det['dist_m'] <= DISTANCE_WARN_M

                candidate_priority = None
                if in_orange and in_path and is_static:
                    candidate_priority = (0, det['dist_m'])
                elif ttc <= TTC_HORIZON:
                    candidate_priority = (1, ttc)

                if candidate_priority is not None and candidate_priority < best_priority:
                    best_priority = candidate_priority
                    best_side     = compute_vo_side(
                        det['x_m'], det['y_m'], ovx, ovy,
                        rob_x, rob_y, rob_vx, rob_vy)
                    is_yolo_avoid = True

                    centeredness = max(0.0, 1.0 - lateral_offset / path_half)
                    proximity_urgency = max(0.0, min(1.0,
                        1.0 - (det['dist_m'] - DISTANCE_DANGER_M)
                              / (DISTANCE_WARN_M - DISTANCE_DANGER_M)))
                    best_turn_factor = max(centeredness, proximity_urgency)

            # ── Side Hysteresis ─────────────────────────────────────────
            if best_side is not None:
                if self._last_avoid_side is None or best_side == self._last_avoid_side:
                    self._last_avoid_side = best_side
                    self._side_streak     = 0
                else:
                    self._side_streak += 1
                    if self._side_streak >= SIDE_FLIP_STREAK:
                        self._last_avoid_side = best_side
                        self._side_streak     = 0
                    else:
                        best_side = self._last_avoid_side
            else:
                self._last_avoid_side = None
                self._side_streak     = 0

            if best_side is not None:
                turn_mag  = AVOID_ANGULAR_Z + (AVOID_ANGULAR_MAX - AVOID_ANGULAR_Z) * best_turn_factor
                avoid_cmd = generate_avoidance_cmd(best_side, turn_mag)

            # ── B. Reflex Optical Flow Avoidance (Independent) ─────────
            elif (abs(flow_r - flow_l) >= FLOW_DIFF_AVOID
                  or flow_l_spike or flow_r_spike):
                diff_flow = flow_r - flow_l
                avoid_cmd          = Twist()
                avoid_cmd.linear.x = AVOID_LINEAR_X

                if abs(diff_flow) >= FLOW_DIFF_AVOID:
                    raw_angular = diff_flow * FLOW_BALANCE_K
                    avoid_cmd.angular.z = max(-FLOW_MAX_TURN_RAD,
                                              min(FLOW_MAX_TURN_RAD, raw_angular))
                else:
                    if flow_l_spike and not flow_r_spike:
                        avoid_cmd.angular.z = -FLOW_MAX_TURN_RAD * 0.6
                    elif flow_r_spike and not flow_l_spike:
                        avoid_cmd.angular.z = +FLOW_MAX_TURN_RAD * 0.6
                    else:
                        avoid_cmd.angular.z = (FLOW_MAX_TURN_RAD * 0.6
                            if diff_flow >= 0 else -FLOW_MAX_TURN_RAD * 0.6)

            # ── C. Heading Recovery ────────────────────────────────────
            if (avoid_cmd is None
                    and abs(user_cmd_z_safe) < 0.05
                    and not warning
                    and abs(yaw_error) > YAW_RECOVERY_MIN):
                correction          = float(np.clip(
                    yaw_error * YAW_RECOVERY_KP,
                    -RECOVERY_MAX_TURN, RECOVERY_MAX_TURN))
                avoid_cmd           = Twist()
                avoid_cmd.linear.x  = RECOVERY_LINEAR_X
                avoid_cmd.angular.z = correction
                is_recovery         = True

        # ── EMA Command Smoothing (Adaptive Alpha) ─────────────────────
        if avoid_cmd is not None:
            if is_yolo_avoid:
                alpha = (AVOID_SMOOTH_ALPHA
                        + (AVOID_SMOOTH_ALPHA_URGENT - AVOID_SMOOTH_ALPHA) * best_turn_factor)
            else:
                alpha = AVOID_SMOOTH_ALPHA
            self._avoid_linear_smooth = (
                alpha * avoid_cmd.linear.x
                + (1.0 - alpha) * self._avoid_linear_smooth)
            self._avoid_angular_smooth = (
                alpha * avoid_cmd.angular.z
                + (1.0 - alpha) * self._avoid_angular_smooth)
            avoid_cmd.linear.x  = self._avoid_linear_smooth
            avoid_cmd.angular.z = self._avoid_angular_smooth
        else:
            self._avoid_linear_smooth  = 0.0
            self._avoid_angular_smooth = 0.0

        self._min_ttc       = min_ttc
        self._avoidance_cmd = avoid_cmd
        self._avoid_was_active = (avoid_cmd is not None)

        if avoid_cmd is not None:
            self.pub_avoid.publish(avoid_cmd)

        # ── 8. Main State Machine ──────────────────────────────────────
        reason_log = ""

        if danger and not user_reversing:
            self._current_state    = RobotState.EMERGENCY
            self.stop_active       = True
            self.clean_cycle_count = 0

            reasons = []
            if flow_c >= FLOW_CENTER_DANGER:
                reasons.append(f'flowC={flow_c:.1f}px')
            elif flow_c_spike:
                reasons.append(f'flowC_spike={flow_c_max:.1f}px(mean={flow_c:.1f})')
            if distance_danger: reasons.append(f'dist<{DISTANCE_DANGER_M}m')
            if stuck_exceeded:  reasons.append(
                f'spin_lock={np.degrees(self._avoid_cum_yaw):.0f}°')
            reason_log = ' '.join(reasons)
            self._engage_stop(reason_log)

        else:
            if self.stop_active:
                if user_reversing:
                    self.stop_active       = False
                    self.clean_cycle_count = 0
                    self._release_stop()
                else:
                    self.clean_cycle_count += 1
                    self._engage_stop('holding — waiting clear cycles')
                    if self.clean_cycle_count >= STOP_RELEASE_CYCLES:
                        self.stop_active       = False
                        self.clean_cycle_count = 0
                        self._release_stop()
            else:
                self._publish_clear()

            if user_reversing:
                self._current_state = RobotState.CLEAR
            elif warning:
                self._current_state = RobotState.WARNING
            elif is_recovery:
                self._current_state = RobotState.RECOVERY
            elif avoid_cmd is not None:
                self._current_state = RobotState.PREDICTING
            else:
                self._current_state = RobotState.CLEAR

        try:
            diag = {
                'state': self._current_state.name if hasattr(self._current_state, 'name') else str(self._current_state),
                'ttc': None if self._min_ttc == float('inf') else round(float(self._min_ttc), 3),
                'min_dist_m': None if current_min_dist == float('inf') else round(float(current_min_dist), 3),
                'n_obstacles': len(all_dets),
                'joy_fwd': bool(user_moving_forward),
                'joy_rev': bool(user_reversing),
                'joy_rot': None, 'left_blocked': None, 'right_blocked': None,
                'avoid_source': 'YOLO' if is_yolo_avoid else ('FLOW' if avoid_cmd is not None else None),
            }
            self.pub_diag.publish(String(data=json.dumps(diag)))
        except Exception as e:
            self.get_logger().warning(f'/flow_diagnostics publish failed: {e}', throttle_duration_sec=5.0)

        # ── 9. WARNING Deceleration Ratio ──────────────────────────────
        if self._current_state == RobotState.EMERGENCY:
            raw_ratio = 0.0
        elif self._current_state == RobotState.WARNING:
            ratio_candidates = []

            if distance_warning:
                ratio_candidates.append(max(0.0, min(1.0,
                    (corridor_min_dist - DISTANCE_DANGER_M)
                    / (DISTANCE_WARN_M - DISTANCE_DANGER_M))))

            if flow_warning:
                flow_ratio = (1.0 - (flow_c - FLOW_CENTER_WARN)
                             / (FLOW_CENTER_DANGER - FLOW_CENTER_WARN))
                ratio_candidates.append(max(0.0, min(1.0, flow_ratio)))

            raw_ratio = min(ratio_candidates) if ratio_candidates else 1.0
        else:
            raw_ratio = 1.0

        self._smooth_ratio = max(0.0, min(1.0,
            EMA_ALPHA * raw_ratio + (1.0 - EMA_ALPHA) * self._smooth_ratio))

        # ── 10. Optical Flow Annotation ────────────────────────────────
        for i, (p0, p1, mag) in enumerate(flow_vectors):
            if i % 3 != 0:
                continue
            fc = (C_DANGER  if mag >= FLOW_CENTER_DANGER else
                  C_WARNING if mag >= FLOW_CENTER_WARN  else
                  (180, 180, 180))
            cv2.arrowedLine(frame, p0, p1, fc, 1, tipLength=0.4)

        third_w = w_img // 3
        cv2.line(frame, (third_w, 0), (third_w, h_img), (80, 80, 80), 1)
        cv2.line(frame, (2 * third_w, 0), (2 * third_w, h_img), (80, 80, 80), 1)
        cv2.putText(frame, f'L:{flow_l:.1f}', (5, h_img - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_OK, 1)
        cv2.putText(frame, f'C:{flow_c:.1f}', (third_w + 5, h_img - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    C_DANGER if flow_danger else (C_WARNING if flow_warning else C_OK), 1)
        cv2.putText(frame, f'R:{flow_r:.1f}', (2 * third_w + 5, h_img - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_OK, 1)

        # ── 11. Camera HUD ────────────────────────────────────────────
        if self._current_state == RobotState.EMERGENCY:
            hud_txt   = f'EMERGENCY STOP ({reason_log})'
            hud_color = C_DANGER
        elif self._current_state == RobotState.RECOVERY:
            hud_txt   = f'RECOVERY  err={yaw_error:.2f}rad  ω={avoid_cmd.angular.z:.2f}rad/s'
            hud_color = C_RECOVERY
        elif self._current_state == RobotState.PREDICTING:
            src = 'YOLO' if min_ttc <= TTC_HORIZON else ('STATIC' if distance_warning else 'FLOW')
            hud_txt   = f'PREDICTING [{src}]  ω={avoid_cmd.angular.z:.2f}  TTC={min_ttc:.1f}s'
            hud_color = C_PREDICT
        elif self._current_state == RobotState.WARNING:
            if distance_warning:
                hud_txt = f'WARN dist={current_min_dist:.1f}m  r={self._smooth_ratio:.2f}'
            else:
                hud_txt = f'WARN flow_c={flow_c:.1f}px  r={self._smooth_ratio:.2f}'
            hud_color = C_WARNING
        else:
            hud_txt   = (f'CLEAR  v={self._robot_vx:.2f}m/s  '
                         f'[L:{flow_l:.1f} C:{flow_c:.1f} R:{flow_r:.1f}]')
            hud_color = C_OK

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_img, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, hud_txt, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2)

        # ── 12. Image Publications ────────────────────────────────────
        bird_view = self._build_bird_eye(all_dets)
        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(frame,     encoding='bgr8'))
        self.pub_bird.publish( self.bridge.cv2_to_imgmsg(bird_view, encoding='bgr8'))

        if ENABLE_DISPLAY:
            h_cam     = frame.shape[0]
            bird_h    = bird_view.shape[0]
            scale     = h_cam / bird_h
            bird_rs   = cv2.resize(bird_view,
                                   (int(MAP_W * scale), h_cam),
                                   interpolation=cv2.INTER_NEAREST)
            sep   = np.full((h_cam, 3, 3), (100, 100, 100), dtype=np.uint8)
            debug = np.hstack([frame, sep, bird_rs])
            debug = cv2.resize(debug,
                               (int(debug.shape[1] * 0.7), int(debug.shape[0] * 0.7)))
            cv2.imshow('Emergency Stop', debug)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                rclpy.shutdown()


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = EmergencyStopNode()
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