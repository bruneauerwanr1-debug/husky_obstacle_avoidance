# -*- coding: utf-8 -*-
"""
raft_avoidance_node.py — 100% Dense Optical Flow (RAFT) Reactive Avoidance Node
=============================================================================
Identical I/O contract to emergency_stop_node.py to remain fully swappable in
twist_mux / base_launch.py WITHOUT ANY CONFIGURATION CHANGES:

  Inputs : /camera/color/image_raw, /cmd_vel_in, /odometry/filtered,
           /joy_teleop/cmd_vel, /homography_matrix, /robot_ground_position
  Outputs: /cmd_vel (filtered), /avoidance_cmd_vel (priority 50 in twist_mux),
           /emergency_stop (lock priority 255), /raft/image (debug),
           /raft/bird_eye, /flow_diagnostics (JSON for bag extraction/analysis)

All decision logic (flow estimation, ego-motion, zones, risk, side selection,
smoothing, anti-lock guard) lives in raft_flow_core.py — WITHOUT ROS dependencies —
to allow isolated testing on rosbag frames. This node only handles ROS plumbing:
topic subscription, orchestration of raft_flow_core classes, and publication.

Manual launch:
    python3 raft_avoidance_node.py

Dependencies: rclpy, cv_bridge, opencv-python, torch, torchvision>=0.12.
"""

import json
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64MultiArray, String

import raft_flow_core as core

# ── Topics (Identical to emergency_stop_node.py) ───────────────────────────
TOPIC_COLOR     = '/camera/color/image_raw'
TOPIC_CMD_IN    = '/cmd_vel_in'
TOPIC_ODOM      = '/odometry/filtered'
TOPIC_JOY       = '/joy_teleop/cmd_vel'
TOPIC_H         = '/homography_matrix'
TOPIC_ROBOT_POS = '/robot_ground_position'

# ── Loop / Safety Timing ──────────────────────────────────────────────────
TIMER_PERIOD = 0.1           # s -> 10 Hz, bounded by RAFT inference cost
WATCHDOG_PERIOD = 0.05        # s -> 20 Hz, camera-independent safety watchdog
STOP_RELEASE_CYCLES = 15
EMA_ALPHA = 0.15
ROBOT_SPEED_TIMEOUT = 0.5
CMD_SOURCE_TIMEOUT = 0.3

# ── Colors (BGR) ──────────────────────────────────────────────────────────
C_OK       = (0,   255,   0)
C_YOLO     = (255, 200,   0)
C_WARNING  = (0,   165, 255)
C_DANGER   = (0,     0, 255)
C_PREDICT  = (255, 200,   0)
C_RECOVERY = (0,   255, 255)
C_MEMORY   = (180,  80, 255)

FLOW_PANEL_H = 92    # Height of optical flow panel stacked beneath bird-eye map

STATE_COLORS = {
    core.RobotState.CLEAR: C_OK,
    core.RobotState.WARNING: C_WARNING,
    core.RobotState.PREDICTING: C_PREDICT,
    core.RobotState.RECOVERY: C_RECOVERY,
    core.RobotState.EMERGENCY: C_DANGER,
}


class RaftAvoidanceNode(Node):

    def __init__(self):
        super().__init__('raft_avoidance_node')

        self.bridge = CvBridge()

        self.get_logger().info('Loading RAFT model...')
        self.flow_estimator = core.RaftFlowEstimator()
        self.get_logger().info(f'RAFT ready on {self.flow_estimator.device} \u2713')

        self.ego_compensator = core.EgoMotionCompensator()
        self.zone_analyzer = core.ZoneAnalyzer()
        self.risk_estimator = core.RiskEstimator()
        self.side_selector = core.SideSelector()
        self.cmd_smoother = core.CommandSmoother()
        self.spin_guard = core.YawSpinGuard()

        # ── Homography / Distance ──────────────────────────────────────────
        self.distance_estimator = core.HomographyDistanceEstimator()
        if self.distance_estimator.load_from_file():
            self.get_logger().info('Homography loaded from calibration file \u2713')
            self.ego_compensator.set_homography(self.distance_estimator.H)
        else:
            self.get_logger().warn(
                f'Homography file not found ({core.H_SAVE_PATH}) '
                '— waiting for topic /homography_matrix')

        # ── Explicit Floor AprilTag Masking ────────────────────────────────
        self.apriltag_masker = (core.AprilTagFloorMasker()
                                 if core.ENABLE_APRILTAG_MASKING else None)
        self._last_apriltag_log_time = 0.0

        # ── Frame Buffers ─────────────────────────────────────────────────
        self.latest_frame = None
        self.prev_frame_for_flow = None
        self._last_flow_residual = None
        self._last_obstacles = []

        # ── Safety / State ────────────────────────────────────────────────
        self.stop_active = False
        self.clean_cycle_count = 0
        self._current_state = core.RobotState.CLEAR
        self._smooth_ratio = 1.0
        self._last_avoid_active = False

        # ── Odometry ──────────────────────────────────────────────────────
        self._robot_vx = 0.0
        self._omega_z = 0.0
        self._robot_speed_stamp = 0.0
        self._current_yaw = 0.0
        self._target_yaw = 0.0

        # ── User Intent (Raw Joystick — never /cmd_vel_in) ─────────────────
        self._user_cmd_x = 0.0
        self._user_cmd_z = 0.0
        self._user_cmd_stamp = 0.0
        self._last_cmd_in_stamp = 0.0

        self._min_ttc = float('inf')

        qos_img = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_rel = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST, depth=10)

        self.create_subscription(Image, TOPIC_COLOR, self.cb_image, qos_img)
        self.create_subscription(Odometry, TOPIC_ODOM, self.cb_odom, qos_rel)
        self.create_subscription(Twist, TOPIC_CMD_IN, self.cb_cmd_vel_in, 10)
        self.create_subscription(Twist, TOPIC_JOY, self.cb_joy, 10)
        self.create_subscription(
            Float64MultiArray, TOPIC_H, self.cb_H, qos_rel)
        self.create_subscription(
            Point, TOPIC_ROBOT_POS, self.cb_robot_pos, qos_rel)

        self.pub_estop = self.create_publisher(Bool, '/emergency_stop', 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_avoid = self.create_publisher(Twist, '/avoidance_cmd_vel', 10)
        self.pub_debug = self.create_publisher(Image, '/raft/image', 10)
        self.pub_bird = self.create_publisher(Image, '/raft/bird_eye', 10)
        self.pub_diag = self.create_publisher(String, '/flow_diagnostics', 10)

        self.create_timer(TIMER_PERIOD, self.run_inference)
        self.create_timer(WATCHDOG_PERIOD, self._cmd_watchdog)

        filt_parts = []
        if core.ENABLE_ZONAL_MEDIAN:
            filt_parts.append('ZONAL_MEDIAN')
        elif core.ENABLE_MEDIAN_FALLBACK:
            filt_parts.append('MEDIAN_FALLBACK')
        filt_str = '+'.join(filt_parts) if filt_parts else 'RAW (no filter)'

        self.get_logger().info(
            'RAFT avoidance node ready.\n'
            f'  Inference resolution   : {core.RAFT_INFER_SIZE}\n'
            f'  Compute device         : {self.flow_estimator.device}\n'
            f'  Mag thresholds WARN/DANGER : {core.FLOW_CENTER_WARN}/{core.FLOW_CENTER_DANGER} px\n'
            f'  TTC thresholds WARN/DANGER : {core.TTC_WARN_S}/{core.TTC_DANGER_S} s '
            f'(div equiv. {core.DIVERGENCE_WARN:.3f}/{core.DIVERGENCE_DANGER:.3f})\n'
            f'  Distance WARN/DANGER/MAX   : {core.DISTANCE_WARN_M}/{core.DISTANCE_DANGER_M}/{core.MAX_OBSTACLE_DIST_M} m\n'
            f'  Homography                 : {"LOADED" if self.distance_estimator.ready else "PENDING"}\n'
            f'  --- Farneback Fix Ports ---\n'
            f'  Rotation compensation : {"ON" if core.ENABLE_ROTATION_COMPENSATION else "OFF"}\n'
            f'  Residual filter       : {filt_str}\n'
            f'  Ground mask (H)       : {"ON" if core.ENABLE_GROUND_MASK else "OFF"}\n'
            f'  AprilTag floor mask   : {"ON" if core.ENABLE_APRILTAG_MASKING else "OFF"}\n'
            f'  Wall dead-zone detect : {"ON" if core.ENABLE_WALL_DEAD_ZONE_DETECTION else "OFF"} '
            f'(streak={core.WALL_STREAK_REQUIRED})\n'
            f'  Zone occupancy turn   : EMERGENCY={core.ZONE_OCCUPANCY_EMERGENCY_TURN_FACTOR} '
            f'WARNING={core.ZONE_OCCUPANCY_WARNING_TURN_FACTOR}\n'
            f'  Recovery              : KP={core.YAW_RECOVERY_KP} max_turn={core.RECOVERY_MAX_TURN} rad/s\n'
            f'  Pure rotation bypass  : lin<{core.PURE_ROTATION_LINEAR_MAX} ang>{core.PURE_ROTATION_ANGULAR_MIN}\n'
        )

    # ══════════════════════════════════════════════════════════════════
    #  CALLBACKS
    # ══════════════════════════════════════════════════════════════════

    def cb_image(self, msg: Image):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def cb_odom(self, msg: Odometry):
        self._robot_vx = msg.twist.twist.linear.x
        self._omega_z = msg.twist.twist.angular.z
        self._robot_speed_stamp = time.monotonic()
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._current_yaw = float(np.arctan2(siny, cosy))

    def cb_joy(self, msg: Twist):
        self._user_cmd_x = msg.linear.x
        self._user_cmd_z = msg.angular.z
        self._user_cmd_stamp = time.monotonic()

    def cb_H(self, msg: Float64MultiArray):
        """Real-time homography matrix reception."""
        H = np.array(msg.data, dtype=np.float64).reshape(3, 3)
        self.distance_estimator.set_homography(H)
        self.ego_compensator.set_homography(H)
        self.get_logger().info('Homography updated from topic \u2713 '
                               '(distance + geometric ground masking)',
                               throttle_duration_sec=5.0)

    def cb_robot_pos(self, msg: Point):
        """Robot ground position."""
        self.distance_estimator.set_robot_ground_pos(msg.x, msg.y)

    def cb_cmd_vel_in(self, msg: Twist):
        """
        Filters incoming commands from twist_mux before sending to motors:
        - Reversing commands are always allowed through and release E-Stop.
        - Pure rotation in place is allowed through and releases E-Stop.
        - EMERGENCY state blocks all forward commands.
        - WARNING state scales linear and angular commands via smooth EMA ratio.
        - Otherwise commands pass through unmodified.
        """
        self._last_cmd_in_stamp = time.monotonic()

        if time.monotonic() - self._robot_speed_stamp > ROBOT_SPEED_TIMEOUT:
            self._robot_vx = msg.linear.x
            self._robot_speed_stamp = time.monotonic()

        is_reversing = msg.linear.x < -0.01
        is_pure_rotation = (abs(msg.linear.x) < core.PURE_ROTATION_LINEAR_MAX
                             and abs(msg.angular.z) > core.PURE_ROTATION_ANGULAR_MIN)

        if is_reversing or is_pure_rotation:
            if self.stop_active:
                self.stop_active = False
                self.clean_cycle_count = 0
                self._release_stop()
            self.pub_cmd.publish(msg)
            return

        if self.stop_active:
            self.pub_cmd.publish(Twist())
            return

        if self._current_state == core.RobotState.WARNING and self._smooth_ratio < 1.0:
            safe = Twist()
            safe.linear.x = msg.linear.x * self._smooth_ratio
            safe.angular.z = msg.angular.z * self._smooth_ratio
            self.pub_cmd.publish(safe)
            return

        self.pub_cmd.publish(msg)

    # ══════════════════════════════════════════════════════════════════
    #  SAFETY
    # ══════════════════════════════════════════════════════════════════

    def _engage_stop(self, reason):
        self.pub_cmd.publish(Twist())
        m = Bool()
        m.data = True
        self.pub_estop.publish(m)
        self.get_logger().error(f'EMERGENCY STOP - {reason}', throttle_duration_sec=0.5)

    def _release_stop(self):
        m = Bool()
        m.data = False
        self.pub_estop.publish(m)
        self.get_logger().info('Emergency stop RELEASED.')

    def _publish_clear(self):
        m = Bool()
        m.data = False
        self.pub_estop.publish(m)

    def _cmd_watchdog(self):
        """Motor watchdog: republishes zero Twist if /cmd_vel_in goes silent."""
        if (time.monotonic() - self._last_cmd_in_stamp) > CMD_SOURCE_TIMEOUT:
            self.pub_cmd.publish(Twist())

    # ══════════════════════════════════════════════════════════════════
    #  MAIN INFERENCE LOOP
    # ══════════════════════════════════════════════════════════════════

    def run_inference(self):
        if self.latest_frame is None:
            return
        frame = self.latest_frame.copy()

        if self.prev_frame_for_flow is None:
            self.prev_frame_for_flow = frame.copy()
            return   # Requires two consecutive frames to compute flow

        # ── 1. RAFT Dense Flow Estimation ───────────────────────────────
        flow_raw = self.flow_estimator.compute(self.prev_frame_for_flow, frame)
        self.prev_frame_for_flow = frame.copy()

        # ── 2. Ego-Motion Compensation (Rotation + Ground Mask) ─────────
        dt = TIMER_PERIOD
        flow_residual = self.ego_compensator.compensate(flow_raw, self._omega_z, dt)

        # ── 2b. Explicit Floor AprilTag Masking ─────────────────────────
        if core.ENABLE_APRILTAG_MASKING and self.apriltag_masker is not None:
            tag_mask = self.apriltag_masker.build_mask(
                frame, flow_residual.shape[1], flow_residual.shape[0])
            if np.any(tag_mask):
                flow_residual[tag_mask, 0] = 0.0
                flow_residual[tag_mask, 1] = 0.0
            now_tag = time.monotonic()
            if (now_tag - self._last_apriltag_log_time) >= 1.0:
                self._last_apriltag_log_time = now_tag
                self.get_logger().info(
                    f'[AprilTag] {self.apriltag_masker.last_n_detected} tags '
                    f'detected and masked', throttle_duration_sec=5.0)

        self._last_flow_residual = flow_residual

        # ── 3. Zonal Metrics + Smoothed Risk ────────────────────────────
        zone_metrics = self.zone_analyzer.compute_zone_metrics(flow_residual)
        risk = self.risk_estimator.update(zone_metrics)

        flow_danger, flow_warning, left_blocked, right_blocked = core.combine_zone_risk(risk)
        self._min_ttc = min((r['ttc'] for r in risk.values()), default=float('inf'))

        # ── 3b. Homography Distance Estimation ──────────────────────────
        obstacles = self.distance_estimator.estimate_obstacle_distances(
            flow_residual, frame.shape)
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
            in_corridor = lateral_offset < core.WARNING_CORRIDOR_HALF_WIDTH

            if dist_m <= core.DISTANCE_DANGER_M:
                distance_danger = True
            elif dist_m <= core.DISTANCE_WARN_M and in_corridor:
                distance_warning = True
                if dist_m < corridor_min_dist:
                    corridor_min_dist = dist_m

        # Combine optical flow risk and metric distance risk
        danger  = flow_danger  or distance_danger
        warning = flow_warning or distance_warning

        # ── 4. User Intent Evaluation ───────────────────────────────────
        now = time.monotonic()
        joy_fresh = (now - self._user_cmd_stamp) <= CMD_SOURCE_TIMEOUT
        user_cmd_x_safe = self._user_cmd_x if joy_fresh else 0.0
        user_cmd_z_safe = self._user_cmd_z if joy_fresh else 0.0
        user_moving_forward = joy_fresh and (user_cmd_x_safe > 0.05)
        user_reversing = joy_fresh and (user_cmd_x_safe < -0.01)
        user_pure_rotation = (joy_fresh
                               and abs(user_cmd_x_safe) < core.PURE_ROTATION_LINEAR_MAX
                               and abs(user_cmd_z_safe) > core.PURE_ROTATION_ANGULAR_MIN)

        # ── 5. Target Heading Memory ────────────────────────────────────
        if abs(user_cmd_z_safe) > 0.05 or not user_moving_forward or danger:
            self._target_yaw = self._current_yaw
        yaw_error = float(np.arctan2(
            np.sin(self._target_yaw - self._current_yaw),
            np.cos(self._target_yaw - self._current_yaw)))

        # ── 6. Anti-Spin Guard ──────────────────────────────────────────
        stuck = self.spin_guard.update(self._current_yaw, self._last_avoid_active)
        danger = danger or stuck

        # ── 7. Avoidance / Recovery Decision ────────────────────────────
        avoid_linear = 0.0
        avoid_angular = 0.0
        is_recovery = False
        avoid_active = False
        avoid_source = ''

        if user_moving_forward and not danger:
            # Priority 0: Zone Occupancy Steering
            zocc_side, zocc_sev = core.zone_occupancy_side(risk)

            if zocc_side is not None:
                side = self.side_selector.resolve(zocc_side)
                turn_factor = (core.ZONE_OCCUPANCY_EMERGENCY_TURN_FACTOR if zocc_sev >= 2
                               else core.ZONE_OCCUPANCY_WARNING_TURN_FACTOR)
                turn_mag = (core.AVOID_ANGULAR_BASE
                            + (core.AVOID_ANGULAR_MAX - core.AVOID_ANGULAR_BASE) * turn_factor)
                target_lin, target_ang = core.generate_avoidance_cmd(side, turn_mag)
                avoid_linear, avoid_angular = self.cmd_smoother.update(
                    target_lin, target_ang, turn_factor)
                avoid_active = True
                avoid_source = 'ZONE_OCCUPANCY' if zocc_sev >= 2 else 'ZONE_OCCUPANCY_WARN'
            else:
                def zone_score(z):
                    div_scaled = (z['div'] / core.DIVERGENCE_DANGER) * core.FLOW_CENTER_DANGER
                    return max(z['mag'], div_scaled)

                risk_l = zone_score(risk['left'])
                risk_r = zone_score(risk['right'])
                side = self.side_selector.select(risk_l, risk_r)

                if side is not None:
                    urgency = max(0.0, min(1.0, max(risk_l, risk_r) / core.FLOW_CENTER_DANGER))
                    turn_mag = (core.AVOID_ANGULAR_BASE
                                + (core.AVOID_ANGULAR_MAX - core.AVOID_ANGULAR_BASE) * urgency)
                    target_lin, target_ang = core.generate_avoidance_cmd(side, turn_mag)
                    avoid_linear, avoid_angular = self.cmd_smoother.update(target_lin, target_ang, urgency)
                    avoid_active = True
                    avoid_source = 'FLOW_DIFF'
                elif (abs(user_cmd_z_safe) < 0.05 and not warning
                      and abs(yaw_error) > core.YAW_RECOVERY_MIN):
                    correction = float(np.clip(
                        yaw_error * core.YAW_RECOVERY_KP,
                        -core.RECOVERY_MAX_TURN, core.RECOVERY_MAX_TURN))
                    avoid_linear, avoid_angular = core.RECOVERY_LINEAR_X, correction
                    is_recovery = True
                    avoid_source = 'RECOVERY'
                else:
                    self.cmd_smoother.reset()
        else:
            self.cmd_smoother.reset()

        if avoid_active or is_recovery:
            cmd = Twist()
            cmd.linear.x = avoid_linear
            cmd.angular.z = avoid_angular
            self.pub_avoid.publish(cmd)

        # ── 8. State Machine Transition ─────────────────────────────────
        bypass_emergency = user_reversing or user_pure_rotation

        reason_log = ''
        if danger and not bypass_emergency:
            self._current_state = core.RobotState.EMERGENCY
            self.stop_active = True
            self.clean_cycle_count = 0
            reasons = []
            if flow_danger:
                walls = [k for k, r in risk.items() if r.get('wall')]
                wall_str = (' wall=' + ','.join(walls)) if walls else ''
                reasons.append('flow_danger=center'
                                + (',left' if left_blocked else '')
                                + (',right' if right_blocked else '')
                                + wall_str)
            if distance_danger:
                reasons.append(f'dist<{core.DISTANCE_DANGER_M}m'
                               f'(min={current_min_dist:.2f}m)')
            if stuck:
                reasons.append(
                    f'spin_lock={np.degrees(self.spin_guard.cumulative_yaw):.0f}deg')
            reason_log = ' '.join(reasons) if reasons else 'unknown'
            self._engage_stop(reason_log)
        else:
            if self.stop_active:
                if bypass_emergency:
                    self.stop_active = False
                    self.clean_cycle_count = 0
                    self._release_stop()
                else:
                    self.clean_cycle_count += 1
                    self._engage_stop('holding - waiting clear cycles')
                    if self.clean_cycle_count >= STOP_RELEASE_CYCLES:
                        self.stop_active = False
                        self.clean_cycle_count = 0
                        self._release_stop()
            else:
                self._publish_clear()

            if bypass_emergency:
                self._current_state = core.RobotState.CLEAR
            elif is_recovery:
                self._current_state = core.RobotState.RECOVERY
            elif avoid_active:
                self._current_state = core.RobotState.PREDICTING
            elif warning:
                self._current_state = core.RobotState.WARNING
            else:
                self._current_state = core.RobotState.CLEAR

        # ── 9. WARNING EMA Deceleration Ratio ───────────────────────────
        if self._current_state == core.RobotState.EMERGENCY:
            raw_ratio = 0.0
        elif self._current_state == core.RobotState.WARNING:
            ratio_candidates = []
            worst_mag = max(r['mag'] for r in risk.values())
            if flow_warning:
                ratio_candidates.append(max(0.0, min(1.0,
                    1.0 - (worst_mag - core.FLOW_CENTER_WARN)
                    / (core.FLOW_CENTER_DANGER - core.FLOW_CENTER_WARN))))
            if distance_warning:
                ratio_candidates.append(max(0.0, min(1.0,
                    (corridor_min_dist - core.DISTANCE_DANGER_M)
                    / (core.DISTANCE_WARN_M - core.DISTANCE_DANGER_M))))
            raw_ratio = min(ratio_candidates) if ratio_candidates else 1.0
        else:
            raw_ratio = 1.0
        self._smooth_ratio = max(0.0, min(1.0,
            EMA_ALPHA * raw_ratio + (1.0 - EMA_ALPHA) * self._smooth_ratio))

        # ── 10. Update state for next cycle ─────────────────────────────
        self._last_avoid_active = avoid_active

        # ── 11. Diagnostic Output ───────────────────────────────────────
        try:
            diag = {
                'state': self._current_state,
                'ttc': None if self._min_ttc == float('inf') else round(self._min_ttc, 2),
                'omega_z': round(self._omega_z, 3),
                'min_dist_m': round(current_min_dist, 2) if current_min_dist < float('inf') else None,
                'n_obstacles': len(obstacles),
                'joy_fwd': user_moving_forward, 'joy_rev': user_reversing,
                'joy_rot': user_pure_rotation,
                'avoid_source': avoid_source,
                'left_blocked': left_blocked, 'right_blocked': right_blocked,
                'zones': {
                    k: {'mag': round(v['mag'], 3), 'div': round(v['div'], 5),
                        'danger': v['danger'], 'warning': v['warning'],
                        'wall': v.get('wall', False)}
                    for k, v in risk.items()
                },
            }
            m = String()
            m.data = json.dumps(diag)
            self.pub_diag.publish(m)
        except Exception as e:
            self.get_logger().warning(f'/flow_diagnostics publish failed: {e}',
                                       throttle_duration_sec=5.0)

        # ── 12. Debug & Bird-Eye Visualizations ─────────────────────────
        self._publish_debug_image(frame, risk, yaw_error, avoid_angular,
                                  reason_log, current_min_dist)

    # ══════════════════════════════════════════════════════════════════
    #  BIRD-EYE VIEW
    # ══════════════════════════════════════════════════════════════════

    def _build_bird_eye(self, obstacles, risk):
        """Constructs Bird-Eye View with grid, distance radii, flow obstacles,
        and bottom flow panel."""
        bird = np.full((core.MAP_H, core.MAP_W, 3), (30, 30, 30), dtype=np.uint8)

        # Grid
        for step, color in [(int(0.5 / core.MAP_SCALE), (55, 55, 55)),
                             (int(1.0 / core.MAP_SCALE), (75, 75, 75))]:
            for x in range(0, core.MAP_W, step):
                cv2.line(bird, (x, 0), (x, core.MAP_H), color, 1)
            for y in range(0, core.MAP_H, step):
                cv2.line(bird, (0, y), (core.MAP_W, y), color, 1)

        # Robot Representation
        rx_m, ry_m = self.distance_estimator.robot_ground_pos
        rbx, rby = self.distance_estimator.ground_m_to_bird_px(rx_m, ry_m)
        rob_in_view = 0 <= rbx < core.MAP_W and 0 <= rby < core.MAP_H

        if rob_in_view:
            cv2.circle(bird, (rbx, rby),
                       int(core.DISTANCE_WARN_M / core.MAP_SCALE), C_WARNING, 1, cv2.LINE_AA)
            cv2.circle(bird, (rbx, rby),
                       int(core.DISTANCE_DANGER_M / core.MAP_SCALE), C_DANGER, 1, cv2.LINE_AA)
            r_pred = int((abs(self._robot_vx) * core.TTC_HORIZON
                          + core.ROBOT_HALF_WIDTH) / core.MAP_SCALE)
            cv2.circle(bird, (rbx, rby), r_pred, C_PREDICT, 1, cv2.LINE_AA)
            tri = np.array([(rbx, rby - 12), (rbx - 8, rby + 8),
                            (rbx + 8, rby + 8)], np.int32)
            cv2.fillPoly(bird, [tri], (0, 255, 255))

        # Optical Flow Detected Obstacles
        for obs in obstacles:
            bx, by = self.distance_estimator.ground_m_to_bird_px(
                obs['x_m'], obs['y_m'])
            if not (0 <= bx < core.MAP_W and 0 <= by < core.MAP_H):
                continue

            dist_m = obs['dist_m']
            if dist_m <= core.DISTANCE_DANGER_M:
                color = C_DANGER
            elif dist_m <= core.DISTANCE_WARN_M:
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

        # HUD Info
        state_color = STATE_COLORS.get(self._current_state, C_OK)
        cv2.putText(bird, f'State: {self._current_state}',
                    (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, state_color, 1)
        cv2.putText(bird, f'vRobot: {self._robot_vx:.2f}m/s  Obs:{len(obstacles)}',
                    (4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_MEMORY, 1)
        cv2.putText(bird, f'Ratio: {self._smooth_ratio:.2f}',
                    (4, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_OK, 1)
        if self._min_ttc < core.TTC_HORIZON:
            cv2.putText(bird, f'TTC: {self._min_ttc:.2f}s',
                        (4, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_PREDICT, 1)

        # Optical Flow Panel
        flow_panel = self._draw_flow_panel(risk)
        combined = np.vstack([bird, flow_panel])
        return combined

    def _draw_flow_panel(self, risk):
        """Flow panel with L/C/R horizontal magnitude indicator bars."""
        panel = np.full((FLOW_PANEL_H, core.MAP_W, 3), (15, 15, 15), dtype=np.uint8)
        bar_x0 = 86
        bar_w = core.MAP_W - bar_x0 - 46
        bar_h = 14
        flow_max = max(core.FLOW_CENTER_DANGER * 1.5,
                       core.FLOW_SPIKE_THRESHOLD * 1.2, 12.0)

        cv2.putText(panel, 'RAFT OPTICAL FLOW (px/frame)', (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

        labels_display = [('L', 'left'), ('C', 'center'), ('R', 'right')]
        thresholds_for_center = (core.FLOW_CENTER_WARN, core.FLOW_CENTER_DANGER)

        for i, (short, label) in enumerate(labels_display):
            r = risk.get(label, {'mag': 0.0})
            val = r['mag']
            y = 24 + i * (bar_h + 6)

            thresholds = thresholds_for_center if label == 'center' else None
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

            cv2.putText(panel, short, (4, y + bar_h - 3),
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

            cv2.putText(panel, f'{val:.1f}', (bar_x0 + bar_w + 4, y + bar_h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1)

        if self._last_obstacles:
            closest = self._last_obstacles[0]
            dist_color = C_DANGER if closest['dist_m'] <= core.DISTANCE_DANGER_M else (
                C_WARNING if closest['dist_m'] <= core.DISTANCE_WARN_M else C_OK)
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

    # ══════════════════════════════════════════════════════════════════
    #  DEBUG VISUALIZATION WINDOWS
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _flow_to_hsv(flow):
        """Converts optical flow field (H,W,2) to BGR visualization via HSV mapping."""
        u = flow[..., 0]
        v = flow[..., 1]
        mag = np.sqrt(u ** 2 + v ** 2)
        ang = np.arctan2(v, u)
        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
        hsv[..., 1] = 255
        max_mag = max(mag.max(), 1e-5)
        hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _publish_debug_image(self, frame, risk, yaw_error, angular_cmd,
                             reason_log='', current_min_dist=float('inf')):
        h, w = frame.shape[:2]
        third = w // 3

        color = STATE_COLORS.get(self._current_state, C_OK)
        if self._current_state == core.RobotState.EMERGENCY:
            hud = f'EMERGENCY STOP ({reason_log})'
        elif self._current_state == core.RobotState.RECOVERY:
            hud = f'RECOVERY err={yaw_error:.2f}rad w={angular_cmd:.2f}rad/s'
        elif self._current_state == core.RobotState.PREDICTING:
            dist_str = (f' d={current_min_dist:.1f}m'
                        if current_min_dist < float('inf') else '')
            hud = f'PREDICTING [RAFT] w={angular_cmd:.2f}{dist_str}'
        elif self._current_state == core.RobotState.WARNING:
            if current_min_dist < float('inf'):
                hud = (f'WARN dist={current_min_dist:.1f}m '
                       f'r={self._smooth_ratio:.2f}')
            else:
                hud = (f'WARN flow r={self._smooth_ratio:.2f}')
        else:
            hud = f'{self._current_state}  v={self._robot_vx:.2f}m/s'

        # ──────────────────────────────────────────────────────────────
        #  WINDOW 1: Camera + Zones + State
        # ──────────────────────────────────────────────────────────────
        cam_vis = frame.copy()

        cv2.line(cam_vis, (third, 0), (third, h), (200, 200, 200), 2)
        cv2.line(cam_vis, (2 * third, 0), (2 * third, h), (200, 200, 200), 2)

        zone_labels_display = ['LEFT', 'CENTER', 'RIGHT']
        for i, zl in enumerate(zone_labels_display):
            cx = i * third + third // 2 - 25
            cv2.putText(cam_vis, zl, (cx, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for i, label in enumerate(core.ZONE_LABELS):
            r = risk.get(label, {'mag': 0.0, 'div': 0.0, 'danger': False, 'warning': False})
            x0 = i * third
            x1 = (i + 1) * third if i < core.N_ZONES - 1 else w
            if r.get('danger', False):
                bar_color = C_DANGER
            elif r.get('warning', False):
                bar_color = C_WARNING
            else:
                bar_color = C_OK
            overlay_zone = cam_vis.copy()
            cv2.rectangle(overlay_zone, (x0, h - 40), (x1, h), bar_color, -1)
            cv2.addWeighted(overlay_zone, 0.35, cam_vis, 0.65, 0, cam_vis)
            txt = f'mag:{r["mag"]:.1f} div:{r["div"]:.3f}' + (' WALL' if r.get('wall') else '')
            cv2.putText(cam_vis, txt, (x0 + 5, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        overlay_top = cam_vis.copy()
        cv2.rectangle(overlay_top, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.addWeighted(overlay_top, 0.6, cam_vis, 0.4, 0, cam_vis)
        cv2.putText(cam_vis, hud, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.rectangle(cam_vis, (0, 0), (w - 1, h - 1), color, 3)
        cv2.imshow('Camera + Zones + State', cam_vis)

        # ──────────────────────────────────────────────────────────────
        #  WINDOW 2: RAFT - Optical Flow
        # ──────────────────────────────────────────────────────────────
        if self._last_flow_residual is not None:
            flow = self._last_flow_residual
            flow_vis = self._flow_to_hsv(flow)

            cv2.line(flow_vis, (third, 0), (third, h), (200, 200, 200), 1)
            cv2.line(flow_vis, (2 * third, 0), (2 * third, h), (200, 200, 200), 1)

            mag_global = np.sqrt(
                flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()

            overlay_flow = flow_vis.copy()
            cv2.rectangle(overlay_flow, (0, 0), (w, 36), (0, 0, 0), -1)
            cv2.addWeighted(overlay_flow, 0.6, flow_vis, 0.4, 0, flow_vis)
            flow_hud = f'RAFT Flow  mean_mag={mag_global:.2f}px  omega_z={self._omega_z:.3f}rad/s'
            cv2.putText(flow_vis, flow_hud, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            for i, label in enumerate(core.ZONE_LABELS):
                r = risk.get(label, {'mag': 0.0, 'div': 0.0})
                txt = f'{label[0].upper()}: {r["mag"]:.1f}px / div={r["div"]:.3f}'
                cv2.putText(flow_vis, txt, (i * third + 5, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

            cv2.imshow('RAFT - Optical Flow', flow_vis)

        # ──────────────────────────────────────────────────────────────
        #  WINDOW 3: Bird-Eye View
        # ──────────────────────────────────────────────────────────────
        bird_view = self._build_bird_eye(self._last_obstacles, risk)
        cv2.imshow('RAFT - Bird Eye', bird_view)

        cv2.waitKey(1)

        # ── ROS Image Publication ──────────────────────────────────────
        debug_frame = frame.copy()
        cv2.line(debug_frame, (third, 0), (third, h), (80, 80, 80), 1)
        cv2.line(debug_frame, (2 * third, 0), (2 * third, h), (80, 80, 80), 1)
        overlay = debug_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 32), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, debug_frame, 0.4, 0, debug_frame)
        cv2.putText(debug_frame, hud, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        for i, label in enumerate(core.ZONE_LABELS):
            r = risk.get(label, {'mag': 0.0, 'div': 0.0})
            txt = f'{label[0].upper()}:{r["mag"]:.1f}px/{r["div"]:.3f}'
            cv2.putText(debug_frame, txt, (i * third + 5, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_OK, 1)
        self.pub_debug.publish(
            self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8'))
        self.pub_bird.publish(
            self.bridge.cv2_to_imgmsg(bird_view, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = RaftAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
