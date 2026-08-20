# Functional Obstacle Avoidance Nodes & Robot Deployment Guide

This directory contains the real-time ROS 2 obstacle avoidance and calibration nodes for the **Clearpath Husky A200** mobile robot.

---

## 1. Robot Configuration & Bringup

### A. Multiplexer Setup (`twist_mux.yaml`)
To ensure safety and command arbitration, copy or link [`../config/twist_mux.yaml`](file:///../config/twist_mux.yaml) into your robot's base configuration (`husky_base/config/twist_mux.yaml`).

The multiplexing priority scheme is:
| Priority | Type | Topic | Source | Description |
| :---: | :---: | :--- | :--- | :--- |
| **255** | **Lock** | `/emergency_stop` | Avoidance Node | Immediate hardware brake lock on imminent collision |
| **50** | **Velocity** | `/avoidance_cmd_vel` | Avoidance Node | Reactive evasion command (steer around obstacle) |
| **10** | **Velocity** | `/joy_teleop/cmd_vel` | Joystick | Manual operator driving commands |
| **8** | **Velocity** | `/twist_marker_server/cmd_vel` | Interactive Marker | Rviz navigation / marker teleop |

> **Safety Bypass**: Reversing (`linear.x < 0`) or pure in-place rotation (`|linear.x| ≈ 0`, `|angular.z| > 0`) automatically releases the vision emergency stop lock, allowing the operator to maneuver out of confined spaces.

### B. Launching the Robot Base
On the Clearpath Husky onboard computer, initialize the hardware base, sensors, and multiplexer:

```bash
cd SSM-Nav2/ssm-nav2
source install/setup.bash
ros2 launch husky_base base.launch-collision.py
```

---

## 2. Overview of Obstacle Avoidance Architecture

Each avoidance node consumes the live RGB stream from `/camera/color/image_raw` and odometry from `/odometry/filtered`, processing the scene to publish steering commands to `/avoidance_cmd_vel` and safety locks to `/emergency_stop`.

```
                       ┌────────────────────────────┐
                       │    RGB Camera (RealSense)  │
                       └─────────────┬──────────────┘
                                     │ /camera/color/image_raw
                                     ▼
                       ┌────────────────────────────┐
                       │    Obstacle Avoidance Node │
                       │ (Farneback / RAFT / YOLO)  │
                       └──────┬──────────────┬──────┘
  /avoidance_cmd_vel         │              │ /emergency_stop
  (Priority 50)              │              │ (Priority 255 Lock)
                             ▼              ▼
                       ┌────────────────────────────┐
/joy_teleop/cmd_vel ─►│         twist_mux          │
(Priority 10)         └─────────────┬──────────────┘
                                     │ /cmd_vel_in
                                     ▼
                       ┌────────────────────────────┐
                       │   Command Filter & Motors  │
                       └────────────────────────────┘
```

---

## 3. High-Level Node Explanations

### 1. CPU Dense Optical Flow (`optical_flow_farneback.py`)
- **What it does**: Provides fully autonomous, real-time motion-field obstacle perception running entirely on CPU.
- **How it works**: Uses the Farneback polynomial expansion algorithm to compute 2D motion vectors for every pixel. It subtracts the robot's ego-motion (linear & angular velocities), splits the scene into Left, Center, and Right zones, and computes risk based on flow magnitude and expansion divergence. When an obstacle is detected in the path, it selects the clearer side to steer around it, or triggers an emergency stop if the path is fully blocked.

### 2. Deep Optical Flow (`raft_avoidance_node.py` + `raft_flow_core.py`)
- **What it does**: Delivers high-precision, deep-learning optical flow perception robust to low textures and dark environments on GPU.
- **How it works**: Employs the RAFT (Recurrent All-Pairs Field Transforms) neural network to compute sharp pixel displacement fields via 4D correlation volumes and recurrent GRU updates. It filters ground plane motion, isolates coherent obstacle clusters, estimates Time-to-Collision (TTC), and executes smooth evasive trajectories.

### 3. Predictive Object Detection (`predictive_yolo.py`)
- **What it does**: Performs semantic object recognition and metric 2D ground-plane distance tracking.
- **How it works**: Runs YOLOv8 to detect surrounding obstacles (pedestrians, obstacles, furniture) and tracks them across frames. It projects each object's ground contact point (bottom-center of bounding box) onto metric floor coordinates using the calibrated homography matrix $H$. It calculates dynamic Velocity Obstacles (VO) and Time-to-Collision (TTC) to proactively steer or stop before entering danger zones.

### 4. Ground Homography Calibrator (`homography_node.py`)
- **What it does**: Calibrates the geometric transformation matrix between camera pixel coordinates and metric ground coordinates ($Z=0$).
- **How it works**: Detects 8 ground-placed AprilTags (`DICT_APRILTAG_36h11`) with known physical coordinates, solves a least-squares perspective transformation, and automatically outputs the matrix to [`../config/homography_calibration.json`](file:///../config/homography_calibration.json).

---

## 4. Execution Commands

### Prerequisites
```bash
# Python dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy scipy
```

### Running Individual Nodes
```bash
# Option A: Farneback Dense Optical Flow (CPU)
ros2 run husky_obstacle_avoidance optical_flow_farneback.py

# Option B: RAFT Deep Optical Flow (GPU)
ros2 run husky_obstacle_avoidance raft_avoidance_node.py

# Option C: YOLOv8 Predictive Avoidance (GPU)
ros2 run husky_obstacle_avoidance predictive_yolo.py

# Option D: Homography Calibration Tool
ros2 run husky_obstacle_avoidance homography_node.py
```

### Running via Unified Launch File
```bash
# Launch with the chosen method: farneback (default), raft, yolo, or homography
ros2 launch husky_obstacle_avoidance avoidance.launch.py method:=farneback
ros2 launch husky_obstacle_avoidance avoidance.launch.py method:=raft
ros2 launch husky_obstacle_avoidance avoidance.launch.py method:=yolo
ros2 launch husky_obstacle_avoidance avoidance.launch.py method:=homography
```

---

## 5. ROS 2 Interface Specification

| Topic | Type | Direction | Description |
| :--- | :--- | :---: | :--- |
| `/camera/color/image_raw` | `sensor_msgs/msg/Image` | In | Raw RGB camera stream (1280x720 or 640x480) |
| `/odometry/filtered` | `nav_msgs/msg/Odometry` | In | Filtered robot wheel odometry & velocities |
| `/joy_teleop/cmd_vel` | `geometry_msgs/msg/Twist` | In | Operator joystick input (for override detection) |
| `/cmd_vel_in` | `geometry_msgs/msg/Twist` | In | Multiplexer output command |
| `/avoidance_cmd_vel` | `geometry_msgs/msg/Twist` | Out | Proactive evasive velocity command (Priority 50) |
| `/emergency_stop` | `std_msgs/msg/Bool` | Out | Collision safety brake lock (Priority 255) |
| `/movement_command` | `std_msgs/msg/String` | Out | Diagnostic state string (`CLEAR`, `WARNING`, `AVOID`, `EMERGENCY`) |
| `/optical_flow_zones` | `std_msgs/msg/Float64MultiArray` | Out | 18-channel optical flow feature diagnostics |
| `/ground_detections` | `std_msgs/msg/String` | Out | JSON-encoded metric ground detections (YOLO) |
| `/flow_diagnostics` | `std_msgs/msg/String` | Out | JSON-encoded flow diagnostic telemetry (RAFT) |
