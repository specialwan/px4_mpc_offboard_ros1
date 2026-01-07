#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Publisher with LOS Guidance DIRECT to PX4 (ROS1 + MAVROS)
Untuk membandingkan PID bawaan PX4 dengan LOS Guidance vs MPC dengan LOS Guidance

Perbedaan dengan trajectory_publisher_px4_direct.py:
- Menggunakan algoritma LOS Guidance untuk menghitung posisi referensi
- LOS lookahead point dikirim sebagai position setpoint
- Yaw mengikuti LOS angle (arah ke lookahead point)
- TIDAK mengirim referensi kecepatan (PX4 PID yang handle)

Published Topics:
    /mavros/setpoint_position/local (PoseStamped, ENU) - Position reference ke PX4
    /waypoint/target               (PoseStamped, ENU)  - Current target waypoint
    /trajectory/status             (String)            - Status messages
    /trajectory/los_info           (TwistStamped)      - LOS debug info

Subscribed Topics:
    /mavros/state                      - Drone state
    /mavros/local_position/pose        - Current pose (ENU)

Author: Generated for PX4 PID with LOS Guidance comparison
"""

import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import String


class Px4LOSGuidanceTrajectoryPublisher(object):
    """
    LOS Guidance Trajectory Publisher untuk PX4 PID Controller.
    
    Algoritma LOS (sama seperti MPC version):
    1. Cari segmen path terdekat dengan posisi drone
    2. Hitung cross-track error (e) = jarak tegak lurus ke path
    3. Hitung along-track position (s) = proyeksi pada path
    4. Lookahead point: titik di depan pada path dengan jarak delta
    5. LOS angle: sudut dari posisi drone ke lookahead point
    
    Perbedaan dengan MPC version:
    - HANYA kirim position reference (lookahead point)
    - TIDAK kirim velocity reference
    - PX4 PID controller yang handle tracking
    """

    def __init__(self):
        rospy.loginfo("="*60)
        rospy.loginfo("LOS Guidance Trajectory Publisher DIRECT to PX4")
        rospy.loginfo("Mode: PID Controller with LOS Guidance")
        rospy.loginfo("Output: /mavros/setpoint_position/local")
        rospy.loginfo("="*60)

        # ===================== LOS GUIDANCE PARAMETERS =====================
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 2.0)  # meter
        self.lookahead_min = rospy.get_param("~lookahead_min", 1.0)             # meter
        self.lookahead_max = rospy.get_param("~lookahead_max", 5.0)             # meter
        self.lookahead_adaptive = rospy.get_param("~lookahead_adaptive", False)  # Disabled untuk PX4
        self.lookahead_speed_gain = rospy.get_param("~lookahead_speed_gain", 0.5)
        
        # Altitude control (untuk 3D LOS)
        self.altitude_gain = rospy.get_param("~altitude_gain", 1.0)

        # ===================== PARAM TRAJECTORY =====================
        self.waypoint_mode = rospy.get_param("~waypoint_mode", "default")
        self.ref_speed = rospy.get_param("~ref_speed", 2.0)
        self.loop_trajectory = rospy.get_param("~loop_trajectory", False)
        self.start_at_current_pose = rospy.get_param("~start_at_current_pose", True)

        # Hover / gate params
        self.hover_altitude = rospy.get_param("~hover_altitude", 2.5)   # ENU +up (m)
        self.alt_tolerance = rospy.get_param("~alt_tolerance", 0.3)     # m
        self.hover_stable_time = rospy.get_param("~hover_stable_time", 1.0)  # s
        self.gate_on_altitude = rospy.get_param("~gate_on_altitude", True)

        # Acceptance / reach detection
        self.acceptance_radius_xy = rospy.get_param("~acceptance_radius_xy", 0.8)
        self.acceptance_alt_tol = rospy.get_param("~acceptance_alt_tol", 0.5)

        # Auto ARM / OFFBOARD
        self.auto_arm = rospy.get_param("~auto_arm", False)
        self.auto_offboard = rospy.get_param("~auto_offboard", False)

        # Circle params
        self.circle_center_n = rospy.get_param("~circle_center_n", 0.0)
        self.circle_center_e = rospy.get_param("~circle_center_e", 0.0)
        self.circle_radius = rospy.get_param("~circle_radius", 10.0)
        self.circle_altitude = rospy.get_param("~circle_altitude", -5.0)
        self.circle_points = int(rospy.get_param("~circle_points", 80))

        # Square params
        self.square_center_n = rospy.get_param("~square_center_n", 0.0)
        self.square_center_e = rospy.get_param("~square_center_e", 0.0)
        self.square_size = rospy.get_param("~square_size", 15.0)
        self.square_altitude = rospy.get_param("~square_altitude", -2.5)
        self.square_points_per_side = int(rospy.get_param("~square_points_per_side", 0))
        self.square_constant_yaw = rospy.get_param("~square_constant_yaw", False)

        # Helix params
        self.helix_center_n = rospy.get_param("~helix_center_n", 0.0)
        self.helix_center_e = rospy.get_param("~helix_center_e", 0.0)
        self.helix_radius = rospy.get_param("~helix_radius", 1.0)
        self.helix_start_altitude = rospy.get_param("~helix_start_altitude", -10.0)
        self.helix_end_altitude = rospy.get_param("~helix_end_altitude", -30.0)
        self.helix_turns = rospy.get_param("~helix_turns", 3.0)
        self.helix_points = int(rospy.get_param("~helix_points", 120))
        self.helix_direction = rospy.get_param("~helix_direction", "ccw")

        # Diamond ascending params
        self.diamond_center_n = rospy.get_param("~diamond_center_n", 0.0)
        self.diamond_center_e = rospy.get_param("~diamond_center_e", 0.0)
        self.diamond_size = rospy.get_param("~diamond_size", 7.5)
        self.diamond_start_altitude = rospy.get_param("~diamond_start_altitude", -5.0)
        self.diamond_end_altitude = rospy.get_param("~diamond_end_altitude", -10.0)
        self.diamond_points_per_side = int(rospy.get_param("~diamond_points_per_side", 0))

        # ===================== STATE TRAJECTORY =====================
        self.positions = None       # NED (N,E,D) array
        self.yaws = None
        self.path_points = None
        self.traj_initialized = False
        
        # LOS state
        self.current_segment_idx = 0
        self.along_track_distance = 0.0
        
        # Segment info untuk LOS
        self.segment_lengths = None
        self.cum_lengths = None
        self.total_path_length = 0.0

        # Mode internal
        self.mode = "WAIT_ALT"
        self.alt_ready_since = None
        
        self.trajectory_finished = False
        self.finish_published = False
        self.reached_flags = None
        self.final_hold_yaw = 0.0

        # ===================== STATE MAVROS =====================
        self.current_state = State()
        self.current_pos_enu = np.zeros(3)
        self.current_vel_enu = np.zeros(3)
        self.prev_pos_enu = np.zeros(3)
        self.prev_time = None
        self.current_yaw = 0.0
        self.pose_received = False

        # Auto ARM/OFFBOARD state
        self.setpoint_stream_started = False
        self.offboard_requested = False
        self.arm_requested = False
        self.setpoint_count = 0
        self.required_setpoints = 100

        # ===================== PUB / SUB =====================
        # PUBLISH LANGSUNG KE PX4 POSITION CONTROLLER
        self.setpoint_pub = rospy.Publisher(
            "/mavros/setpoint_position/local", PoseStamped, queue_size=20
        )
        
        # Publish monitoring topics
        self.waypoint_pub = rospy.Publisher(
            "/waypoint/target", PoseStamped, queue_size=10
        )
        self.status_pub = rospy.Publisher(
            "/trajectory/status", String, queue_size=10
        )
        self.los_info_pub = rospy.Publisher(
            "/trajectory/los_info", TwistStamped, queue_size=10
        )
        # Publisher untuk desired path (titik proyeksi pada path asli)
        self.desired_path_pub = rospy.Publisher(
            "/trajectory/desired_pose", PoseStamped, queue_size=10
        )

        self.state_sub = rospy.Subscriber(
            "/mavros/state", State, self.state_callback, queue_size=10
        )
        self.pos_sub = rospy.Subscriber(
            "/mavros/local_position/pose",
            PoseStamped,
            self.position_callback,
            queue_size=10
        )
        
        # Optional: subscribe to velocity untuk adaptive lookahead
        self.vel_sub = rospy.Subscriber(
            "/mavros/local_position/velocity_local",
            TwistStamped,
            self.velocity_callback,
            queue_size=10
        )

        # Service clients untuk ARM dan SET_MODE
        rospy.loginfo("Waiting for MAVROS services...")
        rospy.wait_for_service("/mavros/cmd/arming", timeout=30)
        rospy.wait_for_service("/mavros/set_mode", timeout=30)
        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        rospy.loginfo("✓ MAVROS services ready")

        if not self.start_at_current_pose:
            self.build_path(anchor_to_current=False)
        else:
            rospy.loginfo("start_at_current_pose = True → path akan di-anchor saat pose pertama diterima.")

        # Timer publish 50 Hz (sama dengan MPC LOS version)
        self.timer = rospy.Timer(rospy.Duration(0.02), self.timer_cb)
        
        rospy.loginfo("="*60)
        rospy.loginfo("✓ LOS Guidance Trajectory Publisher (PX4 Direct) initialized")
        rospy.loginfo("  Publishing to: /mavros/setpoint_position/local at 50Hz")
        rospy.loginfo("  Lookahead distance: %.2f m", self.lookahead_distance)
        rospy.loginfo("  Auto ARM: %s", self.auto_arm)
        rospy.loginfo("  Auto OFFBOARD: %s", self.auto_offboard)
        rospy.loginfo("="*60)

    # ======================================================================
    # MAVROS callbacks
    # ======================================================================
    def state_callback(self, msg: State):
        was_connected = self.current_state.connected if hasattr(self.current_state, 'connected') else False
        self.current_state = msg
        
        if msg.connected and not was_connected:
            rospy.loginfo("🔗 MAVROS connected to PX4!")
        elif not msg.connected and was_connected:
            rospy.logwarn("⚠️  MAVROS disconnected from PX4!")

    def position_callback(self, msg: PoseStamped):
        self.current_pos_enu[:] = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ]
        
        # Extract yaw from quaternion
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        self.current_yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # Estimate velocity from position change
        now = rospy.Time.now().to_sec()
        if self.prev_time is not None:
            dt = now - self.prev_time
            if dt > 0.001:
                self.current_vel_enu = (self.current_pos_enu - self.prev_pos_enu) / dt
        self.prev_pos_enu = self.current_pos_enu.copy()
        self.prev_time = now

        if not self.pose_received:
            self.pose_received = True
            rospy.loginfo("✓ First pose received: ENU (%.2f, %.2f, %.2f), Yaw: %.1f°",
                         self.current_pos_enu[0],
                         self.current_pos_enu[1],
                         self.current_pos_enu[2],
                         np.degrees(self.current_yaw))

            if self.start_at_current_pose and not self.traj_initialized:
                self.build_path(anchor_to_current=True)

    def velocity_callback(self, msg: TwistStamped):
        """Optional velocity callback untuk estimasi kecepatan."""
        self.current_vel_enu = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z
        ])

    # ======================================================================
    # BUILD PATH
    # ======================================================================
    def build_path(self, anchor_to_current=False):
        if anchor_to_current and not self.pose_received:
            rospy.logwarn("build_path dipanggil sebelum pose diterima, skip.")
            return

        if anchor_to_current:
            # Generate path at origin, then offset
            if self.waypoint_mode == "circle":
                path = self.generate_circle_waypoints(
                    center_n=0.0, center_e=0.0,
                    radius=self.circle_radius,
                    altitude=0.0,
                    num_points=self.circle_points
                )
            elif self.waypoint_mode == "square":
                path = self.generate_square_waypoints(
                    center_n=0.0, center_e=0.0,
                    size=self.square_size,
                    altitude=0.0,
                    points_per_side=self.square_points_per_side,
                    constant_yaw=self.square_constant_yaw
                )
            elif self.waypoint_mode == "helix":
                path = self.generate_helix_waypoints(
                    center_n=0.0, center_e=0.0,
                    radius=self.helix_radius,
                    start_altitude=0.0,
                    end_altitude=self.helix_end_altitude - self.helix_start_altitude,
                    turns=self.helix_turns,
                    num_points=self.helix_points,
                    direction=self.helix_direction
                )
            elif self.waypoint_mode == "diamond":
                path = self.generate_diamond_ascending_waypoints(
                    center_n=0.0, center_e=0.0,
                    size=self.diamond_size,
                    start_altitude=0.0,
                    end_altitude=self.diamond_end_altitude - self.diamond_start_altitude,
                    num_points_per_side=self.diamond_points_per_side,
                    initial_yaw=self.current_yaw
                )
            else:
                path = [
                    [0.0, 0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0, 0.0],
                    [5.0, 5.0, 0.0, np.pi/2],
                    [0.0, 5.0, 0.0, np.pi],
                    [0.0, 0.0, 0.0, -np.pi/2],
                ]

            # Offset ke posisi UAV
            n_uav = self.current_pos_enu[1]  # ENU Y -> NED N
            e_uav = self.current_pos_enu[0]  # ENU X -> NED E
            d_hover = -self.hover_altitude   # NED (down)

            first_n, first_e, first_d, _ = path[0]
            dn = n_uav - first_n
            de = e_uav - first_e

            if self.waypoint_mode == "helix" or self.waypoint_mode == "diamond":
                dd = d_hover - first_d
            else:
                dd = 0.0

            for wp in path:
                wp[0] += dn
                wp[1] += de
                if self.waypoint_mode == "helix" or self.waypoint_mode == "diamond":
                    wp[2] += dd
                else:
                    wp[2] = d_hover

            # Remove duplicate last point if same as first
            if len(path) > 1:
                first = path[0]
                last = path[-1]
                if (abs(first[0] - last[0]) < 1e-3 and 
                    abs(first[1] - last[1]) < 1e-3 and 
                    abs(first[2] - last[2]) < 1e-3):
                    path.pop(-1)

            rospy.loginfo(
                "🚀 Trajectory anchored ke UAV: start NED ≈ (%.2f, %.2f, %.2f)",
                n_uav, e_uav, d_hover
            )
        else:
            # Absolute path
            if self.waypoint_mode == "circle":
                path = self.generate_circle_waypoints(
                    center_n=self.circle_center_n,
                    center_e=self.circle_center_e,
                    radius=self.circle_radius,
                    altitude=self.circle_altitude,
                    num_points=self.circle_points
                )
            elif self.waypoint_mode == "square":
                path = self.generate_square_waypoints(
                    center_n=self.square_center_n,
                    center_e=self.square_center_e,
                    size=self.square_size,
                    altitude=self.square_altitude,
                    points_per_side=self.square_points_per_side,
                    constant_yaw=self.square_constant_yaw
                )
            elif self.waypoint_mode == "helix":
                path = self.generate_helix_waypoints(
                    center_n=self.helix_center_n,
                    center_e=self.helix_center_e,
                    radius=self.helix_radius,
                    start_altitude=self.helix_start_altitude,
                    end_altitude=self.helix_end_altitude,
                    turns=self.helix_turns,
                    num_points=self.helix_points,
                    direction=self.helix_direction
                )
            elif self.waypoint_mode == "diamond":
                path = self.generate_diamond_ascending_waypoints(
                    center_n=self.diamond_center_n,
                    center_e=self.diamond_center_e,
                    size=self.diamond_size,
                    start_altitude=self.diamond_start_altitude,
                    end_altitude=self.diamond_end_altitude,
                    num_points_per_side=self.diamond_points_per_side
                )
            else:
                path = [
                    [0.0, 0.0, -self.hover_altitude, 0.0],
                    [5.0, 0.0, -self.hover_altitude, 0.0],
                    [5.0, 5.0, -self.hover_altitude, np.pi/2],
                    [0.0, 5.0, -self.hover_altitude, np.pi],
                    [0.0, 0.0, -self.hover_altitude, -np.pi/2],
                ]

        if len(path) < 2:
            rospy.logwarn("Path hanya punya <=1 titik, menyalin titik pertama.")
            path.append(path[0])

        self.path_points = path
        self.reached_flags = [False] * len(self.path_points)
        if len(self.path_points) > 0:
            self.reached_flags[0] = True

        # Precompute segment info for LOS
        self._precompute_segments()
        self.traj_initialized = True

        # Reset LOS state
        self.current_segment_idx = 0
        self.along_track_distance = 0.0

        self.mode = "WAIT_ALT"
        self.alt_ready_since = None
        self.trajectory_finished = False
        self.finish_published = False
        self.final_hold_yaw = 0.0

        rospy.loginfo("=" * 60)
        rospy.loginfo("LOS Guidance Trajectory Publisher (PX4 Direct)")
        rospy.loginfo("Mode trajektori      : %s", self.waypoint_mode)
        rospy.loginfo("Jumlah titik         : %d", len(self.path_points))
        rospy.loginfo("Total path length    : %.2f m", self.total_path_length)
        rospy.loginfo("Lookahead distance   : %.2f m", self.lookahead_distance)
        rospy.loginfo("Loop trajectory      : %s", "Yes" if self.loop_trajectory else "No")
        rospy.loginfo("Hover altitude       : %.2f m ENU", self.hover_altitude)
        rospy.loginfo("Altitude gate        : %s", "ON" if self.gate_on_altitude else "OFF")
        rospy.loginfo("=" * 60)

    def _precompute_segments(self):
        """Precompute segment lengths dan cumulative lengths untuk LOS."""
        pts = np.array(self.path_points)
        self.positions = pts[:, 0:3]
        self.yaws = pts[:, 3]

        deltas = self.positions[1:, :] - self.positions[:-1, :]
        self.segment_lengths = np.linalg.norm(deltas, axis=1)
        
        # Cumulative lengths
        self.cum_lengths = np.zeros(len(self.segment_lengths) + 1)
        self.cum_lengths[1:] = np.cumsum(self.segment_lengths)
        self.total_path_length = float(self.cum_lengths[-1])

    # ======================================================================
    # LOS GUIDANCE ALGORITHM
    # ======================================================================
    def compute_los_guidance(self, pos_ned):
        """
        Hitung LOS guidance berdasarkan posisi saat ini.
        
        Args:
            pos_ned: Current position dalam NED frame [N, E, D]
            
        Returns:
            los_point_ned: Lookahead point position (NED)
            desired_point_ned: Titik proyeksi pada path (NED) - desired path asli
            yaw_ref: Yaw reference (rad)
            cross_track_error: Cross-track error (meter)
            along_track: Along-track position (meter)
        """
        if self.positions is None or len(self.positions) < 2:
            return pos_ned, pos_ned, 0.0, 0.0, 0.0

        pos = pos_ned[:2]  # XY position (N, E)
        pos_z = pos_ned[2]  # Altitude (D)
        
        # 1. Cari segmen terdekat dengan posisi saat ini
        min_dist = float('inf')
        closest_seg_idx = self.current_segment_idx
        closest_projection = None
        closest_tau = 0.0
        
        # Search in a window around current segment
        search_start = max(0, self.current_segment_idx - 2)
        search_end = min(len(self.positions) - 1, self.current_segment_idx + 5)
        
        for i in range(search_start, search_end):
            p0 = self.positions[i][:2]
            p1 = self.positions[i + 1][:2]
            
            # Project posisi ke segmen
            seg_vec = p1 - p0
            seg_len = np.linalg.norm(seg_vec)
            
            if seg_len < 1e-6:
                tau = 0.0
                proj = p0
            else:
                tau = np.dot(pos - p0, seg_vec) / (seg_len * seg_len)
                tau = np.clip(tau, 0.0, 1.0)
                proj = p0 + tau * seg_vec
            
            dist = np.linalg.norm(pos - proj)
            
            if dist < min_dist:
                min_dist = dist
                closest_seg_idx = i
                closest_projection = proj
                closest_tau = tau
        
        # Update current segment index
        self.current_segment_idx = closest_seg_idx
        
        # 2. Hitung cross-track error dan along-track position
        cross_track_error = min_dist
        
        # Determine sign of cross-track error
        p0 = self.positions[closest_seg_idx][:2]
        p1 = self.positions[closest_seg_idx + 1][:2]
        seg_vec = p1 - p0
        
        # Normal vector (perpendicular, pointing right)
        normal = np.array([seg_vec[1], -seg_vec[0]])
        normal_len = np.linalg.norm(normal)
        if normal_len > 1e-6:
            normal = normal / normal_len
        
        # Cross-track error dengan sign
        cross_track_signed = np.dot(pos - closest_projection, normal)
        
        # Along-track distance dari start
        along_track = self.cum_lengths[closest_seg_idx] + closest_tau * self.segment_lengths[closest_seg_idx]
        self.along_track_distance = along_track
        
        # 2b. Desired point = waypoint tujuan yang sebenarnya
        desired_point_ned = self.positions[closest_seg_idx + 1].copy()
        
        # 3. Lookahead distance (fixed atau adaptive)
        if self.lookahead_adaptive:
            speed = np.linalg.norm(self.current_vel_enu[:2])
            delta = self.lookahead_distance + self.lookahead_speed_gain * speed
            delta = np.clip(delta, self.lookahead_min, self.lookahead_max)
        else:
            delta = self.lookahead_distance
        
        # 4. Cari lookahead point pada path
        lookahead_along = along_track + delta
        
        # Handle looping atau end of path
        if self.loop_trajectory:
            lookahead_along = np.fmod(lookahead_along, self.total_path_length)
            if lookahead_along < 0:
                lookahead_along += self.total_path_length
        else:
            lookahead_along = np.clip(lookahead_along, 0, self.total_path_length)
        
        # Cari segmen untuk lookahead point
        los_seg_idx = np.searchsorted(self.cum_lengths, lookahead_along, side='right') - 1
        los_seg_idx = int(np.clip(los_seg_idx, 0, len(self.segment_lengths) - 1))
        
        # Interpolasi posisi lookahead
        if self.segment_lengths[los_seg_idx] > 1e-6:
            tau_los = (lookahead_along - self.cum_lengths[los_seg_idx]) / self.segment_lengths[los_seg_idx]
            tau_los = np.clip(tau_los, 0.0, 1.0)
        else:
            tau_los = 0.0
        
        p0_los = self.positions[los_seg_idx]
        p1_los = self.positions[los_seg_idx + 1]
        los_point_ned = (1.0 - tau_los) * p0_los + tau_los * p1_los
        
        # 5. Hitung LOS angle (yaw reference)
        los_vec_xy = los_point_ned[:2] - pos
        los_dist_xy = np.linalg.norm(los_vec_xy)
        
        if los_dist_xy > 1e-6:
            # LOS angle: arctan2(E, N) untuk NED frame
            los_angle = np.arctan2(los_vec_xy[1], los_vec_xy[0])
        else:
            # Fallback ke path direction
            los_angle = np.arctan2(seg_vec[1], seg_vec[0])
        
        yaw_ref = los_angle
        
        return los_point_ned, desired_point_ned, yaw_ref, cross_track_signed, along_track

    # ======================================================================
    # TIMER CALLBACK - PUBLISH KE PX4
    # ======================================================================
    def timer_cb(self, event):
        # ======== SETPOINT STREAM COUNTER ========
        if not self.setpoint_stream_started:
            self.setpoint_stream_started = True
            rospy.loginfo("📡 Starting setpoint stream (need %d points before OFFBOARD)", 
                         self.required_setpoints)

        if self.setpoint_count < self.required_setpoints:
            self.setpoint_count += 1
            if self.setpoint_count % 20 == 0:
                rospy.loginfo("📡 Setpoint stream: %d/%d", 
                             self.setpoint_count, self.required_setpoints)
        elif self.setpoint_count == self.required_setpoints:
            self.setpoint_count += 1
            if not (self.auto_arm or self.auto_offboard):
                rospy.loginfo("✅ Setpoint stream ready! You can now switch to OFFBOARD mode manually.")

        # ======== AUTO ARM / OFFBOARD SEQUENCE ========
        if self.auto_arm or self.auto_offboard:
            if self.auto_offboard and \
               self.setpoint_count >= self.required_setpoints and \
               not self.offboard_requested:
                if self.current_state.mode != "OFFBOARD":
                    try:
                        resp = self.set_mode_client(custom_mode="OFFBOARD")
                        if resp.mode_sent:
                            rospy.loginfo("✈️  OFFBOARD mode requested")
                            self.offboard_requested = True
                        else:
                            rospy.logwarn("OFFBOARD request failed")
                    except rospy.ServiceException as e:
                        rospy.logerr("Set mode service call failed: %s", e)
                else:
                    self.offboard_requested = True

            if self.auto_arm and \
               self.offboard_requested and \
               self.current_state.mode == "OFFBOARD" and \
               not self.arm_requested:
                if not self.current_state.armed:
                    try:
                        resp = self.arming_client(value=True)
                        if resp.success:
                            rospy.loginfo("🚁 ARM requested")
                            self.arm_requested = True
                        else:
                            rospy.logwarn("ARM request failed")
                    except rospy.ServiceException as e:
                        rospy.logerr("Arming service call failed: %s", e)
                else:
                    self.arm_requested = True

        # ======== PUBLISH SETPOINT ========
        if not self.traj_initialized:
            if self.pose_received:
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "map"
                pose_msg.pose.position.x = float(self.current_pos_enu[0])
                pose_msg.pose.position.y = float(self.current_pos_enu[1])
                pose_msg.pose.position.z = float(self.current_pos_enu[2])
                
                yaw_enu = self.current_yaw
                qz = np.sin(yaw_enu / 2.0)
                qw = np.cos(yaw_enu / 2.0)
                pose_msg.pose.orientation.x = 0.0
                pose_msg.pose.orientation.y = 0.0
                pose_msg.pose.orientation.z = float(qz)
                pose_msg.pose.orientation.w = float(qw)
                
                self.setpoint_pub.publish(pose_msg)
                rospy.loginfo_throttle(2.0, "📡 Publishing setpoint (hover at current pos) - waiting for trajectory init...")
            else:
                rospy.loginfo_throttle(2.0, "⏳ Waiting for first pose from MAVROS...")
            return

        # ======== Altitude gate check ========
        if self.mode == "WAIT_ALT" and self.gate_on_altitude and self.pose_received:
            alt = self.current_pos_enu[2]
            if abs(alt - self.hover_altitude) <= self.alt_tolerance:
                if self.alt_ready_since is None:
                    self.alt_ready_since = rospy.Time.now().to_sec()
                else:
                    if (rospy.Time.now().to_sec() - self.alt_ready_since) >= self.hover_stable_time:
                        self.mode = "TRACK"
                        rospy.loginfo(
                            "✅ Altitude gate passed: z=%.2f m, start LOS tracking.",
                            alt
                        )
            else:
                self.alt_ready_since = None

        if not self.gate_on_altitude and self.mode == "WAIT_ALT":
            self.mode = "TRACK"
            rospy.loginfo("Altitude gate disabled → langsung TRACK mode.")

        # ==========================================================
        # Hitung referensi
        # ==========================================================
        if self.mode == "WAIT_ALT":
            # Hover at first waypoint
            p0 = self.positions[0]
            yaw0 = self.yaws[0]

            p_hover = np.array([p0[0], p0[1], -self.hover_altitude])
            enu_pos = np.array([p_hover[1], p_hover[0], -p_hover[2]])
            enu_desired = enu_pos.copy()
            yaw_ref = yaw0
            cross_track_error = 0.0
            along_track = 0.0
            
        else:  # TRACK mode dengan LOS
            # Convert current position ENU -> NED
            pos_ned = np.array([
                self.current_pos_enu[1],  # N
                self.current_pos_enu[0],  # E
                -self.current_pos_enu[2]  # D
            ])
            
            # Jika trajectory sudah selesai, HOLD di posisi terakhir
            if self.trajectory_finished:
                last_wp = self.positions[-1]
                last_yaw = self.yaws[-1]
                
                enu_pos = np.array([last_wp[1], last_wp[0], -last_wp[2]])
                enu_desired = enu_pos.copy()
                yaw_ref = last_yaw
                cross_track_error = 0.0
                along_track = self.total_path_length
                
                rospy.loginfo_throttle(5.0, "🛑 HOLDING at final waypoint (yaw=%.1f°)", np.degrees(last_yaw))
            else:
                # Compute LOS guidance
                los_point_ned, desired_point_ned, yaw_ref, cross_track_error, along_track = \
                    self.compute_los_guidance(pos_ned)
                
                # Check trajectory completion
                if not self.loop_trajectory:
                    progress = along_track / self.total_path_length if self.total_path_length > 0 else 0.0
                    rospy.loginfo_throttle(2.0, 
                        f"🛤️ LOS Progress: {progress*100:.1f}% | Cross-track: {cross_track_error:.2f}m | Along-track: {along_track:.1f}m")
                    
                    # Check if reached end
                    if along_track >= self.total_path_length - 0.5:
                        last_wp = self.positions[-1]
                        dist_to_end = np.linalg.norm(pos_ned[:2] - last_wp[:2])
                        
                        if dist_to_end < self.acceptance_radius_xy:
                            if not self.trajectory_finished:
                                self.trajectory_finished = True
                                self.final_hold_yaw = self.yaws[-1]
                                rospy.loginfo("✅ Trajectory FINISHED (LOS: reached end of path)")
                                rospy.loginfo("🛑 Switching to HOLD mode at final waypoint")
                                if not self.finish_published:
                                    status_msg = String()
                                    status_msg.data = "FINISHED"
                                    self.status_pub.publish(status_msg)
                                    self.finish_published = True
                
                # Convert NED -> ENU
                enu_pos = np.array([los_point_ned[1], los_point_ned[0], -los_point_ned[2]])
                enu_desired = np.array([desired_point_ned[1], desired_point_ned[0], -desired_point_ned[2]])

        # ===================== PUBLISH KE PX4 =====================
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(enu_pos[0])
        pose_msg.pose.position.y = float(enu_pos[1])
        pose_msg.pose.position.z = float(enu_pos[2])

        # Yaw: NED ke ENU frame
        # NED: 0° = North, 90° = East (clockwise)
        # ENU: 0° = East, 90° = North (counter-clockwise)
        # Konversi: yaw_enu = pi/2 - yaw_ned
        yaw_enu = np.pi / 2.0 - yaw_ref
        
        # Yaw → quaternion
        qz = np.sin(yaw_enu / 2.0)
        qw = np.cos(yaw_enu / 2.0)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = float(qz)
        pose_msg.pose.orientation.w = float(qw)

        # Desired pose (titik proyeksi pada path asli)
        desired_msg = PoseStamped()
        desired_msg.header.stamp = pose_msg.header.stamp
        desired_msg.header.frame_id = "map"
        desired_msg.pose.position.x = float(enu_desired[0])
        desired_msg.pose.position.y = float(enu_desired[1])
        desired_msg.pose.position.z = float(enu_desired[2])
        desired_msg.pose.orientation = pose_msg.pose.orientation

        # LOS info (for debugging/logging)
        los_info_msg = TwistStamped()
        los_info_msg.header.stamp = pose_msg.header.stamp
        los_info_msg.header.frame_id = "map"
        los_info_msg.twist.linear.x = float(cross_track_error)  # Cross-track error
        los_info_msg.twist.linear.y = float(along_track)        # Along-track distance
        los_info_msg.twist.linear.z = float(self.current_segment_idx)  # Current segment
        los_info_msg.twist.angular.x = 0.0  # No integral for PX4 version
        los_info_msg.twist.angular.y = float(self.lookahead_distance)
        los_info_msg.twist.angular.z = float(yaw_ref)  # LOS yaw

        # PUBLISH!
        self.setpoint_pub.publish(pose_msg)
        self.waypoint_pub.publish(pose_msg)
        self.los_info_pub.publish(los_info_msg)
        self.desired_path_pub.publish(desired_msg)

        # Publish trajectory status
        status_msg = String()
        if self.trajectory_finished:
            status_msg.data = "HOLD"
        else:
            status_msg.data = self.mode
        self.status_pub.publish(status_msg)

    # ======================================================================
    # TRAJECTORY GENERATORS (NED)
    # ======================================================================
    def generate_circle_waypoints(self, center_n=0.0, center_e=0.0,
                                  radius=25.0, altitude=-5.0, num_points=16):
        waypoints = []
        for i in range(num_points):
            angle = 2.0 * np.pi * i / num_points
            n = center_n + radius * np.cos(angle)
            e = center_e + radius * np.sin(angle)
            d = altitude
            yaw = angle + np.pi / 2.0
            waypoints.append([float(n), float(e), float(d), float(yaw)])

        circumference = 2.0 * np.pi * radius
        avg_spacing = circumference / num_points
        rospy.loginfo(
            "Circle trajectory: center=(%.1f,%.1f), radius=%.1f m, alt=%.1f m, %d points, spacing=%.2f m",
            center_n, center_e, radius, -altitude, num_points, avg_spacing
        )
        return waypoints

    def generate_square_waypoints(self, center_n=0.0, center_e=0.0, size=40.0,
                                  altitude=-5.0, points_per_side=3, constant_yaw=True):
        waypoints = []
        half = size / 2.0

        corners = [
            [center_n - half, center_e - half],
            [center_n + half, center_e - half],
            [center_n + half, center_e + half],
            [center_n - half, center_e + half],
        ]

        for i in range(4):
            start_corner = corners[i]
            end_corner = corners[(i + 1) % 4]

            dn = end_corner[0] - start_corner[0]
            de = end_corner[1] - start_corner[1]
            yaw_current = np.arctan2(de, dn)

            next_corner = corners[(i + 2) % 4]
            dn_next = next_corner[0] - end_corner[0]
            de_next = next_corner[1] - end_corner[1]
            yaw_next = np.arctan2(de_next, dn_next)

            num_points = points_per_side + 1
            for j in range(num_points):
                t = float(j) / float(points_per_side + 1)
                n = start_corner[0] + t * dn
                e = start_corner[1] + t * de

                if constant_yaw:
                    yaw = 0.0
                elif t < 0.8:
                    yaw = yaw_current
                else:
                    blend = (t - 0.8) / 0.2
                    yaw_diff = yaw_next - yaw_current
                    yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
                    yaw = yaw_current + blend * yaw_diff

                waypoints.append([float(n), float(e), float(altitude), float(yaw)])

        if len(waypoints) > 0:
            first = waypoints[0]
            waypoints.append([first[0], first[1], first[2], first[3]])

        return waypoints

    def generate_helix_waypoints(self, center_n=0.0, center_e=0.0, radius=1.0,
                                 start_altitude=-10.0, end_altitude=-30.0,
                                 turns=3.0, num_points=120, add_transition=True,
                                 direction="ccw"):
        waypoints = []
        if add_transition:
            waypoints.append([float(center_n), float(center_e),
                              float(start_altitude), 0.0])

        dir_mult = -1.0 if direction.lower() == "cw" else 1.0
        
        for i in range(num_points):
            t = float(i) / float(num_points - 1)
            angle = dir_mult * 2.0 * np.pi * turns * t

            n = center_n + radius * np.cos(angle)
            e = center_e + radius * np.sin(angle)
            d = start_altitude + t * (end_altitude - start_altitude)
            yaw = angle + dir_mult * np.pi / 2.0
            waypoints.append([float(n), float(e), float(d), float(yaw)])

        return waypoints

    def generate_diamond_ascending_waypoints(self, center_n=0.0, center_e=0.0, 
                                            size=15.0, start_altitude=-5.0, 
                                            end_altitude=-10.0, num_points_per_side=20,
                                            initial_yaw=0.0):
        waypoints = []
        
        corners_world = [
            [-size, 0.0],
            [0.0, size],
            [size, 0.0],
            [0.0, -size],
        ]
        
        if num_points_per_side == 0:
            altitude_step = (end_altitude - start_altitude) / 4.0
            
            for i in range(5):
                corner_idx = i % 4
                corner = corners_world[corner_idx]
                
                n = center_n + corner[0]
                e = center_e + corner[1]
                d = start_altitude + i * altitude_step
                
                if i == 0:
                    next_corner = corners_world[1]
                    dn = next_corner[0] - corner[0]
                    de = next_corner[1] - corner[1]
                else:
                    prev_corner_idx = (corner_idx - 1 + 4) % 4
                    prev_corner = corners_world[prev_corner_idx]
                    dn = corner[0] - prev_corner[0]
                    de = corner[1] - prev_corner[1]
                
                yaw = float(np.arctan2(de, dn))
                waypoints.append([float(n), float(e), float(d), yaw])
            
            return waypoints
        
        total_points = num_points_per_side * 4
        altitude_step = (end_altitude - start_altitude) / float(total_points - 1) if total_points > 1 else 0.0
        
        point_idx = 0
        
        for side_idx in range(4):
            corner_start = corners_world[side_idx]
            corner_end = corners_world[(side_idx + 1) % 4]
            
            dn = corner_end[0] - corner_start[0]
            de = corner_end[1] - corner_start[1]
            side_yaw = float(np.arctan2(de, dn))
            
            for i in range(num_points_per_side):
                t = float(i) / float(num_points_per_side)
                n = center_n + (1 - t) * corner_start[0] + t * corner_end[0]
                e = center_e + (1 - t) * corner_start[1] + t * corner_end[1]
                d = start_altitude + point_idx * altitude_step
                
                waypoints.append([float(n), float(e), float(d), side_yaw])
                point_idx += 1
        
        return waypoints


# ======================================================================
# MAIN
# ======================================================================
def main():
    rospy.init_node("trajectory_publisher_px4_los_guidance", anonymous=False)
    
    try:
        node = Px4LOSGuidanceTrajectoryPublisher()
        rospy.loginfo("LOS Guidance Trajectory Publisher (PX4 Direct) started.")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    
    rospy.loginfo("LOS Guidance Trajectory Publisher (PX4 Direct) shutdown.")


if __name__ == "__main__":
    main()
