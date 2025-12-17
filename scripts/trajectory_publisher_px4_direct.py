#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Publisher DIRECT to PX4 (ROS1 + MAVROS)
Untuk membandingkan PID bawaan PX4 vs MPC

- Generate trajectory yang SAMA dengan trajectory_publisher_mavros_gated.py
- Publish LANGSUNG ke /mavros/setpoint_position/local (PX4 position controller)
- BUKAN ke /trajectory/ref_pose (MPC)
- PX4 akan menggunakan controller PID bawaannya
- Altitude gate, waypoint follow, semua fitur tetap sama

Perbedaan:
- Publisher: /mavros/setpoint_position/local (PoseStamped)
- Target: PX4 position controller (PID)
- Logging disesuaikan untuk membandingkan performa
"""

import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from std_msgs.msg import String


class Px4TrajectoryPublisherDirect(object):
    def __init__(self):
        rospy.loginfo("="*60)
        rospy.loginfo("Trajectory Publisher DIRECT to PX4 Position Controller")
        rospy.loginfo("Mode: PID Controller Comparison (PX4 vs MPC)")
        rospy.loginfo("Output: /mavros/setpoint_position/local")
        rospy.loginfo("="*60)

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
        self.advance_on_reach = rospy.get_param("~advance_on_reach", True)

        # Waypoint follow mode
        self.waypoint_follow_mode = rospy.get_param("~waypoint_follow_mode", True)
        self.current_wp_idx = 1

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
        self.square_constant_yaw = rospy.get_param("~square_constant_yaw", True)

        # Helix params
        self.helix_center_n = rospy.get_param("~helix_center_n", 0.0)
        self.helix_center_e = rospy.get_param("~helix_center_e", 0.0)
        self.helix_radius = rospy.get_param("~helix_radius", 1.0)
        self.helix_start_altitude = rospy.get_param("~helix_start_altitude", -10.0)
        self.helix_end_altitude = rospy.get_param("~helix_end_altitude", -30.0)
        self.helix_turns = rospy.get_param("~helix_turns", 3.0)
        self.helix_points = int(rospy.get_param("~helix_points", 120))

        # Diamond ascending params
        self.diamond_center_n = rospy.get_param("~diamond_center_n", 0.0)
        self.diamond_center_e = rospy.get_param("~diamond_center_e", 0.0)
        self.diamond_size = rospy.get_param("~diamond_size", 7.5)
        self.diamond_start_altitude = rospy.get_param("~diamond_start_altitude", -5.0)
        self.diamond_end_altitude = rospy.get_param("~diamond_end_altitude", -10.0)
        self.diamond_points_per_side = int(rospy.get_param("~diamond_points_per_side", 0))

        # ===================== STATE TRAJECTORY =====================
        self.positions = None  # NED (N,E,D)
        self.yaws = None
        self.segment_lengths = None
        self.segment_times = None
        self.cum_times = None
        self.total_time = 0.0
        self.path_points = None
        self.traj_initialized = False

        self.t0 = None
        self.mode = "WAIT_ALT"
        self.alt_ready_since = None
        
        self.trajectory_finished = False
        self.finish_published = False
        self.reached_flags = None
        
        # Flag untuk dynamic yaw (mengikuti arah gerakan)
        # False jika square dengan constant_yaw=True
        self.use_dynamic_yaw = True
        
        # Previous yaw untuk kontinuitas (hindari flip 180°)
        self.previous_yaw = 0.0
        self.just_advanced_waypoint = False  # Flag untuk track apakah baru saja advance waypoint

        # ===================== STATE MAVROS =====================
        self.current_state = State()
        self.current_pos_enu = np.zeros(3)
        self.current_yaw = 0.0  # Current heading/yaw dari drone (rad)
        self.pose_received = False

        # Auto ARM/OFFBOARD state
        self.setpoint_stream_started = False
        self.offboard_requested = False
        self.arm_requested = False
        self.setpoint_count = 0
        self.required_setpoints = 100  # Stream 100 setpoints sebelum OFFBOARD

        # ===================== PUB / SUB =====================
        # PUBLISH LANGSUNG KE PX4 POSITION CONTROLLER
        self.setpoint_pub = rospy.Publisher(
            "/mavros/setpoint_position/local", PoseStamped, queue_size=10
        )
        
        # Publish juga ke topic monitoring (opsional)
        self.waypoint_pub = rospy.Publisher(
            "/waypoint/target", PoseStamped, queue_size=10
        )
        self.status_pub = rospy.Publisher(
            "/trajectory/status", String, queue_size=10
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

        # Timer publish 20 Hz
        self.timer = rospy.Timer(rospy.Duration(0.05), self.timer_cb)
        
        rospy.loginfo("="*60)
        rospy.loginfo("✓ Trajectory Publisher (PX4 Direct) initialized")
        rospy.loginfo("  Publishing to: /mavros/setpoint_position/local at 20Hz")
        rospy.loginfo("  Auto ARM: %s", self.auto_arm)
        rospy.loginfo("  Auto OFFBOARD: %s", self.auto_offboard)
        if not (self.auto_arm or self.auto_offboard):
            rospy.loginfo("  Manual mode: You need to ARM and switch to OFFBOARD manually")
        rospy.loginfo("="*60)

    # ======================================================================
    # MAVROS callbacks
    # ======================================================================
    def state_callback(self, msg: State):
        was_connected = self.current_state.connected if hasattr(self.current_state, 'connected') else False
        self.current_state = msg
        
        # Log connection state changes
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
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = np.arctan2(siny_cosp, cosy_cosp)

        if not self.pose_received:
            self.pose_received = True
            rospy.loginfo("✓ First pose received: ENU (%.2f, %.2f, %.2f), Yaw: %.1f°",
                         self.current_pos_enu[0],
                         self.current_pos_enu[1],
                         self.current_pos_enu[2],
                         np.degrees(self.current_yaw))

            if self.start_at_current_pose and not self.traj_initialized:
                self.build_path(anchor_to_current=True)

    # ======================================================================
    # YAW HELPER FUNCTIONS
    # ======================================================================
    def normalize_angle(self, angle):
        """Normalize angle ke range [-pi, pi]"""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
    
    def get_continuous_yaw(self, direction_vector, prev_yaw):
        """
        Hitung yaw dari direction vector dengan mempertimbangkan kontinuitas.
        Direction vector: [N, E] dalam NED frame (dari posisi current ke target)
        
        Problem: Saat waypoint berpindah, prev_yaw bisa berbeda jauh,
        menyebabkan yaw baru perlu dipilih yang kontinyu.
        """
        # Normalize direction vector
        norm = np.linalg.norm(direction_vector)
        if norm < 1e-6:
            return prev_yaw  # Jika vector terlalu kecil, gunakan prev_yaw
        
        # Hitung yaw dari direction vector: yaw = atan2(E, N)
        yaw_base = float(np.arctan2(direction_vector[1], direction_vector[0]))
        yaw_base = self.normalize_angle(yaw_base)
        
        # Coba 2 kemungkinan: yaw_base dan yaw_base + 180°
        # (karena kadang prev_yaw bisa wrap around -180/180)
        candidates = [
            yaw_base,
            self.normalize_angle(yaw_base + np.pi),
            self.normalize_angle(yaw_base - np.pi)
        ]
        
        # Pilih yang paling dekat dengan prev_yaw
        best_yaw = yaw_base
        min_diff = abs(self.normalize_angle(yaw_base - prev_yaw))
        
        for candidate in candidates:
            diff = abs(self.normalize_angle(candidate - prev_yaw))
            if diff < min_diff:
                min_diff = diff
                best_yaw = candidate
        
        return self.normalize_angle(best_yaw)

    # ======================================================================
    # BUILD PATH + PRECOMPUTE (SAMA dengan trajectory_publisher_mavros_gated.py)
    # ======================================================================
    def build_path(self, anchor_to_current=False):
        if anchor_to_current and not self.pose_received:
            rospy.logwarn("build_path dipanggil sebelum pose diterima, skip.")
            return

        if anchor_to_current:
            # Generate path relative to origin
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
                    constant_yaw=self.square_constant_yaw,
                    # initial_yaw=self.current_yaw
                )
            elif self.waypoint_mode == "helix":
                path = self.generate_helix_waypoints(
                    center_n=0.0, center_e=0.0,
                    radius=self.helix_radius,
                    start_altitude=0.0,
                    end_altitude=self.helix_end_altitude - self.helix_start_altitude,
                    turns=self.helix_turns,
                    num_points=self.helix_points
                )
            elif self.waypoint_mode == "diamond":
                path = self.generate_diamond_ascending_waypoints(
                    center_n=0.0, center_e=0.0,
                    size=self.diamond_size,
                    start_altitude=0.0,
                    end_altitude=self.diamond_end_altitude - self.diamond_start_altitude,
                    num_points_per_side=self.diamond_points_per_side
                )
            else:
                # DEFAULT: Diamond/belah ketupat dengan altitude meningkat
                # Generate path dengan yaw yang benar (dari direction antar waypoint)
                positions_temp = [
                    [0.0, 0.0, 0.0],           # wp0: start (5m)
                    [7.5, 0.0, -1.25],         # wp1: North (6.25m)
                    [0.0, 7.5, -2.5],          # wp2: East (7.5m)
                    [-7.5, 0.0, -3.75],        # wp3: South (8.75m)
                    [0.0, -7.5, -5.0],         # wp4: West (10m)
                    [7.5, 0.0, 0.0],           # wp5: back to North (5m)
                ]
                
                # Hitung yaw dari direction waypoint ke waypoint
                path = []
                for i in range(len(positions_temp)):
                    pos = positions_temp[i]
                    if i < len(positions_temp) - 1:
                        # Yaw dari current ke next waypoint
                        next_pos = positions_temp[i + 1]
                        dn = next_pos[0] - pos[0]
                        de = next_pos[1] - pos[1]
                        yaw = float(np.arctan2(de, dn))
                    else:
                        # Last waypoint, use previous yaw
                        yaw = path[-1][3] if len(path) > 0 else 0.0
                    
                    path.append([pos[0], pos[1], pos[2], yaw])

            # Offset to UAV position
            n_uav = self.current_pos_enu[1]
            e_uav = self.current_pos_enu[0]
            d_hover = -self.hover_altitude

            first_n, first_e, first_d, _ = path[0]
            dn = n_uav - first_n
            de = e_uav - first_e
            
            # Untuk semua mode, preserve altitude variation relative to first waypoint
            dd = d_hover - first_d

            for wp in path:
                wp[0] += dn  # Offset North
                wp[1] += de  # Offset East
                wp[2] += dd  # Offset altitude (preserving relative variation)

            # Jika titik terakhir sama dengan titik pertama (penutupan loop), hapus duplikat terakhir
            if len(path) > 1:
                first = path[0]
                last = path[-1]
                # Bandingkan N, E, dan altitude (toleransi kecil)
                if (abs(first[0] - last[0]) < 1e-3) and (abs(first[1] - last[1]) < 1e-3) and (abs(first[2] - last[2]) < 1e-3):
                    path.pop(-1)
                    rospy.loginfo("🔄 Removed duplicate closing waypoint")

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
                    num_points=self.helix_points
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
                # DEFAULT: 5 waypoint dengan altitude bervariasi
                # Cocok untuk testing dengan waypoint follow mode
                path = [
                    [0.0, 0.0, -self.hover_altitude, 0.0],      # wp0: start
                    [7.5, 0.0, -self.hover_altitude, 0.0],      # wp1: 7.5m North
                    [-7.5, 7.5, -(self.hover_altitude + 2.5), np.pi/2],  # wp2: Southwest, +2.5m higher
                    [-7.5, -7.5, -(self.hover_altitude + 5.0), np.pi],   # wp3: Northwest, +5m higher
                    [7.5, -7.5, -(self.hover_altitude + 2.5), -np.pi/2], # wp4: Northeast, +2.5m higher
                    [0.0, 0.0, -self.hover_altitude, 0.0],      # wp5: back to start
                ]

        # Store trajectory
        self.path_points = np.array(path, dtype=float)
        self.positions = self.path_points[:, 0:3]
        self.yaws = self.path_points[:, 3]
        
        # DEBUG: Log generated yaws
        rospy.loginfo("=" * 60)
        rospy.loginfo("Generated waypoint yaws:")
        for i, yaw in enumerate(self.yaws):
            rospy.loginfo("  wp[%d]: yaw=%.1f° (%.3f rad)", i, np.degrees(yaw), yaw)
        rospy.loginfo("=" * 60)

        # Compute segment lengths and times
        n_seg = len(self.positions) - 1
        self.segment_lengths = np.zeros(n_seg, dtype=float)
        self.segment_times = np.zeros(n_seg, dtype=float)

        for i in range(n_seg):
            dx = self.positions[i+1] - self.positions[i]
            length = float(np.linalg.norm(dx))
            self.segment_lengths[i] = length

            if self.ref_speed > 0.0:
                self.segment_times[i] = length / self.ref_speed
            else:
                self.segment_times[i] = 1.0

        self.cum_times = np.zeros(len(self.segment_times) + 1, dtype=float)
        for i in range(len(self.segment_times)):
            self.cum_times[i+1] = self.cum_times[i] + self.segment_times[i]

        self.total_time = float(self.cum_times[-1])
        self.traj_initialized = True
        
        # Set dynamic yaw behavior
        # Hanya square dengan constant_yaw=True yang matikan dynamic yaw
        if self.waypoint_mode == "square" and self.square_constant_yaw:
            self.use_dynamic_yaw = False
            rospy.loginfo("✓ Yaw mode: CONSTANT (square_constant_yaw=True)")
        else:
            self.use_dynamic_yaw = True
            rospy.loginfo("✓ Yaw mode: DYNAMIC (mengikuti arah gerakan)")

        # Initialize reached flags
        self.reached_flags = [False] * len(self.positions)
        if len(self.reached_flags) > 0:
            self.reached_flags[0] = True

        rospy.loginfo(
            "Path built: %d waypoints, total length: %.2f m, total time: %.2f s",
            len(self.positions),
            float(np.sum(self.segment_lengths)),
            self.total_time
        )

        self.mode = "WAIT_ALT"
        self.alt_ready_since = None

    # ======================================================================
    # TIMER CALLBACK - PUBLISH KE PX4
    # ======================================================================
    def timer_cb(self, event):
        # ======== SETPOINT STREAM COUNTER (ALWAYS COUNT) ========
        # PX4 memerlukan continuous setpoint stream sebelum OFFBOARD mode
        # HARUS publish setpoint bahkan sebelum trajectory initialized!
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
            # Hanya log sekali saat mencapai required setpoints
            self.setpoint_count += 1  # Increment agar tidak log terus
            if not (self.auto_arm or self.auto_offboard):
                rospy.loginfo("✅ Setpoint stream ready! You can now switch to OFFBOARD mode manually.")
                rospy.loginfo("   Use QGroundControl or: rosservice call /mavros/set_mode \"custom_mode: 'OFFBOARD'\"")

        # ======== AUTO ARM / OFFBOARD SEQUENCE (OPTIONAL) ========
        if self.auto_arm or self.auto_offboard:
            # Step 2: Set OFFBOARD mode
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
                    rospy.loginfo("Already in OFFBOARD mode")
                    self.offboard_requested = True

            # Step 3: ARM vehicle
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
                    rospy.loginfo("Already ARMED")
                    self.arm_requested = True

        # ======== PUBLISH SETPOINT (ALWAYS, EVEN BEFORE TRAJECTORY READY) ========
        # Jika trajectory belum initialized, publish current position sebagai setpoint
        if not self.traj_initialized:
            if self.pose_received:
                # Hover di current position
                pose_msg = PoseStamped()
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.header.frame_id = "map"
                pose_msg.pose.position.x = float(self.current_pos_enu[0])
                pose_msg.pose.position.y = float(self.current_pos_enu[1])
                pose_msg.pose.position.z = float(self.current_pos_enu[2])
                
                # Keep current yaw
                yaw_enu = self.current_yaw
                qz = np.sin(yaw_enu / 2.0)
                qw = np.cos(yaw_enu / 2.0)
                pose_msg.pose.orientation.x = 0.0
                pose_msg.pose.orientation.y = 0.0
                pose_msg.pose.orientation.z = float(qz)
                pose_msg.pose.orientation.w = float(qw)
                
                # PUBLISH!
                self.setpoint_pub.publish(pose_msg)
                rospy.loginfo_throttle(2.0, "📡 Publishing setpoint (hover at current pos) - waiting for trajectory init...")
            else:
                rospy.loginfo_throttle(2.0, "⏳ Waiting for first pose from MAVROS...")
            return

        # Trajectory sudah ready, lanjutkan dengan logika normal
        now = rospy.Time.now().to_sec()
        yaw0 = self.yaws[0]

        # Check altitude gate
        if self.mode == "WAIT_ALT" and self.gate_on_altitude and self.pose_received:
            z_enu = self.current_pos_enu[2]
            z_err = abs(z_enu - self.hover_altitude)

            if z_err <= self.alt_tolerance:
                if self.alt_ready_since is None:
                    self.alt_ready_since = now
                else:
                    if (now - self.alt_ready_since) >= self.hover_stable_time:
                        self.mode = "TRACK"
                        self.t0 = now
                        rospy.loginfo(
                            "✅ Altitude gate passed: z=%.2f m, start trajectory tracking.",
                            z_enu
                        )
            else:
                self.alt_ready_since = None

        if not self.gate_on_altitude and self.mode == "WAIT_ALT":
            self.mode = "TRACK"
            self.t0 = rospy.Time.now().to_sec()
            rospy.loginfo("Altitude gate disabled → langsung TRACK mode.")

        # Compute reference position
        if self.mode == "WAIT_ALT":
            p0 = self.positions[0]
            p_hover = np.array([p0[0], p0[1], -self.hover_altitude])
            enu_pos = np.array([p_hover[1], p_hover[0], -p_hover[2]])
            
            # YAW: dynamic atau constant sesuai trajectory mode
            if self.use_dynamic_yaw and self.pose_received:
                cur_ned = np.array([self.current_pos_enu[1], self.current_pos_enu[0], -self.current_pos_enu[2]])
                direction = p0[:2] - cur_ned[:2]  # Direction ke waypoint 0 (N,E)
                distance_xy = float(np.linalg.norm(direction))
                
                if distance_xy > 0.1:  # Jarak > 10cm
                    # Pass direction vector, bukan yaw calculated
                    yaw_ref = self.get_continuous_yaw(direction, self.previous_yaw)
                    self.previous_yaw = yaw_ref
                else:
                    yaw_ref = yaw0
                    self.previous_yaw = yaw_ref
            else:
                # Constant yaw (gunakan yaw dari waypoint)
                yaw_ref = yaw0
                self.previous_yaw = yaw_ref
                rospy.loginfo_throttle(2.0, f"[WAIT_ALT] yaw_ref={np.degrees(yaw_ref):.1f}° (constant mode)")
        else:  # TRACK mode
            if self.t0 is None:
                self.t0 = now
            t_rel = now - self.t0

            if self.loop_trajectory:
                t_rel = np.fmod(t_rel, self.total_time)
                if t_rel < 0:
                    t_rel += self.total_time
            else:
                t_rel = max(0.0, min(t_rel, self.total_time))
                if not self.waypoint_follow_mode:
                    if t_rel >= self.total_time and not self.trajectory_finished:
                        self.trajectory_finished = True
                        rospy.loginfo("✅ Trajectory FINISHED (time-based)")
                        if not self.finish_published:
                            status_msg = String()
                            status_msg.data = "FINISHED"
                            self.status_pub.publish(status_msg)
                            self.finish_published = True

                progress_pct = (t_rel / self.total_time) * 100.0 if self.total_time > 0 else 0.0
                rospy.loginfo_throttle(2.0, f"🛤️  Progress: {progress_pct:.1f}% ({t_rel:.1f}s / {self.total_time:.1f}s)")

            # Waypoint follow mode
            if self.waypoint_follow_mode:
                if self.current_wp_idx >= len(self.positions):
                    p_ref = self.positions[-1].copy()
                    yaw_ref = self.yaws[-1]
                    self.trajectory_finished = True
                else:
                    p_ref = self.positions[self.current_wp_idx].copy()
                    # Gunakan yaw dari trajectory yang sudah di-generate (seperti mavros_gated)
                    yaw_ref = self.yaws[self.current_wp_idx]
                    rospy.loginfo_throttle(2.0, f"[WF] yaw_ref={np.degrees(yaw_ref):.1f}° from self.yaws[{self.current_wp_idx}]")

                # Check if reached
                if self.pose_received and self.reached_flags is not None:
                    cur_ned = np.array([self.current_pos_enu[1], self.current_pos_enu[0], -self.current_pos_enu[2]])
                    wp = self.positions[self.current_wp_idx]
                    d_xy = float(np.linalg.norm(cur_ned[0:2] - wp[0:2]))
                    d_z = float(abs(cur_ned[2] - wp[2]))

                    rospy.loginfo_throttle(1.0, f"[WF] wp_idx={self.current_wp_idx}, d_xy={d_xy:.2f}, d_z={d_z:.2f}")

                    if (not self.reached_flags[self.current_wp_idx]) and \
                       (d_xy <= self.acceptance_radius_xy) and \
                       (d_z <= self.acceptance_alt_tol):
                        self.reached_flags[self.current_wp_idx] = True
                        tnow = rospy.Time.now().to_sec()
                        rospy.loginfo("📍 REACHED waypoint %d (PX4 PID)", self.current_wp_idx)
                        
                        status_msg = String()
                        status_msg.data = f"REACHED {self.current_wp_idx}"
                        self.status_pub.publish(status_msg)

                        if self.current_wp_idx == (len(self.positions) - 1):
                            if not self.finish_published:
                                status_finish = String()
                                status_finish.data = "FINISHED"
                                self.status_pub.publish(status_finish)
                                self.finish_published = True
                                self.trajectory_finished = True
                                rospy.loginfo("📢 Trajectory FINISHED (PX4 PID)")
                        else:
                            self.current_wp_idx += 1
                            self.just_advanced_waypoint = True  # Set flag
                            if self.advance_on_reach:
                                idx_for_time = min(self.current_wp_idx, len(self.cum_times)-1)
                                desired_t_rel = float(self.cum_times[idx_for_time])
                                self.t0 = rospy.Time.now().to_sec() - desired_t_rel
                                rospy.loginfo("⏩ advance_on_reach → t=%.2fs", desired_t_rel)
            else:
                # Time-based interpolation
                j = np.searchsorted(self.cum_times, t_rel, side="right") - 1
                j = int(np.clip(j, 0, len(self.segment_times) - 1))

                t_start = self.cum_times[j]
                dt_seg = self.segment_times[j]
                if dt_seg <= 0.0:
                    tau = 0.0
                else:
                    tau = (t_rel - t_start) / dt_seg
                    tau = float(np.clip(tau, 0.0, 1.0))

                p0 = self.positions[j]
                p1 = self.positions[j + 1]
                p_ref = (1.0 - tau) * p0 + tau * p1

                yaw0_seg = self.yaws[j]
                yaw1_seg = self.yaws[j + 1]
                yaw_diff = np.arctan2(np.sin(yaw1_seg - yaw0_seg), np.cos(yaw1_seg - yaw0_seg))
                yaw_ref = yaw0_seg + tau * yaw_diff

            # NED → ENU
            enu_pos = np.array([p_ref[1], p_ref[0], -p_ref[2]])

        # ===================== PUBLISH KE PX4 POSITION CONTROLLER =====================
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(enu_pos[0])
        pose_msg.pose.position.y = float(enu_pos[1])
        pose_msg.pose.position.z = float(enu_pos[2])

        # Yaw: Konversi NED ke ENU frame
        # NED: North=0°, East=90°
        # ENU: East=0°, North=90°
        # Jadi: yaw_enu = yaw_ned + 90°
        yaw_enu = yaw_ref + np.pi / 2.0
        
        # Yaw → quaternion
        qz = np.sin(yaw_enu / 2.0)
        qw = np.cos(yaw_enu / 2.0)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = float(qz)
        pose_msg.pose.orientation.w = float(qw)
        
        # DEBUG: Log published yaw
        if self.mode == "WAIT_ALT":
            rospy.loginfo_throttle(1.0, f"[PUBLISH] pos=({enu_pos[0]:.2f}, {enu_pos[1]:.2f}, {enu_pos[2]:.2f}) yaw={np.degrees(yaw_enu):.1f}°")

        # PUBLISH LANGSUNG KE PX4!
        self.setpoint_pub.publish(pose_msg)
        self.waypoint_pub.publish(pose_msg)  # For monitoring
        
        # Log publish rate untuk debugging
        if self.setpoint_count <= self.required_setpoints + 10:
            rospy.loginfo_throttle(1.0, f"✅ Publishing setpoint to /mavros/setpoint_position/local (count: {self.setpoint_count})")

    # ======================================================================
    # TRAJECTORY GENERATORS (SAMA dengan trajectory_publisher_mavros_gated.py)
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
        
        rospy.loginfo("Circle: center=(%.1f,%.1f), r=%.1f, alt=%.1f, pts=%d",
                     center_n, center_e, radius, -altitude, num_points)
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

        yaw_mode = "constant (0°)" if constant_yaw else "following path direction"
        rospy.loginfo(
            "Square trajectory: center=(%.1f,%.1f), size=%.1f m, alt=%.1f m, points=%d, yaw=%s",
            center_n, center_e, size, -altitude, len(waypoints), yaw_mode
        )
        return waypoints

    def generate_helix_waypoints(self, center_n=0.0, center_e=0.0, radius=1.0,
                                 start_altitude=-10.0, end_altitude=-30.0,
                                 turns=3.0, num_points=120, add_transition=True):
        waypoints = []
        if add_transition:
            waypoints.append([float(center_n), float(center_e),
                              float(start_altitude), 0.0])

        for i in range(num_points):
            t = float(i) / float(num_points - 1)
            angle = 2.0 * np.pi * turns * t

            n = center_n + radius * np.cos(angle)
            e = center_e + radius * np.sin(angle)
            d = start_altitude + t * (end_altitude - start_altitude)
            
            # Hitung yaw dari velocity tangent vector
            # Helix parametric: (r*cos(θ), r*sin(θ), z(t))
            # Velocity: (-r*sin(θ), r*cos(θ), dz/dt)
            # Untuk yaw (horizontal), hanya komponen NE
            # dN/dθ = -r*sin(θ), dE/dθ = r*cos(θ)
            # Yaw = atan2(dE, dN) = atan2(r*cos(θ), -r*sin(θ))
            yaw = float(np.arctan2(np.cos(angle), -np.sin(angle)))
            
            # Normalize yaw ke [-pi, pi]
            yaw = self.normalize_angle(yaw)
            
            waypoints.append([float(n), float(e), float(d), float(yaw)])

        total_height = abs(end_altitude - start_altitude)
        direction = "descending" if end_altitude < start_altitude else "ascending"
        rospy.loginfo("Helix: r=%.1f, %.1f→%.1f m (%s), turns=%.1f, pts=%d",
                     radius, -start_altitude, -end_altitude, direction, turns, len(waypoints))
        return waypoints

    def generate_diamond_ascending_waypoints(self, center_n=0.0, center_e=0.0, 
                                            size=7.5, start_altitude=-5.0, 
                                            end_altitude=-10.0, num_points_per_side=0, initial_yaw=0.0):
        """
        Generate diamond/belah ketupat trajectory dengan altitude naik gradual.
        
        Diamond shape (dilihat dari atas):
        
                    WP1 (North/Depan)
                   /              \
                  /                \
           WP4 (West/Kiri)    WP2 (East/Kanan)
                  \                /
                   \              /
                    WP3 (South/Belakang)
        
        Urutan: Start(center) → WP1(N) → WP2(E) → WP3(S) → WP4(W) → WP5(N/kembali)
        
        Altitude naik secara linear dari start_altitude ke end_altitude
        Yaw mengikuti arah lintasan (arah kedatangan ke setiap waypoint)
        
        Args:
            center_n, center_e: Center position dalam NED
            size: Jarak dari center ke corner (meter)
            start_altitude: Altitude awal dalam NED (negative = up)
            end_altitude: Altitude akhir dalam NED
            num_points_per_side: Jumlah waypoint per sisi (0 = hanya corners)
        """
        waypoints = []
        
        # Define 4 corners dalam NED frame:
        # North = +N, East = +E, South = -N, West = -E
        # Diamond TIDAK dirotasi - selalu fixed di N/E/S/W
        # Urutan: N → E → S → W (clockwise)
        corners_world = [
            [-size, 0.0],      # WP1: North (depan)
            [0.0, size],      # WP2: East (kanan)
            [size, 0.0],     # WP3: South (belakang)
            [0.0, -size],     # WP4: West (kiri)
        ]
        
        # Jika num_points_per_side == 0, hanya generate corner points + kembali ke start
        if num_points_per_side == 0:
            # 5 waypoints: 4 corners + kembali ke corner 1
            altitude_step = (end_altitude - start_altitude) / 4.0  # 5 points, 4 intervals
            
            for i in range(5):  # 0, 1, 2, 3, 4
                corner_idx = i % 4
                corner = corners_world[corner_idx]
                
                n = center_n + corner[0]
                e = center_e + corner[1]
                d = start_altitude + i * altitude_step
                
                # Yaw dengan swap pasangan:
                # WP1 pakai yaw segment 1→2, WP2 pakai yaw segment 0→1
                # WP3 pakai yaw segment 3→4, WP4 pakai yaw segment 2→3
                # Gunakan XOR dengan 1 untuk swap index ganjil/genap dalam pasangan
                swapped_idx = corner_idx ^ 1  # 0↔1, 2↔3
                
                # Hitung yaw dari corner swapped_idx ke corner berikutnya
                from_corner = corners_world[swapped_idx]
                to_corner = corners_world[(swapped_idx - 1) % 4]
                
                dn = to_corner[0] - from_corner[0]
                de = to_corner[1] - from_corner[1]
                
                yaw = float(np.arctan2(de, dn))
                
                waypoints.append([float(n), float(e), float(d), yaw])
                
                rospy.loginfo("  WP%d: N=%.2f, E=%.2f, D=%.2f (alt=%.1fm), yaw=%.1f°", 
                             i+1, n, e, d, -d, np.degrees(yaw))
            
            rospy.loginfo("Diamond ascending trajectory (point-to-point):")
            rospy.loginfo("  Center: (%.1f, %.1f) m", center_n, center_e)
            rospy.loginfo("  Size: %.1f m (center to corner)", size)
            rospy.loginfo(
                "  Altitude: %.1f → %.1f m (Δ=%.1f m per waypoint)",
                -start_altitude, -end_altitude, -(end_altitude - start_altitude) / 4.0
            )
            rospy.loginfo("  Total points: %d (4 corners + return to start)", len(waypoints))
            
            return waypoints
        
        # Else: generate interpolated points along each side
        total_points = num_points_per_side * 4
        altitude_step = (end_altitude - start_altitude) / float(total_points - 1) if total_points > 1 else 0.0
        
        point_idx = 0
        
        for side_idx in range(4):
            corner_start = corners_world[side_idx]
            corner_end = corners_world[(side_idx + 1) % 4]
            
            # Calculate yaw for this side (direction from start to end)
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
        
        total_height = abs(end_altitude - start_altitude)
        direction = "descending" if end_altitude < start_altitude else "ascending"
        
        rospy.loginfo("Diamond ascending trajectory (interpolated):")
        rospy.loginfo("  Center: (%.1f, %.1f) m", center_n, center_e)
        rospy.loginfo("  Size: %.1f m (center to corner)", size)
        rospy.loginfo(
            "  Altitude: %.1f → %.1f m (%s, Δ=%.1f m)",
            -start_altitude, -end_altitude, direction, total_height
        )
        rospy.loginfo("  Total points: %d (4 sides × %d points/side)", 
                     len(waypoints), num_points_per_side)
        
        return waypoints


# ======================================================================
# MAIN
# ======================================================================
def main():
    rospy.init_node("trajectory_publisher_px4_direct", anonymous=False)
    
    try:
        node = Px4TrajectoryPublisherDirect()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    
    rospy.loginfo("Trajectory Publisher (PX4 Direct) shutdown.")


if __name__ == "__main__":
    main()
