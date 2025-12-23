#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Publisher POSITION ONLY (ROS1 + MAVROS)

Struktur SAMA dengan trajectory_publisher_mavros_gated.py:
- Hover dulu di ketinggian hover_altitude
- Setelah stabil, baru mulai trajectory
- Trajectory start TIDAK dimulai di posisi drone saat ini (absolut)

Perbedaan:
- HANYA publish referensi posisi ke /trajectory/ref_pose_only
- TIDAK ada publish referensi kecepatan
- Digunakan bersama dengan mpc_position_only.py

Publisher:
    /trajectory/ref_pose_only  (PoseStamped, ENU) - hanya posisi + yaw
    /waypoint/target           (PoseStamped, ENU) - untuk monitoring
    /trajectory/status         (String) - status waypoint

Tidak ada:
    /trajectory/ref_vel (tidak dipublish)
"""

import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from std_msgs.msg import String


class TrajectoryPublisherPositionOnly(object):
    def __init__(self):
        rospy.loginfo("="*60)
        rospy.loginfo("Trajectory Publisher POSITION ONLY")
        rospy.loginfo("Output: /trajectory/ref_pose (NO velocity ref)")
        rospy.loginfo("="*60)

        # ===================== PARAM TRAJECTORY =====================
        self.waypoint_mode = rospy.get_param("~waypoint_mode", "default")
        self.ref_speed = rospy.get_param("~ref_speed", 2.0)
        self.loop_trajectory = rospy.get_param("~loop_trajectory", False)
        # Default FALSE - trajectory start di posisi absolut, bukan posisi drone saat ini
        self.start_at_current_pose = rospy.get_param("~start_at_current_pose", False)

        # Hover / gate params
        self.hover_altitude = rospy.get_param("~hover_altitude", 2.5)
        self.alt_tolerance = rospy.get_param("~alt_tolerance", 0.3)
        self.hover_stable_time = rospy.get_param("~hover_stable_time", 2.0)
        self.gate_on_altitude = rospy.get_param("~gate_on_altitude", True)

        # Acceptance / reach detection
        self.acceptance_radius_xy = rospy.get_param("~acceptance_radius_xy", 0.8)
        self.acceptance_alt_tol = rospy.get_param("~acceptance_alt_tol", 0.5)
        self.advance_on_reach = rospy.get_param("~advance_on_reach", True)

        # Waypoint follow mode
        self.waypoint_follow_mode = rospy.get_param("~waypoint_follow_mode", True)
        self.current_wp_idx = 1

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
        
        self.use_dynamic_yaw = False
        self.previous_yaw = 0.0

        # ===================== STATE MAVROS =====================
        self.current_state = State()
        self.current_pos_enu = np.zeros(3)
        self.current_yaw = 0.0
        self.pose_received = False

        # ===================== SUBSCRIBERS =====================
        self.state_sub = rospy.Subscriber(
            "/mavros/state", State, self.state_callback, queue_size=10
        )
        self.local_pose_sub = rospy.Subscriber(
            "/mavros/local_position/pose", PoseStamped, self.position_callback, queue_size=10
        )

        # ===================== PUBLISHERS =====================
        # HANYA ref_pose, TIDAK ada ref_vel
        self.ref_pose_pub = rospy.Publisher(
            "/trajectory/ref_pose", PoseStamped, queue_size=10
        )
        self.waypoint_pub = rospy.Publisher(
            "/waypoint/target", PoseStamped, queue_size=10
        )
        self.status_pub = rospy.Publisher(
            "/trajectory/status", String, queue_size=10
        )

        # ===================== TIMERS =====================
        rate = rospy.get_param("~rate", 50.0)
        self.timer = rospy.Timer(rospy.Duration(1.0/rate), self.timer_cb)

        rospy.loginfo(f"Trajectory Publisher (Position Only) ready. Mode={self.waypoint_mode}")

    # ======================================================================
    # MAVROS callbacks
    # ======================================================================
    def state_callback(self, msg: State):
        self.current_state = msg

    def position_callback(self, msg: PoseStamped):
        self.current_pos_enu[0] = msg.pose.position.x
        self.current_pos_enu[1] = msg.pose.position.y
        self.current_pos_enu[2] = msg.pose.position.z

        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y**2 + q.z**2)
        self.current_yaw = np.arctan2(siny_cosp, cosy_cosp)

        if not self.pose_received:
            self.pose_received = True
            rospy.loginfo("✅ First pose received from MAVROS.")

        # Build path setelah dapat pose pertama
        if self.pose_received and not self.traj_initialized:
            self.build_path(anchor_to_current=self.start_at_current_pose)

    # ======================================================================
    # YAW HELPER FUNCTIONS
    # ======================================================================
    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    # ======================================================================
    # BUILD PATH + PRECOMPUTE (SAMA dengan trajectory_publisher_mavros_gated.py)
    # ======================================================================
    def build_path(self, anchor_to_current=False):
        """
        Build trajectory path.
        
        Jika anchor_to_current=True: 
            - Path di-offset ke posisi UAV saat ini
            - Titik pertama = posisi UAV
            
        Jika anchor_to_current=False (DEFAULT):
            - Path absolut sesuai parameter
            - Drone akan terbang ke titik pertama trajectory
        """
        
        if anchor_to_current and self.pose_received:
            # Path relatif ke posisi UAV saat ini
            rospy.loginfo("Building path ANCHORED to current UAV position...")
            
            # Generate path dengan center di (0,0), nanti di-offset
            if self.waypoint_mode == "circle":
                path = self.generate_circle_waypoints(
                    center_n=0.0,
                    center_e=0.0,
                    radius=self.circle_radius,
                    altitude=0.0,  # akan di-set nanti
                    num_points=self.circle_points
                )
            elif self.waypoint_mode == "square":
                path = self.generate_square_waypoints(
                    center_n=0.0,
                    center_e=0.0,
                    size=self.square_size,
                    altitude=0.0,
                    points_per_side=self.square_points_per_side,
                    constant_yaw=self.square_constant_yaw
                )
            elif self.waypoint_mode == "diamond":
                path = self.generate_diamond_ascending_waypoints(
                    center_n=0.0,
                    center_e=0.0,
                    size=self.diamond_size,
                    start_altitude=0.0,
                    end_altitude=self.diamond_end_altitude - self.diamond_start_altitude,
                    num_points_per_side=self.diamond_points_per_side
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
            n_uav = self.current_pos_enu[1]  # ENU Y → NED N
            e_uav = self.current_pos_enu[0]  # ENU X → NED E
            d_hover = -self.hover_altitude   # NED down

            first_n, first_e, first_d, _ = path[0]
            dn = n_uav - first_n
            de = e_uav - first_e
            
            rospy.loginfo("Anchoring path to UAV position:")
            rospy.loginfo("  UAV: N=%.2f, E=%.2f, D=%.2f", n_uav, e_uav, d_hover)
            rospy.loginfo("  Offset: dN=%.2f, dE=%.2f", dn, de)

            # Offset altitude
            if self.waypoint_mode == "diamond":
                dd = d_hover - first_d
            else:
                dd = 0.0

            for wp in path:
                wp[0] += dn
                wp[1] += de
                if self.waypoint_mode == "diamond":
                    wp[2] += dd
                else:
                    wp[2] = d_hover  # Paksa ke hover altitude

            rospy.loginfo("🚀 Trajectory anchored ke UAV: start NED ≈ (%.2f, %.2f, %.2f)", n_uav, e_uav, d_hover)

        else:
            # PATH ABSOLUT - TIDAK di-anchor ke posisi UAV
            rospy.loginfo("Building ABSOLUTE path (not anchored to UAV)...")
            
            if self.waypoint_mode == "circle":
                path = self.generate_circle_waypoints(
                    center_n=self.circle_center_n,
                    center_e=self.circle_center_e,
                    radius=self.circle_radius,
                    altitude=self.circle_altitude,
                    num_points=self.circle_points
                )
                self.use_dynamic_yaw = True

            elif self.waypoint_mode == "square":
                path = self.generate_square_waypoints(
                    center_n=self.square_center_n,
                    center_e=self.square_center_e,
                    size=self.square_size,
                    altitude=self.square_altitude,
                    points_per_side=self.square_points_per_side,
                    constant_yaw=self.square_constant_yaw
                )
                self.use_dynamic_yaw = not self.square_constant_yaw

            elif self.waypoint_mode == "diamond":
                path = self.generate_diamond_ascending_waypoints(
                    center_n=self.diamond_center_n,
                    center_e=self.diamond_center_e,
                    size=self.diamond_size,
                    start_altitude=self.diamond_start_altitude,
                    end_altitude=self.diamond_end_altitude,
                    num_points_per_side=self.diamond_points_per_side
                )
                self.use_dynamic_yaw = False

            else:  # default
                alt = -self.hover_altitude
                path = [
                    [0.0, 0.0, alt, 0.0],
                    [5.0, 0.0, alt, 0.0],
                    [5.0, 5.0, alt, np.pi/2],
                    [0.0, 5.0, alt, np.pi],
                    [0.0, 0.0, alt, -np.pi/2],
                ]
                self.use_dynamic_yaw = True

        # Validate path
        if len(path) < 2:
            rospy.logwarn("Path hanya punya <=1 titik, menyalin titik pertama.")
            path.append(list(path[0]))

        self.path_points = path

        # Initialize reached flags
        self.reached_flags = [False] * len(self.path_points)
        if len(self.path_points) > 0:
            self.reached_flags[0] = True  # Start point dianggap sudah tercapai

        self.current_wp_idx = 1

        # Precompute segments
        self._precompute_segments()
        
        self.traj_initialized = True
        self.t0 = rospy.Time.now().to_sec()
        self.mode = "WAIT_ALT"
        self.alt_ready_since = None
        self.trajectory_finished = False
        self.finish_published = False

        # Log info
        rospy.loginfo("=" * 60)
        rospy.loginfo("Trajectory Publisher POSITION ONLY")
        rospy.loginfo("Mode trajektori   : %s", self.waypoint_mode)
        rospy.loginfo("Jumlah titik      : %d", len(self.path_points))
        rospy.loginfo("Kecepatan ref     : %.2f m/s", self.ref_speed)
        rospy.loginfo("Total waktu       : %.2f s", self.total_time)
        rospy.loginfo("Loop trajectory   : %s", "Yes" if self.loop_trajectory else "No")
        rospy.loginfo("Start at curr pos : %s", "Yes" if self.start_at_current_pose else "No (ABSOLUTE)")
        rospy.loginfo("Hover altitude    : %.2f m ENU", self.hover_altitude)
        rospy.loginfo("Altitude gate     : %s", "ON" if self.gate_on_altitude else "OFF")
        rospy.loginfo("Acceptance XY/Alt : %.2f m / %.2f m", self.acceptance_radius_xy, self.acceptance_alt_tol)
        rospy.loginfo("waypoint_follow   : %s", "Yes" if self.waypoint_follow_mode else "No")
        rospy.loginfo("Output topic      : /trajectory/ref_pose (NO VELOCITY)")
        rospy.loginfo("=" * 60)
        
        # Log waypoints
        rospy.loginfo("Waypoints (NED):")
        for i, wp in enumerate(self.path_points):
            rospy.loginfo("  WP%d: N=%.2f, E=%.2f, D=%.2f (alt=%.1fm), yaw=%.1f°",
                         i, wp[0], wp[1], wp[2], -wp[2], np.degrees(wp[3]))

    def _precompute_segments(self):
        """Precompute segment lengths and times."""
        pts = np.array(self.path_points)
        positions = pts[:, 0:3]
        yaws = pts[:, 3]

        deltas = positions[1:, :] - positions[:-1, :]
        seg_lengths = np.linalg.norm(deltas, axis=1)
        seg_lengths = np.where(seg_lengths < 1e-6, 1e-6, seg_lengths)
        seg_times = seg_lengths / max(self.ref_speed, 1e-3)

        cum_times = np.zeros(seg_times.shape[0] + 1)
        cum_times[1:] = np.cumsum(seg_times)

        self.positions = positions
        self.yaws = yaws
        self.segment_lengths = seg_lengths
        self.segment_times = seg_times
        self.cum_times = cum_times
        self.total_time = float(cum_times[-1])

    # ======================================================================
    # TIMER CALLBACK - PUBLISH POSISI ONLY
    # ======================================================================
    def timer_cb(self, event):
        if not self.traj_initialized or self.total_time <= 0.0:
            return

        now = rospy.Time.now().to_sec()

        # ===== MODE: WAIT_ALT (hover dulu sampai ketinggian stabil) =====
        if self.mode == "WAIT_ALT" and self.gate_on_altitude and self.pose_received:
            alt = self.current_pos_enu[2]  # ENU z
            if abs(alt - self.hover_altitude) <= self.alt_tolerance:
                if self.alt_ready_since is None:
                    self.alt_ready_since = now
                else:
                    if (now - self.alt_ready_since) >= self.hover_stable_time:
                        self.mode = "TRACK"
                        self.t0 = now
                        rospy.loginfo("✅ Altitude gate passed: z=%.2f m, start trajectory tracking.", alt)
            else:
                self.alt_ready_since = None

        if not self.gate_on_altitude and self.mode == "WAIT_ALT":
            self.mode = "TRACK"
            self.t0 = now
            rospy.loginfo("Altitude gate disabled → langsung TRACK mode.")

        # ===== Compute reference position =====
        if self.mode == "WAIT_ALT":
            # Hover di waypoint pertama pada ketinggian hover_altitude
            p0 = self.positions[0]
            # Override altitude ke hover_altitude (NED down = negative)
            p_hover = np.array([p0[0], p0[1], -self.hover_altitude])
            yaw_ref = self.yaws[0]
            
            # NED → ENU
            enu_pos = np.array([p_hover[1], p_hover[0], -p_hover[2]])

            rospy.loginfo_throttle(2.0, 
                f"[WAIT_ALT] Hover at WP0: ENU=({enu_pos[0]:.1f}, {enu_pos[1]:.1f}, {enu_pos[2]:.1f}), "
                f"current_z={self.current_pos_enu[2]:.2f}m, target_z={self.hover_altitude:.2f}m")

        else:  # MODE: TRACK
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
                        self.status_pub.publish(String(data="FINISHED"))
                        rospy.loginfo("📢 Trajectory FINISHED (time-based)")

            # ===== WAYPOINT FOLLOW MODE =====
            if self.waypoint_follow_mode:
                cur_ned = np.array([
                    self.current_pos_enu[1],   # N
                    self.current_pos_enu[0],   # E
                    -self.current_pos_enu[2]   # D
                ])
                
                target_pos = self.positions[self.current_wp_idx]
                err_xy = np.linalg.norm(cur_ned[:2] - target_pos[:2])
                err_alt = abs(cur_ned[2] - target_pos[2])

                # Check if waypoint reached
                if err_xy <= self.acceptance_radius_xy and err_alt <= self.acceptance_alt_tol:
                    if not self.reached_flags[self.current_wp_idx]:
                        self.reached_flags[self.current_wp_idx] = True
                        self.status_pub.publish(String(data=f"REACHED {self.current_wp_idx}"))
                        rospy.loginfo(f"✅ Waypoint {self.current_wp_idx} REACHED (err_xy={err_xy:.2f}m, err_alt={err_alt:.2f}m)")

                        if self.current_wp_idx >= len(self.positions) - 1:
                            if not self.trajectory_finished:
                                self.trajectory_finished = True
                                self.status_pub.publish(String(data="FINISHED"))
                                rospy.loginfo("📢 Trajectory FINISHED (all waypoints reached)")
                        else:
                            self.current_wp_idx += 1
                            rospy.loginfo(f"➡️ Advancing to waypoint {self.current_wp_idx}")
                            if self.advance_on_reach:
                                idx_for_time = min(self.current_wp_idx, len(self.cum_times)-1)
                                desired_t_rel = float(self.cum_times[idx_for_time])
                                self.t0 = now - desired_t_rel

                p_ref = target_pos
                yaw_ref = self.yaws[self.current_wp_idx]

            else:
                # ===== TIME-BASED INTERPOLATION =====
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

        # ===================== PUBLISH POSISI ONLY =====================
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(enu_pos[0])
        pose_msg.pose.position.y = float(enu_pos[1])
        pose_msg.pose.position.z = float(enu_pos[2])

        # Yaw → quaternion (TANPA konversi, sama seperti mavros_gated)
        qz = np.sin(yaw_ref / 2.0)
        qw = np.cos(yaw_ref / 2.0)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = float(qz)
        pose_msg.pose.orientation.w = float(qw)

        # PUBLISH (HANYA POSISI, TIDAK ADA VELOCITY)
        self.ref_pose_pub.publish(pose_msg)
        self.waypoint_pub.publish(pose_msg)

        # Log (throttled)
        if self.mode == "TRACK":
            rospy.loginfo_throttle(2.0, 
                f"[TRACK] WP{self.current_wp_idx}/{len(self.positions)-1} "
                f"ref=({enu_pos[0]:.1f}, {enu_pos[1]:.1f}, {enu_pos[2]:.1f}) "
                f"cur=({self.current_pos_enu[0]:.1f}, {self.current_pos_enu[1]:.1f}, {self.current_pos_enu[2]:.1f})")

    # ======================================================================
    # TRAJECTORY GENERATORS
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
        
        rospy.loginfo(f"Circle: center=({center_n:.1f},{center_e:.1f}), r={radius:.1f}, alt={-altitude:.1f}m, pts={num_points}")
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

        # TIDAK kembali ke titik awal - stop di waypoint terakhir
        # (hapus kode yang menambahkan first waypoint di akhir)

        yaw_mode = "constant (0°)" if constant_yaw else "following path direction"
        rospy.loginfo(
            "Square trajectory: center=(%.1f,%.1f), size=%.1f m, alt=%.1f m, points=%d, yaw=%s",
            center_n, center_e, size, -altitude, len(waypoints), yaw_mode
        )
        return waypoints

    def generate_diamond_ascending_waypoints(self, center_n=0.0, center_e=0.0, 
                                            size=7.5, start_altitude=-5.0, 
                                            end_altitude=-10.0, num_points_per_side=0):
        waypoints = []
        
        # Diamond corners: N, E, S, W
        corners_world = [
            [-size, 0.0],   # North
            [0.0, size],    # East
            [size, 0.0],    # South
            [0.0, -size],   # West
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
                    # WP0: menghadap ke WP1
                    next_corner = corners_world[1]
                    dn = next_corner[0] - corner[0]
                    de = next_corner[1] - corner[1]
                else:
                    # WP1-4: menghadap ke arah dari WP sebelumnya (arah kedatangan)
                    prev_corner_idx = (corner_idx - 1 + 4) % 4
                    prev_corner = corners_world[prev_corner_idx]
                    dn = corner[0] - prev_corner[0]
                    de = corner[1] - prev_corner[1]
                
                yaw = float(np.arctan2(de, dn))
                
                waypoints.append([float(n), float(e), float(d), yaw])
                
                rospy.loginfo(f"  WP{i}: N={n:.2f}, E={e:.2f}, D={d:.2f}, yaw={np.degrees(yaw):.1f}°")
        else:
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
        
        rospy.loginfo(f"Diamond: center=({center_n:.1f},{center_e:.1f}), size={size:.1f}m")
        rospy.loginfo(f"  Altitude: {-start_altitude:.1f} → {-end_altitude:.1f}m")
        return waypoints


# ======================================================================
# MAIN
# ======================================================================
def main():
    rospy.init_node("trajectory_publisher_position_only", anonymous=False)
    
    try:
        node = TrajectoryPublisherPositionOnly()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    
    rospy.loginfo("Trajectory Publisher (Position Only) shutdown.")


if __name__ == "__main__":
    main()
