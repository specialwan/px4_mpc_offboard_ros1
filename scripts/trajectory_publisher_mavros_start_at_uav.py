#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-based Trajectory Publisher (ROS1 + MAVROS) with option
to start the trajectory at the CURRENT UAV POSE.

Perbedaan utama dari versi sebelumnya:
- Param baru: ~start_at_current_pose (default: True)
- Jika True:
    * Node menunggu pose pertama dari /mavros/local_position/pose
    * Bentuk lintasan (square/circle/helix/default) dibikin di sekitar origin
    * Lalu digeser (offset) supaya titik pertama path = posisi UAV saat ini
- Jika False:
    * Perilaku lama: center_n, center_e, altitude, dst adalah koordinat absolut NED

Output:
- /trajectory/ref_pose  (PoseStamped, ENU)
- /trajectory/ref_vel   (TwistStamped, ENU)
- /waypoint/target      (PoseStamped, ENU)  -> kompatibel dengan MPC lama / logger
"""

import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped, TwistStamped


class TrajectoryPublisherMavros(object):
    def __init__(self):
        rospy.loginfo("Init TrajectoryPublisherMavros (time-based trajectory)")

        # ===================== PARAMETER =====================
        self.waypoint_mode = rospy.get_param("~waypoint_mode", "default")

        # Trajectory speed (NED) [m/s]
        self.ref_speed = rospy.get_param("~ref_speed", 2.0)
        self.loop_trajectory = rospy.get_param("~loop_trajectory", False)

        # NEW: anchor trajectory to current UAV pose?
        self.start_at_current_pose = rospy.get_param("~start_at_current_pose", True)

        # Circle params
        self.circle_center_n = rospy.get_param("~circle_center_n", 0.0)
        self.circle_center_e = rospy.get_param("~circle_center_e", 0.0)
        self.circle_radius = rospy.get_param("~circle_radius", 25.0)
        self.circle_altitude = rospy.get_param("~circle_altitude", -5.0)
        self.circle_points = int(rospy.get_param("~circle_points", 80))

        # Square params
        self.square_center_n = rospy.get_param("~square_center_n", 0.0)
        self.square_center_e = rospy.get_param("~square_center_e", 0.0)
        self.square_size = rospy.get_param("~square_size", 20.0)
        self.square_altitude = rospy.get_param("~square_altitude", -5.0)
        self.square_points_per_side = int(rospy.get_param("~square_points_per_side", 10))
        self.square_constant_yaw = rospy.get_param("~square_constant_yaw", False)

        # Helix params
        self.helix_center_n = rospy.get_param("~helix_center_n", 0.0)
        self.helix_center_e = rospy.get_param("~helix_center_e", 0.0)
        self.helix_radius = rospy.get_param("~helix_radius", 1.0)
        self.helix_start_altitude = rospy.get_param("~helix_start_altitude", -10.0)
        self.helix_end_altitude = rospy.get_param("~helix_end_altitude", -30.0)
        self.helix_turns = rospy.get_param("~helix_turns", 3.0)
        self.helix_points = int(rospy.get_param("~helix_points", 120))

        # ===================== STATE INTERNAL =====================
        self.positions = None          # (N,3) NED
        self.yaws = None               # (N,)
        self.segment_lengths = None
        self.segment_times = None
        self.cum_times = None
        self.total_time = 0.0

        self.path_points = None        # list of [n,e,d,yaw]
        self.traj_initialized = False

        # Posisi UAV (ENU) dari MAVROS
        self.current_pos_enu = np.zeros(3)
        self.pose_received = False

        # Waktu awal traj
        self.t0 = None

        # ===================== PUB / SUB =====================
        # Referensi untuk MPC
        self.pose_pub = rospy.Publisher(
            "/trajectory/ref_pose", PoseStamped, queue_size=10
        )
        self.vel_pub = rospy.Publisher(
            "/trajectory/ref_vel", TwistStamped, queue_size=10
        )

        # Kompatibel dengan MPC lama & data logger
        self.waypoint_pub = rospy.Publisher(
            "/waypoint/target", PoseStamped, queue_size=10
        )

        # SUB pose UAV → untuk anchor trajektori bila diperlukan
        self.pos_sub = rospy.Subscriber(
            "/mavros/local_position/pose",
            PoseStamped,
            self.position_callback,
            queue_size=1
        )

        # Jika TIDAK start_at_current_pose → bangun trajektori langsung
        if not self.start_at_current_pose:
            self.build_path(anchor_to_current=False)
        else:
            rospy.loginfo("start_at_current_pose = True → menunggu pose awal UAV...")

        # Timer publish (20 Hz)
        self.timer = rospy.Timer(rospy.Duration(0.05), self.timer_cb)

    # ================================================================
    # POSITION CALLBACK (UNTUK ANCHOR TRAJECTORY)
    # ================================================================
    def position_callback(self, msg: PoseStamped):
        # ENU
        self.current_pos_enu[:] = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ]
        if not self.pose_received:
            self.pose_received = True
            rospy.loginfo("✅ Pose pertama dari /mavros/local_position/pose diterima")

        # Jika kita mau anchor ke posisi UAV dan path belum dibangun → bangun sekarang
        if self.start_at_current_pose and not self.traj_initialized:
            self.build_path(anchor_to_current=True)

    # ================================================================
    # BUILD PATH (NED) + PRECOMPUTE SEGMENTS
    # ================================================================
    def build_path(self, anchor_to_current=False):
        """
        Bangun self.path_points (list [n,e,d,yaw]) dan precompute segments.

        Jika anchor_to_current=True:
          - bentuk lintasan dibuat di sekitar ORIGIN (center=(0,0))
          - lalu digeser supaya titik pertama = posisi UAV saat ini (NED)
        Jika anchor_to_current=False:
          - gunakan center_n, center_e, altitude params seperti biasa (absolut)
        """
        if anchor_to_current and not self.pose_received:
            rospy.logwarn("build_path dipanggil sebelum pose diterima, skip dulu.")
            return

        # ---------- 1) Generate base path di NED ----------
        if anchor_to_current:
            # Kita buat shape di sekitar origin dengan altitude 0,
            # nanti seluruh path di-offset ke posisi UAV (n,e,d) sekarang.
            if self.waypoint_mode == "circle":
                path = self.generate_circle_waypoints(
                    center_n=0.0,
                    center_e=0.0,
                    radius=self.circle_radius,
                    altitude=0.0,
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
            elif self.waypoint_mode == "helix":
                path = self.generate_helix_waypoints(
                    center_n=0.0,
                    center_e=0.0,
                    radius=self.helix_radius,
                    start_altitude=0.0,
                    end_altitude=self.helix_end_altitude - self.helix_start_altitude,
                    turns=self.helix_turns,
                    num_points=self.helix_points
                )
            else:
                # default: kotak kecil 5x5 di sekitar origin, alt=0
                path = [
                    [0.0, 0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0, 0.0],
                    [5.0, 5.0, 0.0, np.pi/2],
                    [0.0, 5.0, 0.0, np.pi],
                    [0.0, 0.0, 0.0, -np.pi/2],
                ]

            # Offset seluruh path supaya titik pertama = posisi UAV sekarang
            # Konversi pose ENU → NED
            n_uav = self.current_pos_enu[1]
            e_uav = self.current_pos_enu[0]
            d_uav = -self.current_pos_enu[2]

            first_n = path[0][0]
            first_e = path[0][1]
            first_d = path[0][2]

            dn = n_uav - first_n
            de = e_uav - first_e
            dd = d_uav - first_d

            for wp in path:
                wp[0] += dn
                wp[1] += de
                wp[2] += dd

            rospy.loginfo(
                "🚀 Trajectory anchored ke posisi UAV: NED start=(%.2f, %.2f, %.2f)",
                n_uav, e_uav, d_uav
            )

        else:
            # Perilaku lama: gunakan center_* & altitude absolut
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
            else:
                path = [
                    [0.0, 0.0, -5.0, 0.0],
                    [5.0, 0.0, -5.0, 0.0],
                    [5.0, 5.0, -5.0, np.pi/2],
                    [0.0, 5.0, -5.0, np.pi],
                    [0.0, 0.0, -5.0, -np.pi/2],
                ]

        if len(path) < 2:
            rospy.logwarn("Path hanya punya <=1 point. Menambah satu titik dummy.")
            path.append(path[0])

        self.path_points = path

        # ---------- 2) Precompute segments ----------
        self._precompute_segments()

        self.traj_initialized = True
        self.t0 = rospy.Time.now().to_sec()

        rospy.loginfo("=" * 60)
        rospy.loginfo("TrajectoryPublisherMavros (time-based)")
        rospy.loginfo("Mode trajektori : %s", self.waypoint_mode)
        rospy.loginfo("Jumlah titik    : %d", len(self.path_points))
        rospy.loginfo("Kecepatan ref   : %.2f m/s", self.ref_speed)
        rospy.loginfo("Total waktu     : %.2f s", self.total_time)
        rospy.loginfo("Loop trajectory : %s", "Yes" if self.loop_trajectory else "No")
        rospy.loginfo("Start at current pose : %s", "Yes" if self.start_at_current_pose else "No")
        rospy.loginfo("Output pose     : /trajectory/ref_pose & /waypoint/target")
        rospy.loginfo("Output vel      : /trajectory/ref_vel")
        rospy.loginfo("=" * 60)

    # --------------------------------------------------------
    def _precompute_segments(self):
        pts = np.array(self.path_points)  # (N,4)
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

    # ================================================================
    # TIMER CALLBACK → HITUNG p_ref(t), v_ref(t)
    # ================================================================
    def timer_cb(self, event):
        # Kalau path belum siap, jangan publish dulu
        if not self.traj_initialized or self.total_time <= 0.0:
            return

        now = rospy.Time.now().to_sec()
        if self.t0 is None:
            self.t0 = now
        t_rel = now - self.t0

        if self.loop_trajectory:
            t_rel = np.fmod(t_rel, self.total_time)
            if t_rel < 0:
                t_rel += self.total_time
        else:
            t_rel = max(0.0, min(t_rel, self.total_time))

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

        yaw0 = self.yaws[j]
        yaw1 = self.yaws[j + 1]
        yaw_diff = np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))
        yaw_ref = yaw0 + tau * yaw_diff

        dp = p1 - p0
        seg_len = self.segment_lengths[j]
        dir_vec = dp / max(seg_len, 1e-6)
        v_ref_ned = dir_vec * self.ref_speed

        # NED → ENU
        enu_pos = np.array([p_ref[1], p_ref[0], -p_ref[2]])
        enu_vel = np.array([v_ref_ned[1], v_ref_ned[0], -v_ref_ned[2]])

        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(enu_pos[0])
        pose_msg.pose.position.y = float(enu_pos[1])
        pose_msg.pose.position.z = float(enu_pos[2])

        qz = np.sin(yaw_ref / 2.0)
        qw = np.cos(yaw_ref / 2.0)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = float(qz)
        pose_msg.pose.orientation.w = float(qw)

        vel_msg = TwistStamped()
        vel_msg.header.stamp = pose_msg.header.stamp
        vel_msg.header.frame_id = "map"
        vel_msg.twist.linear.x = float(enu_vel[0])
        vel_msg.twist.linear.y = float(enu_vel[1])
        vel_msg.twist.linear.z = float(enu_vel[2])

        self.pose_pub.publish(pose_msg)
        self.vel_pub.publish(vel_msg)
        self.waypoint_pub.publish(pose_msg)

    # ================================================================
    # TRAJECTORY GENERATORS (NED)
    # ================================================================
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
            "Circle trajectory: center=(%.1f,%.1f), radius=%.1f m, alt=%.1f m, "
            "%d points, spacing=%.2f m",
            center_n, center_e, radius, -altitude, num_points, avg_spacing
        )
        return waypoints

    def generate_square_waypoints(self, center_n=0.0, center_e=0.0, size=40.0,
                                  altitude=-5.0, points_per_side=3, constant_yaw=False):
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

            direction_n = end_corner[0] - start_corner[0]
            direction_e = end_corner[1] - start_corner[1]
            yaw_current = np.arctan2(direction_e, direction_n)

            next_corner = corners[(i + 2) % 4]
            direction_n_next = next_corner[0] - end_corner[0]
            direction_e_next = next_corner[1] - end_corner[1]
            yaw_next = np.arctan2(direction_e_next, direction_n_next)

            num_points = points_per_side + 1
            for j in range(num_points):
                t = float(j) / float(points_per_side + 1)
                n = start_corner[0] + t * direction_n
                e = start_corner[1] + t * direction_e

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
            "Square trajectory: center=(%.1f,%.1f), size=%.1f m, alt=%.1f m, "
            "%d points, yaw=%s",
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
            yaw = angle + np.pi / 2.0

            waypoints.append([float(n), float(e), float(d), float(yaw)])

        total_height = abs(end_altitude - start_altitude)
        direction = "descending" if end_altitude < start_altitude else "ascending"
        transition_count = 1 if add_transition else 0

        rospy.loginfo("📐 Helix trajectory generated:")
        rospy.loginfo("   Center: (%.1f, %.1f) m", center_n, center_e)
        rospy.loginfo("   Radius: %.1f m", radius)
        rospy.loginfo(
            "   Altitude: %.1f → %.1f m (%s, Δ=%.1f m)",
            -start_altitude, -end_altitude, direction, total_height
        )
        rospy.loginfo("   Turns: %.1f", turns)
        rospy.loginfo(
            "   Points: %d helix + %d transition = %d total",
            num_points, transition_count, len(waypoints)
        )
        rospy.loginfo(
            "   Arc length: ~%.1f m horizontal + %.1f m vertical",
            2 * np.pi * radius * turns, total_height
        )

        return waypoints


def main():
    rospy.init_node("trajectory_publisher_mavros_traj", anonymous=False)
    node = TrajectoryPublisherMavros()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()