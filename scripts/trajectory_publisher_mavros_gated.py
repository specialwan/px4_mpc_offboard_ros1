#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-based Trajectory Publisher to PX4 (ROS1 + MAVROS) dengan ALTITUDE GATE

- Generate path di NED: default / circle / square / helix
- Precompute panjang segmen → waktu tiap segmen untuk ref_speed konstan
- Publish referensi:
    /trajectory/ref_pose   (PoseStamped, ENU)
    /trajectory/ref_vel    (TwistStamped, ENU)
    /waypoint/target       (PoseStamped, ENU)
- Tambahan: altitude gate
  * Selama drone belum stabil di ketinggian hover_altitude → hanya kirim setpoint hover
  * Setelah stabil (mis. 5 m ± 0.3 m selama 2 s) → baru start lintasan time-based
"""

import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State


class Px4TrajectoryPublisherGated(object):
    def __init__(self):
        rospy.loginfo("Init Px4TrajectoryPublisherGated (time-based + altitude gate)")

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

        # Circle params
        self.circle_center_n = rospy.get_param("~circle_center_n", 0.0)
        self.circle_center_e = rospy.get_param("~circle_center_e", 0.0)
        self.circle_radius = rospy.get_param("~circle_radius", 10.0)
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

        # ===================== STATE TRAJECTORY =====================
        self.positions = None  # NED (N,E,D)
        self.yaws = None
        self.segment_lengths = None
        self.segment_times = None
        self.cum_times = None
        self.total_time = 0.0
        self.path_points = None
        self.traj_initialized = False

        self.t0 = None  # waktu start TRACK mode

        # Mode internal: "WAIT_ALT" atau "TRACK"
        self.mode = "WAIT_ALT"
        self.alt_ready_since = None

        # ===================== STATE MAVROS =====================
        self.current_state = State()
        self.current_pos_enu = np.zeros(3)
        self.pose_received = False

        # ===================== PUB / SUB =====================
        self.pose_pub = rospy.Publisher(
            "/trajectory/ref_pose", PoseStamped, queue_size=10
        )
        self.vel_pub = rospy.Publisher(
            "/trajectory/ref_vel", TwistStamped, queue_size=10
        )
        self.waypoint_pub = rospy.Publisher(
            "/waypoint/target", PoseStamped, queue_size=10
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

        # Kalau tidak anchor ke posisi UAV, build path langsung
        if not self.start_at_current_pose:
            self.build_path(anchor_to_current=False)
        else:
            rospy.loginfo("start_at_current_pose = True → path akan di-anchor saat pose pertama diterima.")

        # Timer publish 20 Hz
        self.timer = rospy.Timer(rospy.Duration(0.05), self.timer_cb)

    # ======================================================================
    # MAVROS callbacks
    # ======================================================================
    def state_callback(self, msg: State):
        self.current_state = msg

    def position_callback(self, msg: PoseStamped):
        # ENU posisi
        self.current_pos_enu[:] = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ]
        if not self.pose_received:
            self.pose_received = True
            rospy.loginfo("✅ Pose pertama dari /mavros/local_position/pose diterima.")

            if self.start_at_current_pose and not self.traj_initialized:
                self.build_path(anchor_to_current=True)

    # ======================================================================
    # BUILD PATH + PRECOMPUTE
    # ======================================================================
    def build_path(self, anchor_to_current=False):
        if anchor_to_current and not self.pose_received:
            rospy.logwarn("build_path dipanggil sebelum pose diterima, skip.")
            return

        if anchor_to_current:
            # bentuk path di sekitar origin (altitude 0), nanti di-offset ke posisi UAV
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
                path = [
                    [0.0, 0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0, 0.0],
                    [5.0, 5.0, 0.0, np.pi/2],
                    [0.0, 5.0, 0.0, np.pi],
                    [0.0, 0.0, 0.0, -np.pi/2],
                ]

            # offset XY ke posisi UAV; ketinggian path kita set ke -hover_altitude
            n_uav = self.current_pos_enu[1]
            e_uav = self.current_pos_enu[0]
            d_hover = -self.hover_altitude  # NED (down)

            first_n, first_e, _, _ = path[0]
            dn = n_uav - first_n
            de = e_uav - first_e

            for wp in path:
                wp[0] += dn
                wp[1] += de
                wp[2] = d_hover   # paksa semua titik di ketinggian hover

            rospy.loginfo(
                "🚀 Trajectory anchored ke UAV: start NED ≈ (%.2f, %.2f, %.2f)",
                n_uav, e_uav, d_hover
            )
        else:
            # path absolut (kalau tidak pakai anchor)
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
        self._precompute_segments()
        self.traj_initialized = True
        self.t0 = rospy.Time.now().to_sec()

        self.mode = "WAIT_ALT"
        self.alt_ready_since = None

        rospy.loginfo("=" * 60)
        rospy.loginfo("PX4 Trajectory Publisher (time-based, gated)")
        rospy.loginfo("Mode trajektori   : %s", self.waypoint_mode)
        rospy.loginfo("Jumlah titik      : %d", len(self.path_points))
        rospy.loginfo("Kecepatan ref     : %.2f m/s", self.ref_speed)
        rospy.loginfo("Total waktu       : %.2f s", self.total_time)
        rospy.loginfo("Loop trajectory   : %s", "Yes" if self.loop_trajectory else "No")
        rospy.loginfo("Start at curr pos : %s", "Yes" if self.start_at_current_pose else "No")
        rospy.loginfo("Hover altitude    : %.2f m ENU", self.hover_altitude)
        rospy.loginfo("Altitude gate     : %s", "ON" if self.gate_on_altitude else "OFF")
        rospy.loginfo("Ref topics        : /trajectory/ref_pose, /trajectory/ref_vel, /waypoint/target")
        rospy.loginfo("=" * 60)

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

    # ======================================================================
    # TIMER → publish ref
    # ======================================================================
    def timer_cb(self, event):
        if not self.traj_initialized or self.total_time <= 0.0:
            return

        # --- Tentukan mode: masih WAIT_ALT atau sudah boleh TRACK? ---
        if self.mode == "WAIT_ALT" and self.gate_on_altitude and self.pose_received:
            alt = self.current_pos_enu[2]  # ENU z
            if abs(alt - self.hover_altitude) <= self.alt_tolerance:
                if self.alt_ready_since is None:
                    self.alt_ready_since = rospy.Time.now().to_sec()
                else:
                    if (rospy.Time.now().to_sec() - self.alt_ready_since) >= self.hover_stable_time:
                        self.mode = "TRACK"
                        self.t0 = rospy.Time.now().to_sec()
                        rospy.loginfo(
                            "✅ Altitude gate passed: z=%.2f m, start trajectory tracking.",
                            alt
                        )
            else:
                self.alt_ready_since = None

        if not self.gate_on_altitude and self.mode == "WAIT_ALT":
            self.mode = "TRACK"
            self.t0 = rospy.Time.now().to_sec()
            rospy.loginfo("Altitude gate disabled → langsung TRACK mode.")

        # ==========================================================
        # Hitung referensi
        # ==========================================================
        if self.mode == "WAIT_ALT":
            # Kirim titik pertama path (XY) di ketinggian hover, kecepatan nol
            p0 = self.positions[0]
            yaw0 = self.yaws[0]

            p_hover = np.array([p0[0], p0[1], -self.hover_altitude])  # NED (D negative)
            # NED → ENU
            enu_pos = np.array([p_hover[1], p_hover[0], -p_hover[2]])
            enu_vel = np.zeros(3)

            yaw_ref = yaw0
        else:  # TRACK mode
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

        # ===================== PUBLISH =====================
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(enu_pos[0])
        pose_msg.pose.position.y = float(enu_pos[1])
        pose_msg.pose.position.z = float(enu_pos[2])

        # yaw → quaternion (roll=0, pitch=0)
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

    # ======================================================================
    # TRAJECTORY GENERATORS (NED) – sama seperti sebelumnya
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
            yaw = angle + np.pi / 2.0
            waypoints.append([float(n), float(e), float(d), float(yaw)])

        total_height = abs(end_altitude - start_altitude)
        direction = "descending" if end_altitude < start_altitude else "ascending"
        transition_count = 1 if add_transition else 0

        rospy.loginfo("Helix trajectory:")
        rospy.loginfo("  Center: (%.1f, %.1f) m", center_n, center_e)
        rospy.loginfo("  Radius: %.1f m", radius)
        rospy.loginfo(
            "  Altitude: %.1f → %.1f m (%s, Δ=%.1f m)",
            -start_altitude, -end_altitude, direction, total_height
        )
        rospy.loginfo(
            "  Turns: %.1f, Points: %d helix + %d transition",
            turns, num_points, transition_count
        )
        return waypoints


def main():
    rospy.init_node("px4_trajectory_publisher_gated", anonymous=False)
    node = Px4TrajectoryPublisherGated()
    rospy.loginfo("PX4 Trajectory Publisher (gated) started.")
    rospy.spin()


if __name__ == "__main__":
    main()
