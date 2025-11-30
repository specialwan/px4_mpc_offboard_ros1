#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-based Trajectory Publisher (ROS1 + MAVROS)

Berbeda dengan waypoint publisher lama (yang advance berdasarkan jarak),
node ini menghasilkan TRAJEKTORI SEBAGAI FUNGSI WAKTU:

- Precompute titik-titik path (NED) dengan generator: default / circle / square / helix
- Hitung panjang tiap segmen → waktu tiap segmen untuk kecepatan referensi konstan
- Untuk setiap waktu t:
    p_ref(t), yaw_ref(t) diinterpolasi sepanjang path
    v_ref(t) = v_ref * arah_tangen_path

Output:
- /trajectory/ref_pose  (PoseStamped, ENU)
- /trajectory/ref_vel   (TwistStamped, ENU)
- /waypoint/target      (PoseStamped, ENU)  -> kompatibel dengan node lama / data logger
"""

import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped, TwistStamped


class TrajectoryPublisherMavros(object):
    def __init__(self):
        rospy.loginfo("Init TrajectoryPublisherMavros (time-based trajectory)")

        # ===================== PARAMETER =====================
        self.waypoint_mode = rospy.get_param("~waypoint_mode", "default")

        # Kecepatan referensi sepanjang lintasan (NED) [m/s]
        self.ref_speed = rospy.get_param("~ref_speed", 2.0)  # konstanta
        self.loop_trajectory = rospy.get_param("~loop_trajectory", True)
        
        # Circle params
        circle_center_n = rospy.get_param("~circle_center_n", 0.0)
        circle_center_e = rospy.get_param("~circle_center_e", 0.0)
        circle_radius = rospy.get_param("~circle_radius", 25.0)
        circle_altitude = rospy.get_param("~circle_altitude", -5.0)
        circle_points = int(rospy.get_param("~circle_points", 80))

        # Square params
        square_center_n = rospy.get_param("~square_center_n", 0.0)
        square_center_e = rospy.get_param("~square_center_e", 0.0)
        square_size = rospy.get_param("~square_size", 20.0)
        square_altitude = rospy.get_param("~square_altitude", -5.0)
        square_points_per_side = int(rospy.get_param("~square_points_per_side", 10))
        square_constant_yaw = rospy.get_param("~square_constant_yaw", False)

        # Helix params
        helix_center_n = rospy.get_param("~helix_center_n", 0.0)
        helix_center_e = rospy.get_param("~helix_center_e", 0.0)
        helix_radius = rospy.get_param("~helix_radius", 1.0)
        helix_start_altitude = rospy.get_param("~helix_start_altitude", -10.0)
        helix_end_altitude = rospy.get_param("~helix_end_altitude", -30.0)
        helix_turns = rospy.get_param("~helix_turns", 3.0)
        helix_points = int(rospy.get_param("~helix_points", 120))

        # ===================== GENERATE PATH (NED) =====================
        # Format internal path_points: list of [n, e, d, yaw]
        if self.waypoint_mode == "circle":
            self.path_points = self.generate_circle_waypoints(
                center_n=circle_center_n,
                center_e=circle_center_e,
                radius=circle_radius,
                altitude=circle_altitude,
                num_points=circle_points
            )
        elif self.waypoint_mode == "square":
            self.path_points = self.generate_square_waypoints(
                center_n=square_center_n,
                center_e=square_center_e,
                size=square_size,
                altitude=square_altitude,
                points_per_side=square_points_per_side,
                constant_yaw=square_constant_yaw
            )
        elif self.waypoint_mode == "helix":
            self.path_points = self.generate_helix_waypoints(
                center_n=helix_center_n,
                center_e=helix_center_e,
                radius=helix_radius,
                start_altitude=helix_start_altitude,
                end_altitude=helix_end_altitude,
                turns=helix_turns,
                num_points=helix_points
            )
        else:
            # default: pakai path kotak kecil 5x5 di sekitar origin
            self.path_points = [
                [0.0, 0.0, -5.0, 0.0],
                [5.0, 0.0, -5.0, 0.0],
                [5.0, 5.0, -5.0, np.pi/2],
                [0.0, 5.0, -5.0, np.pi],
                [0.0, 0.0, -5.0, -np.pi/2],
            ]

        # Minimal 2 titik
        if len(self.path_points) < 2:
            rospy.logwarn("Path hanya punya <=1 point. Menambah satu titik dummy.")
            self.path_points.append(self.path_points[0])

        # ===================== PRECOMPUTE SEGMENTS =====================
        self._precompute_segments()

        # ===================== PUB / SUB =====================
        # → referensi utama untuk MPC (trajectory tracking)
        self.pose_pub = rospy.Publisher(
            "/trajectory/ref_pose", PoseStamped, queue_size=10
        )
        self.vel_pub = rospy.Publisher(
            "/trajectory/ref_vel", TwistStamped, queue_size=10
        )

        # → kompatibel dengan node MPC lama & data logger (hanya pose)
        self.waypoint_pub = rospy.Publisher(
            "/waypoint/target", PoseStamped, queue_size=10
        )

        # Optional: subscribe posisi drone untuk log error (kalau mau nanti)
        # self.pos_sub = rospy.Subscriber(
        #     "/mavros/local_position/pose", PoseStamped, self.position_callback, queue_size=1
        # )

        # Waktu awal trajektori
        self.t0 = rospy.Time.now().to_sec()

        rospy.loginfo("=" * 60)
        rospy.loginfo("TrajectoryPublisherMavros (time-based)")
        rospy.loginfo("Mode trajektori : %s", self.waypoint_mode)
        rospy.loginfo("Jumlah titik    : %d", len(self.path_points))
        rospy.loginfo("Kecepatan ref   : %.2f m/s", self.ref_speed)
        rospy.loginfo("Total waktu     : %.2f s", self.total_time)
        rospy.loginfo("Loop trajectory : %s", "Yes" if self.loop_trajectory else "No")
        rospy.loginfo("Output pose     : /trajectory/ref_pose & /waypoint/target")
        rospy.loginfo("Output vel      : /trajectory/ref_vel")
        rospy.loginfo("=" * 60)

        # Timer publish (misal 20 Hz)
        self.timer = rospy.Timer(rospy.Duration(0.05), self.timer_cb)

    # ================================================================
    # PRECOMPUTE SEGMENTS
    # ================================================================
    def _precompute_segments(self):
        """
        Dari path_points (NED: [n,e,d,yaw]) → hitung:
        - segment_lengths[i] : panjang (m) dari titik i → i+1
        - segment_times[i]   : waktu (s) di segmen i untuk ref_speed
        - cum_times[i]       : waktu kumulatif awal segmen i
        """
        pts = np.array(self.path_points)  # shape (N,4)
        positions = pts[:, 0:3]           # (n,e,d)
        yaws = pts[:, 3]                  # yaw

        # Segment dari i → i+1 (N-1 segmen)
        deltas = positions[1:, :] - positions[:-1, :]
        seg_lengths = np.linalg.norm(deltas, axis=1)

        # Safety: kalau ada segmen dengan panjang 0, kasih panjang kecil
        seg_lengths = np.where(seg_lengths < 1e-6, 1e-6, seg_lengths)

        # Waktu per segmen (s) = panjang / v_ref
        seg_times = seg_lengths / max(self.ref_speed, 1e-3)

        # Waktu kumulatif
        cum_times = np.zeros(seg_times.shape[0] + 1)
        cum_times[1:] = np.cumsum(seg_times)

        self.positions = positions        # (N,3)
        self.yaws = yaws                  # (N,)
        self.segment_lengths = seg_lengths
        self.segment_times = seg_times
        self.cum_times = cum_times        # panjang N
        self.total_time = float(cum_times[-1])

    # ================================================================
    # TIMER CALLBACK → HITUNG p_ref(t), v_ref(t)
    # ================================================================
    def timer_cb(self, event):
        now = rospy.Time.now().to_sec()
        t_rel = now - self.t0

        if self.total_time <= 0.0:
            return

        if self.loop_trajectory:
            # wrap modulo total_time
            t_rel = np.fmod(t_rel, self.total_time)
            if t_rel < 0:
                t_rel += self.total_time
        else:
            # clamp
            if t_rel < 0.0:
                t_rel = 0.0
            if t_rel > self.total_time:
                t_rel = self.total_time

        # Cari segmen i yang memenuhi: cum_times[i] <= t_rel < cum_times[i+1]
        # (cum_times panjang Nseg+1)
        # np.searchsorted mengembalikan index j sehingga cum_times[j-1] <= t_rel < cum_times[j]
        # kita butuh i = j-1
        j = np.searchsorted(self.cum_times, t_rel, side="right") - 1
        j = int(np.clip(j, 0, len(self.segment_times) - 1))

        t_start = self.cum_times[j]
        dt_seg = self.segment_times[j]

        if dt_seg <= 0.0:
            tau = 0.0
        else:
            tau = (t_rel - t_start) / dt_seg
            tau = float(np.clip(tau, 0.0, 1.0))

        # Posisi di segmen j: interpolasi linear
        p0 = self.positions[j]
        p1 = self.positions[j + 1]
        p_ref = (1.0 - tau) * p0 + tau * p1   # NED: [n,e,d]

        # Yaw di segmen j: interpolasi sudut terpendek
        yaw0 = self.yaws[j]
        yaw1 = self.yaws[j + 1]
        yaw_diff = np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))
        yaw_ref = yaw0 + tau * yaw_diff

        # Kecepatan referensi: arah tangen segmen * ref_speed
        dp = p1 - p0
        seg_len = self.segment_lengths[j]
        dir_vec = dp / max(seg_len, 1e-6)  # NED
        v_ref_ned = dir_vec * self.ref_speed

        # ====================================================
        # KONVERSI NED → ENU DAN PUBLISH
        # ====================================================
        # Posisi: NED [n,e,d] → ENU [x=e, y=n, z=-d]
        enu_pos = np.array([
            p_ref[1],
            p_ref[0],
            -p_ref[2]
        ])

        # Velocity: NED [vn, ve, vd] → ENU [vx=ve, vy=vn, vz=-vd]
        enu_vel = np.array([
            v_ref_ned[1],
            v_ref_ned[0],
            -v_ref_ned[2]
        ])

        # PoseStamped (ref_pose & waypoint/target)
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

        # TwistStamped (ref_vel)
        vel_msg = TwistStamped()
        vel_msg.header.stamp = pose_msg.header.stamp
        vel_msg.header.frame_id = "map"
        vel_msg.twist.linear.x = float(enu_vel[0])
        vel_msg.twist.linear.y = float(enu_vel[1])
        vel_msg.twist.linear.z = float(enu_vel[2])

        # Publish
        self.pose_pub.publish(pose_msg)
        self.vel_pub.publish(vel_msg)
        # Backward-compatible
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
    rospy.init_node("trajectory_publisher_mavros", anonymous=False)
    node = TrajectoryPublisherMavros()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
