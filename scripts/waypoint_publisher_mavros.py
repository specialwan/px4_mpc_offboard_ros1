#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Waypoint Publisher with Auto-Advance (ROS1 + MAVROS)

Publishes waypoints ke /waypoint/target untuk MPC Waypoint Follower

Fitur:
- Auto-advance ke waypoint berikut saat waypoint sekarang "tercapai"
- Subscribe posisi drone dari /mavros/local_position/pose
- Support beberapa mode trajektori: default, circle, square, helix
"""

import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np


class WaypointPublisherMavros(object):
    def __init__(self):
        # Inisialisasi node
        # (di main() kita sudah panggil rospy.init_node)
        rospy.loginfo("Init WaypointPublisherMavros (ROS1)")

        # Publisher untuk waypoint
        self.waypoint_pub = rospy.Publisher(
            "/waypoint/target",
            PoseStamped,
            queue_size=10
        )

        # Subscriber untuk posisi drone dari MAVROS (ENU frame)
        self.position_sub = rospy.Subscriber(
            "/mavros/local_position/pose",
            PoseStamped,
            self.position_callback,
            queue_size=1
        )

        # ===== State variables =====
        self.current_position = np.array([0.0, 0.0, 0.0])  # NED internal
        self.waypoint_reached = False
        self.mission_complete = False
        self.current_waypoint_index = 0

        self.last_log_time = rospy.Time.now()

        # ===== Parameter (rosparam) =====
        # Gunakan private namespace (~) supaya bisa di-set lewat launch file
        self.continuous_mode = rospy.get_param("~continuous_mode", True)
        self.acceptance_radius = rospy.get_param("~acceptance_radius", 0.8)
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 2.0)
        self.loop_mission = rospy.get_param("~loop_mission", False)

        # Default waypoints (NED: [North, East, Down, yaw])
        self.waypoints = [
            [0.0, 0.0, -5.0, 0.0],
            [5.0, 0.0, -5.0, 0.0],
            [5.0, 5.0, -5.0, 0.0],
            [0.0, 5.0, -5.0, 0.0],
            [0.0, 0.0, -5.0, 0.0],
        ]

        # ===== Trajectory Mode Selection =====
        # Supported: 'default', 'circle', 'square', 'helix'
        self.waypoint_mode = rospy.get_param("~waypoint_mode", "default")

        # Circle parameters
        circle_center_n = rospy.get_param("~circle_center_n", 0.0)
        circle_center_e = rospy.get_param("~circle_center_e", 0.0)
        circle_radius = rospy.get_param("~circle_radius", 25.0)
        circle_altitude = rospy.get_param("~circle_altitude", -5.0)
        circle_points = int(rospy.get_param("~circle_points", 80))

        # Square parameters
        square_center_n = rospy.get_param("~square_center_n", 0.0)
        square_center_e = rospy.get_param("~square_center_e", 0.0)
        square_size = rospy.get_param("~square_size", 20.0)
        square_altitude = rospy.get_param("~square_altitude", -5.0)
        square_points_per_side = int(rospy.get_param("~square_points_per_side", 10))
        square_constant_yaw = rospy.get_param("~square_constant_yaw", False)

        # Helix parameters
        helix_center_n = rospy.get_param("~helix_center_n", 0.0)
        helix_center_e = rospy.get_param("~helix_center_e", 0.0)
        helix_radius = rospy.get_param("~helix_radius", 1.0)
        helix_start_altitude = rospy.get_param("~helix_start_altitude", -10.0)
        helix_end_altitude = rospy.get_param("~helix_end_altitude", -30.0)
        helix_turns = rospy.get_param("~helix_turns", 3.0)
        helix_points = int(rospy.get_param("~helix_points", 120))

        # Generate trajectory sesuai mode
        if self.waypoint_mode == "circle":
            self.waypoints = self.generate_circle_waypoints(
                center_n=circle_center_n,
                center_e=circle_center_e,
                radius=circle_radius,
                altitude=circle_altitude,
                num_points=circle_points
            )
        elif self.waypoint_mode == "square":
            self.waypoints = self.generate_square_waypoints(
                center_n=square_center_n,
                center_e=square_center_e,
                size=square_size,
                altitude=square_altitude,
                points_per_side=square_points_per_side,
                constant_yaw=square_constant_yaw
            )
        elif self.waypoint_mode == "helix":
            self.waypoints = self.generate_helix_waypoints(
                center_n=helix_center_n,
                center_e=helix_center_e,
                radius=helix_radius,
                start_altitude=helix_start_altitude,
                end_altitude=helix_end_altitude,
                turns=helix_turns,
                num_points=helix_points
            )

        rospy.loginfo("=" * 60)
        rospy.loginfo("Smart Waypoint Publisher with Auto-Advance (ROS1)")
        rospy.loginfo("Waypoint Mode       : %s", self.waypoint_mode)
        rospy.loginfo("Total waypoints     : %d", len(self.waypoints))
        rospy.loginfo("Tracking Mode       : %s",
                      "CONTINUOUS (smooth)" if self.continuous_mode else "DISCRETE (stop at each WP)")
        rospy.loginfo("Acceptance radius   : %.2f m", self.acceptance_radius)
        rospy.loginfo("Lookahead distance  : %.2f m", self.lookahead_distance)
        rospy.loginfo("Loop mission        : %s", "Yes" if self.loop_mission else "No")
        rospy.loginfo("Publishing to topic : /waypoint/target")
        rospy.loginfo("=" * 60)

        # Timer untuk publish waypoint (10 Hz)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_waypoint)

    # =====================================================================
    # CALLBACK POSISI (dari MAVROS)
    # =====================================================================
    def position_callback(self, msg):
        """Update posisi drone dan handle perpindahan waypoint."""
        # /mavros/local_position/pose → ENU
        # Konversi ENU → NED internal
        self.current_position = np.array([
            msg.pose.position.y,      # ENU Y → NED X (North)
            msg.pose.position.x,      # ENU X → NED Y (East)
            -msg.pose.position.z      # ENU Z → NED Z (Down)
        ])

        # Kalau mission sudah selesai dan tidak loop → tidak usah apa-apa
        if self.mission_complete and not self.loop_mission:
            return

        if self.current_waypoint_index >= len(self.waypoints):
            return

        # Waypoint aktif
        wp = self.waypoints[self.current_waypoint_index]
        target_position = np.array([wp[0], wp[1], wp[2]])
        distance = np.linalg.norm(self.current_position - target_position)

        # Log jarak tiap 2 detik
        now = rospy.Time.now()
        if (now - self.last_log_time).to_sec() > 2.0:
            rospy.loginfo(
                "WP %d/%d: distance = %.2f m",
                self.current_waypoint_index + 1,
                len(self.waypoints),
                distance
            )
            self.last_log_time = now

        # ===== CONTINUOUS MODE: pakai lookahead_distance =====
        if self.continuous_mode:
            if distance < self.lookahead_distance:
                if self.current_waypoint_index < len(self.waypoints) - 1:
                    self.current_waypoint_index += 1
                    rospy.loginfo(
                        "→ Advancing to WP %d (continuous mode)",
                        self.current_waypoint_index + 1
                    )
                else:
                    # Sudah di waypoint terakhir
                    if self.loop_mission:
                        rospy.loginfo("↻ Looping trajectory ke WP1")
                        self.current_waypoint_index = 0
                    else:
                        if not self.mission_complete:
                            rospy.loginfo("✓ Mission complete - hover di waypoint terakhir")
                            self.mission_complete = True

        # ===== DISCRETE MODE: stop di tiap waypoint =====
        else:
            if distance < self.acceptance_radius and not self.waypoint_reached:
                self.waypoint_reached = True
                rospy.loginfo(
                    "✓ Waypoint %d REACHED (distance: %.2f m)",
                    self.current_waypoint_index + 1,
                    distance
                )

                if self.current_waypoint_index < len(self.waypoints) - 1:
                    self.current_waypoint_index += 1
                    self.waypoint_reached = False
                    rospy.loginfo(
                        "➜ Moving to waypoint %d/%d",
                        self.current_waypoint_index + 1,
                        len(self.waypoints)
                    )
                else:
                    rospy.loginfo("=" * 60)
                    rospy.loginfo("🎉 MISSION COMPLETE! All waypoints reached.")
                    if self.loop_mission:
                        rospy.loginfo("Looping back ke waypoint pertama...")
                        rospy.loginfo("=" * 60)
                        self.current_waypoint_index = 0
                        self.waypoint_reached = False
                    else:
                        rospy.loginfo("Mission akan hover di waypoint terakhir.")
                        rospy.loginfo("=" * 60)
                        self.mission_complete = True

    # =====================================================================
    # PUBLISH WAYPOINT (dipanggil oleh rospy.Timer)
    # =====================================================================
    def publish_waypoint(self, event):
        """Publish waypoint aktif (konversi NED → ENU)."""
        if self.current_waypoint_index >= len(self.waypoints):
            return

        wp = self.waypoints[self.current_waypoint_index]
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        # NED → ENU
        # NED: [North, East, Down]
        # ENU: [East, North, Up]
        msg.pose.position.x = wp[1]        # NED East → ENU X
        msg.pose.position.y = wp[0]        # NED North → ENU Y
        msg.pose.position.z = -wp[2]       # NED Down → ENU Z

        # Yaw → quaternion (hanya yaw, roll/pitch=0)
        yaw = wp[3]
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = np.sin(yaw / 2.0)
        msg.pose.orientation.w = np.cos(yaw / 2.0)

        self.waypoint_pub.publish(msg)

    # =====================================================================
    # TRAJECTORY GENERATORS
    # =====================================================================
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
    rospy.init_node("waypoint_publisher_mavros", anonymous=False)
    node = WaypointPublisherMavros()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
