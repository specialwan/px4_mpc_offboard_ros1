#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Waypoint Publisher for PX4 Internal Controller (PID) - ROS1

- Generate trajectory (default / circle / square / helix) di frame NED
- Publish ke:
    /mavros/setpoint_position/local  (PX4 internal PID position controller)
    /waypoint/target                 (untuk data logger)
- Auto-advance waypoint (continuous / discrete)
- Auto OFFBOARD + ARM setelah stream setpoint beberapa detik
"""

import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


class WaypointPublisherPX4(object):
    def __init__(self):
        # ------------------------------------------------------------------
        # Node init
        # ------------------------------------------------------------------
        rospy.loginfo("Initializing WaypointPublisherPX4 (ROS1)...")

        # Publisher ke PX4 internal controller
        self.setpoint_pub = rospy.Publisher(
            "/mavros/setpoint_position/local",
            PoseStamped,
            queue_size=10
        )

        # Publisher tambahan untuk data logger
        self.waypoint_pub = rospy.Publisher(
            "/waypoint/target",
            PoseStamped,
            queue_size=10
        )

        # Subscriber posisi lokal
        self.position_sub = rospy.Subscriber(
            "/mavros/local_position/pose",
            PoseStamped,
            self.position_callback,
            queue_size=10
        )

        # Subscriber state MAVROS
        self.state_sub = rospy.Subscriber(
            "/mavros/state",
            State,
            self.state_callback,
            queue_size=10
        )

        # Service MAVROS
        self.arming_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        # ------------------------------------------------------------------
        # State variables
        # ------------------------------------------------------------------
        self.current_position = np.array([0.0, 0.0, 0.0])  # NED
        self.current_state = State()
        self.waypoint_reached = False
        self.last_log_time = rospy.Time.now()
        self.mission_complete = False
        self.offboard_enabled = False
        self.armed = False
        self.setpoint_counter = 0  # counter untuk pre-arm streaming

        # ------------------------------------------------------------------
        # Parameter trajectory & behaviour (ROS params)
        # ------------------------------------------------------------------
        self.continuous_mode = rospy.get_param("~continuous_mode", True)
        self.acceptance_radius = rospy.get_param("~acceptance_radius", 0.8)
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 2.0)
        self.loop_mission = rospy.get_param("~loop_mission", False)

        # Default waypoints (NED: [N, E, D, yaw])
        self.waypoints = [
            [0.0, 0.0, -5.0, 0.0],
            [5.0, 0.0, -5.0, 0.0],
            [5.0, 5.0, -5.0, 0.0],
            [0.0, 5.0, -5.0, 0.0],
            [0.0, 0.0, -5.0, 0.0],
        ]

        # Mode trajectory
        self.waypoint_mode = rospy.get_param("~waypoint_mode", "default")

        # Circle params
        circle_center_n = rospy.get_param("~circle_center_n", 0.0)
        circle_center_e = rospy.get_param("~circle_center_e", 0.0)
        circle_radius = rospy.get_param("~circle_radius", 25.0)
        circle_altitude = rospy.get_param("~circle_altitude", -5.0)
        circle_points = rospy.get_param("~circle_points", 80)

        # Square params
        square_center_n = rospy.get_param("~square_center_n", 0.0)
        square_center_e = rospy.get_param("~square_center_e", 0.0)
        square_size = rospy.get_param("~square_size", 20.0)
        square_altitude = rospy.get_param("~square_altitude", -5.0)
        square_points_per_side = rospy.get_param("~square_points_per_side", 10)
        square_constant_yaw = rospy.get_param("~square_constant_yaw", False)
        self.square_desired_spacing = rospy.get_param("~square_desired_spacing", 1.0)

        # Helix params
        helix_center_n = rospy.get_param("~helix_center_n", 0.0)
        helix_center_e = rospy.get_param("~helix_center_e", 0.0)
        helix_radius = rospy.get_param("~helix_radius", 1.0)
        helix_start_alt = rospy.get_param("~helix_start_altitude", -10.0)
        helix_end_alt = rospy.get_param("~helix_end_altitude", -30.0)
        helix_turns = rospy.get_param("~helix_turns", 3.0)
        helix_points = rospy.get_param("~helix_points", 120)

        # Generate trajectory sesuai mode
        if self.waypoint_mode == "circle":
            self.waypoints = self.generate_circle_waypoints(
                center_n=circle_center_n,
                center_e=circle_center_e,
                radius=circle_radius,
                altitude=circle_altitude,
                num_points=circle_points,
            )
        elif self.waypoint_mode == "square":
            self.waypoints = self.generate_square_waypoints(
                center_n=square_center_n,
                center_e=square_center_e,
                size=square_size,
                altitude=square_altitude,
                points_per_side=square_points_per_side,
                constant_yaw=square_constant_yaw,
            )
        elif self.waypoint_mode == "helix":
            self.waypoints = self.generate_helix_waypoints(
                center_n=helix_center_n,
                center_e=helix_center_e,
                radius=helix_radius,
                start_altitude=helix_start_alt,
                end_altitude=helix_end_alt,
                turns=helix_turns,
                num_points=helix_points,
            )

        self.current_waypoint_index = 0

        rospy.loginfo("=" * 60)
        rospy.loginfo("PX4 Waypoint Publisher (PID Controller) - ROS1")
        rospy.loginfo("Waypoint Mode: %s", self.waypoint_mode)
        rospy.loginfo("Total waypoints: %d", len(self.waypoints))
        rospy.loginfo("Tracking Mode: %s",
                      "CONTINUOUS (smooth)" if self.continuous_mode else "DISCRETE (stop at each WP)")
        rospy.loginfo("Acceptance radius: %.2f m", self.acceptance_radius)
        rospy.loginfo("Lookahead distance: %.2f m", self.lookahead_distance)
        rospy.loginfo("Loop mission: %s", "Yes" if self.loop_mission else "No")
        rospy.loginfo("Publishing to /mavros/setpoint_position/local (PX4 Controller)")
        rospy.loginfo("=" * 60)
        rospy.loginfo("IMPORTANT: Sending setpoints before arming...")
        rospy.loginfo("          Will auto-arm and switch to OFFBOARD after ~2s")
        rospy.loginfo("=" * 60)

    # ----------------------------------------------------------------------
    # Callbacks dan service helpers
    # ----------------------------------------------------------------------
    def state_callback(self, msg):
        self.current_state = msg

        if self.current_state.mode == "OFFBOARD" and not self.offboard_enabled:
            self.offboard_enabled = True
            rospy.loginfo("✓ OFFBOARD mode enabled!")

        if self.current_state.armed and not self.armed:
            self.armed = True
            rospy.loginfo("✓ Vehicle armed!")

    def arm_vehicle(self):
        """Arm via MAVROS"""
        try:
            rospy.wait_for_service("/mavros/cmd/arming", timeout=2.0)
        except rospy.ROSException:
            rospy.logwarn("Arming service not available")
            return False

        try:
            resp = self.arming_srv(True)
            if resp.success:
                rospy.loginfo("✓ Arming command sent successfully")
                return True
            else:
                rospy.logwarn("✗ Arming command failed")
                return False
        except rospy.ServiceException as e:
            rospy.logwarn("Arming service call failed: %s", e)
            return False

    def set_offboard_mode(self):
        """Set mode OFFBOARD"""
        try:
            rospy.wait_for_service("/mavros/set_mode", timeout=2.0)
        except rospy.ROSException:
            rospy.logwarn("Set mode service not available")
            return False

        try:
            # SetMode.srv: uint8 base_mode, string custom_mode
            resp = self.set_mode_srv(base_mode=0, custom_mode="OFFBOARD")
            if resp.mode_sent:
                rospy.loginfo("✓ OFFBOARD mode command sent successfully")
                return True
            else:
                rospy.logwarn("✗ OFFBOARD mode command failed")
                return False
        except rospy.ServiceException as e:
            rospy.logwarn("Set mode service call failed: %s", e)
            return False

    def position_callback(self, msg):
        """
        Update posisi drone (convert ENU → NED) dan handle waypoint advance.
        ENU: x=East, y=North, z=Up
        NED: x=North, y=East, z=Down
        """
        self.current_position = np.array([
            msg.pose.position.y,     # ENU Y → NED X
            msg.pose.position.x,     # ENU X → NED Y
            -msg.pose.position.z     # ENU Z (up) → NED Z (down)
        ])

        if self.mission_complete and not self.loop_mission:
            return

        if self.current_waypoint_index >= len(self.waypoints):
            return

        wp = self.waypoints[self.current_waypoint_index]
        target_pos = np.array([wp[0], wp[1], wp[2]])
        distance = np.linalg.norm(self.current_position - target_pos)

        now = rospy.Time.now()
        if (now - self.last_log_time).to_sec() > 2.0:
            rospy.loginfo("WP %d/%d distance: %.2f m",
                          self.current_waypoint_index + 1, len(self.waypoints), distance)
            self.last_log_time = now

        # CONTINUOUS MODE
        if self.continuous_mode:
            if distance < self.lookahead_distance:
                if self.current_waypoint_index < len(self.waypoints) - 1:
                    self.current_waypoint_index += 1
                    rospy.loginfo("→ Advancing to WP %d (continuous mode)",
                                  self.current_waypoint_index + 1)
                else:
                    if self.loop_mission:
                        rospy.loginfo("↻ Looping trajectory...")
                        self.current_waypoint_index = 0
                    else:
                        if not self.mission_complete:
                            rospy.loginfo("✓ Mission complete - hovering at final waypoint")
                            self.mission_complete = True
        # DISCRETE MODE
        else:
            if distance < self.acceptance_radius and not self.waypoint_reached:
                self.waypoint_reached = True
                rospy.loginfo("✓ Waypoint %d REACHED (distance: %.2f m)",
                              self.current_waypoint_index + 1, distance)

                if self.current_waypoint_index < len(self.waypoints) - 1:
                    self.current_waypoint_index += 1
                    self.waypoint_reached = False
                    rospy.loginfo("➜ Moving to Waypoint %d/%d",
                                  self.current_waypoint_index + 1, len(self.waypoints))
                else:
                    rospy.loginfo("=" * 60)
                    rospy.loginfo("🎉 MISSION COMPLETE! All waypoints reached.")
                    if self.loop_mission:
                        rospy.loginfo("Looping back to first waypoint...")
                        rospy.loginfo("=" * 60)
                        self.current_waypoint_index = 0
                        self.waypoint_reached = False
                    else:
                        rospy.loginfo("Mission will hover at final waypoint.")
                        rospy.loginfo("=" * 60)
                        self.mission_complete = True

    # ----------------------------------------------------------------------
    # Main publisher (dipanggil di loop utama @10 Hz)
    # ----------------------------------------------------------------------
    def publish_setpoint(self):
        self.setpoint_counter += 1

        if self.current_waypoint_index < len(self.waypoints):
            wp = self.waypoints[self.current_waypoint_index]
        else:
            wp = self.waypoints[-1]

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        # NED → ENU
        # NED: [N, E, D]
        # ENU: [E, N, U]
        msg.pose.position.x = wp[1]      # E
        msg.pose.position.y = wp[0]      # N
        msg.pose.position.z = -wp[2]     # -D

        yaw = wp[3]
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = np.sin(yaw / 2.0)
        msg.pose.orientation.w = np.cos(yaw / 2.0)

        # Publish ke PX4 controller
        self.setpoint_pub.publish(msg)
        # Mirror ke /waypoint/target (logger)
        self.waypoint_pub.publish(msg)

        # Setelah ±2 detik stream (20 * 0.1s) → set OFFBOARD + ARM
        if self.setpoint_counter == 20:
            rospy.loginfo("=" * 60)
            rospy.loginfo("✓ Setpoints streaming established (~2s)")
            rospy.loginfo("  Trying to switch to OFFBOARD...")
            rospy.loginfo("=" * 60)

        if self.setpoint_counter == 21:
            if self.current_state.mode != "OFFBOARD":
                self.set_offboard_mode()

        if self.setpoint_counter == 25:
            if not self.current_state.armed:
                self.arm_vehicle()

        if (not self.armed) and (self.setpoint_counter % 50 == 0):
            rospy.loginfo("Status: mode=%s, armed=%s, setpoints=%d",
                          self.current_state.mode,
                          str(self.current_state.armed),
                          self.setpoint_counter)

    # ----------------------------------------------------------------------
    # Trajectory generators (sama seperti ROS2)
    # ----------------------------------------------------------------------
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
        rospy.loginfo("Circle trajectory: center=(%.1f,%.1f), radius=%.1fm, alt=%.1fm, %d points, spacing=%.2fm",
                      center_n, center_e, radius, -altitude, num_points, avg_spacing)
        return waypoints

    def generate_square_waypoints(self, center_n=0.0, center_e=0.0, size=40.0,
                                  altitude=-5.0, points_per_side=3, constant_yaw=False):
        waypoints = []
        half = size / 2.0

        corners = [
            [center_n - half, center_e - half],  # SW
            [center_n + half, center_e - half],  # SE
            [center_n + half, center_e + half],  # NE
            [center_n - half, center_e + half],  # NW
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
            first_wp = waypoints[0]
            waypoints.append([first_wp[0], first_wp[1], first_wp[2], first_wp[3]])

        yaw_mode = "constant (0°)" if constant_yaw else "following path direction"
        rospy.loginfo("Square trajectory: center=(%.1f,%.1f), size=%.1fm, alt=%.1fm, points=%d, yaw=%s",
                      center_n, center_e, size, -altitude, len(waypoints), yaw_mode)
        return waypoints

    def generate_helix_waypoints(self, center_n=0.0, center_e=0.0, radius=1.0,
                                 start_altitude=-10.0, end_altitude=-30.0,
                                 turns=3.0, num_points=120, add_transition=True):
        waypoints = []

        if add_transition:
            waypoints.append([float(center_n), float(center_e), float(start_altitude), 0.0])

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
        rospy.loginfo("  Altitude: %.1fm → %.1fm (%s, Δ=%.1fm)",
                      -start_altitude, -end_altitude, direction, total_height)
        rospy.loginfo("  Turns: %.1f rotations", turns)
        rospy.loginfo("  Points: %d helix + %d transition = %d total",
                      num_points, transition_count, len(waypoints))

        return waypoints

    # ----------------------------------------------------------------------
    def spin(self):
        rate = rospy.Rate(10.0)  # 10 Hz (PX4 butuh >2 Hz OFFBOARD)
        while not rospy.is_shutdown():
            self.publish_setpoint()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("waypoint_publisher_px4", anonymous=False)
    node = WaypointPublisherPX4()
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass
