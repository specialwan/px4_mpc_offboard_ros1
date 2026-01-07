#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Logger untuk LOS Guidance Trajectory Following

Menyimpan data untuk analisis dan perhitungan metrik error:
1. Desired Path (titik proyeksi pada path asli)
2. LOS Reference (lookahead point dari LOS)
3. Actual Pose (posisi drone sebenarnya)
4. Velocity Reference dan Actual
5. Cross-track error, along-track distance
6. Yaw reference dan actual

Output: CSV file dengan timestamp

Subscribed Topics:
    /trajectory/desired_pose   - Desired path (proyeksi pada path asli)
    /trajectory/ref_pose       - LOS reference (lookahead point)
    /trajectory/ref_vel        - Velocity reference
    /trajectory/los_info       - LOS info (cross-track, along-track, etc.)
    /trajectory/status         - Trajectory status
    /mavros/local_position/pose     - Actual pose
    /mavros/local_position/velocity_local - Actual velocity

Author: Generated for MPC path following analysis
"""

import rospy
import numpy as np
import csv
import os
from datetime import datetime

from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import String


class LOSDataLogger(object):
    def __init__(self):
        rospy.loginfo("Init LOSDataLogger")

        # ===================== PARAMETERS =====================
        self.log_dir = rospy.get_param("~log_dir", os.path.expanduser("~/catkin_ws_ros/src/mpc_offboard/logs"))
        self.log_rate = rospy.get_param("~log_rate", 20.0)  # Hz
        self.auto_start = rospy.get_param("~auto_start", True)
        self.stop_on_finish = rospy.get_param("~stop_on_finish", True)
        
        # Create log directory if not exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            rospy.loginfo(f"Created log directory: {self.log_dir}")

        # ===================== STATE =====================
        self.is_logging = False
        self.log_file = None
        self.csv_writer = None
        self.start_time = None
        self.data_count = 0
        
        # Data buffers (latest received data)
        self.desired_pose = None      # Desired path (proyeksi)
        self.los_ref_pose = None      # LOS lookahead point
        self.actual_pose = None       # Actual drone pose
        self.ref_vel = None           # Reference velocity
        self.actual_vel = None        # Actual velocity
        self.los_info = None          # LOS info (cross-track, along-track, etc.)
        self.trajectory_status = "INIT"
        
        # Timestamps untuk sinkronisasi
        self.desired_pose_time = None
        self.los_ref_pose_time = None
        self.actual_pose_time = None
        
        # ===================== SUBSCRIBERS =====================
        self.desired_pose_sub = rospy.Subscriber(
            "/trajectory/desired_pose", PoseStamped,
            self.desired_pose_callback, queue_size=10
        )
        self.los_ref_pose_sub = rospy.Subscriber(
            "/trajectory/ref_pose", PoseStamped,
            self.los_ref_pose_callback, queue_size=10
        )
        self.ref_vel_sub = rospy.Subscriber(
            "/trajectory/ref_vel", TwistStamped,
            self.ref_vel_callback, queue_size=10
        )
        self.los_info_sub = rospy.Subscriber(
            "/trajectory/los_info", TwistStamped,
            self.los_info_callback, queue_size=10
        )
        self.status_sub = rospy.Subscriber(
            "/trajectory/status", String,
            self.status_callback, queue_size=10
        )
        self.actual_pose_sub = rospy.Subscriber(
            "/mavros/local_position/pose", PoseStamped,
            self.actual_pose_callback, queue_size=10
        )
        self.actual_vel_sub = rospy.Subscriber(
            "/mavros/local_position/velocity_local", TwistStamped,
            self.actual_vel_callback, queue_size=10
        )
        
        # ===================== TIMER =====================
        self.log_timer = rospy.Timer(rospy.Duration(1.0 / self.log_rate), self.log_timer_cb)
        
        # Auto start logging jika diaktifkan
        if self.auto_start:
            rospy.sleep(1.0)  # Wait for subscribers to connect
            self.start_logging()
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("LOS Data Logger initialized")
        rospy.loginfo(f"Log directory: {self.log_dir}")
        rospy.loginfo(f"Log rate: {self.log_rate} Hz")
        rospy.loginfo(f"Auto start: {self.auto_start}")
        rospy.loginfo(f"Stop on finish: {self.stop_on_finish}")
        rospy.loginfo("=" * 60)

    # ======================================================================
    # CALLBACKS
    # ======================================================================
    def desired_pose_callback(self, msg: PoseStamped):
        self.desired_pose = msg
        self.desired_pose_time = rospy.Time.now()
    
    def los_ref_pose_callback(self, msg: PoseStamped):
        self.los_ref_pose = msg
        self.los_ref_pose_time = rospy.Time.now()
    
    def ref_vel_callback(self, msg: TwistStamped):
        self.ref_vel = msg
    
    def los_info_callback(self, msg: TwistStamped):
        self.los_info = msg
    
    def status_callback(self, msg: String):
        prev_status = self.trajectory_status
        self.trajectory_status = msg.data
        
        rospy.loginfo(f"Trajectory status: {self.trajectory_status}")
        
        # Stop logging when trajectory finished
        if self.stop_on_finish and self.trajectory_status == "FINISHED" and self.is_logging:
            rospy.loginfo("Trajectory finished, stopping logger...")
            self.stop_logging()
    
    def actual_pose_callback(self, msg: PoseStamped):
        self.actual_pose = msg
        self.actual_pose_time = rospy.Time.now()
    
    def actual_vel_callback(self, msg: TwistStamped):
        self.actual_vel = msg

    # ======================================================================
    # LOGGING CONTROL
    # ======================================================================
    def start_logging(self):
        if self.is_logging:
            rospy.logwarn("Already logging!")
            return
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"los_log_{timestamp}.csv"
        filepath = os.path.join(self.log_dir, filename)
        
        try:
            self.log_file = open(filepath, 'w', newline='')
            self.csv_writer = csv.writer(self.log_file)
            
            # Write header
            header = [
                'timestamp',
                'elapsed_time',
                # Desired path (proyeksi pada path asli)
                'desired_x', 'desired_y', 'desired_z', 'desired_yaw',
                # LOS reference (lookahead point)
                'los_ref_x', 'los_ref_y', 'los_ref_z', 'los_ref_yaw',
                # Actual pose
                'actual_x', 'actual_y', 'actual_z', 'actual_yaw',
                # Velocity reference
                'ref_vx', 'ref_vy', 'ref_vz',
                # Actual velocity
                'actual_vx', 'actual_vy', 'actual_vz',
                # LOS info
                'cross_track_error', 'along_track_distance', 'segment_idx',
                # Computed errors
                'error_x', 'error_y', 'error_z', 'error_xy', 'error_3d',
                'error_from_desired_x', 'error_from_desired_y', 'error_from_desired_z',
                'error_from_desired_xy', 'error_from_desired_3d',
                # Status
                'status'
            ]
            self.csv_writer.writerow(header)
            
            self.start_time = rospy.Time.now()
            self.is_logging = True
            self.data_count = 0
            
            rospy.loginfo(f"📝 Started logging to: {filepath}")
            
        except Exception as e:
            rospy.logerr(f"Failed to start logging: {e}")
    
    def stop_logging(self):
        if not self.is_logging:
            rospy.logwarn("Not currently logging!")
            return
        
        self.is_logging = False
        
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            self.csv_writer = None
        
        rospy.loginfo(f"📝 Stopped logging. Total data points: {self.data_count}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary statistics setelah logging selesai."""
        rospy.loginfo("=" * 60)
        rospy.loginfo("LOGGING SUMMARY")
        rospy.loginfo(f"Total data points: {self.data_count}")
        if self.start_time:
            duration = (rospy.Time.now() - self.start_time).to_sec()
            rospy.loginfo(f"Duration: {duration:.2f} seconds")
            rospy.loginfo(f"Effective rate: {self.data_count / duration:.2f} Hz")
        rospy.loginfo("=" * 60)

    # ======================================================================
    # LOG TIMER
    # ======================================================================
    def log_timer_cb(self, event):
        if not self.is_logging:
            return
        
        # Check if we have all required data
        if self.actual_pose is None:
            return
        
        now = rospy.Time.now()
        elapsed = (now - self.start_time).to_sec() if self.start_time else 0.0
        
        # Extract actual pose
        actual_x = self.actual_pose.pose.position.x
        actual_y = self.actual_pose.pose.position.y
        actual_z = self.actual_pose.pose.position.z
        actual_yaw = self.quaternion_to_yaw(self.actual_pose.pose.orientation)
        
        # Extract desired pose (proyeksi pada path)
        if self.desired_pose:
            desired_x = self.desired_pose.pose.position.x
            desired_y = self.desired_pose.pose.position.y
            desired_z = self.desired_pose.pose.position.z
            desired_yaw = self.quaternion_to_yaw(self.desired_pose.pose.orientation)
        else:
            desired_x = desired_y = desired_z = desired_yaw = 0.0
        
        # Extract LOS reference pose (lookahead point)
        if self.los_ref_pose:
            los_ref_x = self.los_ref_pose.pose.position.x
            los_ref_y = self.los_ref_pose.pose.position.y
            los_ref_z = self.los_ref_pose.pose.position.z
            los_ref_yaw = self.quaternion_to_yaw(self.los_ref_pose.pose.orientation)
        else:
            los_ref_x = los_ref_y = los_ref_z = los_ref_yaw = 0.0
        
        # Extract reference velocity
        if self.ref_vel:
            ref_vx = self.ref_vel.twist.linear.x
            ref_vy = self.ref_vel.twist.linear.y
            ref_vz = self.ref_vel.twist.linear.z
        else:
            ref_vx = ref_vy = ref_vz = 0.0
        
        # Extract actual velocity
        if self.actual_vel:
            actual_vx = self.actual_vel.twist.linear.x
            actual_vy = self.actual_vel.twist.linear.y
            actual_vz = self.actual_vel.twist.linear.z
        else:
            actual_vx = actual_vy = actual_vz = 0.0
        
        # Extract LOS info
        if self.los_info:
            cross_track_error = self.los_info.twist.linear.x
            along_track_distance = self.los_info.twist.linear.y
            segment_idx = int(self.los_info.twist.linear.z)
        else:
            cross_track_error = along_track_distance = segment_idx = 0.0
        
        # Compute errors from LOS reference
        error_x = los_ref_x - actual_x
        error_y = los_ref_y - actual_y
        error_z = los_ref_z - actual_z
        error_xy = np.sqrt(error_x**2 + error_y**2)
        error_3d = np.sqrt(error_x**2 + error_y**2 + error_z**2)
        
        # Compute errors from desired path (path asli)
        error_from_desired_x = desired_x - actual_x
        error_from_desired_y = desired_y - actual_y
        error_from_desired_z = desired_z - actual_z
        error_from_desired_xy = np.sqrt(error_from_desired_x**2 + error_from_desired_y**2)
        error_from_desired_3d = np.sqrt(error_from_desired_x**2 + error_from_desired_y**2 + error_from_desired_z**2)
        
        # Write row
        row = [
            now.to_sec(),
            elapsed,
            # Desired path
            desired_x, desired_y, desired_z, desired_yaw,
            # LOS reference
            los_ref_x, los_ref_y, los_ref_z, los_ref_yaw,
            # Actual pose
            actual_x, actual_y, actual_z, actual_yaw,
            # Velocity reference
            ref_vx, ref_vy, ref_vz,
            # Actual velocity
            actual_vx, actual_vy, actual_vz,
            # LOS info
            cross_track_error, along_track_distance, segment_idx,
            # Computed errors (from LOS ref)
            error_x, error_y, error_z, error_xy, error_3d,
            # Computed errors (from desired path)
            error_from_desired_x, error_from_desired_y, error_from_desired_z,
            error_from_desired_xy, error_from_desired_3d,
            # Status
            self.trajectory_status
        ]
        
        self.csv_writer.writerow(row)
        self.data_count += 1
        
        # Flush periodically
        if self.data_count % 100 == 0:
            self.log_file.flush()
            rospy.loginfo_throttle(5.0, f"📝 Logged {self.data_count} data points...")

    # ======================================================================
    # UTILITY
    # ======================================================================
    def quaternion_to_yaw(self, q):
        """Extract yaw from quaternion."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)
    
    def shutdown_hook(self):
        """Called when node is shutting down."""
        if self.is_logging:
            rospy.loginfo("Shutting down, stopping logging...")
            self.stop_logging()


def main():
    rospy.init_node("los_data_logger", anonymous=False)
    node = LOSDataLogger()
    rospy.on_shutdown(node.shutdown_hook)
    rospy.loginfo("LOS Data Logger started.")
    rospy.spin()


if __name__ == "__main__":
    main()
