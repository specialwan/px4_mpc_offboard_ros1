#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Logger Node for PX4 LOS Guidance Trajectory Tracking (ROS1 version)

Sama seperti data_logger_los_guidance.py, tetapi untuk PX4 direct:
- LOS ref subscribe ke /mavros/setpoint_position/local (setpoint yang dikirim ke PX4)
- Target (desired path) dari /trajectory/desired_pose
- TIDAK ada control output dari MPC (karena menggunakan PID PX4)

Fitur:
- Logging → CSV (append) dengan urutan kolom tetap
- Waktu 'time' konsisten relatif t0 (sampel pertama), semua batch
- Saat ROS shutdown: flush CSV terakhir → build Excel (Summary + sheets)
- Menggunakan topik MAVROS + trajectory:
  - /mavros/local_position/pose
  - /mavros/local_position/velocity_local
  - /trajectory/desired_pose        (🔁 DESIRED PATH - titik proyeksi pada path asli)
  - /mavros/setpoint_position/local (🔁 LOS reference - setpoint yang dikirim ke PX4)
  - /trajectory/los_info            (LOS debug info: cross-track, along-track, etc.)
"""

import os
import csv
import time
import atexit
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import AttitudeTarget
from std_msgs.msg import String


class DataLoggerPx4LOSGuidance(object):
    def __init__(self):
        # =======================
        # PARAMETER
        # =======================
        home_dir = os.path.expanduser('~')
        default_log_dir = os.path.join(home_dir, 'flight_logs')

        self.output_dir = rospy.get_param('~output_dir', default_log_dir)
        self.log_rate = float(rospy.get_param('~log_rate_hz', 20.0))          # Hz
        self.auto_save_interval = float(rospy.get_param('~auto_save_interval', 30.0))  # detik
        self.auto_stop_on_finish = rospy.get_param('~auto_stop_on_finish', True)  # Auto stop saat trajectory selesai

        self.trajectory_finished = False

        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = f'flight_log_px4_los_{timestamp}'
        self.csv_filename = os.path.join(self.output_dir, base + '.csv')
        self.excel_filename = os.path.join(self.output_dir, base + '.xlsx')

        # clock steady (pakai time.monotonic)
        self._t0 = None               # t0 monotonic
        self.last_ref_time = None     # waktu terakhir trajectory ref pose diterima

        # =======================
        # DATA STORAGE
        # =======================
        self.data_buffer = []
        self._csv_header_written = False
        self._buf_flush_every = 200

        # urutan kolom fix
        self._columns = [
            'time',
            'pos_x', 'pos_y', 'pos_z',
            'vel_x', 'vel_y', 'vel_z', 'vel_mag',
            'roll_deg', 'pitch_deg', 'yaw_deg',
            # Target = desired path (proyeksi pada path asli)
            'target_pos_x', 'target_pos_y', 'target_pos_z',
            'target_yaw_deg',
            # LOS reference (lookahead point) - dari setpoint_position/local
            'los_ref_x', 'los_ref_y', 'los_ref_z', 'los_ref_yaw_deg',
            # LOS info (cross-track, along-track, etc.)
            'cross_track_error', 'along_track_distance', 'current_segment',
            'lookahead_distance', 'los_yaw_ned_deg',
            # Attitude target (dari PX4)
            'attitude_target_roll_deg', 'attitude_target_pitch_deg', 'attitude_target_yaw_deg',
            'attitude_target_thrust',
            # Error terhadap desired path (path asli)
            'error_x', 'error_y', 'error_z', 'error_mag',
            'error_yaw_deg',
            # Error terhadap LOS reference (lookahead point)
            'error_los_x', 'error_los_y', 'error_los_z', 'error_los_mag',
            'waypoint_fresh',   # sekarang artinya "ref point fresh" (trajectory point baru < 0.5 s)
            'trajectory_status',  # status: WAIT_ALT, TRACK, HOLD, FINISHED
        ]

        # state sekarang (ENU dari MAVROS)
        self.current_pos = np.zeros(3)
        self.current_vel = np.zeros(3)
        self.current_euler = np.zeros(3)  # roll, pitch, yaw (rad)

        # desired path (target) - titik proyeksi pada path asli
        self.target_pos = np.zeros(3)
        self.target_yaw = 0.0

        # LOS reference (lookahead point) - dari setpoint_position/local
        self.los_ref_pos = np.zeros(3)
        self.los_ref_yaw = 0.0

        # LOS info
        self.cross_track_error = 0.0
        self.along_track_distance = 0.0
        self.current_segment = 0
        self.lookahead_distance = 0.0
        self.los_yaw_ned = 0.0  # LOS yaw dalam NED frame

        # attitude target (dari PX4)
        self.attitude_target_euler = np.zeros(3)  # roll, pitch, yaw (rad)
        self.attitude_target_thrust = 0.0

        # flags
        self.position_received = False
        self.velocity_received = False
        self.desired_ref_received = False
        self.los_ref_received = False
        self.los_info_received = False
        self.attitude_target_received = False

        # trajectory status (WAIT_ALT, TRACK, HOLD, FINISHED)
        self.current_trajectory_status = "UNKNOWN"

        # info controller (untuk sheet Summary)
        self.controller_type = 'PX4_PID_LOS_GUIDANCE'

        # graceful shutdown
        self._shutting_down = False
        self._io_lock = threading.Lock()

        # =======================
        # SUBSCRIBERS
        # =======================
        queue_size = 10

        # Posisi & kecepatan aktual dari MAVROS (ENU)
        self.position_sub = rospy.Subscriber(
            '/mavros/local_position/pose',
            PoseStamped,
            self.position_callback,
            queue_size=queue_size
        )

        self.velocity_sub = rospy.Subscriber(
            '/mavros/local_position/velocity_local',
            TwistStamped,
            self.velocity_callback,
            queue_size=queue_size
        )

        # 🔁 DESIRED PATH - titik proyeksi pada path asli (untuk hitung error)
        self.desired_pose_sub = rospy.Subscriber(
            '/trajectory/desired_pose',
            PoseStamped,
            self.desired_pose_callback,
            queue_size=queue_size
        )

        # 🔁 LOS reference - setpoint yang dikirim ke PX4
        self.los_ref_pose_sub = rospy.Subscriber(
            '/mavros/setpoint_position/local',
            PoseStamped,
            self.los_ref_pose_callback,
            queue_size=queue_size
        )

        # LOS info (cross-track, along-track, etc.)
        self.los_info_sub = rospy.Subscriber(
            '/trajectory/los_info',
            TwistStamped,
            self.los_info_callback,
            queue_size=queue_size
        )

        # Attitude target yang dikirim ke PX4
        self.attitude_target_sub = rospy.Subscriber(
            '/mavros/setpoint_raw/target_attitude',
            AttitudeTarget,
            self.attitude_target_callback,
            queue_size=queue_size
        )

        # Subscribe ke trajectory status untuk auto-stop
        self.trajectory_status_sub = rospy.Subscriber(
            '/trajectory/status',
            String,
            self.trajectory_status_callback,
            queue_size=10
        )

        # =======================
        # TIMERS
        # =======================
        self.log_timer = rospy.Timer(
            rospy.Duration(1.0 / self.log_rate),
            self.log_data
        )
        self.save_timer = rospy.Timer(
            rospy.Duration(self.auto_save_interval),
            self.auto_flush_csv
        )

        # info awal
        rospy.loginfo('=' * 70)
        rospy.loginfo('📊 DATA LOGGER PX4 LOS GUIDANCE (ROS1) STARTED')
        rospy.loginfo('=' * 70)
        rospy.loginfo('Output dir : %s', self.output_dir)
        rospy.loginfo('CSV file   : %s', self.csv_filename)
        rospy.loginfo('Excel file : %s (dibuat saat shutdown)', self.excel_filename)
        rospy.loginfo('Log rate   : %.1f Hz', self.log_rate)
        rospy.loginfo('Flush intv : %.1f s', self.auto_save_interval)
        rospy.loginfo('=' * 70)
        rospy.loginfo('Target reference: /trajectory/desired_pose (desired path asli)')
        rospy.loginfo('LOS reference   : /mavros/setpoint_position/local (setpoint ke PX4)')
        rospy.loginfo('=' * 70)
        rospy.loginfo('Waiting for MAVROS & trajectory data...')

        # ROS shutdown hook + atexit
        rospy.on_shutdown(self.on_shutdown)
        atexit.register(self._atexit_handler)

    # =======================
    # CALLBACKS
    # =======================
    def position_callback(self, msg: PoseStamped):
        """MAVROS local_position/pose (ENU)"""
        if self._shutting_down:
            return

        self.current_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        q = msg.pose.orientation
        self.current_euler = self.quaternion_to_euler(q.w, q.x, q.y, q.z)

        if not self.position_received:
            self.position_received = True
            rospy.loginfo('✅ Position data received (MAVROS /mavros/local_position/pose)')

    def velocity_callback(self, msg: TwistStamped):
        """MAVROS local_position/velocity_local (ENU)"""
        if self._shutting_down:
            return

        self.current_vel = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z
        ])

        if not self.velocity_received:
            self.velocity_received = True
            rospy.loginfo('✅ Velocity data received (MAVROS /mavros/local_position/velocity_local)')

    def desired_pose_callback(self, msg: PoseStamped):
        """
        DESIRED PATH - titik proyeksi pada path asli (ENU).
        Ini adalah lintasan referensi yang sebenarnya untuk menghitung trajectory error.
        """
        if self._shutting_down:
            return

        self.target_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        q = msg.pose.orientation
        _, _, yaw = self.quaternion_to_euler(q.w, q.x, q.y, q.z)
        self.target_yaw = yaw

        # untuk flag "fresh point"
        self.last_ref_time = time.monotonic()

        if not self.desired_ref_received:
            self.desired_ref_received = True
            rospy.loginfo('✅ Desired path reference received (/trajectory/desired_pose)')

    def los_ref_pose_callback(self, msg: PoseStamped):
        """
        LOS reference (lookahead point) - setpoint yang dikirim ke PX4.
        Ini adalah posisi yang sebenarnya di-command ke PX4 PID controller.
        """
        if self._shutting_down:
            return

        self.los_ref_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        q = msg.pose.orientation
        _, _, yaw = self.quaternion_to_euler(q.w, q.x, q.y, q.z)
        self.los_ref_yaw = yaw

        if not self.los_ref_received:
            self.los_ref_received = True
            rospy.loginfo('✅ LOS reference pose received (/mavros/setpoint_position/local)')

    def los_info_callback(self, msg: TwistStamped):
        """
        LOS info dari trajectory publisher:
        - linear.x: cross-track error
        - linear.y: along-track distance
        - linear.z: current segment index
        - angular.x: (unused in PX4 version)
        - angular.y: lookahead distance
        - angular.z: LOS yaw (NED)
        """
        if self._shutting_down:
            return

        self.cross_track_error = msg.twist.linear.x
        self.along_track_distance = msg.twist.linear.y
        self.current_segment = int(msg.twist.linear.z)
        self.lookahead_distance = msg.twist.angular.y
        self.los_yaw_ned = msg.twist.angular.z

        if not self.los_info_received:
            self.los_info_received = True
            rospy.loginfo('✅ LOS info received (/trajectory/los_info)')

    def attitude_target_callback(self, msg: AttitudeTarget):
        """Attitude target dari PX4 (via MAVROS)."""
        if self._shutting_down:
            return

        q = msg.orientation
        self.attitude_target_euler = self.quaternion_to_euler(q.w, q.x, q.y, q.z)
        self.attitude_target_thrust = msg.thrust

        if not self.attitude_target_received:
            self.attitude_target_received = True
            rospy.loginfo('✅ Attitude target received (/mavros/setpoint_raw/target_attitude)')

    def trajectory_status_callback(self, msg: String):
        """Callback untuk status trajectory - simpan status dan auto stop saat selesai"""
        # Simpan status saat ini untuk logging
        status_text = msg.data.upper()
        
        # Parse status dari pesan (bisa berupa "WAIT_ALT", "TRACK", "HOLD", "FINISHED", dll)
        if "FINISHED" in status_text:
            self.current_trajectory_status = "FINISHED"
        elif "HOLD" in status_text:
            self.current_trajectory_status = "HOLD"
        elif "TRACK" in status_text:
            self.current_trajectory_status = "TRACK"
        elif "WAIT" in status_text or "ALT" in status_text:
            self.current_trajectory_status = "WAIT_ALT"
        else:
            self.current_trajectory_status = status_text[:20]  # Truncate jika terlalu panjang
        
        if msg.data == "FINISHED" and not self.trajectory_finished:
            self.trajectory_finished = True
            rospy.loginfo("\n" + "="*60)
            rospy.loginfo("🏁 Trajectory FINISHED detected!")
            rospy.loginfo("="*60)
            
            if self.auto_stop_on_finish:
                rospy.loginfo("Auto-stopping data logger in 2 seconds...")
                rospy.sleep(2.0)  # Tunggu 2 detik untuk data terakhir
                rospy.loginfo("Shutting down data logger node.")
                rospy.signal_shutdown("Trajectory finished - auto stop")

    # =======================
    # UTIL
    # =======================
    def quaternion_to_euler(self, w, x, y, z):
        """Konversi quaternion → euler (roll, pitch, yaw) rad."""
        # roll
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = np.sign(sinp) * (np.pi / 2.0)
        else:
            pitch = np.arcsin(sinp)

        # yaw
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    # =======================
    # LOGGING
    # =======================
    def log_data(self, event):
        if self._shutting_down:
            return

        if not self.position_received or not self.velocity_received:
            return

        now = time.monotonic()
        timestamp_sec = now

        if self._t0 is None:
            self._t0 = timestamp_sec

        # flag waypoint_fresh (di sini artinya "trajectory ref fresh")
        waypoint_fresh = 0
        if self.last_ref_time is not None:
            dt_wp = now - self.last_ref_time
            waypoint_fresh = 1 if dt_wp < 0.5 else 0

        # trajectory tracking error (terhadap desired path asli)
        pos_error = self.current_pos - self.target_pos

        yaw_error = self.current_euler[2] - self.target_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

        # error terhadap LOS reference (lookahead point)
        los_error = self.current_pos - self.los_ref_pos

        entry = {
            'time': timestamp_sec,

            'pos_x': self.current_pos[0],
            'pos_y': self.current_pos[1],
            'pos_z': self.current_pos[2],

            'vel_x': self.current_vel[0],
            'vel_y': self.current_vel[1],
            'vel_z': self.current_vel[2],
            'vel_mag': float(np.linalg.norm(self.current_vel)),

            'roll_deg': float(np.degrees(self.current_euler[0])),
            'pitch_deg': float(np.degrees(self.current_euler[1])),
            'yaw_deg': float(np.degrees(self.current_euler[2])),

            # Target = desired path (proyeksi pada path asli)
            'target_pos_x': self.target_pos[0],
            'target_pos_y': self.target_pos[1],
            'target_pos_z': self.target_pos[2],
            'target_yaw_deg': float(np.degrees(self.target_yaw)),

            # LOS reference (lookahead point) - dari setpoint_position/local
            'los_ref_x': self.los_ref_pos[0],
            'los_ref_y': self.los_ref_pos[1],
            'los_ref_z': self.los_ref_pos[2],
            'los_ref_yaw_deg': float(np.degrees(self.los_ref_yaw)),

            # LOS info
            'cross_track_error': self.cross_track_error,
            'along_track_distance': self.along_track_distance,
            'current_segment': self.current_segment,
            'lookahead_distance': self.lookahead_distance,
            'los_yaw_ned_deg': float(np.degrees(self.los_yaw_ned)),

            # Attitude target
            'attitude_target_roll_deg': float(np.degrees(self.attitude_target_euler[0])),
            'attitude_target_pitch_deg': float(np.degrees(self.attitude_target_euler[1])),
            'attitude_target_yaw_deg': float(np.degrees(self.attitude_target_euler[2])),
            'attitude_target_thrust': float(self.attitude_target_thrust),

            # Error terhadap desired path (path asli)
            'error_x': pos_error[0],
            'error_y': pos_error[1],
            'error_z': pos_error[2],
            'error_mag': float(np.linalg.norm(pos_error)),
            'error_yaw_deg': float(np.degrees(yaw_error)),

            # Error terhadap LOS reference (lookahead point)
            'error_los_x': los_error[0],
            'error_los_y': los_error[1],
            'error_los_z': los_error[2],
            'error_los_mag': float(np.linalg.norm(los_error)),

            'waypoint_fresh': int(waypoint_fresh),
            'trajectory_status': self.current_trajectory_status,
        }

        self.data_buffer.append(entry)

        if len(self.data_buffer) >= self._buf_flush_every:
            self.flush_to_csv()

    def auto_flush_csv(self, event):
        if self._shutting_down:
            return
        self.flush_to_csv()

    def flush_to_csv(self):
        """Tulis buffer → CSV (append) dengan urutan kolom fix, time relatif t0."""
        if not self.data_buffer:
            return

        with self._io_lock:
            if not self.data_buffer:
                return

            try:
                df = pd.DataFrame(self.data_buffer, columns=self._columns)

                t0 = self._t0 if self._t0 is not None else df['time'].iloc[0]
                df['time'] = df['time'] - t0

                mode = 'a' if self._csv_header_written else 'w'
                header = not self._csv_header_written

                df.to_csv(
                    self.csv_filename,
                    mode=mode,
                    header=header,
                    index=False,
                    quoting=csv.QUOTE_NONNUMERIC
                )

                self._csv_header_written = True
                self.data_buffer.clear()
                rospy.loginfo('💾 Flushed → %s', Path(self.csv_filename).name)

            except Exception as e:
                rospy.logerr('❌ CSV flush failed: %s', str(e))

    def save_to_excel(self):
        """Bangun Excel dari CSV (dipanggil saat shutdown)."""
        with self._io_lock:
            # tulis sisa buffer dulu
            if self.data_buffer:
                try:
                    df_buf = pd.DataFrame(self.data_buffer, columns=self._columns)
                    t0 = self._t0 if self._t0 is not None else df_buf['time'].iloc[0]
                    df_buf['time'] = df_buf['time'] - t0

                    mode = 'a' if self._csv_header_written else 'w'
                    header = not self._csv_header_written
                    df_buf.to_csv(
                        self.csv_filename,
                        mode=mode,
                        header=header,
                        index=False,
                        quoting=csv.QUOTE_NONNUMERIC
                    )
                    self._csv_header_written = True
                    self.data_buffer.clear()
                except Exception as e:
                    rospy.logerr('❌ Final CSV write failed: %s', str(e))

            if not os.path.exists(self.csv_filename):
                rospy.logwarn('No CSV file to convert!')
                return

            try:
                df = pd.read_csv(self.csv_filename)

                with pd.ExcelWriter(self.excel_filename, engine='openpyxl') as writer:
                    # semua data
                    df.to_excel(writer, sheet_name='Flight Data', index=False)

                    # summary
                    summary = pd.DataFrame({
                        'Metric': [
                            'Controller Type',
                            'Duration (s)',
                            'Samples',
                            'Avg Position Error (m) [from desired path]',
                            'Max Position Error (m) [from desired path]',
                            'RMSE Position (m) [from desired path]',
                            'Avg Position Error (m) [from LOS ref]',
                            'Max Position Error (m) [from LOS ref]',
                            'Avg Cross-Track Error (m)',
                            'Max Cross-Track Error (m)',
                            'Avg Yaw Error (deg)',
                            'Max Yaw Error (deg)',
                            'Avg Speed (m/s)',
                            'Max Speed (m/s)',
                        ],
                        'Value': [
                            self.controller_type,
                            float(df['time'].iloc[-1]) if len(df) else 0.0,
                            int(len(df)),
                            float(df['error_mag'].mean()) if 'error_mag' in df else np.nan,
                            float(df['error_mag'].max())  if 'error_mag' in df else np.nan,
                            float(np.sqrt((df['error_mag']**2).mean())) if 'error_mag' in df else np.nan,
                            float(df['error_los_mag'].mean()) if 'error_los_mag' in df else np.nan,
                            float(df['error_los_mag'].max())  if 'error_los_mag' in df else np.nan,
                            float(df['cross_track_error'].abs().mean()) if 'cross_track_error' in df else np.nan,
                            float(df['cross_track_error'].abs().max())  if 'cross_track_error' in df else np.nan,
                            float(df['error_yaw_deg'].abs().mean()) if 'error_yaw_deg' in df else np.nan,
                            float(df['error_yaw_deg'].abs().max())  if 'error_yaw_deg' in df else np.nan,
                            float(df['vel_mag'].mean()) if 'vel_mag' in df else np.nan,
                            float(df['vel_mag'].max())  if 'vel_mag' in df else np.nan,
                        ]
                    })
                    summary.to_excel(writer, sheet_name='Summary', index=False)

                    def write_if(cols, sheet):
                        cols = [c for c in cols if c in df.columns]
                        if cols:
                            df[cols].to_excel(writer, sheet_name=sheet, index=False)

                    write_if(
                        ['time', 'pos_x', 'pos_y', 'pos_z',
                         'target_pos_x', 'target_pos_y', 'target_pos_z',
                         'los_ref_x', 'los_ref_y', 'los_ref_z',
                         'error_x', 'error_y', 'error_z', 'error_mag',
                         'error_los_x', 'error_los_y', 'error_los_z', 'error_los_mag'],
                        'Position'
                    )

                    write_if(
                        ['time', 'vel_x', 'vel_y', 'vel_z', 'vel_mag'],
                        'Velocity'
                    )

                    write_if(
                        ['time', 'roll_deg', 'pitch_deg', 'yaw_deg',
                         'target_yaw_deg', 'los_ref_yaw_deg',
                         'error_yaw_deg'],
                        'Attitude'
                    )

                    write_if(
                        ['time', 'cross_track_error', 'along_track_distance',
                         'current_segment', 'lookahead_distance', 'los_yaw_ned_deg'],
                        'LOS Info'
                    )

                    write_if(
                        ['time', 'attitude_target_roll_deg', 'attitude_target_pitch_deg',
                         'attitude_target_yaw_deg', 'attitude_target_thrust'],
                        'Attitude Target'
                    )

                rospy.loginfo('✅ Excel built from CSV: %s', self.excel_filename)

            except Exception as e:
                rospy.logerr('❌ Failed to build Excel: %s', str(e))

    # =======================
    # SHUTDOWN
    # =======================
    def on_shutdown(self):
        if self._shutting_down:
            return
        self._shutting_down = True
        rospy.loginfo('🔻 ROS shutdown: saving final CSV/Excel (PX4 LOS guidance logger)...')
        try:
            self.save_to_excel()
        except Exception:
            pass
        rospy.loginfo('Final CSV/Excel saved. Bye!')

    def _atexit_handler(self):
        if self._shutting_down:
            return
        self._shutting_down = True
        try:
            self.save_to_excel()
        except Exception:
            pass


def main():
    rospy.init_node('data_logger_px4_los_guidance', anonymous=False)
    node = DataLoggerPx4LOSGuidance()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo('🔻 KeyboardInterrupt received.')
    # on_shutdown akan otomatis dipanggil oleh ROS


if __name__ == '__main__':
    main()
