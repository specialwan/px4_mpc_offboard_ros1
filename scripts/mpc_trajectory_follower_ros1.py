#!/usr/bin/env python3
############################################################################
#
#   MPC Trajectory Tracking Controller (MAVROS Version, ROS1)
#
#   - ROS1 + mavros
#   - MPC 6-state: [x, y, z, vx, vy, vz]
#   - Input referensi dari trajectory publisher:
#       /trajectory/ref_pose (PoseStamped, ENU)
#       /trajectory/ref_vel  (TwistStamped, ENU)
#   - Output: acceleration [ax, ay, az] (NED)
#   - Dikonversi ke Attitude + Thrust → /mavros/setpoint_raw/attitude
#
############################################################################

import rospy
import numpy as np
import quadprog  # pastikan sudah: pip3 install quadprog

from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State, AttitudeTarget
from mavros_msgs.srv import CommandBool, SetMode


class MPCPositionController:
    """
    MPC Controller untuk posisi
    6-state model: [x, y, z, vx, vy, vz]
    3 control inputs: [ax, ay, az] (perintah akselerasi)
    """

    def __init__(self, dt=0.1, Np=6, Nc=3):
        self.dt = dt
        self.Np = Np
        self.Nc = Nc
        self.nx = 6  # [x, y, z, vx, vy, vz]
        self.nu = 3  # [ax, ay, az]

        # System matrices (integrator model)
        self.A = np.array([
            [1, 0, 0, dt, 0,  0],   # x
            [0, 1, 0, 0,  dt, 0],   # y
            [0, 0, 1, 0,  0,  dt],  # z
            [0, 0, 0, 1,  0,  0],   # vx
            [0, 0, 0, 0,  1,  0],   # vy
            [0, 0, 0, 0,  0,  1],   # vz
        ])

        self.B = np.array([
            [0.5*dt**2, 0,         0        ],  # x
            [0,         0.5*dt**2, 0        ],  # y
            [0,         0,         0.5*dt**2],  # z
            [dt,        0,         0        ],  # vx
            [0,         dt,        0        ],  # vy
            [0,         0,         dt       ],  # vz
        ])

        self.C = np.eye(self.nx)

        # Cost matrices - SAFE MODE REAL FLIGHT
        self.Q = np.diag([
            10.0, 10.0, 180.0,   # posisi
            12.0, 12.0, 90.0     # kecepatan
        ])

        self.R = np.diag([0.025, 0.025, 0.025])     # penalti besarnya u
        self.R_delta = np.diag([0.15, 0.15, 0.15])  # penalti perubahan u

        self.u_prev = np.zeros(self.nu)
        self.a_max = 10.0  # m/s^2

        self._build_prediction_matrices()

    def _build_prediction_matrices(self):
        # Phi
        self.Phi = np.zeros((self.Np*self.nx, self.nx))
        A_power = np.eye(self.nx)
        for i in range(self.Np):
            A_power = A_power @ self.A
            self.Phi[i*self.nx:(i+1)*self.nx, :] = A_power

        # Gamma
        self.Gamma = np.zeros((self.Np*self.nx, self.Nc*self.nu))
        for i in range(self.Np):
            for j in range(min(i+1, self.Nc)):
                A_power = np.eye(self.nx)
                for _ in range(i - j):
                    A_power = A_power @ self.A
                self.Gamma[i*self.nx:(i+1)*self.nx, j*self.nu:(j+1)*self.nu] = A_power @ self.B

        self._build_qp_matrices()

    def _build_qp_matrices(self):
        Q_bar = np.kron(np.eye(self.Np), self.Q)
        R_bar = np.kron(np.eye(self.Nc), self.R)

        R_delta_bar = np.zeros((self.Nc*self.nu, self.Nc*self.nu))
        for i in range(self.Nc):
            R_delta_bar[i*self.nu:(i+1)*self.nu, i*self.nu:(i+1)*self.nu] = self.R_delta
            if i > 0:
                R_delta_bar[i*self.nu:(i+1)*self.nu, (i-1)*self.nu:i*self.nu] = -self.R_delta
                R_delta_bar[(i-1)*self.nu:i*self.nu, i*self.nu:(i+1)*self.nu] = -self.R_delta

        H = self.Gamma.T @ Q_bar @ self.Gamma + R_bar + R_delta_bar
        self.H = (H + H.T) / 2.0  # symmetrize

        self.Q_bar = Q_bar
        self.R_delta_bar = R_delta_bar

    def compute_control(self, x, x_ref):
        # x, x_ref: shape (6,)
        r = np.tile(x_ref, self.Np)
        err_pred = self.Phi @ x - r

        u_prev_ext = np.tile(self.u_prev, self.Nc)

        f_tracking = self.Gamma.T @ self.Q_bar @ err_pred
        f_rate = -self.R_delta_bar @ u_prev_ext
        f = f_tracking + f_rate

        C = np.vstack([
            np.eye(self.Nc*self.nu),
            -np.eye(self.Nc*self.nu)
        ])
        b = np.hstack([
            np.full(self.Nc*self.nu, -self.a_max),
            np.full(self.Nc*self.nu, -self.a_max)
        ])

        try:
            u_opt = quadprog.solve_qp(self.H, -f, C.T, b, meq=0)[0]
            u = u_opt[:self.nu]

            # pengaman lateral saat error z besar
            z_err = abs(x[2] - x_ref[2])
            if z_err > 0.3:
                lateral_reduction = np.clip(1.0 - (z_err - 0.3)*2.0, 0.3, 1.0)
                u[0] *= lateral_reduction
                u[1] *= lateral_reduction

            self.u_prev = u.copy()
            u = np.clip(u, -self.a_max, self.a_max)
            return u

        except Exception as e:
            rospy.logwarn(f"MPC QP failed: {e}")
            return np.zeros(self.nu)


def acceleration_to_attitude_thrust_px4(accel_ned, yaw_desired, hover_thrust=0.4, gravity=9.81):
    ax, ay, az = accel_ned

    ax = np.clip(ax, -3.0, 3.0)
    ay = np.clip(ay, -3.0, 3.0)
    az = np.clip(az, -5.0, 5.0)

    specific_force = np.array([ax, ay, az - gravity])

    thrust_mag = np.linalg.norm(specific_force)
    if thrust_mag < gravity:
        thrust_mag = gravity

    body_z = -specific_force / np.linalg.norm(specific_force)

    thrust_norm = (thrust_mag / gravity) * hover_thrust
    thrust_norm = np.clip(thrust_norm, hover_thrust*0.5, 1.0)

    if np.linalg.norm(body_z) < 1e-8:
        body_z = np.array([0.0, 0.0, 1.0])
    body_z = body_z / np.linalg.norm(body_z)

    y_C = np.array([-np.sin(yaw_desired), np.cos(yaw_desired), 0.0])
    body_x = np.cross(y_C, body_z)

    if body_z[2] < 0.0:
        body_x = -body_x

    if abs(body_z[2]) < 1e-6:
        body_x = np.array([0.0, 0.0, 1.0])

    body_x = body_x / np.linalg.norm(body_x)
    body_y = np.cross(body_z, body_x)

    R = np.column_stack([body_x, body_y, body_z])

    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])

    max_tilt = np.radians(30.0)
    roll = np.clip(roll, -max_tilt, max_tilt)
    pitch = np.clip(pitch, -max_tilt, max_tilt)

    return roll, pitch, yaw, thrust_norm, R


def quaternion_to_euler(q):
    w, x, y, z = q

    sinr_cosp = 2.0*(w*x + y*z)
    cosr_cosp = 1.0 - 2.0*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0*(w*y - z*x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi/2, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0*(w*z + x*y)
    cosy_cosp = 1.0 - 2.0*(y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class MPCTrajectoryFollowerROS1:
    """
    Versi TRAJECTORY TRACKING:
    - Referensi posisi + yaw dari /trajectory/ref_pose (ENU)
    - Referensi kecepatan dari /trajectory/ref_vel (ENU)
    - MPC tracking state x = [pos_ned, vel_ned] → x_ref dari traj
    """

    def __init__(self):
        self.node_name = "mpc_trajectory_follower"
        rospy.loginfo("="*60)
        rospy.loginfo("MPC Trajectory Tracking (ROS1 + MAVROS)")
        rospy.loginfo("Input ref: /trajectory/ref_pose & /trajectory/ref_vel")
        rospy.loginfo("Output   : /mavros/setpoint_raw/attitude")
        rospy.loginfo("MPC Optimization: 10 Hz | Control Output: 50 Hz")
        rospy.loginfo("="*60)

        # MPC core
        self.mpc = MPCPositionController(dt=0.1, Np=6, Nc=3)
        rospy.loginfo("MPC Controller initialized: dt=0.1s, Np=6, Nc=3")

        # State
        self.current_state = State()
        self.armed = False
        self.offboard_mode = False

        self.current_position = np.zeros(3)      # NED
        self.current_velocity = np.zeros(3)      # NED
        self.current_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # w,x,y,z

        # Trajectory reference (NED)
        self.ref_position = np.array([0.0, 0.0, -2.5])
        self.ref_velocity = np.zeros(3)
        self.ref_yaw = -45.0

        self.ref_pose_received = False
        self.ref_vel_received = False

        self.mpc_acceleration = np.zeros(3)

        self.attitude_roll = 0.0
        self.attitude_pitch = 0.0
        self.attitude_yaw = 0.0
        self.attitude_thrust = 0.35
        self.attitude_R_matrix = np.eye(3)

        self.setpoint_counter = 0

        # ===================== SUBSCRIBERS =====================
        self.state_sub = rospy.Subscriber(
            "/mavros/state", State, self.state_callback, queue_size=10
        )
        self.local_pose_sub = rospy.Subscriber(
            "/mavros/local_position/pose", PoseStamped, self.local_pose_callback, queue_size=10
        )
        self.local_vel_sub = rospy.Subscriber(
            "/mavros/local_position/velocity_local", TwistStamped, self.local_vel_callback, queue_size=10
        )

        # Trajectory references (ENU)
        self.traj_pose_sub = rospy.Subscriber(
            "/trajectory/ref_pose", PoseStamped, self.traj_pose_callback, queue_size=10
        )
        self.traj_vel_sub = rospy.Subscriber(
            "/trajectory/ref_vel", TwistStamped, self.traj_vel_callback, queue_size=10
        )

        # ===================== PUBLISHERS =====================
        self.attitude_pub = rospy.Publisher(
            "/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=20
        )

        # ===================== SERVICES =====================
        rospy.loginfo("Waiting for mavros services...")
        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")

        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        rospy.loginfo("MAVROS services ready.")

        # ===================== TIMERS =====================
        self.mpc_timer = rospy.Timer(rospy.Duration(0.1), self.mpc_callback)    # 10 Hz
        self.ctrl_timer = rospy.Timer(rospy.Duration(0.02), self.control_loop)  # 50 Hz
        self.sm_timer = rospy.Timer(rospy.Duration(0.5), self.state_machine_callback)  # 2 Hz

    # ===================== Callbacks =====================

    def state_callback(self, msg: State):
        prev_armed = self.armed
        prev_offboard = self.offboard_mode

        self.current_state = msg
        self.armed = msg.armed
        self.offboard_mode = (msg.mode == "OFFBOARD")

        if prev_armed != self.armed:
            if self.armed:
                rospy.loginfo("✓ ARMED")
            else:
                rospy.logwarn("✗ DISARMED")

        if prev_offboard != self.offboard_mode:
            if self.offboard_mode:
                rospy.loginfo("✓ OFFBOARD MODE ACTIVE")
            else:
                rospy.logwarn("✗ OFFBOARD MODE INACTIVE")

    def local_pose_callback(self, msg: PoseStamped):
        # ENU → NED
        self.current_position[0] = msg.pose.position.y      # N
        self.current_position[1] = msg.pose.position.x      # E
        self.current_position[2] = -msg.pose.position.z     # D

        self.current_orientation = np.array([
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        ])

    def local_vel_callback(self, msg: TwistStamped):
        # ENU → NED
        self.current_velocity[0] = msg.twist.linear.y
        self.current_velocity[1] = msg.twist.linear.x
        self.current_velocity[2] = -msg.twist.linear.z

    def traj_pose_callback(self, msg: PoseStamped):
        """
        Pose referensi dari trajectory publisher (ENU) → simpan di NED
        """
        # Posisi ENU → NED
        n = msg.pose.position.y
        e = msg.pose.position.x
        d = -msg.pose.position.z
        self.ref_position = np.array([n, e, d])

        # Yaw dari quaternion (langsung dari orientasi ENU, konsisten dengan cara lama)
        q = [
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        ]
        _, _, yaw = quaternion_to_euler(q)
        self.ref_yaw = yaw

        if not self.ref_pose_received:
            self.ref_pose_received = True
            rospy.loginfo("✅ Trajectory pose reference received.")

    def traj_vel_callback(self, msg: TwistStamped):
        """
        Velocity referensi dari trajectory publisher (ENU) → simpan di NED
        """
        # ENU vel: [vx, vy, vz] → NED: [vn, ve, vd]
        vn = msg.twist.linear.y
        ve = msg.twist.linear.x
        vd = -msg.twist.linear.z
        self.ref_velocity = np.array([vn, ve, vd])

        if not self.ref_vel_received:
            self.ref_vel_received = True
            rospy.loginfo("✅ Trajectory velocity reference received.")

    # ===================== MPC timer =====================

    def mpc_callback(self, event):
        if not self.offboard_mode:
            return

        # Jika belum ada trajectory ref, tahan di posisi sekarang (hover)
        if not self.ref_pose_received:
            self.ref_position[0] = self.current_position[0]
            self.ref_position[1] = self.current_position[1]
            self.ref_position[2] = -2.5
            self.ref_velocity[:] = 0.0
        else:
            # kalau pose sudah ada tapi vel belum pernah diterima, anggap vel=0
            if not self.ref_vel_received:
                self.ref_velocity[:] = 0.0

        # State & reference vector (NED)
        x = np.concatenate([self.current_position, self.current_velocity])
        x_ref = np.concatenate([self.ref_position, self.ref_velocity])

        acc = self.mpc.compute_control(x, x_ref)
        self.mpc_acceleration = acc

        roll, pitch, yaw, thrust, R = acceleration_to_attitude_thrust_px4(
            acc, self.ref_yaw, hover_thrust=0.35, gravity=9.81
        )

        self.attitude_roll = roll
        self.attitude_pitch = pitch
        self.attitude_yaw = yaw
        self.attitude_thrust = thrust
        self.attitude_R_matrix = R

        pos_err = np.linalg.norm(self.ref_position - self.current_position)
        vel_err = np.linalg.norm(self.ref_velocity - self.current_velocity)

        rospy.loginfo_throttle(
            1.0,
            f"🎯 MPC TRAJ STATE:\n"
            f"   Pos: [{self.current_position[0]:.1f}, {self.current_position[1]:.1f}, {self.current_position[2]:.1f}] → "
            f"[{self.ref_position[0]:.1f}, {self.ref_position[1]:.1f}, {self.ref_position[2]:.1f}] | Err: {pos_err:.2f}m\n"
            f"   Vel: [{self.current_velocity[0]:.2f}, {self.current_velocity[1]:.2f}, {self.current_velocity[2]:.2f}] → "
            f"[{self.ref_velocity[0]:.2f}, {self.ref_velocity[1]:.2f}, {self.ref_velocity[2]:.2f}] m/s | Err: {vel_err:.2f}\n"
            f"   Acc: [{acc[0]:.2f}, {acc[1]:.2f}, {acc[2]:.2f}] m/s² | Thrust: {thrust:.3f}"
        )

    # ===================== Attitude setpoint =====================

    def rotmat_nedfrd_to_quat_enuflu(self, R_ned_from_body_frd: np.ndarray):
        T_ENU_NED = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0,-1]
        ], dtype=float)

        T_FRD_FLU = np.diag([1, -1, -1]).astype(float)

        R_enu_from_body_flu = T_ENU_NED @ R_ned_from_body_frd @ T_FRD_FLU
        wxyz = self.rotation_matrix_to_quaternion(R_enu_from_body_flu)
        return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=float)

    def control_loop(self, event):
        att = AttitudeTarget()
        att.header.stamp = rospy.Time.now()
        att.header.frame_id = "map"

        att.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE |
            AttitudeTarget.IGNORE_PITCH_RATE |
            AttitudeTarget.IGNORE_YAW_RATE
        )

        q_xyzw = self.rotmat_nedfrd_to_quat_enuflu(self.attitude_R_matrix)
        att.orientation.x = float(q_xyzw[0])
        att.orientation.y = float(q_xyzw[1])
        att.orientation.z = float(q_xyzw[2])
        att.orientation.w = float(q_xyzw[3])

        att.thrust = float(np.clip(self.attitude_thrust, 0.0, 1.0))
        att.body_rate.x = 0.0
        att.body_rate.y = 0.0
        att.body_rate.z = 0.0

        self.attitude_pub.publish(att)
        self.setpoint_counter += 1

        if self.setpoint_counter % 50 == 0:
            rospy.loginfo(
                f"📤 AttitudeTarget ENU: thrust={att.thrust:.3f} "
                f"q=[{att.orientation.x:.3f},{att.orientation.y:.3f},"
                f"{att.orientation.z:.3f},{att.orientation.w:.3f}]"
            )

    # ===================== State machine =====================

    def state_machine_callback(self, event):
        # kirim beberapa setpoint dulu → lalu OFFBOARD + ARM
        if self.current_state.mode != "OFFBOARD" and self.setpoint_counter > 50:
            self.set_offboard_mode()

        if (not self.current_state.armed) and self.current_state.mode == "OFFBOARD":
            self.arm()

        if self.current_state.mode == "OFFBOARD" and self.current_state.armed:
            err = np.linalg.norm(self.current_position - self.ref_position)
            vel_norm = np.linalg.norm(self.current_velocity)
            acc_norm = np.linalg.norm(self.mpc_acceleration)

            if self.ref_pose_received:
                rospy.loginfo(
                    f"TRK Pos: [{self.current_position[0]:.1f}, {self.current_position[1]:.1f}, {self.current_position[2]:.1f}] | "
                    f"Ref: [{self.ref_position[0]:.1f}, {self.ref_position[1]:.1f}, {self.ref_position[2]:.1f}] | "
                    f"Err: {err:.2f}m | Vel: {vel_norm:.2f} m/s | Acc: {acc_norm:.2f} m/s²"
                )
            else:
                rospy.loginfo(
                    f"Hovering at: [{self.current_position[0]:.1f}, {self.current_position[1]:.1f}, {self.current_position[2]:.1f}] "
                    f"| Waiting for trajectory reference..."
                )

    # ===================== Helpers =====================

    def rotation_matrix_to_quaternion(self, R):
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0.0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z])

    def arm(self):
        try:
            res = self.arming_client(True)
            if res.success:
                rospy.loginfo(">>> ARM command sent via MAVROS <<<")
            else:
                rospy.logerr("Failed to arm via MAVROS")
        except rospy.ServiceException as e:
            rospy.logerr(f"Arming service call failed: {e}")

    def set_offboard_mode(self):
        try:
            res = self.set_mode_client(custom_mode="OFFBOARD")
            if res.mode_sent:
                rospy.loginfo(">>> OFFBOARD mode command sent via MAVROS <<<")
            else:
                rospy.logerr("Failed to set OFFBOARD mode via MAVROS")
        except rospy.ServiceException as e:
            rospy.logerr(f"SetMode service call failed: {e}")


def main():
    rospy.init_node("mpc_trajectory_follower", anonymous=False)
    node = MPCTrajectoryFollowerROS1()
    rospy.loginfo("MPC Trajectory Follower ROS1 started.")
    rospy.spin()


if __name__ == "__main__":
    main()
