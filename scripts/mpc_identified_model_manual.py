#!/usr/bin/env python3
############################################################################
#
#   MPC Trajectory Tracking Controller - IDENTIFIED MODEL (MAVROS, ROS1)
#   VARIAN MANUAL-OFFBOARD:
#
#   - ROS1 + mavros
#   - MPC dengan IDENTIFIED MODEL dari system identification
#   - State: [x, y, z, vx, vy, vz, roll, pitch, yaw, ...]
#   - Control inputs: [thrust, roll_cmd, pitch_cmd, yaw_rate_cmd]
#   - Input referensi dari trajectory publisher:
#       /trajectory/ref_pose (PoseStamped, ENU)
#       /trajectory/ref_vel  (TwistStamped, ENU)
#   - Output: LANGSUNG ke /mavros/setpoint_raw/attitude
#   - TIDAK ADA konversi acceleration → attitude
#
#   PERBEDAAN DENGAN MPC LINEARIZED:
#   - Model: Identified dari data flight test (bukan linearisasi fisika)
#   - Control output: thrust + attitude setpoint (bukan akselerasi)
#   - Lebih akurat untuk karakteristik drone spesifik
#
############################################################################

import rospy
import numpy as np
import quadprog  # pastikan sudah: pip3 install quadprog

from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State, AttitudeTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest


class MPCIdentifiedModelController:
    """
    MPC Controller dengan Identified Model
    State: [x, y, z, vx, vy, vz]
    Control inputs: [thrust, roll_cmd, pitch_cmd]
    Yaw setpoint diambil langsung dari trajectory publisher
    
    TODO: Isi matrices A, B dari hasil system identification!
    """

    def __init__(self, dt=0.1, Np=10, Nc=3):
        self.dt = dt
        self.Np = Np
        self.Nc = Nc
        self.nx = 6  # [x, y, z, vx, vy, vz]
        self.nu = 3  # [thrust, roll_cmd, pitch_cmd]

        # =====================================================
        # TODO: ISI MATRICES INI DARI HASIL SYSTEM IDENTIFICATION!
        # =====================================================
        # Format: x(k+1) = A * x(k) + B * u(k)
        # A: state transition matrix (nx x nx)
        # B: control input matrix (nx x nu)
        
        # PLACEHOLDER - GANTI DENGAN NILAI DARI SYSTEM ID!
        self.A = np.array([
            [ 0.5864, -0.6465, -0.4143,  0.4367,  0.1764, -0.0050],
            [-0.1166,  0.5842, -0.4205,  0.4351,  0.0547, -0.4754],
            [ 0.2085,  0.2632,  1.0974, -0.2010, -0.0732,  0.1140],
            [ 0.0455,  0.2284,  0.2584,  0.7459, -0.0253,  0.2619],
            [ 0.3394,  0.1967, -0.1062,  0.1012,  0.8524, -0.6981],
            [ 0.0006,  0.0007,  0.0005, -0.0004, -0.0002,  0.9901]
        ])

        self.B = np.array([
            [ 0.4493, -0.5665,  0.0252],
            [ 0.9590, -0.2516,  0.0868],
            [ 0.1606,  0.2756, -0.0070],
            [-1.2863,  0.1229, -0.0522],
            [ 2.3782,  0.3360,  0.0965],
            [ 0.0000,  0.0008,  0.0000]
        ])
        
        # Contoh struktur yang mungkin (sesuaikan dengan hasil identification):
        # self.A = np.array([
        #     [a11, a12, ..., a16],
        #     [a21, a22, ..., a26],
        #     ...
        #     [a61, a62, ..., a66]
        # ])
        # 
        # self.B = np.array([
        #     [b11, b12, b13],  # efek thrust, roll, pitch ke dx
        #     [b21, b22, b23],  # efek ke dy
        #     ...
        #     [b61, b62, b63]   # efek ke dvz
        # ])

        self.C = np.eye(self.nx)

        # Cost matrices - TUNING SESUAI KEBUTUHAN
        self.Q = np.diag([
            70.0, 70.0, 150.0,   # posisi x,y,z
            6.0, 6.0, 90.0       # kecepatan vx,vy,vz
        ])

        self.R = np.diag([0.1, 0.5, 0.5])  # penalti [thrust, roll, pitch]
        self.R_delta = np.diag([0.2, 0.8, 0.8])  # penalti perubahan control

        self.u_prev = np.zeros(self.nu)
        
        # Control constraints
        self.thrust_min = 0.0
        self.thrust_max = 1.0
        self.roll_max = np.radians(25.0)   # rad
        self.pitch_max = np.radians(25.0)  # rad

        self._build_prediction_matrices()

    def _build_prediction_matrices(self):
        """Build prediction matrices untuk MPC"""
        # Phi matrix (state prediction)
        self.Phi = np.zeros((self.Np*self.nx, self.nx))
        A_power = np.eye(self.nx)
        for i in range(self.Np):
            A_power = A_power @ self.A
            self.Phi[i*self.nx:(i+1)*self.nx, :] = A_power

        # Gamma matrix (control effect)
        self.Gamma = np.zeros((self.Np*self.nx, self.Nc*self.nu))
        for i in range(self.Np):
            for j in range(min(i+1, self.Nc)):
                A_power = np.eye(self.nx)
                for _ in range(i - j):
                    A_power = A_power @ self.A
                self.Gamma[i*self.nx:(i+1)*self.nx, j*self.nu:(j+1)*self.nu] = A_power @ self.B

        self._build_qp_matrices()

    def _build_qp_matrices(self):
        """Build QP problem matrices"""
        Q_bar = np.kron(np.eye(self.Np), self.Q)
        R_bar = np.kron(np.eye(self.Nc), self.R)
        
        # Delta-u matrices for control rate penalty
        M = np.zeros((self.Nc*self.nu, self.Nc*self.nu))
        for i in range(self.Nc):
            M[i*self.nu:(i+1)*self.nu, i*self.nu:(i+1)*self.nu] = np.eye(self.nu)
            if i > 0:
                M[i*self.nu:(i+1)*self.nu, (i-1)*self.nu:i*self.nu] = -np.eye(self.nu)

        R_delta_bar = M.T @ np.kron(np.eye(self.Nc), self.R_delta) @ M

        self.H = 2.0 * (self.Gamma.T @ Q_bar @ self.Gamma + R_bar + R_delta_bar)
        self.H = 0.5 * (self.H + self.H.T)  # symmetrize

        self.Gamma_T_Q = self.Gamma.T @ Q_bar
        self.M_T_Rd = M.T @ np.kron(np.eye(self.Nc), self.R_delta)

    def compute_control(self, x, x_ref):
        """
        Compute optimal control using MPC
        
        Args:
            x: current state [x,y,z,vx,vy,vz] (NED)
            x_ref: reference state [x,y,z,vx,vy,vz] (NED)
            
        Returns:
            u: control [thrust, roll_cmd, pitch_cmd, yaw_rate_cmd]
        """
        try:
            # Error
            e = x - x_ref
            
            # Reference trajectory (constant reference)
            x_ref_bar = np.tile(x_ref, self.Np)
            
            # Prediction error
            e_pred = self.Phi @ e
            
            # QP objective: f = Gamma^T Q (Phi e + Gamma U - x_ref_bar) + R U + R_delta M U
            f_temp = self.Gamma_T_Q @ (e_pred - x_ref_bar)
            
            # Delta-u term
            u_rep = np.tile(self.u_prev, self.Nc)
            f_delta = self.M_T_Rd @ u_rep
            
            f = f_temp + f_delta
            
            # Constraints: lower_bound <= u <= upper_bound
            # Format: G * U >= h
            n_vars = self.Nc * self.nu
            
            # Build constraint matrices
            # Upper bounds: -u <= -lower
            # Lower bounds: u <= upper
            G_list = []
            h_list = []
            
            for i in range(self.Nc):
                # Thrust constraints
                G_thrust = np.zeros((2, n_vars))
                G_thrust[0, i*self.nu] = -1.0  # -thrust >= -thrust_max
                G_thrust[1, i*self.nu] = 1.0   # thrust >= thrust_min
                G_list.append(G_thrust)
                h_list.extend([-self.thrust_max, self.thrust_min])
                
                # Roll constraints
                G_roll = np.zeros((2, n_vars))
                G_roll[0, i*self.nu+1] = -1.0
                G_roll[1, i*self.nu+1] = 1.0
                G_list.append(G_roll)
                h_list.extend([-self.roll_max, -self.roll_max])
                
                # Pitch constraints
                G_pitch = np.zeros((2, n_vars))
                G_pitch[0, i*self.nu+2] = -1.0
                G_pitch[1, i*self.nu+2] = 1.0
                G_list.append(G_pitch)
                h_list.extend([-self.pitch_max, -self.pitch_max])
            
            G = np.vstack(G_list)
            h = np.array(h_list)
            
            # Solve QP
            sol = quadprog.solve_qp(self.H, -f, -G.T, -h, meq=0)
            U_opt = sol[0]
            
            u = U_opt[:self.nu]
            
            # Clip for safety
            u[0] = np.clip(u[0], self.thrust_min, self.thrust_max)
            u[1] = np.clip(u[1], -self.roll_max, self.roll_max)
            u[2] = np.clip(u[2], -self.pitch_max, self.pitch_max)
            
            self.u_prev = u.copy()
            return u

        except Exception as e:
            rospy.logwarn(f"MPC QP failed: {e}")
            return np.zeros(self.nu)


def quaternion_to_euler(q):
    """Convert quaternion [w,x,y,z] to euler angles [roll, pitch, yaw]"""
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


def euler_to_quaternion(roll, pitch, yaw):
    """Convert euler angles to quaternion [x,y,z,w] (ENU convention)"""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return np.array([qx, qy, qz, qw], dtype=float)


class MPCIdentifiedModelFollowerROS1:
    """
    ROS1 Node untuk MPC dengan Identified Model
    - Terima trajectory reference dari trajectory publisher
    - Jalankan MPC dengan identified model
    - Kirim thrust + attitude setpoint LANGSUNG ke MAVROS
    """

    def __init__(self):
        rospy.init_node("mpc_identified_model_follower_auto", anonymous=False)
        
        rospy.loginfo("="*60)
        rospy.loginfo("MPC Identified Model Follower (ROS1 + MAVROS) - AUTO OFFBOARD/ARM")
        rospy.loginfo("Input ref: /trajectory/ref_pose & /trajectory/ref_vel")
        rospy.loginfo("Output   : /mavros/setpoint_raw/attitude (DIRECT)")
        rospy.loginfo("Mode & ARM: AUTO via MAVROS services")
        rospy.loginfo("MPC Optimization: 10 Hz | Control Output: 50 Hz")
        rospy.loginfo("="*60)

        # ===================== PARAMETER MPC =====================
        mpc_dt = rospy.get_param("~mpc_dt", 0.1)
        mpc_Np = int(rospy.get_param("~mpc_Np", 10))
        mpc_Nc = int(rospy.get_param("~mpc_Nc", 3))

        self.mpc = MPCIdentifiedModelController(dt=mpc_dt, Np=mpc_Np, Nc=mpc_Nc)
        rospy.loginfo("MPC Controller initialized: dt=%.2fs, Np=%d, Nc=%d", mpc_dt, mpc_Np, mpc_Nc)

        # ===================== AUTO MODE PARAMETERS =====================
        self.auto_arm_offboard = rospy.get_param("~auto_arm_offboard", True)
        self.setpoint_stream_duration = rospy.get_param("~setpoint_stream_duration", 2.0)  # seconds

        # ===================== STATE MAVROS =====================
        self.current_state = State()
        self.offboard_mode = False
        self.armed = False
        
        # State machine for auto arm/offboard
        self.auto_mode_enabled = False
        self.setpoint_stream_start = None
        self.offboard_requested = False
        self.arm_requested = False

        # State UAV (NED)
        self.current_position = np.zeros(3)      # [N,E,D]
        self.current_velocity = np.zeros(3)      # [vN,vE,vD]
        self.current_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # w,x,y,z
        self.current_yaw = 0.0
        self.yaw_initialized = False

        # Trajectory reference (NED)
        self.ref_position = np.array([0.0, 0.0, -2.5])
        self.ref_velocity = np.zeros(3)
        self.ref_yaw = 0.0

        self.ref_pose_received = False
        self.ref_vel_received = False

        # Output MPC (DIRECT attitude + thrust)
        self.control_thrust = 0.35
        self.control_roll = 0.0
        self.control_pitch = 0.0
        # Yaw diambil dari trajectory publisher
        self.attitude_yaw = 0.0

        self.setpoint_counter = 0

        # ===================== PUBLISHERS =====================
        self.attitude_pub = rospy.Publisher(
            "/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10
        )

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

        # ===================== SERVICE CLIENTS =====================
        rospy.loginfo("Waiting for MAVROS services...")
        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/set_mode")
        
        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        rospy.loginfo("✓ MAVROS services connected")

        # ===================== TIMERS =====================
        self.mpc_timer = rospy.Timer(rospy.Duration(0.1), self.mpc_callback)    # 10 Hz
        self.ctrl_timer = rospy.Timer(rospy.Duration(0.02), self.control_loop)  # 50 Hz
        self.sm_timer = rospy.Timer(rospy.Duration(0.5), self.state_monitor_callback)  # 2 Hz

        # Start setpoint streaming for auto mode
        if self.auto_arm_offboard:
            self.setpoint_stream_start = rospy.Time.now()
            self.auto_mode_enabled = True
            rospy.loginfo("🚀 Auto ARM/OFFBOARD enabled - streaming setpoints...")

        rospy.loginfo("MPC Identified Model Follower ROS1 (auto OFFBOARD) started.")

    # ===================== State & sensor callbacks =====================
    def state_callback(self, msg: State):
        self.current_state = msg
        self.offboard_mode = (msg.mode == "OFFBOARD")
        self.armed = msg.armed

        if self.offboard_mode and not hasattr(self, '_offboard_logged'):
            self._offboard_logged = True
            rospy.loginfo("✓ OFFBOARD MODE ACTIVE (set manual)")

    def local_pose_callback(self, msg: PoseStamped):
        """ENU → NED"""
        self.current_position[0] = msg.pose.position.y      # N
        self.current_position[1] = msg.pose.position.x      # E
        self.current_position[2] = -msg.pose.position.z     # D

        self.current_orientation = np.array([
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        ])
        
        # Update current yaw from quaternion
        _, _, yaw_rad = quaternion_to_euler(self.current_orientation)
        self.current_yaw = yaw_rad
        
        # Mark yaw as initialized after first pose update
        if not self.yaw_initialized:
            self.yaw_initialized = True
            rospy.loginfo("✅ Current yaw initialized: %.1f°", np.degrees(self.current_yaw))

    def local_vel_callback(self, msg: TwistStamped):
        """ENU → NED"""
        self.current_velocity[0] = msg.twist.linear.y
        self.current_velocity[1] = msg.twist.linear.x
        self.current_velocity[2] = -msg.twist.linear.z

    def traj_pose_callback(self, msg: PoseStamped):
        """Pose referensi dari trajectory publisher (ENU) → simpan di NED"""
        n = msg.pose.position.y
        e = msg.pose.position.x
        d = -msg.pose.position.z
        self.ref_position = np.array([n, e, d])

        # Extract yaw from quaternion
        qw = msg.pose.orientation.w
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        _, _, yaw_rad = quaternion_to_euler([qw, qx, qy, qz])
        self.ref_yaw = yaw_rad

        if not self.ref_pose_received:
            self.ref_pose_received = True
            rospy.loginfo("✅ Trajectory pose reference received.")

    def traj_vel_callback(self, msg: TwistStamped):
        """Velocity referensi dari trajectory publisher (ENU) → simpan di NED"""
        vn = msg.twist.linear.y
        ve = msg.twist.linear.x
        vd = -msg.twist.linear.z
        self.ref_velocity = np.array([vn, ve, vd])

        if not self.ref_vel_received:
            self.ref_vel_received = True
            rospy.loginfo("✅ Trajectory velocity reference received.")

    # ===================== MPC timer =====================
    def mpc_callback(self, event):
        """Run MPC optimization at 10 Hz"""
        # MPC hanya aktif jika mode OFFBOARD & armed
        if not (self.offboard_mode and self.armed):
            return
        
        # Wait until yaw is initialized before sending control commands
        if not self.yaw_initialized:
            rospy.logwarn_throttle(2.0, "⏳ Waiting for yaw initialization...")
            return

        # Jika belum ada trajectory ref, hover di posisi sekarang
        if not self.ref_pose_received:
            self.ref_position[0] = self.current_position[0]
            self.ref_position[1] = self.current_position[1]
            self.ref_position[2] = -2.5
            self.ref_velocity[:] = 0.0
        else:
            if not self.ref_vel_received:
                self.ref_velocity[:] = 0.0

        # State & reference vector (NED)
        x = np.concatenate([self.current_position, self.current_velocity])
        x_ref = np.concatenate([self.ref_position, self.ref_velocity])

        # Run MPC
        u = self.mpc.compute_control(x, x_ref)
        
        # Extract control outputs: [thrust, roll, pitch]
        self.control_thrust = float(u[0])
        self.control_roll = float(u[1])
        self.control_pitch = float(u[2])
        
        # Yaw diambil dari trajectory reference (bukan dari MPC)
        self.attitude_yaw = self.ref_yaw

        pos_err = np.linalg.norm(self.ref_position - self.current_position)
        vel_err = np.linalg.norm(self.ref_velocity - self.current_velocity)

        rospy.loginfo_throttle(
            1.0,
            f"🎯 MPC IDENTIFIED MODEL:\n"
            f"   Pos: [{self.current_position[0]:.1f}, {self.current_position[1]:.1f}, {self.current_position[2]:.1f}] → "
            f"[{self.ref_position[0]:.1f}, {self.ref_position[1]:.1f}, {self.ref_position[2]:.1f}] | Err: {pos_err:.2f}m\n"
            f"   Vel: [{self.current_velocity[0]:.2f}, {self.current_velocity[1]:.2f}, {self.current_velocity[2]:.2f}] → "
            f"[{self.ref_velocity[0]:.2f}, {self.ref_velocity[1]:.2f}, {self.ref_velocity[2]:.2f}] m/s | Err: {vel_err:.2f}\n"
            f"   Control: Thrust={self.control_thrust:.3f}, Roll={np.degrees(self.control_roll):.1f}°, "
            f"Pitch={np.degrees(self.control_pitch):.1f}°, Yaw={np.degrees(self.attitude_yaw):.1f}° (from traj)"
        )

    # ===================== Control loop (50Hz) =====================
    def control_loop(self, event):
        """Publish attitude setpoint at 50 Hz"""
        att = AttitudeTarget()
        att.header.stamp = rospy.Time.now()
        att.header.frame_id = "map"

        # Type mask: use attitude + thrust, ignore rates
        att.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE |
            AttitudeTarget.IGNORE_PITCH_RATE |
            AttitudeTarget.IGNORE_YAW_RATE
        )

        # Convert roll, pitch, yaw (from trajectory) to quaternion (ENU)
        # Roll/pitch from MPC, yaw from trajectory publisher
        q_xyzw = euler_to_quaternion(
            self.control_roll,
            self.control_pitch,
            self.attitude_yaw  # Yaw dari trajectory publisher
        )
        att.orientation.x = float(q_xyzw[0])
        att.orientation.y = float(q_xyzw[1])
        att.orientation.z = float(q_xyzw[2])
        att.orientation.w = float(q_xyzw[3])

        att.thrust = float(np.clip(self.control_thrust, 0.0, 1.0))
        
        # Body rates ignored (type_mask)
        att.body_rate.x = 0.0
        att.body_rate.y = 0.0
        att.body_rate.z = 0.0

        self.attitude_pub.publish(att)
        self.setpoint_counter += 1

        if self.setpoint_counter % 50 == 0:
            rospy.loginfo(
                f"📤 AttitudeTarget: thrust={att.thrust:.3f}, "
                f"roll={np.degrees(self.control_roll):.1f}°, "
                f"pitch={np.degrees(self.control_pitch):.1f}°, "
                f"yaw={np.degrees(self.attitude_yaw):.1f}° (from traj)"
            )

    # ===================== State monitor =====================
    def state_monitor_callback(self, event):
        """
        Monitor dan AUTO request OFFBOARD + ARM jika enabled
        """
        # Auto mode state machine
        if self.auto_mode_enabled and self.auto_arm_offboard:
            # Step 1: Stream setpoints untuk 2 detik sebelum request OFFBOARD
            if self.setpoint_stream_start is not None:
                elapsed = (rospy.Time.now() - self.setpoint_stream_start).to_sec()
                if elapsed < self.setpoint_stream_duration:
                    rospy.loginfo_throttle(1.0, f"⏳ Streaming setpoints... {elapsed:.1f}s / {self.setpoint_stream_duration:.1f}s")
                    return
                else:
                    self.setpoint_stream_start = None  # Stop logging
            
            # Step 2: Request OFFBOARD mode
            if not self.offboard_mode and not self.offboard_requested:
                try:
                    req = SetModeRequest()
                    req.custom_mode = "OFFBOARD"
                    resp = self.set_mode_client(req)
                    if resp.mode_sent:
                        rospy.loginfo("✓ OFFBOARD mode requested")
                        self.offboard_requested = True
                    else:
                        rospy.logwarn("❌ Failed to request OFFBOARD mode")
                except rospy.ServiceException as e:
                    rospy.logerr(f"Service call failed: {e}")
            
            # Step 3: ARM drone setelah OFFBOARD aktif
            if self.offboard_mode and not self.armed and not self.arm_requested:
                rospy.sleep(0.5)  # Wait a bit after OFFBOARD
                try:
                    req = CommandBoolRequest()
                    req.value = True
                    resp = self.arming_client(req)
                    if resp.success:
                        rospy.loginfo("✓ ARM requested")
                        self.arm_requested = True
                    else:
                        rospy.logwarn("❌ Failed to ARM")
                except rospy.ServiceException as e:
                    rospy.logerr(f"ARM service call failed: {e}")
            
            # Step 4: Confirm armed
            if self.armed and self.arm_requested:
                rospy.loginfo("🚁 DRONE ARMED - MPC control active!")
                self.auto_mode_enabled = False  # Stop auto mode logic

        # Regular status monitoring
        if self.offboard_mode and self.armed:
            err = np.linalg.norm(self.ref_position - self.current_position)
            vel_norm = np.linalg.norm(self.current_velocity)

            if self.ref_pose_received:
                rospy.loginfo(
                    f"[MPC ACTIVE] Pos: [{self.current_position[0]:.1f}, {self.current_position[1]:.1f}, {self.current_position[2]:.1f}] | "
                    f"Ref: [{self.ref_position[0]:.1f}, {self.ref_position[1]:.1f}, {self.ref_position[2]:.1f}] | "
                    f"Err: {err:.2f}m | Vel: {vel_norm:.2f} m/s"
                )
            else:
                rospy.loginfo(
                    f"[MPC ACTIVE] Hovering near current position, waiting for trajectory reference."
                )
        else:
            rospy.loginfo_throttle(
                2.0,
                f"[MPC STANDBY] mode={self.current_state.mode}, armed={self.armed}, "
                f"ref_pose={'OK' if self.ref_pose_received else 'NO'}, "
                f"ref_vel={'OK' if self.ref_vel_received else 'NO'}"
            )


if __name__ == '__main__':
    try:
        controller = MPCIdentifiedModelFollowerROS1()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
