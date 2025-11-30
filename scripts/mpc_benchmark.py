#!/usr/bin/env python3
"""
MPC Benchmark (tanpa ROS/MAVROS)

- Struktur MPC sama dengan MPCPositionController yang kamu pakai di ROS.
- Model 6 state : [x, y, z, vx, vy, vz]
- Input        : [ax, ay, az]
- Simulasi closed loop sederhana mengikuti lintasan lingkaran di ketinggian tetap.
- Fokus utama: mengukur waktu komputasi solve QP MPC per step.

Jalankan di mini PC / Raspberry Pi:
    python3 mpc_benchmark.py
"""

import time
import numpy as np
import quadprog


# ============================================================================
# MPC Position Controller (copy dari versi ROS, tapi tanpa rospy)
# ============================================================================
class MPCPositionController:
    """
    MPC Controller for position control
    6-state model: [x, y, z, vx, vy, vz]
    3 control inputs: [ax, ay, az] (acceleration commands)
    """

    def __init__(self, dt=0.1, Np=6, Nc=3):
        """
        Args:
            dt: sample time (s)
            Np: prediction horizon
            Nc: control horizon
        """
        self.dt = dt
        self.Np = Np
        self.Nc = Nc
        self.nx = 6  # [x, y, z, vx, vy, vz]
        self.nu = 3  # [ax, ay, az]

        # System matrices (integrator model)
        dt = self.dt
        self.A = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1],
        ])

        self.B = np.array([
            [0.5*dt**2, 0,         0        ],
            [0,         0.5*dt**2, 0        ],
            [0,         0,         0.5*dt**2],
            [dt,        0,         0        ],
            [0,         dt,        0        ],
            [0,         0,         dt       ],
        ])

        self.C = np.eye(self.nx)

        # Cost matrices - sama seperti versi "SAFE MODE FOR REAL FLIGHT"
        self.Q = np.diag([
            18.0, 18.0, 140.0,   # posisi
            12.0, 12.0, 90.0     # kecepatan
        ])

        self.R = np.diag([0.025, 0.025, 0.025])      # effort
        self.R_delta = np.diag([0.15, 0.15, 0.15])   # rate penalty

        self.u_prev = np.zeros(self.nu)
        self.a_max = 3.5  # m/s^2

        self._build_prediction_matrices()

    # ------------------- Build prediction matrices -------------------
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
                self.Gamma[i*self.nx:(i+1)*self.nx,
                           j*self.nu:(j+1)*self.nu] = A_power @ self.B

        self._build_qp_matrices()

    def _build_qp_matrices(self):
        Q_bar = np.kron(np.eye(self.Np), self.Q)
        R_bar = np.kron(np.eye(self.Nc), self.R)

        # R_delta untuk penalti perubahan input
        R_delta_bar = np.zeros((self.Nc*self.nu, self.Nc*self.nu))
        for i in range(self.Nc):
            R_delta_bar[i*self.nu:(i+1)*self.nu, i*self.nu:(i+1)*self.nu] = self.R_delta
            if i > 0:
                R_delta_bar[i*self.nu:(i+1)*self.nu,
                            (i-1)*self.nu:i*self.nu] = -self.R_delta
                R_delta_bar[(i-1)*self.nu:i*self.nu,
                            i*self.nu:(i+1)*self.nu] = -self.R_delta

        H = self.Gamma.T @ Q_bar @ self.Gamma + R_bar + R_delta_bar
        self.H = (H + H.T) / 2.0  # symmetrize

        self.Q_bar = Q_bar
        self.R_delta_bar = R_delta_bar

    # ------------------- Solve MPC QP -------------------
    def compute_control(self, x, x_ref):
        """
        x, x_ref: shape (6,)
        Return:
            u : shape (3,) -> [ax, ay, az]
        """
        r = np.tile(x_ref, self.Np)
        err_pred = self.Phi @ x - r

        u_prev_ext = np.tile(self.u_prev, self.Nc)

        f_tracking = self.Gamma.T @ self.Q_bar @ err_pred
        f_rate = -self.R_delta_bar @ u_prev_ext
        f = f_tracking + f_rate

        # Constraint: -a_max <= u_i <= a_max
        C = np.vstack([
            np.eye(self.Nc*self.nu),
            -np.eye(self.Nc*self.nu)
        ])
        b = np.hstack([
            np.full(self.Nc*self.nu, -self.a_max),
            np.full(self.Nc*self.nu, -self.a_max)
        ])

        try:
            # quadprog biasanya mengembalikan: x, f, xu, iters, lagrange, iact
            res = quadprog.solve_qp(self.H, -f, C.T, b, meq=0)
            u_opt = res[0]          # solusi QP ada di elemen pertama
            u = u_opt[:self.nu]

            # Safety: kalau error ketinggian besar, kurangi lateral
            z_err = abs(x[2] - x_ref[2])
            if z_err > 0.3:
                lateral_reduction = np.clip(1.0 - (z_err - 0.3)*2.0, 0.3, 1.0)
                u[0] *= lateral_reduction
                u[1] *= lateral_reduction

            self.u_prev = u.copy()
            u = np.clip(u, -self.a_max, self.a_max)
            return u
        
        except Exception as e:
            print(f"[MPC] QP failed: {e}")
            return np.zeros(self.nu)


# ============================================================================
# Simulasi sederhana + benchmark waktu komputasi
# ============================================================================
def generate_reference_trajectory(T, dt, radius=10.0, z_ref=-5.0, speed=2.0):
    """
    Buat referensi trajektori lingkaran di NED:
    - x = North, y = East, z = Down
    - kecepatan linear kira-kira 'speed' m/s
    """
    t = np.arange(0.0, T, dt)
    # Keliling = 2*pi*R; waktu satu putaran = L / v
    circle_length = 2.0 * np.pi * radius
    T_circle = circle_length / speed
    omega = 2.0 * np.pi / T_circle  # rad/s

    x_ref = radius * np.cos(omega * t)          # N
    y_ref = radius * np.sin(omega * t)          # E
    z_ref_arr = z_ref * np.ones_like(t)         # Down (negatif)

    vx_ref = -radius * omega * np.sin(omega * t)
    vy_ref =  radius * omega * np.cos(omega * t)
    vz_ref = np.zeros_like(t)

    # Stack jadi [x,y,z,vx,vy,vz] per time step
    refs = np.vstack([x_ref, y_ref, z_ref_arr, vx_ref, vy_ref, vz_ref]).T
    return t, refs


def run_mpc_benchmark(
    T_sim=30.0,
    dt=0.1,
    Np=6,
    Nc=3,
    print_trajectory_stats=False
):
    """
    Jalankan simulasi closed-loop dan ukur waktu komputasi MPC.
    """
    print("======================================================")
    print("MPC BENCHMARK (no ROS/MAVROS)")
    print("======================================================")
    print(f"Sim time     : {T_sim:.1f} s")
    print(f"dt           : {dt:.3f} s (freq ~{1.0/dt:.1f} Hz)")
    print(f"Horizon Np   : {Np}")
    print(f"Control Nc   : {Nc}")
    print("======================================================\n")

    mpc = MPCPositionController(dt=dt, Np=Np, Nc=Nc)

    # Trajektori referensi
    t_vec, ref_traj = generate_reference_trajectory(
        T=T_sim,
        dt=dt,
        radius=10.0,
        z_ref=-5.0,
        speed=2.0
    )

    # State awal (misal mulai di origin dengan kecepatan nol)
    x = np.zeros(6)

    comp_times = []
    pos_errors = []

    for k, t in enumerate(t_vec):
        x_ref = ref_traj[k]

        # --- ukur waktu komputasi MPC ---
        t0 = time.perf_counter()
        u = mpc.compute_control(x, x_ref)
        t1 = time.perf_counter()
        comp_ms = (t1 - t0) * 1000.0
        comp_times.append(comp_ms)

        # update state pakai model linier
        x = mpc.A @ x + mpc.B @ u

        # simpan error posisi (opsional, buat info tambahan)
        pos_err = np.linalg.norm(x[0:3] - x_ref[0:3])
        pos_errors.append(pos_err)

    comp_times = np.array(comp_times)
    pos_errors = np.array(pos_errors)

    # ================== Ringkasan waktu komputasi ==================
    print("WAKTU KOMPUTASI MPC PER STEP (solve_qp) [ms]:")
    print(f"  Samples          : {len(comp_times)}")
    print(f"  Mean             : {comp_times.mean():.3f} ms")
    print(f"  Std              : {comp_times.std():.3f} ms")
    print(f"  Min              : {comp_times.min():.3f} ms")
    print(f"  Max              : {comp_times.max():.3f} ms")
    print(f"  95th percentile  : {np.percentile(comp_times,95):.3f} ms")
    print(f"  99th percentile  : {np.percentile(comp_times,99):.3f} ms")
    print("")
    print("Rasio terhadap dt:")
    print(f"  Mean / dt        : {comp_times.mean() / (dt*1000.0):.3f}")
    print(f"  Max  / dt        : {comp_times.max() / (dt*1000.0):.3f}")
    print("  (Harus jauh < 1 supaya real-time aman)\n")

    if print_trajectory_stats:
        print("INFO TAMBAHAN: PERFORMA TRAJECTORY (hanya indikasi kasar)")
        print(f"  Average position error : {pos_errors.mean():.3f} m")
        print(f"  Max position error     : {pos_errors.max():.3f} m")
        print("Ini cuma untuk cek modelnya masuk akal, bukan hasil TA utama.\n")

    return comp_times, pos_errors


if __name__ == "__main__":
    # Kamu bisa ubah parameter ini untuk test berbagai kombinasi
    # misal Np / Nc berbeda dan lihat pengaruhnya ke komputasi.
    run_mpc_benchmark(
        T_sim=60.0,   # durasi simulasi
        dt=0.1,       # harus sama dengan di controller ROS
        Np=6,
        Nc=3,
        print_trajectory_stats=True
    )
