# PX4 MPC Offboard Control (ROS1)

Model Predictive Control (MPC) implementation for PX4 drone using ROS1 and MAVROS.

## 🚁 Features

- **MPC Position Controller** with attitude/thrust output
- **Multiple trajectory modes**: Circle, Square, Helix
- **Dual control modes**:
  - MPC external controller (`mpc_mavros_attitude.py`)
  - PX4 internal PID controller (`waypoint_publisher_px4.py`)
- **Smart waypoint follower** with continuous/discrete tracking
- **Data logging** for flight analysis

## 📦 Dependencies

- ROS1 (Noetic recommended)
- MAVROS
- Python 3
- NumPy
- quadprog (`pip install quadprog`)

## 🔧 Installation

```bash
# Clone repository into your catkin workspace
cd ~/catkin_ws/src
git clone https://github.com/specialwan/px4_mpc_offboard_ros1.git mpc_offboard

# Install Python dependencies
pip3 install numpy quadprog

# Build workspace
cd ~/catkin_ws
catkin_make

# Source workspace
source devel/setup.bash
```

## 🚀 Usage

### Mode 1: MPC External Controller

**Circle Trajectory:**
```bash
# Terminal 1: MAVROS
roslaunch mavros px4.launch

# Terminal 2: MPC Controller
rosrun mpc_offboard mpc_mavros_attitude.py

# Terminal 3: Waypoint Publisher
roslaunch mpc_offboard waypoint_circle.launch
```

**Square Trajectory:**
```bash
roslaunch mpc_offboard waypoint_square.launch
```

**Helix Trajectory:**
```bash
roslaunch mpc_offboard waypoint_helix.launch
```

### Mode 2: PX4 Internal PID Controller

**Circle Trajectory:**
```bash
# Terminal 1: MAVROS
roslaunch mavros px4.launch

# Terminal 2: Waypoint Publisher (PX4)
roslaunch mpc_offboard waypoint_px4_circle.launch
```

**Square Trajectory:**
```bash
roslaunch mpc_offboard waypoint_px4_square.launch
```

**Helix Trajectory:**
```bash
roslaunch mpc_offboard waypoint_px4_helix.launch
```

## ⚙️ Configuration

### MPC Controller Parameters

Edit `scripts/mpc_mavros_attitude.py`:

```python
# Constraint parameters (adjust for your drone)
self.a_max = 3.0  # Max acceleration (m/s²)

# Cost matrices
self.Q = np.diag([12.0, 12.0, 120.0, 8.0, 8.0, 60.0])  # State cost
self.R = np.diag([0.05, 0.05, 0.05])  # Control effort
self.R_delta = np.diag([0.15, 0.15, 0.15])  # Rate of change penalty

# Attitude constraints
max_tilt = np.radians(25.0)  # Max tilt angle (degrees)
hover_thrust = 0.42  # Hover thrust (0.0-1.0)
```

### Trajectory Parameters

Edit launch files in `launch/` directory:

**Circle:**
```xml
<param name="circle_radius" value="25.0"/>    <!-- Radius (m) -->
<param name="circle_altitude" value="-5.0"/>  <!-- Altitude (m, NED) -->
<param name="circle_points" value="80"/>      <!-- Number of waypoints -->
```

**Helix:**
```xml
<param name="helix_radius" value="1.0"/>            <!-- Radius (m) -->
<param name="helix_start_altitude" value="-10.0"/>  <!-- Start altitude (m) -->
<param name="helix_end_altitude" value="-30.0"/>    <!-- End altitude (m) -->
<param name="helix_turns" value="3.0"/>             <!-- Number of turns -->
```

**Square:**
```xml
<param name="square_size" value="20.0"/>      <!-- Side length (m) -->
<param name="square_altitude" value="-5.0"/>  <!-- Altitude (m, NED) -->
```

### Tracking Mode

```xml
<param name="continuous_mode" value="true"/>     <!-- Smooth tracking -->
<param name="lookahead_distance" value="2.0"/>   <!-- Lookahead (m) -->
<param name="loop_mission" value="true"/>        <!-- Loop trajectory -->
<param name="acceptance_radius" value="0.8"/>    <!-- Waypoint radius (m) -->
```

## 📊 Hardware Tested

- **Frame**: DJI F550 Hexacopter
- **Flight Controller**: Pixhawk 2.4.8
- **Motors**: EMAX XA2212 1400KV
- **Propellers**: 8x4.5
- **Total Weight**: ~1.5-2.0 kg

### Recommended Constraints for F550

```python
# MPC constraints
self.a_max = 3.0  # m/s²
max_tilt = np.radians(25.0)  # 25°
hover_thrust = 0.42  # 42%

# Acceleration limits
ax = np.clip(ax, -2.5, 2.5)  # Lateral
az = np.clip(az, -4.0, 4.0)  # Vertical
```

## 📁 Package Structure

```
mpc_offboard/
├── CMakeLists.txt
├── package.xml
├── launch/
│   ├── waypoint_circle.launch       # MPC circle
│   ├── waypoint_square.launch       # MPC square
│   ├── waypoint_helix.launch        # MPC helix
│   ├── waypoint_px4_circle.launch   # PX4 circle
│   ├── waypoint_px4_square.launch   # PX4 square
│   └── waypoint_px4_helix.launch    # PX4 helix
└── scripts/
    ├── mpc_mavros_attitude.py       # MPC controller
    ├── waypoint_publisher_mavros.py # Waypoint publisher (MPC)
    ├── waypoint_publisher_px4.py    # Waypoint publisher (PX4)
    ├── data_logger_mavros.py        # Flight data logger
    ├── offboard_posctl.py           # Basic offboard control
    └── check_mavros.sh              # MAVROS diagnostic tool
```

## 🎯 Topics

### Published (MPC Mode)
- `/mavros/setpoint_raw/attitude` - Attitude and thrust commands
- `/waypoint/target` - Target waypoint

### Published (PX4 Mode)
- `/mavros/setpoint_position/local` - Position setpoints

### Subscribed
- `/mavros/local_position/pose` - Current position
- `/mavros/state` - FCU state

## 🛠️ Troubleshooting

### "No offboard signal" error

1. Ensure setpoints are published at >2Hz
2. Check local position estimate is available:
   ```bash
   rostopic echo /mavros/local_position/pose
   ```
3. Verify GPS lock (outdoor) or vision system (indoor)
4. Run diagnostic tool:
   ```bash
   cd scripts
   ./check_mavros.sh
   ```

### Drone oscillates

- Reduce `a_max` (e.g., from 4.0 to 3.0)
- Increase `R` and `R_delta` in MPC cost matrices
- Reduce `max_tilt` angle
- Tune PX4 rate controller gains

### Slow tracking

- Increase `a_max`
- Increase `lookahead_distance`
- Reduce cost on control effort (`R`)
- Increase `max_tilt` angle

## 📖 References

- [PX4 Offboard Control](https://docs.px4.io/main/en/flight_modes/offboard.html)
- [MAVROS Documentation](http://wiki.ros.org/mavros)
- Model Predictive Control theory

## 📝 License

MIT License

## 👤 Author

specialwan

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

**⚠️ Safety Notice**: Always test in simulation first. Follow local regulations for drone flights. Use failsafe mechanisms.
