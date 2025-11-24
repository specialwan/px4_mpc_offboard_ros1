# MPC Parameter Tuning Guide

## 🛡️ SAFE MODE (Current - for Real Flight Testing)

### MPC Controller (`mpc_mavros_attitude.py`)
```python
# Cost matrices
self.Q = np.diag([18.0, 18.0, 140.0, 12.0, 12.0, 90.0])
self.R = np.diag([0.025, 0.025, 0.025])
self.R_delta = np.diag([0.15, 0.15, 0.15])
self.a_max = 3.5  # m/s²

# Velocity reference
max_speed = 1.8  # m/s
lookahead = 4.0  # m
```

### Circle Trajectory
```xml
<param name="circle_radius" value="12.0"/>
<param name="circle_points" value="80"/>
<param name="lookahead_distance" value="2.5"/>
<param name="acceptance_radius" value="1.2"/>
```

### Helix Trajectory
```xml
<param name="helix_radius" value="5.0"/>
<param name="helix_start_altitude" value="-5.0"/>
<param name="helix_end_altitude" value="-15.0"/>
<param name="helix_turns" value="2.0"/>
<param name="lookahead_distance" value="2.5"/>
```

**Characteristics:**
- ✅ Speed: ~1.8 m/s (SAFE)
- ✅ Acceleration: ±3.5 m/s² (Conservative)
- ✅ Smooth transitions
- ✅ Good for initial real flight testing
- ⚠️ Trade-off: Slightly less precise tracking

---

## ⚡ BALANCED MODE (Good Precision + Reasonable Speed)

### MPC Controller
```python
self.Q = np.diag([22.0, 22.0, 145.0, 13.0, 13.0, 95.0])
self.R = np.diag([0.02, 0.02, 0.02])
self.R_delta = np.diag([0.12, 0.12, 0.12])
self.a_max = 4.5  # m/s²

max_speed = 2.3  # m/s
lookahead = 3.5  # m
```

### Circle Trajectory
```xml
<param name="circle_radius" value="15.0"/>
<param name="circle_points" value="85"/>
<param name="lookahead_distance" value="2.2"/>
<param name="acceptance_radius" value="1.0"/>
```

**Characteristics:**
- ✅ Speed: ~2.3 m/s
- ✅ Good balance between speed and precision
- ✅ Use after successful safe mode testing

---

## 🎯 PRECISION MODE (High Accuracy, Moderate Speed)

### MPC Controller
```python
self.Q = np.diag([25.0, 25.0, 150.0, 15.0, 15.0, 100.0])
self.R = np.diag([0.015, 0.015, 0.015])
self.R_delta = np.diag([0.08, 0.08, 0.08])
self.a_max = 5.0  # m/s²

max_speed = 2.5  # m/s
lookahead = 3.0  # m
```

### Circle Trajectory
```xml
<param name="circle_radius" value="15.0"/>
<param name="circle_points" value="90"/>
<param name="lookahead_distance" value="2.0"/>
<param name="acceptance_radius" value="1.0"/>
```

**Characteristics:**
- ✅ High tracking precision
- ✅ Tight path following
- ⚠️ Requires well-tuned drone
- ⚠️ More aggressive corrections

---

## 🚀 AGGRESSIVE MODE (Fast, for Advanced Testing Only)

### MPC Controller
```python
self.Q = np.diag([15.0, 15.0, 150.0, 12.0, 12.0, 100.0])
self.R = np.diag([0.01, 0.01, 0.01])
self.R_delta = np.diag([0.05, 0.05, 0.05])
self.a_max = 6.0  # m/s²

max_speed = 4.0  # m/s
lookahead = 4.0  # m
```

**Characteristics:**
- ⚡ Speed: ~4.0 m/s
- ⚡ Fast waypoint transitions
- ⚠️ **DANGEROUS** - Only for experienced pilots
- ⚠️ Requires excellent hardware tuning
- ⚠️ Risk of oscillations/crashes

---

## 📊 Parameter Impact Summary

| Parameter | Effect on... | Lower Value | Higher Value |
|-----------|-------------|-------------|--------------|
| **Q (Position)** | Tracking accuracy | Looser tracking | Tighter tracking |
| **Q (Velocity)** | Smoothness | Jerky | Smooth |
| **R** | Control effort | More aggressive | More gentle |
| **R_delta** | Rate changes | Sudden changes | Gradual changes |
| **a_max** | Max acceleration | Slower | Faster |
| **max_speed** | Cruise speed | Slower, safer | Faster, risky |
| **lookahead** | Waypoint advance | Tight tracking | Early transition |
| **acceptance_radius** | Precision | More precise | More forgiving |

---

## 🔧 Progressive Testing Protocol

### Phase 1: Initial Testing (SAFE MODE)
1. Start with SAFE MODE parameters
2. Test hover stability
3. Test small circle (r=5m, 1 loop)
4. Verify smooth operation

### Phase 2: Validation (BALANCED MODE)
1. Switch to BALANCED MODE
2. Test larger circles (r=12-15m)
3. Test helix trajectory
4. Monitor tracking error

### Phase 3: Performance (PRECISION MODE)
1. Use PRECISION MODE
2. Compare with PX4 internal PID
3. Analyze logged data
4. Fine-tune as needed

### Phase 4: Advanced (AGGRESSIVE MODE - OPTIONAL)
⚠️ **Only if:**
- Previous phases successful
- Hardware well-tuned
- Experienced pilot present
- Failsafe mechanisms tested

---

## 🛟 Safety Recommendations

### Before Flight
- ✅ Test in simulation first
- ✅ Check all parameter values
- ✅ Verify geofence active
- ✅ Test manual RC override
- ✅ Check battery voltage

### During Flight
- ✅ Monitor QGC telemetry
- ✅ Watch for oscillations
- ✅ Ready to switch to manual mode
- ✅ Keep safe altitude (>3m)
- ✅ Monitor tracking errors

### Red Flags (Switch to Manual!)
- ❌ Sudden oscillations
- ❌ Rapid altitude changes
- ❌ Loss of position control
- ❌ Excessive tilt angles (>30°)
- ❌ Unusual motor sounds

---

## 📝 Current Configuration Status

**Active Mode:** SAFE MODE ✅  
**Max Speed:** 1.8 m/s  
**Acceleration Limit:** 3.5 m/s²  
**Suitable for:** First real flight testing

**Next Step:** After successful testing, consider BALANCED MODE for better performance.
