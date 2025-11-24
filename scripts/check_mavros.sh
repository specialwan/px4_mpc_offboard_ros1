#!/bin/bash
# Script untuk mengecek status MAVROS

echo "=== Checking MAVROS Topics ==="
echo ""
echo "1. Available setpoint topics:"
rostopic list | grep setpoint
echo ""

echo "2. MAVROS State:"
rostopic echo -n 1 /mavros/state
echo ""

echo "3. Local Position (pastikan ada estimasi posisi):"
timeout 2 rostopic echo /mavros/local_position/pose || echo "No local position data!"
echo ""

echo "4. Checking if setpoint_raw/local is being published:"
timeout 2 rostopic hz /mavros/setpoint_raw/local || echo "No setpoint being published!"
echo ""

echo "5. Global Position (GPS - jika outdoor):"
timeout 2 rostopic echo -n 1 /mavros/global_position/global || echo "No GPS data"
