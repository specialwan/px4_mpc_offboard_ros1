#!/usr/bin/env python3
import rospy
from mavros_msgs.msg import AttitudeTarget, State
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import Quaternion
import math
import time

current_state = State()

def state_cb(msg):
    global current_state
    current_state = msg


def to_quaternion(roll=0.0, pitch=0.0, yaw=0.0):
    """
    Convert Euler angles (rad) → quaternion (w,x,y,z)
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)


def main():
    rospy.init_node("offboard_att_raw")

    rospy.Subscriber("/mavros/state", State, state_cb)

    att_pub = rospy.Publisher(
        "/mavros/setpoint_raw/attitude",
        AttitudeTarget,
        queue_size=50
    )

    arm_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
    mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)

    rate = rospy.Rate(50)  # 50 Hz Wajib!

    # ================================
    # AttitudeTarget Template
    # ================================
    att = AttitudeTarget()
    att.type_mask = (
        AttitudeTarget.IGNORE_ROLL_RATE |
        AttitudeTarget.IGNORE_PITCH_RATE |
        AttitudeTarget.IGNORE_YAW_RATE
    )

    # target attitude
    roll  = 0.0
    pitch = 0.0
    yaw   = 0.0

    # thrust hover (adjust sesuai drone)
    att.thrust = 0.40

    # ===========================================
    # SEND >100 SETPOINTS SEBELUM OFFBOARD
    # ===========================================
    for i in range(150):
        qw, qx, qy, qz = to_quaternion(roll, pitch, yaw)
        att.orientation.w = qw
        att.orientation.x = qx
        att.orientation.y = qy
        att.orientation.z = qz
        att_pub.publish(att)
        rate.sleep()

    # ==== MASUK OFFBOARD ====
    mode_srv(custom_mode="OFFBOARD")
    time.sleep(0.5)

    # ==== ARM ====
    arm_srv(True)

    rospy.loginfo(">>> OFFBOARD ATTITUDE RAW Started <<<")

    # ===========================================
    # MAIN LOOP 50Hz
    # ===========================================
    while not rospy.is_shutdown():

        # contoh: attitude fixed hover
        qw, qx, qy, qz = to_quaternion(roll, pitch, yaw)
        att.orientation.w = qw
        att.orientation.x = qx
        att.orientation.y = qy
        att.orientation.z = qz

        # publish 50 Hz
        att_pub.publish(att)
        rate.sleep()


if __name__ == "__main__":
    main()
