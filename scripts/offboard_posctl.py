#!/usr/bin/env python3
import rospy
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, SetMode

current_state = State()

def state_cb(msg):
    global current_state
    current_state = msg

def main():
    rospy.init_node("offboard_solid")

    rospy.Subscriber("/mavros/state", State, state_cb)

    sp_pub = rospy.Publisher(
        "/mavros/setpoint_raw/local",
        PositionTarget,
        queue_size=20
    )

    arm_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
    mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)

    rate = rospy.Rate(20)

    def create_setpoint():
        sp = PositionTarget()
        sp.header.stamp = rospy.Time.now()
        sp.coordinate_frame = PositionTarget.FRAME_LOCAL_NED

        sp.type_mask = (
            PositionTarget.IGNORE_VX |
            PositionTarget.IGNORE_VY |
            PositionTarget.IGNORE_VZ |
            PositionTarget.IGNORE_AFX |
            PositionTarget.IGNORE_AFY |
            PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW_RATE
        )

        sp.position.x = 0.0
        sp.position.y = 0.0
        sp.position.z = 2.0   # 2 meter altitude (positive Z in NED = down, so use positive for up)

        sp.yaw = 0.0
        return sp

    # ==== WAIT FOR CONNECTION ====
    rospy.loginfo("Waiting for FCU connection...")
    while not rospy.is_shutdown() and not current_state.connected:
        rate.sleep()
    
    rospy.loginfo("FCU connected!")

    # ==== SEND 100 SETPOINTS BEFORE OFFBOARD ====
    rospy.loginfo("Sending initial setpoints...")
    for i in range(100):
        if rospy.is_shutdown():
            break
        sp = create_setpoint()
        sp_pub.publish(sp)
        rate.sleep()
    
    rospy.loginfo("Initial setpoints sent. Ready for OFFBOARD mode.")

    # ==== SET MODE OFFBOARD ====
    last_req = rospy.Time.now()
    offboard_set = False
    
    # Counter for logging
    loop_count = 0
    
    # ==== MAIN LOOP ====
    while not rospy.is_shutdown():
        # Create fresh setpoint with updated timestamp
        sp = create_setpoint()
        
        # Publish setpoint FIRST (most important!)
        sp_pub.publish(sp)
        
        loop_count += 1
        
        # Log every 2 seconds (40 iterations at 20Hz)
        if loop_count % 40 == 0:
            rospy.loginfo("Publishing setpoints... Mode: %s, Armed: %s" % 
                         (current_state.mode, current_state.armed))
        
        # Try to set OFFBOARD mode (with timeout to avoid spamming)
        if current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
            try:
                rospy.loginfo("Attempting to set OFFBOARD mode...")
                result = mode_srv(custom_mode="OFFBOARD")
                if result.mode_sent:
                    rospy.loginfo("OFFBOARD mode request sent successfully")
                    offboard_set = True
                else:
                    rospy.logwarn("OFFBOARD mode request failed")
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s" % e)
            last_req = rospy.Time.now()
        
        # Try to ARM (only after offboard is set, with timeout to avoid spamming)
        elif current_state.mode == "OFFBOARD" and not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
            try:
                rospy.loginfo("Attempting to ARM...")
                result = arm_srv(True)
                if result.success:
                    rospy.loginfo("Vehicle ARMED successfully!")
                else:
                    rospy.logwarn("ARM request failed")
            except rospy.ServiceException as e:
                rospy.logerr("Arming failed: %s" % e)
            last_req = rospy.Time.now()
        
        rate.sleep()

if __name__ == "__main__":
    main()
