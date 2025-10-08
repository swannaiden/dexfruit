#!/usr/bin/env python3
"""
XArm State Publisher Node

This node only publishes the current state of the XArm:
- Current pose
- Previous pose
- Joint states
- Gripper state

It does NOT send any commands to the robot.
"""

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R


class XArmStatePublisher(Node):
    """
    ROS node that publishes XArm state information.
    """
    def __init__(self):
        super().__init__('xarm_state_publisher')
        
        # Declare parameter for xarm IP address
        self.ip = self.declare_parameter('xarm_ip', '192.168.1.213').value
        self.get_logger().info(f'Connecting to xArm at IP: {self.ip}')

        # Initialize XArm for state reading only
        self.arm = XArmAPI(self.ip)
        self.setup_xarm()

        # State tracking
        self.prev_pose = None

        # Publishers
        self.pose_publisher = self.create_publisher(PoseStamped, 'robot_pose', 1)
        self.prev_publisher = self.create_publisher(PoseStamped, 'prev_robot_pose', 1)
        self.joints_publisher = self.create_publisher(JointState, 'xarm_joint_states', 1)
        self.gripper_state_pub = self.create_publisher(Float32, 'gripper_state', 10)

        # Timer for state publishing
        self.state_timer = self.create_timer(1.0 / 30.0, self.state_timer_callback)

        self.get_logger().info('XArm State Publisher initialized successfully')

    def setup_xarm(self):
        """
        Initialize the XArm connection for state reading.
        """
        # Only enable motion for state reading
        self.arm.motion_enable(enable=True)
        time.sleep(1)
        self.get_logger().info('XArm connection established for state publishing')

    def state_timer_callback(self):
        """
        Timer callback to publish current XArm state.
        """
        try:
            self.get_logger().debug('Publishing XArm state...')

            # Publish previous pose if available
            if self.prev_pose is not None:
                self.publish_previous_pose(self.prev_pose)
            # Get current arm position
            # start_time = time.time()
            code, pose = self.arm.get_position(is_radian=True)
            # end_time = time.time()


            # self.get_logger().info(f'get_position took {(end_time - start_time) * 1000:.2f} ms')
            if code != 0:
                self.get_logger().warn(f'Failed to get XArm position: error code {code}')
                return

            # Convert position from mm to m
            pose[0:3] = [x / 1000 for x in pose[0:3]]
            
            # Publish current pose
            self.publish_pose(pose)
            
            # Store current pose for next iteration
            self.prev_pose = pose

            # Get and publish joint angles
            code, angles = self.arm.get_joint_states(is_radian=True)
            if code == 0:
                self.publish_angles(angles[0] if angles else [])

            # Get and publish gripper state
            code, gripper_val = self.arm.get_gripper_position()
            if code == 0:
                # Normalize gripper value from [0-850] to [0-1] range
                normalized_gripper_val = (gripper_val - 850) / -850.0
                
                msg = Float32()
                msg.data = normalized_gripper_val
                self.gripper_state_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f'Error in state callback: {e}')

    def publish_previous_pose(self, pose):
        """
        Publish the previous XArm pose.
        """
        x, y, z, roll, pitch, yaw = pose

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        msg.pose.position.x = x
        msg.pose.position.y = y 
        msg.pose.position.z = z 

        # Convert Euler to quaternion
        quaternion = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        msg.pose.orientation.x = quaternion[0]
        msg.pose.orientation.y = quaternion[1]
        msg.pose.orientation.z = quaternion[2]
        msg.pose.orientation.w = quaternion[3]

        self.prev_publisher.publish(msg)

    def publish_angles(self, angles):
        """
        Publish the current XArm joint angles.
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = angles
        self.joints_publisher.publish(msg)

    def publish_pose(self, pose):
        """
        Publish the current XArm pose.
        """
        x, y, z, roll, pitch, yaw = pose

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z

        # Convert Euler to quaternion
        quaternion = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        msg.pose.orientation.x = quaternion[0]
        msg.pose.orientation.y = quaternion[1]
        msg.pose.orientation.z = quaternion[2]
        msg.pose.orientation.w = quaternion[3]

        self.pose_publisher.publish(msg)

    def destroy_node(self):
        """
        Cleanly disconnect from XArm before shutting down.
        """
        self.get_logger().info('Disconnecting from XArm...')
        self.arm.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    # Create state publisher node
    node = XArmStatePublisher()
    
    try:
        node.get_logger().info('Running XArm State Publisher...')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()