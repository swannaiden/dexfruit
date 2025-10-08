#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from robot_learning.data_collection_ros2.spacemouse import SpaceMouse
from robot_learning.data_collection_ros2.input2action import input2action
import numpy as np
import tkinter as tk
from xarm.wrapper import XArmAPI

import time  # For measuring press durations

from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import TwistStamped, PoseStamped
from scipy.spatial.transform import Rotation

def euler_to_quaternion(roll, pitch, yaw):
    # Assuming degrees for the xArm, as in your original code
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    quat = rot.as_quat()
    # print(quat)
    return quat

class Spacemouse2Xarm(Node):
    def __init__(self):
        super().__init__('spacemouse2xarm')
        ip = self.declare_parameter('xarm_ip', '192.168.1.213').value

        # --- XArm setup ---
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)
        self.arm.set_state(0)

        # --- SpaceMouse device ---
        self.device = SpaceMouse(pos_sensitivity=1.2, rot_sensitivity=1.2)
        self.device.start_control()

        # Track previous button states for edge detection
        self.prev_left_button = False
        self.prev_right_button = False

        # Additional state for tracking long-press on right button
        self.right_button_pressed = False
        self.right_button_press_time = None
        self.go_home_done_for_press = False
        self.long_press_threshold = 1.0  # seconds (adjust as needed)

        # --- Create publishers ---
        self.action_pub = self.create_publisher(TwistStamped, 'robot_action', 10)
        self.position_action_pub = self.create_publisher(PoseStamped, 'robot_position_action', 10)
        self.gripper_pub = self.create_publisher(Float32, 'gripper_position', 10)
        self.start_demo_pub = self.create_publisher(Bool, 'start_demo', 10)
        self.end_demo_pub = self.create_publisher(Bool, 'end_demo', 10)

        # --- Listen for external reset messages ---
        self.reset_sub = self.create_subscription(Bool, 'reset_xarm', self.reset_callback, 10)
        self.is_resetting = False

        # --- GUI for the gripper slider (UPDATED with larger size) ---
        self.root = tk.Tk()
        self.root.title("Gripper Control")
        self.root.geometry("1500x300")  # Larger window
        
        self.slider = tk.Scale(
            self.root,
            from_=0, to=1, 
            resolution=0.01,
            orient='horizontal',
            label='Gripper Open/Close',
            command=self.update_gripper,
            length=1700,  # Longer slider
            width=100,     # Thicker slider
            font=('Arial', 16, 'bold'),  # Larger font
            troughcolor='#E0E0E0',  # Light gray background
            sliderlength=80  # Larger slider handle
        )
        self.slider.pack(fill=tk.X, expand=True, padx=20, pady=50)  # More padding
        self.slider.set(0)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.gripper_position = 0.0
        self.root.update()

        # --- Timer for ~30 Hz loop ---
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)

    def update_gripper(self, value):
        self.gripper_position = float(value)

    def reset_callback(self, msg: Bool):
        """
        Called externally via the "reset_xarm" topic.
        """

        # let's try to modify the reset position and focus only on grasping the object.

        if msg.data:
            self.is_resetting = True
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            self.arm.set_position(
                x=166.9, y=2.1, z=230.5,
                roll=179.2, pitch=0.1, yaw=1.3,
                speed=100, is_radian=False, wait=True
            )
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(1)
            self.arm.set_state(0)
            self.is_resetting = False


    def go_home(self):
        """
        Same motion logic as in reset_callback, to move the arm home.
        """
        self.is_resetting = True
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0) # Position control mode
        self.arm.set_state(0)
        # self.arm.set_position(
        #         x=334.5, y=-142.3, z=128.3,
        #         roll=179.3, pitch=-0.1, yaw=1.2,
        #         speed=100, is_radian=False, wait=True
        #     )  # this one is for graspign the strawberry initial position
        self.arm.set_position(
                x=166.9, y=2.1, z=230.5,
                roll=179.2, pitch=0.1, yaw=1.3,
                speed=100, is_radian=False, wait=True
            )
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1) # Servo control mode
        self.arm.set_state(0)
        self.is_resetting = False

    def action_to_cartesian_velocity(self, actions):
        """
        Convert input2action outputs into linear & angular velocities.
        """
        dt = 1000.0 / 30.0
        scale = 100.0
        ang_scale = 1.0
        vx = -scale * actions[0] / dt
        vy = -scale * actions[1] / dt
        vz =  scale * actions[2] / dt
        wx = -ang_scale * actions[3]
        wy = -ang_scale * actions[4]
        wz =  ang_scale * actions[5]
        return vx, vy, vz, wx, wy, wz

    def timer_callback(self):
        """
        Main 30 Hz update: reads SpaceMouse, publishes robot control messages,
        optionally commands xArm, etc.
        """
        # Check the current state of the SpaceMouse buttons
        left_button  = self.device.buttons[0]
        right_button = self.device.buttons[1]

        # =============================
        #   BUTTON EDGE DETECTIONS
        # =============================
        # Left Button => "start_demo" on rising edge
        if left_button and not self.prev_left_button:
            self.start_demo_pub.publish(Bool(data=True))

        # Right Button => "end_demo" on rising edge
        if right_button and not self.prev_right_button:
            self.end_demo_pub.publish(Bool(data=True))
            # Start tracking how long it's held
            self.right_button_pressed = True
            self.right_button_press_time = time.time()
            self.go_home_done_for_press = False

        # Right Button => falling edge => reset the press flags
        if not right_button and self.prev_right_button:
            self.right_button_pressed = False
            self.right_button_press_time = None
            self.go_home_done_for_press = False

        # ===========================
        #   LONG-PRESS DETECTION
        # ===========================
        if self.right_button_pressed and right_button:
            # It's still held; check duration
            press_duration = time.time() - self.right_button_press_time
            # If we cross the threshold and haven't gone home yet, do it
            if (press_duration >= self.long_press_threshold) and (not self.go_home_done_for_press):
                self.go_home()
                self.go_home_done_for_press = True

        # ======================
        #  Robot Control Loop
        # ======================
        # Skip normal cartesian control if we are actively resetting
        if self.is_resetting:
            self.root.update()
            self.prev_left_button  = left_button
            self.prev_right_button = right_button
            return

        # 1) Get current xArm pose
        curr_pose = self.arm.get_position()[1]
        curr_pose = np.array(curr_pose)

        # 2) Get cartesian input from the SpaceMouse
        actions, _ = input2action(device=self.device, robot="xArm")

        # 3) Convert the slider [0..1] to a gripper command (0 => 850, 1 => -10)
        grasp = 850 - 860 * self.gripper_position

        # 4) Convert to velocity, then produce a new pose
        vx, vy, vz, wx, wy, wz = self.action_to_cartesian_velocity(actions)
        new_pose = curr_pose + np.array([vx, vy, vz, wx, wy, wz])

        # 5) Publish PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = new_pose[0]
        pose_msg.pose.position.y = new_pose[1]
        pose_msg.pose.position.z = new_pose[2]

        roll, pitch, yaw = new_pose[3], new_pose[4], new_pose[5]
        q = euler_to_quaternion(roll, pitch, yaw)
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]
        self.position_action_pub.publish(pose_msg)

        # 6) Publish TwistStamped
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.twist.linear.x = vx
        twist_msg.twist.linear.y = vy
        twist_msg.twist.linear.z = vz
        twist_msg.twist.angular.x = wx
        twist_msg.twist.angular.y = wy
        twist_msg.twist.angular.z = wz
        self.action_pub.publish(twist_msg)

        # 7) Publish the gripper [0..1]
        self.gripper_pub.publish(Float32(data=self.gripper_position))

        # 8) Command the xArm
        self.arm.set_servo_cartesian(new_pose, speed=150, mvacc=2000)
        self.arm.set_gripper_position(grasp)

        # Final step: update GUI & remember button states
        self.root.update()
        self.prev_left_button  = left_button
        self.prev_right_button = right_button

    def on_close(self):
        self.arm.disconnect()
        self.root.quit()

def main(args=None):
    rclpy.init(args=args)
    xarm_spacemouse = Spacemouse2Xarm()
    rclpy.spin(xarm_spacemouse)
    xarm_spacemouse.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()