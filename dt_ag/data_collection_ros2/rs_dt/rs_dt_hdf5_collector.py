#!/usr/bin/env python3
"""
Collector that records

  • robot pose / gripper
  • RealSense RGB (front and wrist only)
  • DenseTact left & right RGB

All image topics are subscribed as *compressed* streams (JPEG for colour,
PNG-16 bit for depth) to reduce transport load.  Frames are saved losslessly
in an HDF5 file per episode (LZF inside the archive).

"""

import os, numpy as np, h5py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import cv2


class XArmDataCollection(Node):
    # ──────────────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__('xarm_data_collection_node')
        self.get_logger().info("Starting ablation collector with compressed streams")

        self._bridge = CvBridge()

        # QoS tuned for high-rate camera streams
        sensor_qos = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT, history=rclpy.qos.HistoryPolicy.KEEP_LAST)

        # ─── Subscribers ────────────────────────────────────────────────
        self.pose_sub = Subscriber(self, PoseStamped, 'robot_position_action', qos_profile=sensor_qos)
        self.grip_sub = Subscriber(self, Float32, 'gripper_position', qos_profile=sensor_qos)

        self.rs_front_rgb_sub = Subscriber(self, CompressedImage, '/rs_front/rs_front/color/image_raw/compressed', qos_profile=sensor_qos)
        self.rs_wrist_rgb_sub = Subscriber(self, CompressedImage, '/rs_wrist/rs_wrist/color/image_raw/compressed', qos_profile=sensor_qos)

        self.dt_left_sub = Subscriber(self, CompressedImage, 'RunCamera/image_raw_20/compressed',  qos_profile=sensor_qos)
        self.dt_right_sub = Subscriber(self, CompressedImage, 'RunCamera/image_raw_22/compressed', qos_profile=sensor_qos)

        # DenseTact difference streams
        self.dt_left_diff_sub = Subscriber(self, CompressedImage, 'RunCamera/image_diff_20/compressed',  qos_profile=sensor_qos)
        self.dt_right_diff_sub = Subscriber(self, CompressedImage, 'RunCamera/image_diff_22/compressed', qos_profile=sensor_qos)

        # Synchronise everything
        self.sync = ApproximateTimeSynchronizer(
            [
                self.pose_sub, 
                self.grip_sub,
                self.rs_front_rgb_sub,
                self.rs_wrist_rgb_sub,
                self.dt_left_sub, 
                self.dt_right_sub,
                self.dt_left_diff_sub,
                self.dt_right_diff_sub
            ],
            queue_size=100, 
            slop=0.1, 
            allow_headerless=True)
        
        self.sync.registerCallback(self.synced_callback)

        # Episode control
        self.create_subscription(Bool, 'start_demo', self.start_demo_callback, 10)
        self.create_subscription(Bool, 'end_demo', self.end_demo_callback, 10)

        # Publisher for reset_dt topic
        self.reset_dt_pub = self.create_publisher(Bool, 'reset_dt', 10)

        # ─── Buffers ────────────────────────────────────────────────────
        self.reset_buffers()
        self.is_collecting = False
        self.demo_count = 1

    # ──────────────────────────────────────────────────────────────────────
    # Episode helpers
    # ──────────────────────────────────────────────────────────────────────
    def reset_buffers(self):
        self.pose_buf = []
        self.grip_buf = []
        self.rs_front_rgb_buf = []
        self.rs_wrist_rgb_buf = []
        self.dt_left_buf = []
        self.dt_right_buf = []
        self.dt_left_diff_buf = []
        self.dt_right_diff_buf = []

    def start_demo_callback(self, msg: Bool):
        if msg.data and not self.is_collecting:
            self.get_logger().info("Starting a new demonstration.")
            self.is_collecting = True
            self.episode_start = self.get_clock().now()
            self.reset_buffers()
            
            # Publish reset signal for DenseTact image difference
            reset_msg = Bool()
            reset_msg.data = True
            self.reset_dt_pub.publish(reset_msg)
            self.get_logger().info("Published reset_dt signal")

    def end_demo_callback(self, msg: Bool):
        if msg.data and self.is_collecting:
            self.get_logger().info("Ending demonstration.")
            self.is_collecting = False
            self.save_episode()
            # stats
            dur = (self.get_clock().now() - self.episode_start).nanoseconds / 1e9
            n = len(self.pose_buf)
            self.get_logger().info(f"⏹ Episode {self.demo_count} | {n} frames | {dur:.1f}s | {n/dur:.2f} Hz")
            self.demo_count += 1

    # ──────────────────────────────────────────────────────────────────────
    # Main sync callback
    # ──────────────────────────────────────────────────────────────────────
    def synced_callback(self,
                        pose_msg: PoseStamped, 
                        grip_msg: Float32,
                        rs_front_rgb_msg: CompressedImage, 
                        rs_wrist_rgb_msg: CompressedImage, 
                        dt_left_msg: CompressedImage, 
                        dt_right_msg: CompressedImage,
                        dt_left_diff_msg: CompressedImage,
                        dt_right_diff_msg: CompressedImage
                        ):
        
        if not self.is_collecting:
            return

        # Robot
        p, o = pose_msg.pose.position, pose_msg.pose.orientation
        self.pose_buf.append([p.x, p.y, p.z, o.w, o.x, o.y, o.z])
        self.grip_buf.append(grip_msg.data)

        # RealSense
        self.rs_front_rgb_buf.append(self.parse_color_image(rs_front_rgb_msg))
        self.rs_wrist_rgb_buf.append(self.parse_color_image(rs_wrist_rgb_msg))

        # DenseTact
        self.dt_left_buf.append(self.parse_color_image(dt_left_msg))
        self.dt_right_buf.append(self.parse_color_image(dt_right_msg))
        
        # DenseTact difference streams
        self.dt_left_diff_buf.append(self.parse_color_image(dt_left_diff_msg))
        self.dt_right_diff_buf.append(self.parse_color_image(dt_right_diff_msg))

        self.get_logger().info(f"Collected {len(self.pose_buf)} frames")

    # ──────────────────────────────────────────────────────────────────────
    # Parsers
    # ──────────────────────────────────────────────────────────────────────
    def parse_color_image(self, msg: CompressedImage) -> np.ndarray:
        """JPEG → CHW RGB (uint8)"""
        bgr = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.transpose(2, 0, 1)                # (3,H,W)

    def parse_depth_image(self, msg: CompressedImage) -> np.ndarray:
        """PNG-16 bit → (H,W) uint16 (millimetres)"""
        depth = self._bridge.compressed_imgmsg_to_cv2(msg)  # returns uint16 already
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16)
        return depth

    # ──────────────────────────────────────────────────────────────────────
    # Saving
    # ──────────────────────────────────────────────────────────────────────
    def save_episode(self):
        pose = np.asarray(self.pose_buf,  dtype=np.float32)
        pose[:, :3] /= 1000.0                          # mm → m
        last_pose = np.roll(pose, -1, axis=0); last_pose[0] = pose[0]
        gripper = np.asarray(self.grip_buf, dtype=np.float32)

        rs_front_rgb = np.asarray(self.rs_front_rgb_buf) if self.rs_front_rgb_buf else None
        rs_wrist_rgb = np.asarray(self.rs_wrist_rgb_buf) if self.rs_wrist_rgb_buf else None
        dt_left = np.asarray(self.dt_left_buf) if self.dt_left_buf else None
        dt_right = np.asarray(self.dt_right_buf) if self.dt_right_buf else None
        dt_left_diff = np.asarray(self.dt_left_diff_buf) if self.dt_left_diff_buf else None
        dt_right_diff = np.asarray(self.dt_right_diff_buf) if self.dt_right_diff_buf else None

        save_dir = os.path.join(os.getcwd(), 'demo_data')
        os.makedirs(save_dir, exist_ok=True)
        fn = os.path.join(save_dir, f'episode_{self.demo_count}.hdf5')

        with h5py.File(fn, 'w') as f:
            f.create_dataset('pose', data=pose)
            f.create_dataset('gripper', data=gripper)
            f.create_dataset('last_pose', data=last_pose)

            if rs_front_rgb is not None: f.create_dataset('rs_front_rgb', data=rs_front_rgb, compression='lzf')
            if rs_wrist_rgb is not None: f.create_dataset('rs_wrist_rgb', data=rs_wrist_rgb, compression='lzf')
            if dt_left is not None: f.create_dataset('dt_left', data=dt_left, compression='lzf')
            if dt_right is not None: f.create_dataset('dt_right', data=dt_right, compression='lzf')
            if dt_left_diff is not None: f.create_dataset('dt_left_diff', data=dt_left_diff, compression='lzf')
            if dt_right_diff is not None: f.create_dataset('dt_right_diff', data=dt_right_diff, compression='lzf')

        self.get_logger().info(f"💾  Saved to {fn}")

    def destroy_node(self):
        return super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = XArmDataCollection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
