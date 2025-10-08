#!/usr/bin/env python3

import sys
import pathlib
from launch import LaunchService
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    dt_ag_root = pathlib.Path(__file__).resolve().parents[2]
    data_collection_root = dt_ag_root / "data_collection_ros2"
    script_1 = str(data_collection_root / "rs_dt" / "rs_dt_hdf5_collector.py")
    script_2 = str(data_collection_root / "xarm_dt_spacemouse_ros2.py")
    script_3 = str(data_collection_root / "publish_dt_data_collection.py")
    script_4 = str(dt_ag_root / "utils" / "publish_dt.py")

    return LaunchDescription([
        ExecuteProcess(
            cmd=['python3', script_1],
            name='xarm_rs_dt_collector',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['python3', script_2],
            name='xarm_spacemouse_ros2',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['python3', script_4],
            name='publish_dt',
            output='screen'
        ),

        # RealSense #1  → namespace/camera1
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='rs_wrist',
            namespace='rs_wrist',
            output='screen',
            parameters=[{
                # change these to your desired resolution / FPS
                'serial_no': '845112071112', # unique to wrist camera
                'camera_name': 'rs_wrist',
                'enable_color': True,
                'enable_depth': False,
                'enable_infra1': False,
                'enable_infra2': False,
                'enable_gyro': False,
                'enable_accel': False,
                'rgb_camera.color_profile': '640x360x30',
            }]
        ),

        # RealSense #2  → namespace /camera2
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='rs_front',
            namespace='rs_front',
            output='screen',
            parameters=[{
                'serial_no': '317222074068', # unique to front camera
                'camera_name': 'rs_front',
                'enable_color': True,
                'enable_depth': False,
                'enable_infra1': False,
                'enable_infra2': False,
                'enable_gyro': False,
                'enable_accel': False,
                'rgb_camera.color_profile': '640x360x30',
            }]
        ),

        # # RealSense #3  → namespace /camera3
        # Node(
        #     package='realsense2_camera',
        #     executable='realsense2_camera_node',
        #     name='rs_side',
        #     namespace='rs_side',
        #     output='screen',
        #     parameters=[{
        #         'serial_no': '_040322073693', # unique to side camera
        #         'camera_name': 'rs_side',
        #         'enable_color': True,
        #         'enable_depth': False,
        #         'enable_infra1': False,
        #         'enable_infra2': False,
        #         'enable_gyro': False,
        #         'enable_accel': False,
        #         'rgb_camera.color_profile': '640x360x30',
        #     }]
        # ),
    ])

def main(argv=sys.argv[1:]):
    ld = generate_launch_description()
    ls = LaunchService(argv=argv)
    ls.include_launch_description(ld)
    return ls.run()

if __name__ == '__main__':
    sys.exit(main())
