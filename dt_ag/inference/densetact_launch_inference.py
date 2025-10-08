#!/usr/bin/env python3

import sys
import pathlib
from launch import LaunchService
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    dt_ag_root = pathlib.Path(__file__).resolve().parents[1]
    script_3 = str(dt_ag_root / "utils" / "xarm_state_publisher.py")
    script_4 = str(dt_ag_root / "utils" / "xarm_position_control.py")
    script_2 = str(dt_ag_root / "utils" / "publish_dt.py")

    return LaunchDescription([
        ExecuteProcess(
            cmd=['python3', script_3],
            name='xarm_state_publisher',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['python3', script_4],
            name='xarm_position_control',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['python3', script_2],
            name='publish_dt',
            output='screen'
        ),

        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='rs_wrist',
            namespace='rs_wrist',
            output='screen',
            parameters=[{
                'serial_no': 'XX',
                'camera_name': 'rs_wrist',
                'enable_color': True,
                'enable_depth': False,
                'rgb_camera.color_profile': '640x360x60',
            }]
        ),

        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='rs_front',
            namespace='rs_front',
            output='screen',
            parameters=[{
                'serial_no': 'XX',
                'camera_name': 'rs_front',
                'enable_color': True,
                'enable_depth': False,
                'rgb_camera.color_profile': '640x360x60',
            }]
        ),
    ])

def main(argv=sys.argv[1:]):
    ld = generate_launch_description()
    ls = LaunchService(argv=argv)
    ls.include_launch_description(ld)
    return ls.run()

if __name__ == '__main__':
    sys.exit(main())
