#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class CameraJointController(Node):
    def __init__(self):
        super().__init__('camera_joint_controller')
        self.client = ActionClient(self, FollowJointTrajectory, '/camera_joint_controller/follow_joint_trajectory')
        self.client.wait_for_server()
        self.get_logger().info("Camera Joint Controller Ready. Enter goals to move the camera.")

    def send_goal(self, position, time_sec=1.0):
        goal_msg = FollowJointTrajectory.Goal()

        # Define trajectory
        goal_msg.trajectory = JointTrajectory()
        goal_msg.trajectory.joint_names = ['camera_joint']

        # Define a single point in the trajectory
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start.sec = int(time_sec)
        point.time_from_start.nanosec = int((time_sec - int(time_sec)) * 1e9)

        goal_msg.trajectory.points.append(point)

        # Send goal and wait for result
        self.get_logger().info(f"Sending goal: position={position}")
        future = self.client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return
        
        self.get_logger().info('Goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result()
        self.get_logger().info('Goal execution completed')

def main(args=None):
    rclpy.init(args=args)
    node = CameraJointController()

    try:
        while True:
            try:
                position = float(input("\nEnter desired camera position (e.g., -1.0 for up, 1.0 for down, 0.0 for center): "))
                node.send_goal(position)
            except ValueError:
                print("Invalid input! Please enter a numeric value.")
    except KeyboardInterrupt:
        print("\nExiting...")
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
