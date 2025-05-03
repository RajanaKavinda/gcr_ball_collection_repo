#!/usr/bin/env python3
###########################################################################################

# ROS Specific Imports
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from gcr_obj_detect.msg import YoloDetectionArray
from cv_bridge import CvBridge

# General Python Imports
import numpy as np
import cv2

# Custom Imports
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml
import argparse


# Camera Joint Controller Import
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

#*#####################################################################################################################################################################################
#*#####################################################################################################################################################################################
class DetectionResults:
    def __init__(self, xywh, conf, cls):
        self.xywhr = xywh  # Rename to `xywhr` if your tracker expects it
        self.conf = conf
        self.cls = cls

# PID Controller Class for Ball Collection Mode
class BallFollowPIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def calculate(self, current_value):
        current_time = time.time()
        if self.last_time is None:
            self.last_time = current_time
        error = current_value
        delta_time = current_time - self.last_time
        if delta_time > 0:
            self.integral = self.integral + error * delta_time
            derivative = (error - self.previous_error) / delta_time
            proportional_component = self.kp * error
            integral_component = self.ki * self.integral
            derivative_component = self.kd * derivative
            output = proportional_component + integral_component + derivative_component
            self.previous_error = error
            self.last_time = current_time
            return output
        else:
            return 0.0

# Main Node Class
class BallCollectionNode(Node):
    def __init__(self):
        super().__init__('ball_collection_bot_sort_node')
        
        # Initialize empty lists to store the detection data
        self.xywh_array = np.empty((0, 4), dtype=np.float32)  # 2D array for bounding boxes
        self.conf_array = np.empty((0,), dtype=np.float32)    # 1D array for confidence scores

        # Load BoT-SORT configuration and initialize the tracker
        tracker_config = yaml_load(check_yaml("/home/rajana/gazebo_ws/src/gcr_ball_collection/config/custom_botsort.yaml"))
        # tracker_config = yaml_load(check_yaml("botsort.yaml"))
        tracker_args = argparse.Namespace(**tracker_config)
        self.tracker = BOTSORT(args=tracker_args , frame_rate=30)  # Initialize BoT-SORT tracker

        # Motor control variables
        self.x_tolerance = 20    # Tolerance range for centered tracking
        self.y_tolerance = 200     # Tolerance range for distance tracking

        # Initialize OpenCV bridge
        self.br = CvBridge()
        
        # Track disappearance counters
        self.golfball_diappeared_counter = 0
        self.golfball_diappeared_t = 60

        # Track disappearance counters
        self.selected_track_id = None

        # Original Image Subscriber
        self.image_sub = self.create_subscription(Image, '/camera_sensor/image_raw', self.listener_callback, 2)
        self.latest_image_msg = None    # Store the latest rgb image message

        # Object Detection data from YOLO Subscriber
        self.yolo_sub = self.create_subscription(YoloDetectionArray, '/golfball', self.yolo_callback, 10)

        # cmd_vel Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Run the follow_me process
        self.create_timer(0.2, self.ball_collection_timer_callback)

        # PID Controller for Ball Collection Mode
        self.pid_linear  = BallFollowPIDController(kp=0.2, ki=0.0, kd=0.1)  
        self.pid_angular = BallFollowPIDController(kp=0.2, ki=0.0, kd=0.2)

        # Camera Joint Controller
        self.client_camera = ActionClient(self, FollowJointTrajectory, '/camera_joint_controller/follow_joint_trajectory')
        self.client_camera.wait_for_server()
        self.get_logger().info("Camera Joint Controller Ready.")

        # Ball Collector Joint Controller
        self.client_ball = ActionClient(self, FollowJointTrajectory, '/ball_collector_joint_controller/follow_joint_trajectory')
        self.client_ball.wait_for_server()
        self.get_logger().info("Ball Collector Joint Controller Ready.")

        # Send initial goal to ball collector joint
        self.send_goal(2, 1.55, 2.0)  # Joint number 2 for ball collector joint
        self.send_goal(1, 0.2, 2.0)
        self.camera_up = True

#^###################################################################################################################################################################################################
#^########################### Ball Collection Mode Logic BELOW ######################################################################################################################################

    def ball_selection(self, tracks):
        max_area = 0
        selected_track_id = None

        for track in tracks:
            track_id = int(track[4])  # Track ID
            x1, y1, x2, y2 = track[:4]  # Bounding box coordinates
            area = (x2 - x1) * (y2 - y1)  # Calculate bounding box area

            if area > max_area:
                max_area = area
                selected_track_id = track_id

        self.selected_track_id = selected_track_id


    # Track the object and follow it
    def track_object_and_follow(self, tracks, frame):

        self.get_logger().info(f"Length of tracks: {len(tracks)}")

        for track in tracks:
            track_id = int(track[4])
            x1, y1, x2, y2 = track[:4]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv2.putText(frame, f"ID-{track_id}", (int(x1), int(y1) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        track_id_available = any(self.selected_track_id == int(track[4]) for track in tracks)
        selected_track = next((track for track in tracks if self.selected_track_id == int(track[4])), None)

        if not track_id_available:
            self.golfball_diappeared_counter += 1
            self.move_robot(0.15, 0.0)
            self.get_logger().info(f"Selected track id is not available in tracks : counter - {self.golfball_diappeared_counter}")
        if self.golfball_diappeared_counter > self.golfball_diappeared_t:
            self.golfball_diappeared_counter = 0
            self.get_logger().info("Golfball disappeared. Resetting Ball Collection mode.")
            self.reset_ball_collection("Ball Collection was reset due to golfball disappearance. Press stop and select the golfball again.")
            return 
        if selected_track is not None:
            self.follow_golfball(selected_track)       

    # Motor Velocity Control for Ball Collection Mode
    def follow_golfball(self, track):
        x1, y1, x2, y2 = track[:4]
        bbox_center_x = (x1 + x2) / 2
        x_deviation = bbox_center_x - 320  #  -320 <= x_deviation <= 320
        y = 480 - y2 # 0 <= y <= 480

        # Calculate the distance to the object
        self.get_logger().info(f"y : {y} x_deviation : {x_deviation}")

        # Calculate linear and angular velocities
        linear_x = self.pid_linear.calculate(y/300)
        angular_z = self.pid_angular.calculate(x_deviation/150)
        
        if abs(y) < self.y_tolerance:
            if self.camera_up:
                self.send_goal(1, 0.5, 3.0)
                self.camera_up = False
            if abs(x_deviation) > self.x_tolerance:
                self.move_robot(linear_x/2, -angular_z)
            else:
                self.move_robot(linear_x + 0.2, 0.0)                     
        else:
            if not self.camera_up:
                self.send_goal(1, 0.2, 2.0)
                self.camera_up = True
            self.move_robot(linear_x, -angular_z)
        
        self.get_logger().info(f"Selected Track ID: {self.selected_track_id}")

    def move_robot(self, linear_x, angular_z):
        adjusted_msg = Twist()
        adjusted_msg.linear.x = linear_x
        adjusted_msg.angular.z = angular_z
        self.cmd_vel_pub.publish(adjusted_msg)
        self.get_logger().info(f"Command published: Linear={adjusted_msg.linear.x }, Angular={adjusted_msg.angular.z}")


    # Execute Ball Collection mode
    def execute_ball_collection(self, detections, frame):
        if frame is None:
            self.get_logger().error("No valid image frame available for processing.")
            return
        tracks = self.tracker.update(detections, frame)
        # print(tracks)
        if self.selected_track_id is None:
            self.ball_selection(tracks)
        else:
            self.track_object_and_follow(tracks, frame)
            cv2.imshow("Ball Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyWindow("Ball Collection")

    # Convert YOLO detection data to numpy arrays if Ball Collection mode is active
    def yolo_callback(self, msg):   
        # Initialize lists to hold bounding box coordinates and confidence scores
        xywh_list = []
        conf_list = []
        # Iterate through the items in the YoloDetectionArray message
        for detection in msg.items:  
            # Extract bounding box and confidence
            x = detection.x
            y = detection.y
            width = detection.width
            height = detection.height
            confidence = detection.confidence
            # Append the bounding box (x, y, width, height) and confidence to the lists
            xywh_list.append([x, y, width, height])
            conf_list.append(confidence)
        # Convert the lists to numpy arrays
        self.xywh_array = np.array(xywh_list, dtype=np.float32).reshape(-1, 4)
        self.conf_array = np.array(conf_list, dtype=np.float32)


    # Modify the listener_callback to store the latest image message unconditionally
    def listener_callback(self, msg):
        # Store the latest image message, regardless of the state of run_follow_me
        self.latest_image_msg = msg


    # Modify the ball_collection_timer_callback to conditionally process the stored image
    def ball_collection_timer_callback(self):
        if self.latest_image_msg is not None:
            # Convert the stored ROS Image message to an OpenCV image
            try:
                self.cv_image = self.br.imgmsg_to_cv2(self.latest_image_msg, 'bgr8')
            except Exception as e:
                self.get_logger().error(f"Failed to convert image: {e}")
                return
            classes = np.zeros(len(self.conf_array), dtype=int)  
            detections = DetectionResults(self.xywh_array, self.conf_array, cls = classes)
            # Execute Ball Collection mode with the converted OpenCV image
            self.execute_ball_collection(detections, self.cv_image)


    # Reset Ball Collection mode
    def reset_ball_collection(self, reset_msg):
        self.move_robot(0.0,0.0)
        self.xywh_array = np.empty((0, 4), dtype=np.float32)  
        self.conf_array = np.empty((0,), dtype=np.float32)  
        self.selected_track_id = None
        try:
            cv2.destroyWindow("Ball Collection")
        except cv2.error:
            pass  
        self.get_logger().info(f"{reset_msg}")


#^########################### Ball Collection Mode Logic ABOVE #######################################################################################################################################

#########################################################################################################################################################################################################
    def send_goal(self, joint_number,  position, time_sec=1.0):
        goal_msg = FollowJointTrajectory.Goal()

        # Define trajectory
        goal_msg.trajectory = JointTrajectory()

        # Define a single point in the trajectory
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start.sec = int(time_sec)
        point.time_from_start.nanosec = int((time_sec - int(time_sec)) * 1e9)

        goal_msg.trajectory.points.append(point)

        # Send goal and wait for result
        self.get_logger().info(f"Sending goal: position={position}")

        if joint_number == 1:
            goal_msg.trajectory.joint_names = ['camera_joint']
            future = self.client_camera.send_goal_async(goal_msg)
        elif joint_number == 2:
            goal_msg.trajectory.joint_names = ['ball_collector_joint']
            future = self.client_ball.send_goal_async(goal_msg)
        
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


#^#####################################################################################################################################################################################################

def main(args=None):
    rclpy.init(args=args)
    mode_manager_node = BallCollectionNode()

    try:
        rclpy.spin(mode_manager_node)
    except KeyboardInterrupt:
        pass 
    finally:
        mode_manager_node.move_robot(0.0,0.0)
        mode_manager_node.destroy_node()  
        rclpy.shutdown() 

if __name__ == '__main__':
    main()

###########################################################################################
















# #!/usr/bin/env python3
# ###########################################################################################

# # ROS Specific Imports
# import time
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist
# from sensor_msgs.msg import Image
# from gcr_obj_detect.msg import YoloDetectionArray
# from cv_bridge import CvBridge

# # General Python Imports
# import numpy as np
# import cv2

# # Custom Imports
# from ultralytics.trackers.bot_sort import BOTSORT
# from ultralytics.utils import yaml_load
# from ultralytics.utils.checks import check_yaml
# import argparse

# #*#####################################################################################################################################################################################
# #*#####################################################################################################################################################################################
# class DetectionResults:
#     def __init__(self, xywh, conf, cls):
#         self.xywhr = xywh  # Rename to `xywhr` if your tracker expects it
#         self.conf = conf
#         self.cls = cls

# # PID Controller Class for Ball Collection Mode
# class BallFollowPIDController:
#     def __init__(self, kp, ki, kd):
#         self.kp = kp
#         self.ki = ki
#         self.kd = kd
#         self.previous_error = 0.0
#         self.integral = 0.0
#         self.last_time = None

#     def calculate(self, current_value):
#         current_time = time.time()
#         if self.last_time is None:
#             self.last_time = current_time
#         error = current_value
#         delta_time = current_time - self.last_time
#         if delta_time > 0:
#             self.integral = self.integral + error * delta_time
#             derivative = (error - self.previous_error) / delta_time
#             proportional_component = self.kp * error
#             integral_component = self.ki * self.integral
#             derivative_component = self.kd * derivative
#             output = proportional_component + integral_component + derivative_component
#             self.previous_error = error
#             self.last_time = current_time
#             return output
#         else:
#             return 0.0

# # Main Node Class
# class BallCollectionNode(Node):
#     def __init__(self):
#         super().__init__('ball_collection_bot_sort_node')
        
#         # Initialize empty lists to store the detection data
#         self.xywh_array = np.empty((0, 4), dtype=np.float32)  # 2D array for bounding boxes
#         self.conf_array = np.empty((0,), dtype=np.float32)    # 1D array for confidence scores

#         # Load BoT-SORT configuration and initialize the tracker
#         tracker_config = yaml_load(check_yaml("/home/rajana/gazebo_ws/src/gcr_ball_collection/config/custom_botsort.yaml"))
#         # tracker_config = yaml_load(check_yaml("botsort.yaml"))
#         tracker_args = argparse.Namespace(**tracker_config)
#         self.tracker = BOTSORT(args=tracker_args , frame_rate=30)  # Initialize BoT-SORT tracker

#         # Motor control variables
#         self.x_tolerance = 20    # Tolerance range for centered tracking
#         self.y_tolerance = 140     # Tolerance range for distance tracking

#         # Initialize OpenCV bridge
#         self.br = CvBridge()
        
#         # Track disappearance counters
#         self.golfball_diappeared_counter = 0
#         self.golfball_diappeared_t = 50

#         # Track disappearance counters
#         self.selected_track_id = None

#         # Original Image Subscriber
#         self.image_sub = self.create_subscription(Image, '/camera_sensor/image_raw', self.listener_callback, 2)
#         self.latest_image_msg = None    # Store the latest rgb image message

#         # Object Detection data from YOLO Subscriber
#         self.yolo_sub = self.create_subscription(YoloDetectionArray, '/golfball', self.yolo_callback, 10)

#         # cmd_vel Publisher
#         self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
#         # Run the follow_me process
#         self.create_timer(0.2, self.ball_collection_timer_callback)

#         # PID Controller for Ball Collection Mode
#         self.pid_linear  = BallFollowPIDController(kp=0.05, ki=0.0, kd=0.0)  
#         self.pid_angular = BallFollowPIDController(kp=0.07, ki=0.0, kd=0.0)

# #^###################################################################################################################################################################################################
# #^########################### Ball Collection Mode Logic BELOW ######################################################################################################################################

#     def ball_selection(self, tracks):
#         max_area = 0
#         selected_track_id = None

#         for track in tracks:
#             track_id = int(track[4])  # Track ID
#             x1, y1, x2, y2 = track[:4]  # Bounding box coordinates
#             area = (x2 - x1) * (y2 - y1)  # Calculate bounding box area

#             if area > max_area:
#                 max_area = area
#                 selected_track_id = track_id

#         self.selected_track_id = selected_track_id


#     # Track the object and follow it
#     def track_object_and_follow(self, tracks, frame):

#         self.get_logger().info(f"Length of tracks: {len(tracks)}")

#         for track in tracks:
#             track_id = int(track[4])
#             x1, y1, x2, y2 = track[:4]
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
#             cv2.putText(frame, f"ID-{track_id}", (int(x1), int(y1) - 5), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

#         track_id_available = any(self.selected_track_id == int(track[4]) for track in tracks)
#         selected_track = next((track for track in tracks if self.selected_track_id == int(track[4])), None)

#         if not track_id_available:
#             self.golfball_diappeared_counter += 1
#             self.move_robot(0.15, 0.0)
#             self.get_logger().info(f"Selected track id is not available in tracks : counter - {self.golfball_diappeared_counter}")
#         if self.golfball_diappeared_counter > self.golfball_diappeared_t:
#             self.golfball_diappeared_counter = 0
#             self.get_logger().info("Golfball disappeared. Resetting Ball Collection mode.")
#             self.reset_ball_collection("Ball Collection was reset due to golfball disappearance. Press stop and select the golfball again.")
#             return 
#         if selected_track is not None:
#             self.follow_golfball(selected_track)       

#     # Motor Velocity Control for Ball Collection Mode
#     def follow_golfball(self, track):
#         x1, y1, x2, y2 = track[:4]
#         bbox_center_x = (x1 + x2) / 2
#         x_deviation = bbox_center_x - 320  #  -320 <= x_deviation <= 320
#         y = 480 - y2 # 0 <= y <= 480

#         # Calculate the distance to the object
#         self.get_logger().info(f"y : {y} x_deviation : {x_deviation}")

#         # Calculate linear and angular velocities
#         linear_x = self.pid_linear.calculate(y/300)
#         angular_z = self.pid_angular.calculate(x_deviation/150)
        
#         if abs(y) < self.y_tolerance:
#             if abs(x_deviation) > self.x_tolerance:
#                 self.move_robot(0.0, -angular_z)
#             else:
#                 self.move_robot(linear_x + 0.2, 0.0) 
#         else:
#             self.move_robot(linear_x, -angular_z)
        
#         self.get_logger().info(f"Selected Track ID: {self.selected_track_id}")

#     def move_robot(self, linear_x, angular_z):
#         adjusted_msg = Twist()
#         adjusted_msg.linear.x = linear_x
#         adjusted_msg.angular.z = angular_z
#         self.cmd_vel_pub.publish(adjusted_msg)
#         self.get_logger().info(f"Command published: Linear={adjusted_msg.linear.x }, Angular={adjusted_msg.angular.z}")


#     # Execute Ball Collection mode
#     def execute_ball_collection(self, detections, frame):
#         if frame is None:
#             self.get_logger().error("No valid image frame available for processing.")
#             return
#         tracks = self.tracker.update(detections, frame)
#         # print(tracks)
#         if self.selected_track_id is None:
#             self.ball_selection(tracks)
#         else:
#             self.track_object_and_follow(tracks, frame)
#             cv2.imshow("Ball Collection", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cv2.destroyWindow("Ball Collection")

#     # Convert YOLO detection data to numpy arrays if Ball Collection mode is active
#     def yolo_callback(self, msg):   
#         # Initialize lists to hold bounding box coordinates and confidence scores
#         xywh_list = []
#         conf_list = []
#         # Iterate through the items in the YoloDetectionArray message
#         for detection in msg.items:  
#             # Extract bounding box and confidence
#             x = detection.x
#             y = detection.y
#             width = detection.width
#             height = detection.height
#             confidence = detection.confidence
#             # Append the bounding box (x, y, width, height) and confidence to the lists
#             xywh_list.append([x, y, width, height])
#             conf_list.append(confidence)
#         # Convert the lists to numpy arrays
#         self.xywh_array = np.array(xywh_list, dtype=np.float32).reshape(-1, 4)
#         self.conf_array = np.array(conf_list, dtype=np.float32)


#     # Modify the listener_callback to store the latest image message unconditionally
#     def listener_callback(self, msg):
#         # Store the latest image message, regardless of the state of run_follow_me
#         self.latest_image_msg = msg


#     # Modify the ball_collection_timer_callback to conditionally process the stored image
#     def ball_collection_timer_callback(self):
#         if self.latest_image_msg is not None:
#             # Convert the stored ROS Image message to an OpenCV image
#             try:
#                 self.cv_image = self.br.imgmsg_to_cv2(self.latest_image_msg, 'bgr8')
#             except Exception as e:
#                 self.get_logger().error(f"Failed to convert image: {e}")
#                 return
#             classes = np.zeros(len(self.conf_array), dtype=int)  
#             detections = DetectionResults(self.xywh_array, self.conf_array, cls = classes)
#             # Execute Ball Collection mode with the converted OpenCV image
#             self.execute_ball_collection(detections, self.cv_image)


#     # Reset Ball Collection mode
#     def reset_ball_collection(self, reset_msg):
#         self.move_robot(0.0,0.0)
#         self.xywh_array = np.empty((0, 4), dtype=np.float32)  
#         self.conf_array = np.empty((0,), dtype=np.float32)  
#         self.selected_track_id = None
#         try:
#             cv2.destroyWindow("Ball Collection")
#         except cv2.error:
#             pass  
#         self.get_logger().info(f"{reset_msg}")


# #^########################### Ball Collection Mode Logic ABOVE #######################################################################################################################################
# #^#####################################################################################################################################################################################################

# def main(args=None):
#     rclpy.init(args=args)
#     mode_manager_node = BallCollectionNode()

#     try:
#         rclpy.spin(mode_manager_node)
#     except KeyboardInterrupt:
#         pass 
#     finally:
#         mode_manager_node.move_robot(0.0,0.0)
#         mode_manager_node.destroy_node()  
#         rclpy.shutdown() 

# if __name__ == '__main__':
#     main()

# ###########################################################################################