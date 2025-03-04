#!/usr/bin/env python3

# ROS Specific Imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from gcr_object_detection.msg import YoloDetectionArray
from cv_bridge import CvBridge



# General Python Imports
import numpy as np
import cv2
import time

# Custom Imports
from deep_sort.deep_sort import DeepSort


#*#####################################################################################################################################################################################
#*#####################################################################################################################################################################################


# PID Controller Class for Human Following Mode
class HumanFollowPIDController:
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


#*#####################################################################################################################################################################################
#*#####################################################################################################################################################################################


# Main Node Class
class BallCollectionNode(Node):
    def __init__(self):
        super().__init__('ball_collection_node')
        
        # Initialize empty lists to store the detection data
        self.xywh_array = np.empty((0, 4), dtype=np.float32)  # 2D array for bounding boxes
        self.conf_array = np.empty((0,), dtype=np.float32)    # 1D array for confidence scores

        # Load DeepSORT and YOLO models
        deep_sort_weights = '/home/rajana/gazebo_ws/src/gcr_fb_bridge/src/deep_sort/deep/checkpoint/ckpt.t7'
        self.tracker = DeepSort(model_path=deep_sort_weights, max_age=80, max_iou_distance=0.7, n_init=3)

        # Motor control variables
        self.x_tolerance = 40    # Tolerance range for centered tracking
        self.y_tolerance = 5     # Tolerance range for distance tracking
        
        # Track disappearance counters
        self.person_diappeared_counter = 0
        self.person_diappeared_t = 10

        # Track disappearance counters
        self.initial_person_not_found_counter = 0
        self.initial_person_not_found_t = 20
        self.selected_track_id = None

        # Original Image Subscriber
        self.image_sub = self.create_subscription(Image, '/camera_sensor/image_raw', self.listener_callback, 2)
        self.latest_image_msg = None    # Store the latest rgb image message

        # Object Detection data from YOLO Subscriber
        self.yolo_sub = self.create_subscription(YoloDetectionArray, '/golfball', self.yolo_callback, 2)

        # Initialize the CvBridge class
        self.br = CvBridge()

        #& PID Constants for Indoor Lab Navigation
        self.pid_linear  = HumanFollowPIDController(kp=0.05, ki=0.0, kd=0.0)  
        self.pid_angular = HumanFollowPIDController(kp=0.07, ki=0.0, kd=0.0)

        # cmd_vel Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Run the follow_me process
        self.create_timer(0.2, self.follow_me_timer_callback)


#*#####################################################################################################################################################################################
#*#####################################################################################################################################################################################

    def follow_me_callback(self, doc_snapshot, changes, read_time):
        if self.current_mode == 'Follow_Me':
            for doc in doc_snapshot:
                data = doc.to_dict()
                if 'Timestamp' in data and 'Mode_Activated' in data:
                    current_mode_activated = data.get('Mode_Activated', False)
                    # Check if we are transitioning from Mode_Activated = True to False
                    if self.prev_mode_activated and not current_mode_activated:
                        # Reset Human Follow Mode
                        self.reset_human_follow("Human following was reset due to mode deactivation.")
                    # If Mode_Activated is True, update follow-me mode
                    if current_mode_activated:
                        self.x_pixel = data.get('Image_X_Pixel', 0)
                        self.y_pixel = data.get('Image_Y_Pixel', 0)
                        self.run_follow_me = True
                        self.get_logger().info(f'Follow Me Mode updated: x_pixel={self.x_pixel}, y_pixel={self.y_pixel}')
                    # Update previous mode state
                    self.prev_mode_activated = current_mode_activated
        else:
            self.get_logger().info('Ignrajanag Follow Me update because current mode is not Follow Me.')


    def ball_collection_callback(self, doc_snapshot, changes, read_time):
        if self.current_mode == 'Ball_Collection':
            for doc in doc_snapshot:
                data = doc.to_dict()
                if 'Timestamp' in data:
                    self.execute_ball_collection()
        else:
            self.get_logger().info('Ignrajanag Ball Collection update because current mode is not Ball Collection.')




#?###################################################################################################################################################################################################
#?########################### Human Following Mode Logic BELOW ######################################################################################################################################


    def update_follow_me_status(self, status_message):
        doc_ref = self.db.collection('FYP_Golf_Robot').document('DATA_Follow_Me')
        doc_ref.update({'Status': status_message})
        self.get_logger().info(f"Updated Follow Me status: {status_message}")


    def person_selection(self, x_pixel, y_pixel, tracks):
        if x_pixel and y_pixel:
            # Find the track ID for the bounding box that contains the selected coordinates
            for track in tracks:
                track_id = int(track[4])  # Track ID
                x1, y1, x2, y2 = track[:4]  # Bounding box coordinates
                if (x1 <= x_pixel <= x2) and (y1 <= y_pixel<= y2):
                    self.selected_track_id = track_id
                    self.get_logger().info(f"Selected Track ID: {self.selected_track_id}")
                    self.update_follow_me_status("Person selected, following initiated")
                    self.publish_audio_feedback("Following the selected human from behind")
                    return
            self.initial_person_not_found_counter += 1
            # If the person is not found, deactivate the human following after a certain number of frames
            if self.initial_person_not_found_counter > self.initial_person_not_found_t:
                self.initial_person_not_found_counter = 0
                self.get_logger().error("Person not found. Press stop and select the person again.")
                self.update_follow_me_status("Person not found, follow mode reset")
                self.reset_human_follow("Human following was reset due to person not found.")


    # Track the object and follow it
    def track_object_and_follow(self, tracks, frame):
        if len(tracks) == 0:
            self.person_diappeared_counter += 1
        if self.person_diappeared_counter > self.person_diappeared_t:
            self.person_diappeared_counter = 0
            self.reset_human_follow("Human following was reset due to person disappearance. Press stop and select the person again.")
            self.publish_audio_feedback("Person disappeared. Please press stop and select the person again.")
            return 
        for track in tracks:
            track_id = int(track[4])
            x1, y1, x2, y2 = track[:4]
            if self.selected_track_id == track_id:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID-{self.selected_track_id}", (int(x1), int(y1) - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                self.follow_person(track, frame)


    # Motor Velocity Control for Human Following Mode
    def follow_person(self, track, frame):
        x1, y1, x2, y2 = track[:4]
        bbox_center_x = (x1 + x2) / 2
        x_deviation = bbox_center_x - 320  #  -320 <= x_deviation <= 320
        y = 480 - y2 # 0 <= y <= 480
        linear_x = self.pid_linear.calculate(y)
        angular_z = self.pid_angular.calculate(x_deviation)
        
        #^# NEED TO ADD A TRAPEZOIDAL VELOCITY PROFILE FOR SMOOTH ACCELERATION AND DECELERATION
        if abs(y) < self.y_tolerance:
            self.stop()
            self.update_follow_me_status("Person is too close, stopping")

        elif self.obstacle_detected:
            self.stop()
            self.update_follow_me_status("Obstacle detected, stopping movement")
            
        else:
            adjusted_msg = Twist()
            adjusted_msg.linear.x = linear_x
            adjusted_msg.angular.z = angular_z
            self.cmd_vel_pub.publish(adjusted_msg)
            self.update_follow_me_status("Following person")


    # Stop the robot
    def stop(self):
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(stop_msg)
        self.get_logger().info('Command published: Linear=0, Angular=0')


    # Execute Follow Me Mode
    def execute_follow_me(self, xywh_data, conf_data, frame):
        if frame is None:
            self.get_logger().error("No valid image frame available for processing.")
            self.update_follow_me_status("No valid image frame")
            return
        tracks = self.tracker.update(xywh_data, conf_data, frame)
        if self.selected_track_id is None:
            self.person_selection(self.x_pixel, self.y_pixel, tracks)
        else:
            self.track_object_and_follow(tracks, frame)
            cv2.imshow("Human Following", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyWindow("Human Following")


    # Convert YOLO detection data to numpy arrays if Follow Me mode is active
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


    # Modify the follow_me_timer_callback to conditionally process the stored image
    def follow_me_timer_callback(self):
        # Check if Follow Me mode is active
        if self.run_follow_me and self.latest_image_msg is not None:
            # Convert the stored ROS Image message to an OpenCV image
            try:
                self.cv_image = self.br.imgmsg_to_cv2(self.latest_image_msg, 'bgr8')
            except Exception as e:
                self.get_logger().error(f"Failed to convert image: {e}")
                return
            # Execute Follow Me mode with the converted OpenCV image
            self.execute_follow_me(self.xywh_array, self.conf_array, self.cv_image)


    # Reset Follow Me Mode
    def reset_human_follow(self, reset_msg):
        self.stop()
        self.run_follow_me = False
        self.x_pixel, self.y_pixel = None, None
        self.xywh_array = np.empty((0, 4), dtype=np.float32)  
        self.conf_array = np.empty((0,), dtype=np.float32)  
        self.og_frame = None
        self.selected_track_id = None
        try:
            cv2.destroyWindow("Human Following")
        except cv2.error:
            pass  
        self.get_logger().info(f"{reset_msg}")
        self.update_follow_me_status("Human follow mode reset")


#?########################### Human Following Mode Logic ABOVE ######################################################################################################################################
#?###################################################################################################################################################################################################
#
#^###################################################################################################################################################################################################
#^########################### Ball Collection Mode Logic BELOW ######################################################################################################################################


    # Execute Ball Collection Mode
    def execute_ball_collection(self):
        self.get_logger().info("Running functions for Ball Collection Mode.")
        #? self.publish_audio_feedback("Starting collection of all visible golf balls")
        #? self.publish_audio_feedback("I have finished collecting all visible golf balls")


#^########################### Ball Collection Mode Logic ABOVE #######################################################################################################################################
#^####################################################################################################################################################################################################


    def shutdown(self):
        self.get_logger().info("Shutting down BallCollectionNode.")
        rclpy.shutdown()


#*###################################################################################################################################################################################################
#*###################################################################################################################################################################################################


def main(args=None):
    rclpy.init(args=args)
    mode_manager_node = BallCollectionNode()
    rclpy.spin(mode_manager_node)
    mode_manager_node.shutdown()


if __name__ == '__main__':
    main()


