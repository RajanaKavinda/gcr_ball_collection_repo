#!/usr/bin/env python3
###########################################################################################

# ROS Specific Imports
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
from deep_sort.deep_sort import DeepSort

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
        deep_sort_weights = '/home/rajana/gazebo_ws/src/gcr_ball_collection/src/deep_sort/deep/checkpoint/ckpt.t7'
        self.tracker = DeepSort(model_path=deep_sort_weights, max_age=80, max_iou_distance=0.6, n_init=3)

        # Motor control variables
        self.x_tolerance = 10    # Tolerance range for centered tracking
        self.y_tolerance = 130     # Tolerance range for distance tracking

        # Initialize OpenCV bridge
        self.br = CvBridge()
        
        # Track disappearance counters
        self.golfball_diappeared_counter = 0
        self.golfball_diappeared_t = 100

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
        self.get_logger().info(f"Selected Track ID: {self.selected_track_id}")


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
            self.move_robot(0.25, 0.0)
        if self.golfball_diappeared_counter > self.golfball_diappeared_t:
            self.golfball_diappeared_counter = 0
            self.get_logger().info("Golfball disappeared. Resetting Ball Collection mode.")
            self.reset_ball_collection("Ball Collection was reset due to golfball disappearance. Press stop and select the golfball again.")
            return 
        if selected_track is not None:
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            # cv2.putText(frame, f"ID-{self.selected_track_id}", (int(x1), int(y1) - 5), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            self.follow_golfball(selected_track)       

    # Motor Velocity Control for Ball Collection Mode
    def follow_golfball(self, track):
        x1, y1, x2, y2 = track[:4]
        bbox_center_x = (x1 + x2) / 2
        x_deviation = bbox_center_x - 320  #  -320 <= x_deviation <= 320
        y = 480 - y2 # 0 <= y <= 480

        # Calculate linear and angular velocities
        linear_x = y / 800 + 0.3
        angular_z = x_deviation / 300
        
        if abs(y) < self.y_tolerance:
            if abs(x_deviation) > self.x_tolerance:
                self.move_robot(0.0, -angular_z)
            else:
                self.move_robot(linear_x, 0.0) 
        else:
            self.move_robot(linear_x, -angular_z)

    def move_robot(self, linear_x, angular_z):
        adjusted_msg = Twist()
        adjusted_msg.linear.x = linear_x
        adjusted_msg.angular.z = angular_z
        self.cmd_vel_pub.publish(adjusted_msg)
        self.get_logger().info(f"Command published: Linear={adjusted_msg.linear.x }, Angular={adjusted_msg.angular.z}")

    # # Execute Ball Collection mode
    # def execute_ball_collection(self, xywh_data, conf_data, frame):
    #     if frame is None:
    #         self.get_logger().error("No valid image frame available for processing.")
    #         return
    #     tracks = self.tracker.update(xywh_data, conf_data, frame)

    #     for track in tracks:
    #         track_id = int(track[4])
    #         x1, y1, x2, y2 = track[:4]
    #         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    #         cv2.putText(frame, f"ID-{track_id}", (int(x1), int(y1) - 5), 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    #     cv2.imshow("Ball Collection", frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         cv2.destroyWindow("Ball Collection")

    # Execute Ball Collection mode
    def execute_ball_collection(self, xywh_data, conf_data, frame):
        if frame is None:
            self.get_logger().error("No valid image frame available for processing.")
            return
        tracks = self.tracker.update(xywh_data, conf_data, frame)
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
            # Execute Ball Collection mode with the converted OpenCV image
            self.execute_ball_collection(self.xywh_array, self.conf_array, self.cv_image)


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
#^####################################################################################################################################################################################################



def main(args=None):
    rclpy.init(args=args)
    mode_manager_node = BallCollectionNode()

    try:
        rclpy.spin(mode_manager_node)
    except KeyboardInterrupt:
        pass 
    finally:
        mode_manager_node.stop() 
        mode_manager_node.shutdown()  
        mode_manager_node.destroy_node()  
        rclpy.shutdown() 



if __name__ == '__main__':
    main()


###########################################################################################