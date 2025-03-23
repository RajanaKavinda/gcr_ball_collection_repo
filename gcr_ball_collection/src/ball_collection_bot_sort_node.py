#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from gcr_obj_detect.msg import YoloDetectionArray
from cv_bridge import CvBridge

import numpy as np
import cv2

# Import BoT-SORT tracking
from ultralytics.trackers import BOTSORT


class BallCollectionNode(Node):
    def __init__(self):
        super().__init__('ball_collection_node')
        
        self.xywh_array = np.empty((0, 4), dtype=np.float32)
        self.conf_array = np.empty((0,), dtype=np.float32)
        
        # Load BoT-SORT Tracker
        self.tracker = BOTSORT()

        self.x_tolerance = 20
        self.y_tolerance = 150

        self.br = CvBridge()
        
        self.golfball_disappeared_counter = 0
        self.golfball_disappeared_t = 50

        self.selected_track_id = None

        self.image_sub = self.create_subscription(Image, '/camera_sensor/image_raw', self.listener_callback, 2)
        self.latest_image_msg = None

        self.yolo_sub = self.create_subscription(YoloDetectionArray, '/golfball', self.yolo_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.create_timer(0.2, self.ball_collection_timer_callback)

    def ball_selection(self, tracks):
        max_area = 0
        selected_track_id = None

        for track in tracks:
            track_id = int(track[4])
            x1, y1, x2, y2 = track[:4]
            area = (x2 - x1) * (y2 - y1)

            if area > max_area:
                max_area = area
                selected_track_id = track_id

        self.selected_track_id = selected_track_id
        self.get_logger().info(f"Selected Track ID: {self.selected_track_id}")

    def track_object_and_follow(self, tracks, frame):
        if len(tracks) == 0:
            self.golfball_disappeared_counter += 1
        
        if self.golfball_disappeared_counter > self.golfball_disappeared_t:
            self.golfball_disappeared_counter = 0
            self.get_logger().info("Golfball disappeared. Resetting Ball Collection mode.")
            self.reset_ball_collection("Ball Collection was reset due to golfball disappearance. Press stop and select the golfball again.")
            return 

        for track in tracks:
            track_id = int(track[4])
            x1, y1, x2, y2 = track[:4]

            if self.selected_track_id == track_id:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                cv2.putText(frame, f"ID-{self.selected_track_id}", (int(x1), int(y1) - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                self.follow_golfball(track, frame)

    def follow_golfball(self, track, frame):
        x1, y1, x2, y2 = track[:4]
        bbox_center_x = (x1 + x2) / 2
        x_deviation = bbox_center_x - 320
        y = 480 - y2

        linear_x = y / 800 + 0.8
        angular_z = x_deviation / 300
        
        adjusted_msg = Twist()
        if abs(y) < self.y_tolerance:
            if abs(x_deviation) > self.x_tolerance:
                adjusted_msg.linear.x = 0.0
                adjusted_msg.angular.z = -angular_z 
            else:
                adjusted_msg.linear.x = linear_x
                adjusted_msg.angular.z = 0.0
        else:
            adjusted_msg.linear.x = linear_x
            adjusted_msg.angular.z = -angular_z 
        
        self.cmd_vel_pub.publish(adjusted_msg)
        self.get_logger().info(f"Command published: Linear={adjusted_msg.linear.x}, Angular={adjusted_msg.angular.z}")   

    def stop(self):
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(stop_msg)
        self.get_logger().info('Command published: Linear=0, Angular=0')

    def execute_ball_collection(self, xywh_data, conf_data, frame):
        if frame is None:
            self.get_logger().error("No valid image frame available for processing.")
            return
        tracks = self.tracker.update(xywh_data, conf_data, frame)
        if self.selected_track_id is None:
            self.ball_selection(tracks)
        else:
            self.track_object_and_follow(tracks, frame)

    def yolo_callback(self, msg):   
        xywh_list = []
        conf_list = []
        for detection in msg.items:  
            x = detection.x
            y = detection.y
            width = detection.width
            height = detection.height
            confidence = detection.confidence
            xywh_list.append([x, y, width, height])
            conf_list.append(confidence)
        self.xywh_array = np.array(xywh_list, dtype=np.float32).reshape(-1, 4)
        self.conf_array = np.array(conf_list, dtype=np.float32)

    def listener_callback(self, msg):
        self.latest_image_msg = msg
        
    def ball_collection_timer_callback(self):
        if self.latest_image_msg is not None:
            try:
                self.og_image = self.br.imgmsg_to_cv2(self.latest_image_msg, 'rgb8')
            except Exception as e:
                self.get_logger().error(f"Failed to convert image: {e}")
                return
            self.execute_ball_collection(self.xywh_array, self.conf_array, self.og_image)

    def reset_ball_collection(self, reset_msg):
        self.stop()
        self.xywh_array = np.empty((0, 4), dtype=np.float32)  
        self.conf_array = np.empty((0,), dtype=np.float32)  
        self.selected_track_id = None
        self.get_logger().info(f"{reset_msg}")


def main(args=None):
    rclpy.init(args=args)
    mode_manager_node = BallCollectionNode()

    try:
        rclpy.spin(mode_manager_node)
    except KeyboardInterrupt:
        pass 
    finally:
        mode_manager_node.stop() 
        mode_manager_node.destroy_node()  
        rclpy.shutdown() 


if __name__ == '__main__':
    main()