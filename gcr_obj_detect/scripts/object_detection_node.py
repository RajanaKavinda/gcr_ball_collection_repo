#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from gcr_obj_detect.msg import YoloDetection
from gcr_obj_detect.msg import YoloDetectionArray
from ultralytics import YOLO
import torch
import cv2


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('obj_detection_node')

        # Ensure CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Load the YOLO model
        self.model = YOLO("/home/rajana/gazebo_ws/src/gcr_obj_detect/custom_yolo_model/new_best_model_yolo11n_V5.pt").to(device)

        # Publishers for each object class
        self.human_pub = self.create_publisher(YoloDetectionArray, '/human', 10)
        self.golfball_pub = self.create_publisher(YoloDetectionArray, '/golfball', 10)
        self.current_frame = self.create_publisher(Image, '/current_frame', 10)
        self.bridge = CvBridge()
        
        # Confidence threshold
        self.confidence_threshold = 0.4
        
        # Subscribe to the undistorted image topic
        self.image_subscriber = self.create_subscription(
            Image, 
            '/camera_sensor/image_raw', 
            self.image_callback, 
            1
        )

        # Buffers to store the latest detection data
        self.final_image=None
        self.pub_image=None
        self.human_data = YoloDetectionArray()
        self.golfball_data = YoloDetectionArray()

        # Publish data at a constant rate of 50 FPS
        self.publish_timer = self.create_timer(0.02, self.publish_detections)
        
        # Log initialization
        self.get_logger().warn("YOLO Object Detection Node Initialized")


    def image_callback(self, msg):
        # Convert the ROS Image message to an OpenCV image
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Run YOLO inference
        self.pub_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process YOLO results and update class variables
        results = self.model(self.pub_image)
        self.process_results(results, img)

        #& Display image with bounding boxes (for debugging)
        #& cv2.imshow("YOLO Object Detection", img)
        #& cv2.waitKey(1)

    def process_results(self, results, img):
        # Initialize detection arrays for each class
        human_data = YoloDetectionArray()
        golfball_data = YoloDetectionArray()

        target_classes = {0: golfball_data, 1: human_data}

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])

                confidence = round(float(box.conf), 2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w // 2
                cy = y1 + h // 2
                
                if cls not in target_classes or confidence < self.confidence_threshold:
                    continue

                # Create detection message
                detection = YoloDetection()
                detection.x = cx
                detection.y = cy
                detection.width = w
                detection.height = h
                detection.confidence = confidence

                # Append detection to the appropriate class array
                target_classes[cls].items.append(detection)

                # Draw bounding box and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                label = f'{self.get_class_name(cls)} {confidence}'
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        self.final_image = img

        # Update buffers for each class detection
        self.human_data = human_data
        self.golfball_data = golfball_data


    def publish_detections(self):
        # Publish the yolo outputted image
        if self.final_image is not None:
            self.current_frame.publish(self.bridge.cv2_to_imgmsg(self.final_image))        

        # Publish detection data for each class
        if self.human_data.items:
            self.human_pub.publish(self.human_data)
        if self.golfball_data.items:
            self.golfball_pub.publish(self.golfball_data)


    def get_class_name(self, cls):
        class_names = ["golfball", "human"]
        return class_names[cls] if cls < len(class_names) else "Unknown"




def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()




if __name__ == '__main__':
    main()









# Inverse Perspective Transform (IPT) code snippet



# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# from gcr_obj_detect.msg import YoloDetection
# from gcr_obj_detect.msg import YoloDetectionArray
# from ultralytics import YOLO
# import numpy as np
# import cv2


# class ObjectDetectionNode(Node):
#     def __init__(self):
#         super().__init__('obj_detection_node')
        
#         # Load the YOLO model
#         self.model = YOLO("/home/rajana/gazebo_ws/src/gcr_obj_detect/custom_yolo_model/new_best_model_yolo11n_V5.pt")

#         # Publishers for each object class
#         self.human_pub = self.create_publisher(YoloDetectionArray, '/human', 10)
#         self.golfball_pub = self.create_publisher(YoloDetectionArray, '/golfball', 10)
#         self.current_frame = self.create_publisher(Image, '/current_frame', 10)
#         self.bridge = CvBridge()
        
#         # Confidence threshold
#         self.confidence_threshold = 0.5
        
#         # Subscribe to the undistorted image topic
#         self.image_subscriber = self.create_subscription(
#             Image, 
#             '/camera_sensor/image_raw', 
#             self.image_callback, 
#             1
#         )

#         # Buffers to store the latest detection data
#         self.final_image=None
#         self.pub_image=None
#         self.human_data = YoloDetectionArray()
#         self.golfball_data = YoloDetectionArray()

#         # Publish data at a constant rate of 50 FPS
#         self.publish_timer = self.create_timer(0.02, self.publish_detections)
        
#         # Log initialization
#         self.get_logger().warn("YOLO Object Detection Node Initialized")


#     def image_callback(self, msg):
#         # Convert the ROS Image message to an OpenCV image
#         img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

#         # Perform Inverse Perspective Transform (IPT)
#         img = self.inverse_perspective_transform(img)

#         # Run YOLO inference
#         self.pub_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Process YOLO results and update class variables
#         results = self.model(self.pub_image)
#         self.process_results(results, img)

#     def inverse_perspective_transform(self, img):
#         # Image size
#         height, width = img.shape[:2]
#         print(height, width)

#         # Define source points (these should be manually tuned based on your camera view)
#         src_pts = np.float32([
#             [width * 0.4, height * 0.9],  # Bottom-left
#             [width * 0.6, height * 0.9],  # Bottom-right
#             [width * 0.1, height * 0.1],  # Top-left
#             [width * 0.9, height * 0.1]   # Top-right
#         ])

#         # Define destination points (where the source points should be mapped)
#         dst_pts = np.float32([
#             [width * 0.1, height],  # Bottom-left
#             [width * 0.9, height],  # Bottom-right
#             [width * 0.1, 0],       # Top-left
#             [width * 0.9, 0]        # Top-right
#         ])

#         # Compute perspective transform matrix
#         M = cv2.getPerspectiveTransform(src_pts, dst_pts)

#         # Apply inverse perspective transform
#         warped_img = cv2.warpPerspective(img, M, (width, height))

#         return warped_img



#     def process_results(self, results, img):
#         # Initialize detection arrays for each class
#         human_data = YoloDetectionArray()
#         golfball_data = YoloDetectionArray()

#         target_classes = {0: golfball_data, 1: human_data}

#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 cls = int(box.cls[0])

#                 confidence = round(float(box.conf), 2)
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 w = x2 - x1
#                 h = y2 - y1
#                 cx = x1 + w // 2
#                 cy = y1 + h // 2
                
#                 if cls not in target_classes or confidence < self.confidence_threshold:
#                     continue

#                 # Create detection message
#                 detection = YoloDetection()
#                 detection.x = cx
#                 detection.y = cy
#                 detection.width = w
#                 detection.height = h
#                 detection.confidence = confidence

#                 # Append detection to the appropriate class array
#                 target_classes[cls].items.append(detection)

#                 # Draw bounding box and label
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
#                 label = f'{self.get_class_name(cls)} {confidence}'
#                 cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

#         self.final_image = img

#         # Update buffers for each class detection
#         self.human_data = human_data
#         self.golfball_data = golfball_data


#     def publish_detections(self):
#         # Publish the yolo outputted image
#         if self.final_image is not None:
#             self.current_frame.publish(self.bridge.cv2_to_imgmsg(self.final_image))        

#         # Publish detection data for each class
#         if self.human_data.items:
#             self.human_pub.publish(self.human_data)
#         if self.golfball_data.items:
#             self.golfball_pub.publish(self.golfball_data)


#     def get_class_name(self, cls):
#         class_names = ["golfball", "human"]
#         return class_names[cls] if cls < len(class_names) else "Unknown"




# def main(args=None):
#     rclpy.init(args=args)
#     node = ObjectDetectionNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.cap.release()
#         cv2.destroyAllWindows()
#         rclpy.shutdown()




# if __name__ == '__main__':
#     main()