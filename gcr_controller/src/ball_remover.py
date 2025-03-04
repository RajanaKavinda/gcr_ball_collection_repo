#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.msg import ContactsState

class BallRemover(Node):
    def __init__(self):
        super().__init__('ball_remover')
        
        # Create service client to delete objects
        self.client = self.create_client(DeleteEntity, '/delete_entity')
        while not self.client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("Waiting for /delete_entity service...")

        # Subscribe to contact sensor topic
        self.subscription = self.create_subscription(
            ContactsState, '/gcr/contact_sensor_topic', self.collision_callback, 10)

    def collision_callback(self, msg):
        for contact in msg.states:
            obj1 = self.extract_entity_name(contact.collision1_name)
            obj2 = self.extract_entity_name(contact.collision2_name)

            self.get_logger().info(f'Collision between {obj1} and {obj2}')

            # Check if either object is a golf ball
            for obj in [obj1, obj2]:
                if obj.startswith('golfball'):
                    self.delete_ball(obj)
    
    def extract_entity_name(self, full_name):
        """ Extract the entity name before '::' to get only the object name. """
        return full_name.split('::')[0]

    def delete_ball(self, ball_name):
        req = DeleteEntity.Request()
        req.name = ball_name
        self.client.call_async(req)
        self.get_logger().info(f'Deleting {ball_name}')

def main():
    rclpy.init()
    node = BallRemover()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
