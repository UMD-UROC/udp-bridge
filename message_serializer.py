import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message

from cdcl_umd_msgs.msg import TargetBoxArray
from sensor_msgs.msg import Image

import json

FILE_DEST = "message.bin"
TOPIC = '/target_locations'

class MessageSerializer(Node):

    def __init__(self):
        super().__init__('message_serializer')
        self.subscription = self.create_subscription(
            TargetBoxArray,
            TOPIC,
            self.listener_callback,
            1)
        self.subscription  # prevent unused variable warning
        self.get_logger().info(f"Listening on {TOPIC}")

    def listener_callback(self, msg):
        self.get_logger().info(f"Heard message from {TOPIC}, serializing and writing to {FILE_DEST}")
        msg.source_img = Image()
        binary_data = serialize_message(msg)
        with open(FILE_DEST, "wb") as f:
            f.write(binary_data)
        self.get_logger().info(f"Wrote file")

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MessageSerializer()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()