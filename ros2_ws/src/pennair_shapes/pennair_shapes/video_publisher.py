"""Streams a video file as sensor_msgs/Image, one frame per timer tick.

This stands in for the aircraft camera: downstream nodes only ever see the
current frame, never the whole file.
"""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        self.declare_parameter('video_path', '')
        self.declare_parameter('fps', 0.0)        # 0 -> use the file's own fps
        self.declare_parameter('loop', True)
        self.declare_parameter('topic', 'camera/image_raw')

        path = self.get_parameter('video_path').value
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f'could not open video: {path!r}')
        fps = self.get_parameter('fps').value or self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.loop = self.get_parameter('loop').value

        self.pub = self.create_publisher(Image, self.get_parameter('topic').value, 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(1.0 / fps, self.tick)
        self.get_logger().info(f'streaming {path} at {fps:.1f} fps')

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            if not self.loop:
                self.get_logger().info('end of video')
                self.timer.cancel()
                return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind and keep going
            ok, frame = self.cap.read()
            if not ok:
                return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
