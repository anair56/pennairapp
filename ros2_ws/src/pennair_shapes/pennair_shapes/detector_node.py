"""Runs the shape detector on every incoming image and publishes the results.

Subscribes: camera/image_raw            (sensor_msgs/Image)
Publishes:  shapes/detections           (pennair_shapes_msgs/ShapeDetectionArray)
            shapes/image_annotated      (sensor_msgs/Image, optional, for rqt_image_view)
"""

import time

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Point32, Polygon
from rclpy.node import Node
from sensor_msgs.msg import Image

from pennair_shapes_msgs.msg import ShapeDetection, ShapeDetectionArray
from shape_detector import CameraModel, ShapeDetector, Tracker, draw


class DetectorNode(Node):
    def __init__(self):
        super().__init__('shape_detector')
        self.declare_parameter('image_topic', 'camera/image_raw')
        self.declare_parameter('use_3d', True)
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('proc_width', 960)

        self.detector = ShapeDetector(proc_width=self.get_parameter('proc_width').value)
        self.tracker = Tracker()
        self.use_3d = self.get_parameter('use_3d').value
        self.cam = CameraModel() if self.use_3d else None
        self.bridge = CvBridge()

        self.sub = self.create_subscription(Image, self.get_parameter('image_topic').value,
                                            self.on_image, 10)
        self.pub = self.create_publisher(ShapeDetectionArray, 'shapes/detections', 10)
        self.pub_img = None
        if self.get_parameter('publish_annotated').value:
            self.pub_img = self.create_publisher(Image, 'shapes/image_annotated', 10)

        self._n, self._t = 0, 0.0

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        t0 = time.perf_counter()
        dets = self.tracker.update(self.detector.detect(frame))
        if self.cam is not None:
            self.cam.locate(dets, frame.shape)
        dt = time.perf_counter() - t0

        out = ShapeDetectionArray()
        out.header = msg.header
        for d in dets:
            m = ShapeDetection()
            m.id = int(d.track_id)
            m.label = d.label
            m.occluded = bool(d.occluded)
            m.center_px = Point(x=float(d.center[0]), y=float(d.center[1]), z=0.0)
            if d.xyz is not None:
                m.position = Point(x=float(d.xyz[0]), y=float(d.xyz[1]), z=float(d.xyz[2]))
            m.outline = Polygon(points=[Point32(x=float(p[0]), y=float(p[1]), z=0.0)
                                        for p in d.contour.reshape(-1, 2)])
            out.detections.append(m)
        self.pub.publish(out)

        if self.pub_img is not None:
            draw(frame, dets, show_3d=self.use_3d, fps=1.0 / max(dt, 1e-6))
            img = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img.header = msg.header
            self.pub_img.publish(img)

        # casual fps log every ~3s worth of frames
        self._n += 1; self._t += dt
        if self._n % 100 == 0:
            self.get_logger().info(f'{len(dets)} shapes, algo {1000*self._t/100:.1f} ms/frame')
            self._t = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
