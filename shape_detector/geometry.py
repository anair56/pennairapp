"""Part 4: pixel centres -> (X, Y, Z) in the camera frame.

Given intrinsics K and the fact that the circle is 10 in across (radius), we get
depth from the circle's apparent size: Z = f * R / r_px. Everything sits on a
flat surface (per the prompt) so that Z applies to every shape, and then
X = u * Z / fx, Y = v * Z / fy with (u, v) measured from the principal point.

K from the prompt has cx = cy = 0, which we read as "pixel coordinates are
relative to the image centre" (that's also what the reference video shows).
"""

import numpy as np

K = np.array([[2564.3186869, 0, 0],
              [0, 2569.70273111, 0],
              [0, 0, 1]])
CIRCLE_RADIUS_IN = 10.0


class CameraModel:
    def __init__(self, K=K, circle_radius=CIRCLE_RADIUS_IN, depth_alpha=0.3):
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.R = circle_radius
        self.Z = None            # last known depth (in)
        self.alpha = depth_alpha  # EMA factor so Z doesn't jitter frame to frame

    def update_depth(self, dets):
        # pick the biggest *unoccluded* circle and use its size for depth
        # (a half-hidden circle has a bogus radius; we just hold the last Z then)
        circles = [d for d in dets if d.label == 'circle' and not d.occluded]
        if not circles:
            return self.Z
        c = max(circles, key=lambda d: d.area)
        z = self.fx * self.R / max(c.radius, 1e-6)
        self.Z = z if self.Z is None else (1 - self.alpha) * self.Z + self.alpha * z
        return self.Z

    def locate(self, dets, frame_shape):
        """Attach (X, Y, Z) in inches to every detection. Frame centre = principal point."""
        H, W = frame_shape[:2]
        Z = self.update_depth(dets)
        if Z is None:
            return dets
        for d in dets:
            u = d.center[0] - (W / 2 + self.cx)
            v = d.center[1] - (H / 2 + self.cy)
            d.xyz = (u * Z / self.fx, v * Z / self.fy, Z)
        return dets
