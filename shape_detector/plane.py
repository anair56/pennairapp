"""Part 6: ground-plane tilt from the circle's ellipse.

Part 4 assumes the camera looks straight down. If the aircraft banks/pitches the
circle projects as an ellipse: minor/major axis ratio = cos(tilt) and the minor
axis points along the direction the plane recedes. From that we get the plane's
normal, then every shape's (X, Y, Z) is the intersection of its pixel ray with
that plane instead of a fixed depth.

Two honest caveats, both handled explicitly here:
  * pose from one circle is two-fold ambiguous ("tilted toward" vs "away"; off
    axis the two solutions differ in magnitude too). We enumerate both and pick
    by `expected_dir_deg` (an IMU on the aircraft) or, failing that, the smaller
    tilt. Above ~30 deg the wrong one also fits visibly worse, so it drops out.
  * the challenge's "circle" asset is ~4% taller than wide (`circle_aspect`,
    measured on a top-down frame), so the model fits an ellipse of that shape
    lying on the plane, not a circle. cos() is flat near 1, so tilts under
    ~8 deg are reported as 0.
"""

import cv2
import numpy as np

from .geometry import K, CIRCLE_RADIUS_IN


class TiltedPlaneCamera:
    def __init__(self, K=K, circle_radius=CIRCLE_RADIUS_IN, circle_aspect=0.961,
                 flat_cost=1e-5, expected_dir_deg=None, alpha=0.3, assume_flat=False):
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.R = circle_radius
        self.circle_aspect = circle_aspect  # width/height of the asset when seen top-down
        self.flat_cost = flat_cost          # if a flat plane fits the ellipse this well, call it flat (~8 deg dead zone)
        self.expected_dir_deg = expected_dir_deg  # hint (e.g. IMU) for the toward/away ambiguity
        self.alpha = alpha
        self.assume_flat = assume_flat      # Part 4 behaviour (depth from the major axis), for comparisons
        # smoothed state
        self.normal = None      # unit plane normal in camera coords (z forward)
        self.C = None           # a point on the plane (the circle centre), inches
        self.tilt_deg = 0.0
        self.tilt_dir_deg = 0.0
        self.Z = None           # depth of the circle centre, for the HUD
        self.ambiguous = False  # last update had two plausible solutions (picked by hint/prior)

    # ---- ellipse -> plane normal ------------------------------------------
    def _observe(self, contour, W, H):
        """Second-order moments of the circle region in normalised camera coords,
        plus the circle centre's normalised position."""
        pts = contour.reshape(-1, 2).astype(np.float32)
        m = cv2.moments(pts)
        ex, ey = m['m10'] / m['m00'], m['m01'] / m['m00']
        cov = np.array([[m['mu20'], m['mu11']], [m['mu11'], m['mu02']]]) / m['m00']
        D = np.diag([1 / self.fx, 1 / self.fy])
        cov = D @ cov @ D
        xc = np.array([(ex - (W / 2 + self.cx)) / self.fx, (ey - (H / 2 + self.cy)) / self.fy])
        return cov, xc

    def _predict(self, n, xc):
        """Shape of the image ellipse for plane normal(s) n and a circle at xc.
        Local projection Jacobian A = [I | -xc]; image covariance ~ A S A^T where
        S is the asset's covariance on the plane. The asset is 1/circle_aspect
        longer along the plane's y axis (taken as the camera y axis projected onto
        the plane, i.e. no yaw), so S = b1 b1^T + b2 b2^T / aspect^2.
        n may be (3,) or (N, 3); returns (2,2) or (N,2,2)."""
        n = np.atleast_2d(n)
        A = np.array([[1, 0, -xc[0]], [0, 1, -xc[1]]])
        y = np.array([0.0, 1.0, 0.0])
        b2 = y[None] - n[:, 1:2] * n                   # y projected onto the plane
        b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
        b1 = np.cross(b2, n)
        Ab1, Ab2 = b1 @ A.T, b2 @ A.T                  # (N, 2)
        k2 = 1.0 / (self.circle_aspect ** 2)
        return Ab1[:, :, None] * Ab1[:, None, :] + k2 * Ab2[:, :, None] * Ab2[:, None, :]

    def _cost(self, n, cov, xc):
        p = self._predict(n, xc)[0]
        return float(((p / np.trace(p) - cov / np.trace(cov)) ** 2).sum())

    def _solve(self, cov, xc):
        """Grid-search plane normals whose predicted ellipse matches the observed
        one. Returns the low-cost local minima as a list of (cost, n), best first.
        There are generically two: pose from a single circle is ambiguous (the two
        circular sections of the back-projected cone). On-axis they're mirror
        images; off-axis the partner has a larger tilt in ~the opposite direction."""
        th = np.radians(np.arange(0, 75, 1.0))
        ps = np.radians(np.arange(0, 360, 3.0))
        T, P = np.meshgrid(th, ps, indexing='ij')
        n = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1).reshape(-1, 3)
        pred = self._predict(n, xc)
        pred /= np.trace(pred, axis1=1, axis2=2)[:, None, None]
        obs = cov / np.trace(cov)
        cost = ((pred - obs) ** 2).sum(axis=(1, 2)).reshape(T.shape)
        # local minima on the grid (psi is periodic, theta is not)
        padded = np.pad(cost, ((1, 1), (0, 0)), constant_values=np.inf)
        is_min = np.ones_like(cost, bool)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    is_min &= cost <= np.roll(padded, dj, axis=1)[1 + di:1 + di + cost.shape[0]]
        best = float(cost.min())
        idx = np.argwhere(is_min & (cost <= max(10 * best, 1e-6)))
        cands = sorted((float(cost[i, j]), n.reshape(T.shape + (3,))[i, j]) for i, j in idx)
        return cands[:2]

    def update(self, dets, frame_shape):
        """Refresh plane estimate from the best visible circle. Returns True if updated."""
        H, W = frame_shape[:2]
        circles = [d for d in dets if d.label == 'circle' and not d.occluded and len(d.contour) >= 5]
        if not circles:
            return False
        cov, xc = self._observe(max(circles, key=lambda d: d.area).contour, W, H)
        evals = np.linalg.eigvalsh(cov)
        self.ambiguous = False
        flat_n = np.array([0.0, 0.0, 1.0])
        if self.assume_flat or self._cost(flat_n, cov, xc) < self.flat_cost:
            n = flat_n                          # a flat plane already explains the shape (within noise)
        else:
            cands = self._solve(cov, xc)
            self.ambiguous = len(cands) > 1
            if self.ambiguous and self.expected_dir_deg is not None:
                # outside hint (IMU): take the candidate whose recession direction agrees
                e = np.radians(self.expected_dir_deg)
                n = max(cands, key=lambda c: c[1][0] * np.cos(e) + c[1][1] * np.sin(e))[1]
            elif self.ambiguous:
                n = max(cands, key=lambda c: c[1][2])[1]   # prior: the smaller tilt
            else:
                n = cands[0][1]
        # depth of the circle centre from the ellipse size: cov = (R/Zc)^2/4 * P(n)
        if self.assume_flat:
            # major axis = un-foreshortened diameter (the asset's long axis, R/aspect)
            Zc = (self.R / self.circle_aspect) / (2 * np.sqrt(evals[1]))
        else:
            P = self._predict(n, xc)[0]
            Zc = self.R * np.sqrt(np.trace(P) / (4 * np.trace(cov)))
        C = Zc * np.array([xc[0], xc[1], 1.0])
        if self.normal is None:
            self.normal, self.C = n, C
        else:
            self.normal = (1 - self.alpha) * self.normal + self.alpha * n
            self.normal /= np.linalg.norm(self.normal)
            self.C = (1 - self.alpha) * self.C + self.alpha * C
        self.Z = float(self.C[2])
        self.tilt_deg = float(np.degrees(np.arccos(np.clip(self.normal[2], -1, 1))))
        self.tilt_dir_deg = float(np.degrees(np.arctan2(self.normal[1], self.normal[0]))) % 360
        return True

    # ---- pixel -> 3D on the plane -----------------------------------------
    def locate(self, dets, frame_shape):
        H, W = frame_shape[:2]
        self.update(dets, frame_shape)
        if self.normal is None:
            return dets
        n, nC = self.normal, float(self.normal @ self.C)
        for d in dets:
            r = np.array([(d.center[0] - (W / 2 + self.cx)) / self.fx,
                          (d.center[1] - (H / 2 + self.cy)) / self.fy, 1.0])
            denom = float(n @ r)
            if denom <= 1e-6:        # ray parallel to / behind the plane, skip
                continue
            P = (nC / denom) * r
            d.xyz = (float(P[0]), float(P[1]), float(P[2]))
        return dets

    def hud(self):
        if self.normal is None:
            return None
        if self.tilt_deg < 0.5:
            return 'plane: flat'
        amb = ' (ambiguous, took smaller)' if self.ambiguous and self.expected_dir_deg is None else ''
        return f'plane tilt {self.tilt_deg:.0f} deg, dir {self.tilt_dir_deg:.0f} deg{amb}'


def orbit_homography(K_full, R, Z0):
    """Image warp for tilting the camera by R while keeping the ground point that
    was on the optical axis (at depth Z0) centred, i.e. orbiting around it.
    For a plane this is an exact homography: H = K (R + t n^T / d) K^-1 with
    n = (0,0,1), d = Z0 and t chosen so the centre point stays put.
    Returns (H, t) so callers can build ground truth as P' = R P + t."""
    n = np.array([0.0, 0.0, 1.0])
    C0 = np.array([0.0, 0.0, Z0])
    t = C0 - R @ C0
    H = K_full @ (R + np.outer(t, n) / Z0) @ np.linalg.inv(K_full)
    return H, t


def rot_about_image_axis(theta_deg, axis_deg):
    """Rotation by theta about an axis lying in the image plane at angle axis_deg
    (0 = x axis / roll-like, 90 = y axis / pitch-like)."""
    t, p = np.radians(theta_deg), np.radians(axis_deg)
    axis = np.array([np.cos(p), np.sin(p), 0.0])
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * Kx + (1 - np.cos(t)) * (Kx @ Kx)
