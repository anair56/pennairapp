"""Core shape detector.

Idea: every background we get (grass, gravel, whatever) is *textured*, while the
shapes are flat or smoothly shaded. So instead of looking for specific colours we
look for patches that are locally smooth. That makes the whole thing background
agnostic for free, and gradient-filled shapes still count as "smooth" because a
gradient has ~zero second derivative.

Pipeline per frame:
  1. downscale (speed)
  2. texture map = box-mean of |Laplacian| (after a tiny gaussian to kill codec noise)
  3. threshold at a fraction of the frame's median texture (median ~= background level)
  4. morphology clean-up -> blobs
  5. watershed from the blobs to snap outlines back to the real edges
     (step 2 leaves a "ring" of high texture around each edge, so blobs are eroded)
  6. contours -> centre, rough shape label
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Detection:
    contour: np.ndarray            # Nx1x2 int32, full-res pixel coords
    center: tuple                  # (x, y) full-res pixels
    area: float                    # full-res px^2
    label: str                     # 'circle', 'triangle', ...
    radius: float                  # min enclosing circle radius (px), used for the 3D part
    occluded: bool = False         # touching another shape -> outline/label not trustworthy
    track_id: int = -1             # filled by the tracker
    xyz: tuple = field(default=None)  # (X, Y, Z) in inches, filled by the 3D part


class ShapeDetector:
    def __init__(self, proc_width=960, blur_sigma=1.0, box=9, thresh_ratio=0.30,
                 min_area_frac=8e-4, min_solidity=0.6, flat_bg_floor=3.0):
        self.proc_width = proc_width
        self.blur_sigma = blur_sigma
        self.box = box
        self.thresh_ratio = thresh_ratio
        self.min_area_frac = min_area_frac  # min blob area as a fraction of the frame
        self.min_solidity = min_solidity    # area / convex hull area; kills ragged noise blobs
        self.flat_bg_floor = flat_bg_floor  # below this median texture we assume a plain bg
        self._open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # how far the watershed may grow a blob outwards (the edge "ring" is ~6px wide)
        self._band_k = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))  # rect = separable = fast
        self._chan_mean = np.array([[1 / 3, 1 / 3, 1 / 3]], np.float32)
        self.last_mask = None  # kept around for debugging / the report

    # ---- texture measure ------------------------------------------------
    def texture_map(self, img):
        # all uint8/int16 on purpose: the float32 version of this was 3x slower
        f = cv2.GaussianBlur(img, (0, 0), self.blur_sigma)
        lap = cv2.convertScaleAbs(cv2.Laplacian(f, cv2.CV_16S, ksize=3))  # |lap| per channel
        lap = cv2.transform(lap, self._chan_mean)                            # mean over BGR
        return cv2.blur(lap, (self.box, self.box))

    def _smooth_mask(self, img):
        """Binary mask of 'smooth' (= probably a shape) pixels."""
        tex = self.texture_map(img)
        med = _median_u8(tex)   # np.median on 0.5MP is ~3ms, a histogram is ~0.3ms
        if med > self.flat_bg_floor:
            mask = (tex < self.thresh_ratio * med)
        else:
            # plain-colour background: texture says nothing, so fall back to
            # "differs from the dominant colour". median colour ~= background.
            bg = np.median(img.reshape(-1, 3), axis=0)
            diff = np.abs(img.astype(np.float32) - bg).sum(axis=2)
            mask = diff > 40
        mask = mask.astype(np.uint8) * 255
        # drop specks, then fill tiny holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._open_k)
        return mask

    # ---- watershed refinement --------------------------------------------
    def _refine(self, img, mask):
        """Grow each blob out to the true edge using watershed. Returns label image
        where 1 = background and 2..N = shapes (and -1 on the boundaries)."""
        n, labels = cv2.connectedComponents(mask)
        if n <= 1:
            return None, 0
        sure_bg = cv2.dilate(mask, self._band_k) == 0
        markers = labels.astype(np.int32) + 1   # shapes become 2..n, unknown -> 1
        markers[markers == 1] = 0               # unknown band
        markers[sure_bg] = 1                    # confident background
        cv2.watershed(img, markers)
        return markers, n

    # ---- main entry -------------------------------------------------------
    def detect(self, frame):
        H, W = frame.shape[:2]
        scale = self.proc_width / W
        small = cv2.resize(frame, (self.proc_width, int(round(H * scale))),
                           interpolation=cv2.INTER_AREA) if scale < 1 else frame
        scale = small.shape[1] / W

        mask = self._smooth_mask(small)
        self.last_mask = mask
        markers, n = self._refine(small, mask)
        if markers is None:
            return []

        min_area = self.min_area_frac * small.shape[0] * small.shape[1]
        dets = []
        for lbl in range(2, n + 1):
            comp = (markers == lbl).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(c))
            if hull_area == 0 or area / hull_area < self.min_solidity:
                continue
            m = cv2.moments(c)
            cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']
            (_, _), r = cv2.minEnclosingCircle(c)
            inv = 1.0 / scale
            dets.append(Detection(
                contour=np.round(c * inv).astype(np.int32),
                center=(cx * inv, cy * inv),
                area=area * inv * inv,
                label=classify(c),
                radius=r * inv,
                # touching another shape OR cut off by the frame edge: outline
                # is partial, so don't trust its label/radius
                occluded=self._touches_other(markers, comp, c, lbl) or _clipped(c, small.shape),
            ))
        return dets

    @staticmethod
    def _touches_other(markers, comp, contour, lbl):
        """True if this region shares a border with another shape region (overlap)."""
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0 = max(x - 3, 0), max(y - 3, 0)
        roi_m = markers[y0:y + h + 3, x0:x + w + 3]
        roi_c = cv2.dilate(comp[y0:y + h + 3, x0:x + w + 3], None, iterations=3)
        other = (roi_m >= 2) & (roi_m != lbl)
        return bool(np.any(other & (roi_c > 0)))


def _clipped(contour, shape, margin=2):
    x, y, w, h = cv2.boundingRect(contour)
    return x <= margin or y <= margin or x + w >= shape[1] - margin or y + h >= shape[0] - margin


def _median_u8(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    return float(np.searchsorted(np.cumsum(hist), img.size / 2))


def _densify(pts, step=3.0):
    """Insert points along each polygon edge so there's one every ~step px."""
    out = []
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        k = max(int(np.linalg.norm(b - a) / step), 1)
        out.append(a + (b - a) * (np.arange(k) / k)[:, None])
    return np.concatenate(out).astype(np.float32)


def _ellipse_residual(contour):
    """Mean |r - 1| of the contour points in the best-fit ellipse's own frame.
    ~0.01 for a real ellipse, ~0.08 for a pentagon, worse for anything pointier."""
    # contours are CHAIN_APPROX_SIMPLE (corners only), and an ellipse fits a
    # rectangle's 4 corners perfectly, so resample the outline densely first
    pts = _densify(contour.reshape(-1, 2).astype(np.float32))
    (ex, ey), (w, h), ang = cv2.fitEllipse(pts)
    if w < 1 or h < 1:
        return 1.0
    pts = pts - (ex, ey)
    c, s_ = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    x = (pts[:, 0] * c + pts[:, 1] * s_) / (w / 2)
    y = (-pts[:, 0] * s_ + pts[:, 1] * c) / (h / 2)
    return float(np.mean(np.abs(np.hypot(x, y) - 1)))


def classify(contour):
    """Very rough polygon classification, just for nicer labels."""
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return 'unknown'
    if len(contour) >= 5 and _ellipse_residual(contour) < 0.04:
        return 'circle'   # or an ellipse: a circle seen at an angle (Part 6) still counts
    approx = cv2.approxPolyDP(contour, 0.045 * peri, True)  # loose enough to ignore a rounded corner
    v = len(approx)
    if v == 3:
        return 'triangle'
    if v == 4:
        # rectangle vs trapezoid: compare the two "horizontal" side lengths
        pts = approx.reshape(-1, 2).astype(np.float32)
        sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
        ratio = min(sides[0], sides[2]) / max(sides[0], sides[2])
        ratio2 = min(sides[1], sides[3]) / max(sides[1], sides[3])
        return 'rectangle' if min(ratio, ratio2) > 0.8 else 'trapezoid'
    if v == 5:
        return 'pentagon'
    return f'{v}-gon'
