"""Quick sanity checks. Run with:  python -m pytest tests/   (or just python tests/test_detector.py)"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shape_detector import ShapeDetector  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), '..')
STATIC = os.path.join(ROOT, 'PennAir 2024 App Static.png')
EXPECTED = {'circle', 'triangle', 'rectangle', 'trapezoid', 'pentagon'}


def test_static_image():
    img = cv2.imread(STATIC)
    dets = ShapeDetector().detect(img)
    assert len(dets) == 5
    assert {d.label for d in dets} == EXPECTED


def test_flat_background():
    # same shapes, but paste them onto a plain grey background: the texture
    # trick has nothing to work with, so the colour fallback has to kick in
    img = cv2.imread(STATIC)
    det = ShapeDetector()
    dets = det.detect(img)
    shapes = np.zeros(img.shape[:2], np.uint8)
    for d in dets:
        cv2.drawContours(shapes, [d.contour], -1, 255, -1)
    flat = np.full_like(img, 120)
    flat[shapes > 0] = img[shapes > 0]
    dets2 = det.detect(flat)
    assert len(dets2) == 5
    assert {d.label for d in dets2} == EXPECTED


def test_no_shapes():
    noise = np.random.default_rng(0).integers(0, 255, (540, 960, 3), np.uint8)
    assert ShapeDetector().detect(noise) == []


if __name__ == '__main__':
    test_static_image(); test_flat_background(); test_no_shapes()
    print('all good')
