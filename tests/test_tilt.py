"""Part 6: tilt estimation on synthetically rotated frames.  python tests/test_tilt.py"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
from shape_detector import ShapeDetector  # noqa: E402
from tilt_benchmark import clean_frame, run_case, flat_cam  # noqa: E402


def _setup():
    frame, flat_dets = clean_frame()
    flat_cam().locate(flat_dets, frame.shape)
    return frame, flat_dets, ShapeDetector()


def test_flat_frame_reads_flat():
    frame, flat_dets, det = _setup()
    r = run_case(frame, flat_dets, 0, 90, det)
    assert r['est_theta'] < 1.0


def test_tilt_recovered():
    frame, flat_dets, det = _setup()
    for theta, axis in [(20, 90), (30, 0), (25, 45)]:
        r = run_case(frame, flat_dets, theta, axis, det)
        assert abs(r['est_theta'] - theta) < 4, (theta, axis, r['est_theta'])
        assert r['dir_err'] < 10, (theta, axis, r['dir_err'])
        # tilt-aware 3D should beat the flat model clearly
        assert r['tilt_err'] < 0.5 * r['flat_err'], (theta, axis, r['tilt_err'], r['flat_err'])


if __name__ == '__main__':
    test_flat_frame_reads_flat(); test_tilt_recovered()
    print('all good')
