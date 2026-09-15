#!/usr/bin/env python3
"""Part 6 benchmark: fake a tilted camera and check the plane estimate + 3D positions.

The given videos are shot straight down, so we make our own tilted frames: take a
clean top-down frame, rotate the camera by a known R (pure rotation = exact
homography x' = K R K^-1 x), and see whether the estimator recovers the tilt and
whether the corrected X/Y/Z match the ground truth (R applied to the flat-frame
positions).

    python tools/tilt_benchmark.py            # prints a markdown table
    python tools/tilt_benchmark.py --demo results/part6_tilt_demo.png
    python tools/tilt_benchmark.py --video results/part6_tilt_demo.mp4   # hard video, tilt sweeping over time
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shape_detector import ShapeDetector, draw  # noqa: E402
from shape_detector.plane import TiltedPlaneCamera, orbit_homography, rot_about_image_axis  # noqa: E402
from shape_detector.geometry import K  # noqa: E402

VIDEO = os.path.join(os.path.dirname(__file__), '..', 'PennAir 2024 App Dynamic Hard.mp4')


def clean_frame(video=VIDEO, start=1450):
    """First frame from `start` with 5 unoccluded shapes incl. a circle."""
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    det = ShapeDetector()
    while True:
        ok, f = cap.read()
        if not ok:
            raise SystemExit('no clean frame found')
        dets = det.detect(f)
        if len(dets) == 5 and all(not d.occluded for d in dets) and any(d.label == 'circle' for d in dets):
            return f, dets


def k_full(shape):
    H, W = shape[:2]
    Kf = K.copy()
    Kf[0, 2] += W / 2
    Kf[1, 2] += H / 2
    return Kf


def flat_cam():
    # what Part 4 does: flat plane, depth from the un-foreshortened diameter
    return TiltedPlaneCamera(assume_flat=True)


def run_case(frame, flat_dets, theta, axis_deg, det):
    Kf = k_full(frame.shape)
    R = rot_about_image_axis(theta, axis_deg)
    Z0 = flat_dets[0].xyz[2]
    H, t = orbit_homography(Kf, R, Z0)
    warped = cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # ground truth: flat-frame positions moved into the new camera frame
    truth = {}
    for d in flat_dets:
        P = R @ np.array(d.xyz) + t
        px = Kf @ P
        truth[d.label] = (P, px[:2] / px[2])
    n_true = R @ np.array([0, 0, 1.0])

    dets = det.detect(warped)
    cam = TiltedPlaneCamera()          # no hint: the two-fold ambiguity is settled by the prior
    cam.locate(dets, warped.shape)
    flat_copy = det.detect(warped)
    flat_cam().locate(flat_copy, warped.shape)

    est_theta = cam.tilt_deg
    est_dir = np.degrees(np.arctan2(cam.normal[1], cam.normal[0])) if cam.normal is not None else np.nan
    true_dir = np.degrees(np.arctan2(n_true[1], n_true[0]))
    dir_err = abs((est_dir - true_dir + 180) % 360 - 180) if theta > 0 else 0.0

    # match detections to truth by pixel distance, compare 3D
    def pos_err(ds):
        errs = []
        for label, (P, px) in truth.items():
            cand = [d for d in ds if d.xyz is not None]
            if not cand:
                continue
            d = min(cand, key=lambda d: np.hypot(d.center[0] - px[0], d.center[1] - px[1]))
            if np.hypot(d.center[0] - px[0], d.center[1] - px[1]) < 60:
                errs.append(np.linalg.norm(np.array(d.xyz) - P))
        return (np.mean(errs) if errs else np.nan), len(errs)

    tilt_err, n_ok = pos_err(dets)
    flat_err, _ = pos_err(flat_copy)
    if os.environ.get('TILT_DEBUG'):
        for label, (P, px) in truth.items():
            near = lambda ds: min(ds, key=lambda d: np.hypot(d.center[0] - px[0], d.center[1] - px[1]))
            a, b = near(dets), near(flat_copy)
            print(f'    {label:10s} truth {np.round(P).astype(int)}  tilt {None if a.xyz is None else np.round(a.xyz).astype(int)}'
                  f'  flat {None if b.xyz is None else np.round(b.xyz).astype(int)}')
    return dict(theta=theta, axis=axis_deg, est_theta=est_theta, dir_err=dir_err,
                tilt_err=tilt_err, flat_err=flat_err, n=n_ok, n_det=len(dets), ambiguous=cam.ambiguous,
                circle=any(d.label == 'circle' and not d.occluded for d in dets),
                warped=warped, dets=dets, cam=cam)


def make_video(out_path, seconds, video=VIDEO, start=1450):
    """Warp the real video with a tilt that sweeps in magnitude and direction, and
    run the tilt-aware pipeline on it. Truth is printed next to the estimate."""
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames = int(seconds * fps)
    det, cam = ShapeDetector(), TiltedPlaneCamera()
    writer = None
    Z0 = None
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if Z0 is None:
            d0 = det.detect(frame)
            flat_cam().locate(d0, frame.shape)
            Z0 = next((d.xyz[2] for d in d0 if d.xyz), 258.0)
            Kf = k_full(frame.shape)
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
        s = i / n_frames
        theta = 35 * np.sin(np.pi * s) ** 2          # 0 -> 35 -> 0 deg
        axis = (360 * s) % 360                        # direction sweeps once around
        R = rot_about_image_axis(theta, axis)
        H, _ = orbit_homography(Kf, R, Z0)
        warped = cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)
        dets = det.detect(warped)
        cam.locate(dets, warped.shape)
        n_true = R @ np.array([0, 0, 1.0])
        true_txt = f'true tilt {theta:.0f} deg, dir {np.degrees(np.arctan2(n_true[1], n_true[0])) % 360:.0f} deg'
        draw(warped, dets, show_3d=True, hud=cam.hud() or 'plane: (no circle)')
        cv2.putText(warped, true_txt, (15, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(warped, true_txt, (15, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(warped)
    writer.release()
    print('wrote', out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo', help='write an annotated tilted frame here')
    ap.add_argument('--demo-angle', type=float, default=30)
    ap.add_argument('--video', help='write a video where the tilt sweeps over time (true vs estimated)')
    ap.add_argument('--seconds', type=float, default=20)
    args = ap.parse_args()

    if args.video:
        make_video(args.video, args.seconds)
        return

    frame, flat_dets = clean_frame()
    flat_cam().locate(flat_dets, frame.shape)   # ground-truth flat positions
    det = ShapeDetector()

    cases = [(0, 90), (10, 90), (20, 90), (30, 90), (40, 90), (10, 0), (20, 0), (30, 0), (40, 0), (25, 45)]
    print('| true tilt | axis | est. tilt | dir err | solutions | 3D err, tilt model | 3D err, flat (Part 4) model |')
    print('|---|---|---|---|---|---|---|')
    for theta, axis in cases:
        r = run_case(frame, flat_dets, theta, axis, det)
        sign = 'n/a' if theta == 0 else ('2, took smaller' if r['ambiguous'] else '1')
        print(f"| {theta}° | {axis}° | {r['est_theta']:.1f}° | {r['dir_err']:.1f}° | {sign} | "
              f"{r['tilt_err']:.1f} in | {r['flat_err']:.1f} in |")

    if args.demo:
        r = run_case(frame, flat_dets, args.demo_angle, 90, det)
        out = draw(r['warped'].copy(), r['dets'], show_3d=True)
        cv2.putText(out, r['cam'].hud(), (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, r['cam'].hud(), (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(args.demo, out)
        print('wrote', args.demo)


if __name__ == '__main__':
    main()
