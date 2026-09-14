#!/usr/bin/env python3
"""Parts 2-4: run the detector on a video, one frame at a time (streamed).

    python detect_video.py "PennAir 2024 App Dynamic.mp4" -o results/dynamic_result.mp4
    python detect_video.py "PennAir 2024 App Dynamic Hard.mp4" -o results/hard_result.mp4
    python detect_video.py "PennAir 2024 App Dynamic.mp4" --3d -o results/dynamic_3d.mp4

Nothing here peeks ahead in the video: each frame goes through detect() on its
own, plus a small tracker that only remembers the previous frames.
"""

import argparse
import time

import cv2

from shape_detector import ShapeDetector, Tracker, CameraModel, draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('-o', '--output', help='write annotated video here (mp4)')
    ap.add_argument('--3d', dest='three_d', action='store_true', help='Part 4: print X/Y/Z in inches')
    ap.add_argument('--show', action='store_true', help='display live window')
    ap.add_argument('--max-frames', type=int, default=0, help='stop early (debug)')
    ap.add_argument('--proc-width', type=int, default=960, help='internal processing width')
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f'could not open {args.video}')
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (W, H))

    det = ShapeDetector(proc_width=args.proc_width)
    tracker = Tracker()
    cam = CameraModel() if args.three_d else None

    n, total_algo = 0, 0.0
    fps_smooth = None
    while True:
        ok, frame = cap.read()   # <- the "stream": one frame at a time
        if not ok:
            break
        t0 = time.perf_counter()
        dets = det.detect(frame)
        dets = tracker.update(dets)
        if cam is not None:
            cam.locate(dets, frame.shape)
        dt = time.perf_counter() - t0
        total_algo += dt
        inst = 1.0 / max(dt, 1e-6)
        fps_smooth = inst if fps_smooth is None else 0.9 * fps_smooth + 0.1 * inst

        draw(frame, dets, show_3d=args.three_d, fps=fps_smooth)
        if writer:
            writer.write(frame)
        if args.show:
            cv2.imshow('shapes', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        n += 1
        if n % 100 == 0:
            print(f'frame {n}: {len(dets)} shapes, algo {1000*dt:.1f} ms', flush=True)
        if args.max_frames and n >= args.max_frames:
            break

    cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f'{n} frames, mean algo time {1000*total_algo/max(n,1):.1f} ms '
          f'({n/max(total_algo,1e-6):.0f} fps), video is {fps_in:.1f} fps')


if __name__ == '__main__':
    main()
