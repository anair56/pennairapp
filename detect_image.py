#!/usr/bin/env python3
"""Part 1: run the detector on a single image.

    python detect_image.py "PennAir 2024 App Static.png" -o results/static_result.png
"""

import argparse
import time

import cv2

from shape_detector import ShapeDetector, draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('-o', '--output', default='results/static_result.png')
    ap.add_argument('--mask', help='also dump the intermediate smooth-mask here (debug)')
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f'could not read {args.image}')

    det = ShapeDetector()
    t0 = time.perf_counter()
    dets = det.detect(img)
    dt = time.perf_counter() - t0

    for d in dets:
        print(f'{d.label:10s} center=({d.center[0]:.1f}, {d.center[1]:.1f})  area={d.area:.0f}px')
    print(f'{len(dets)} shapes in {dt*1000:.1f} ms')

    out = draw(img.copy(), dets)
    cv2.imwrite(args.output, out)
    if args.mask:
        cv2.imwrite(args.mask, det.last_mask)
    if args.show:
        cv2.imshow('result', out); cv2.waitKey(0)


if __name__ == '__main__':
    main()
