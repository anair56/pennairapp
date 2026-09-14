"""Tiny nearest-neighbour tracker so shapes keep a stable ID across frames.

Nothing fancy: match each new detection to the closest live track (within a
pixel radius), spawn new tracks for leftovers, drop tracks that go missing for a
few frames. Used to keep labels stable and to smooth the depth estimate.
"""

from collections import Counter, deque

import numpy as np


class Tracker:
    def __init__(self, max_dist=120, max_missing=10):
        self.max_dist = max_dist
        self.max_missing = max_missing
        self.tracks = {}     # id -> {'center': (x,y), 'missing': int, 'labels': deque}
        self._next_id = 0

    def update(self, dets):
        ids = list(self.tracks.keys())
        used = set()
        # greedy: closest pairs first
        pairs = []
        for i, d in enumerate(dets):
            for tid in ids:
                t = self.tracks[tid]
                # shapes move ~30px/frame here, so match against where we
                # expect the track to be (constant velocity), not where it was
                px = t['center'][0] + t['vel'][0]
                py = t['center'][1] + t['vel'][1]
                dist = np.hypot(d.center[0] - px, d.center[1] - py)
                if dist < self.max_dist:
                    pairs.append((dist, i, tid))
        pairs.sort()
        matched_det = set()
        for dist, i, tid in pairs:
            if i in matched_det or tid in used:
                continue
            matched_det.add(i); used.add(tid)
            dets[i].track_id = tid
            t = self.tracks[tid]
            vx = dets[i].center[0] - t['center'][0]
            vy = dets[i].center[1] - t['center'][1]
            t['vel'] = (0.5 * t['vel'][0] + 0.5 * vx, 0.5 * t['vel'][1] + 0.5 * vy)
            t['center'] = dets[i].center
            t['missing'] = 0
            if not dets[i].occluded:   # only trust the label when the shape is on its own
                t['labels'].append(dets[i].label)
            dets[i].label = _vote(t['labels'])

        for i, d in enumerate(dets):
            if i not in matched_det:
                d.track_id = self._next_id
                self.tracks[self._next_id] = {'center': d.center, 'vel': (0.0, 0.0), 'missing': 0,
                                              'labels': deque([] if d.occluded else [d.label], maxlen=30)}
                d.label = _vote(self.tracks[self._next_id]['labels'])
                self._next_id += 1

        for tid in ids:
            if tid not in used:
                t = self.tracks[tid]
                t['missing'] += 1
                # coast along the last velocity so it can be picked up again
                t['center'] = (t['center'][0] + t['vel'][0], t['center'][1] + t['vel'][1])
                if t['missing'] > self.max_missing:
                    del self.tracks[tid]
        return dets


def _vote(labels):
    """Majority label over the last ~second, ignoring the vague ones. A shape
    that's briefly half-hidden behind another one keeps its real name."""
    good = [l for l in labels if l in ('circle', 'triangle', 'rectangle', 'trapezoid', 'pentagon')]
    return Counter(good).most_common(1)[0][0] if good else 'shape'
