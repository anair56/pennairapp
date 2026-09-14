"""Drawing helpers: outline, centre dot, label text, FPS counter."""

import cv2

OUTLINE = (0, 255, 0)
OUTLINE_OCCLUDED = (0, 200, 255)   # orange-ish: shape is touching/overlapping another
CENTER = (0, 0, 255)
TEXT = (255, 255, 255)


def _put(img, text, org, scale=0.6):
    # black-ish shadow then white text so it's readable on any background
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT, 1, cv2.LINE_AA)


def draw(frame, dets, show_3d=False, fps=None):
    out = frame  # draw in place; caller copies if it cares
    for d in dets:
        cv2.drawContours(out, [d.contour], -1, OUTLINE_OCCLUDED if d.occluded else OUTLINE, 3)
        cx, cy = int(round(d.center[0])), int(round(d.center[1]))
        cv2.circle(out, (cx, cy), 6, CENTER, -1)
        cv2.circle(out, (cx, cy), 6, (255, 255, 255), 1)
        tag = d.label if d.track_id < 0 else f'#{d.track_id} {d.label}'
        _put(out, tag, (cx + 10, cy - 8))
        if show_3d and d.xyz is not None:
            X, Y, Z = d.xyz
            _put(out, f'X={X:+.0f} Y={Y:+.0f} Z={Z:.0f} in', (cx + 10, cy + 16), 0.55)
        else:
            _put(out, f'({cx}, {cy})', (cx + 10, cy + 16), 0.55)
    if fps is not None:
        _put(out, f'{fps:.0f} fps (algo only)', (15, 30), 0.8)
    return out
