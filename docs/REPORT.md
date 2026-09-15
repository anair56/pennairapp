# Report

Short notes per part: what I did, what went wrong, what I changed.

## Part 1 – static image

**Approach.** My first instinct was HSV colour thresholding ("not green" = shape),
but Part 3 was already asking for background agnosticism so I went straight for
something that doesn't care about colour. The observation that carries the whole
project: the background is *textured* and the shapes are *smooth*. Grass has
strong high-frequency detail everywhere, the shapes have none.

So the detector builds a "texture map" (local mean of |Laplacian|) and keeps the
pixels where it's low. The threshold is relative to the frame's median texture,
which is essentially a measurement of the background, so it self-tunes.

**Challenges.**
- The texture window straddles the shape edge, so every blob comes out eroded by
  ~6 px (at 960 px wide) and sharp corners get rounded. Just dilating back would
  merge overlapping shapes and blur the outline. Instead I use the eroded blobs as
  watershed markers and let the watershed grow them out to the real edge – it
  stops where the gradient is strongest, which is the true boundary.
- Triangle tips still come out slightly rounded (the tip region is too thin to
  survive the texture window). The polygon approximation tolerance was loosened
  so the label logic doesn't see an extra vertex there.

Result: 5/5 shapes, outlines and centres on, correct labels
(`results/static_result.png`).

## Part 2 – video, streamed

`detect_video.py` pulls frames with `cap.read()` one at a time and never seeks or
looks ahead, so the algorithm is the same one-frame function as Part 1 plus a
little memory (tracker) that only knows the past.

**Performance.** First working version was ~30 ms/frame (right at the 30 fps
budget, no margin). Changes that mattered, in order:

| change | ms/frame |
|---|---|
| float32 texture map, numpy channel sum | ~30 |
| uint8/int16 pipeline, `cv2.transform` for the channel mean | ~16 |
| histogram median instead of `np.median` (3 ms → 0.3 ms), rectangular (separable) dilation kernel | ~11–14 |

Everything runs at 960×540 and contours are scaled back to 1080p, which costs
essentially nothing in outline accuracy. Numbers are from an Apple laptop,
single-threaded Python. 640 px wide gives ~8 ms if it ever needed to run on a
weaker board.

**Consistency.** Shapes move ~30 px/frame (up to ~120 px) so a plain
nearest-neighbour tracker mixed up IDs when shapes crossed. Two fixes:

- constant-velocity prediction: match new detections against where each track
  *should* be, and coast missing tracks along their velocity for a few frames;
- shape labels come from a majority vote over the last ~30 frames, and frames
  where the shape is touching another one or is cut off by the frame edge don't
  vote. A half-hidden pentagon looks like a trapezoid and a circle sliding in
  from the edge looks like a D; this keeps them from being misnamed.

**Overlap.** When two shapes overlap, the colour edge between them shows up in
the texture map, so the watershed naturally splits them into two regions with a
shared border. That's why overlapping shapes still get separate outlines and
centres (drawn in orange). The occluded shape's centre is the centroid of its
*visible* part, so it drifts toward the visible side. Proper occlusion handling
(Part 6) would need per-shape models and is out of scope here.

## Part 3 – background agnostic

Nothing background-specific was ever in the code, so the same script runs on the
"Hard" video with zero parameter changes. Two things did have to change though:

- **Gradient-filled shapes.** My first texture measure was local standard
  deviation. That flags a steep colour gradient (the navy→yellow pentagon) as
  texture. Switching to the Laplacian fixed it: a linear ramp has zero second
  derivative no matter how steep it is, while gravel/grass have a lot.
- **Codec noise inside gradients.** The H.264 gradients aren't perfectly linear
  (banding), which leaked through at the fine scale. A σ=1 Gaussian before the
  Laplacian and a 9×9 mean after it were enough.
- **Low-contrast edges.** The white→grey trapezoid's bottom edge is nearly the
  same brightness as the dark gravel. The watershed occasionally wanders a few
  pixels there. Limiting how far it may grow (a ~9 px band around each blob)
  keeps that bounded.
- **Plain-colour backgrounds.** Not in the test data, but "agnostic" should cover
  it: if the frame's median texture is near zero the texture trick has nothing to
  work with, so the detector falls back to "differs from the dominant colour".
  Covered by `tests/test_detector.py::test_flat_background`.
- The hard video has a thin smooth strip at its right edge which showed up as a
  false positive; a min-area and solidity (area / convex-hull area) filter removed it.

## Part 4 – 3D

Intrinsics from the prompt: `fx = 2564.32`, `fy = 2569.70`, `cx = cy = 0`.
`cx = cy = 0` is read as "pixel coordinates are measured from the image centre"
(that's also what the reference video shows), so `u = x − W/2`, `v = y − H/2`.

Depth comes from the circle, whose real radius is 10 in:

```
Z = fx · R / r_px          (r_px from cv2.minEnclosingCircle on the circle's contour)
X = u · Z / fx
Y = v · Z / fy
```

The surface is flat so that one Z is applied to every shape. Z is smoothed with
an EMA and only updated from an *unoccluded* circle (a half-hidden circle has a
bogus radius); when the circle is off-screen the last Z is held. For these
videos Z works out to ~247 in (gravel) and ~246 in (grass), i.e. about 20.5 ft,
which is a plausible camera height.
