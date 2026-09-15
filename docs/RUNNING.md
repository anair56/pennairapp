# Running the code

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Part 1
python detect_image.py "PennAir 2024 App Static.png" -o results/static_result.png

# Part 2 / 3  (add --show for a live window, q to quit)
python detect_video.py "PennAir 2024 App Dynamic.mp4"      -o results/part2_dynamic.mp4
python detect_video.py "PennAir 2024 App Dynamic Hard.mp4" -o results/part3_dynamic_hard.mp4

# Part 4
python detect_video.py "PennAir 2024 App Dynamic Hard.mp4" --3d -o results/part4_dynamic_hard_3d.mp4

# Part 6: tilt-aware 3D (the given videos are flat, so the HUD just says "plane: flat")
python detect_video.py "PennAir 2024 App Dynamic Hard.mp4" --tilt -o results/part6_hard_tilt.mp4

# Part 6 benchmark on synthetically tilted frames: markdown table, a still, and a sweep video
python tools/tilt_benchmark.py
python tools/tilt_benchmark.py --demo results/part6_tilt_demo.png
python tools/tilt_benchmark.py --video results/part6_tilt_demo.mp4

# sanity tests
python tests/test_detector.py
python tests/test_tilt.py
```

`detect_video.py` reads frames one at a time and never looks ahead, so it behaves
like a live camera feed. The videos it writes are raw `mp4v`; the ones in
`results/` were re-encoded to H.264 with ffmpeg so they play in a browser.

`PennAir 2024 App Dynamic.mp4` is 102 MB, over GitHub's file limit, so it's not
in the repo; drop it in the project root from the challenge doc.

## Part 5 – ROS2

Two nodes and a launch file live in `ros2_ws/src`:

- `pennair_shapes_msgs` – `ShapeDetection` / `ShapeDetectionArray` messages (id, label, pixel centre, 3D position, outline polygon)
- `pennair_shapes`
  - `video_publisher` – streams a video file as `sensor_msgs/Image` on `camera/image_raw`
  - `detector_node` – runs the same `shape_detector` package on each image, publishes `shapes/detections` and an annotated image on `shapes/image_annotated`
  - `launch/shapes.launch.py` – runs both

```bash
# Ubuntu 24.04 + ROS 2 Jazzy (macOS: Ubuntu VM via UTM)
sudo apt install ros-jazzy-ros-base ros-jazzy-cv-bridge ros-jazzy-rqt-image-view ros-dev-tools python3-opencv python3-numpy
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch pennair_shapes shapes.launch.py video:=/abs/path/PennAir\ 2024\ App\ Dynamic\ Hard.mp4 use_3d:=true

# in another terminal
ros2 topic echo /shapes/detections
ros2 run rqt_image_view rqt_image_view /shapes/image_annotated
```

The ROS package pulls in the algorithm through a symlink
(`ros2_ws/src/pennair_shapes/shape_detector → ../../../shape_detector`), so the
nodes run exactly the same code as the scripts above.

## Layout

```
shape_detector/     the algorithm (detector, tracker, 3D geometry, tilted plane, drawing)
detect_image.py     Part 1 CLI
detect_video.py     Parts 2–4 (and 6 with --tilt) CLI
tools/              Part 6 synthetic-tilt benchmark
tests/              sanity tests
results/            processed image / videos / gifs
docs/REPORT.md      full write-up
ros2_ws/            Part 5
```
