# One detected shape in one frame.
int32 id                       # tracker id (stable across frames)
string label                   # circle / triangle / rectangle / trapezoid / pentagon / shape
bool occluded                  # touching another shape, outline+label less trustworthy
geometry_msgs/Point center_px  # pixel centre (x, y); z unused
geometry_msgs/Point position   # (X, Y, Z) in inches, camera frame; all zero if depth unknown
geometry_msgs/Polygon outline  # traced outline, pixel coords
