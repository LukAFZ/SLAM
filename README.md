# imglistener

ROS 2 package for visual SLAM on a mobile robot with a Kinect RGB-D camera.
ORB keypoints are detected in the RGB stream, lifted into 3D with the
registered depth image, and matched frame-to-frame to estimate the robot's
motion via RANSAC + a closed-form 2D Kabsch alignment. Instead of a single
pose estimate, the package runs `NUM_ROBOTS` independent "virtual robot"
instances in parallel (see `config.py`), each with its own landmark map and
a per-landmark Extended Kalman Filter; the instance with the best
accumulated log-likelihood is published as the result.

## How a frame is processed

1. **`slam_node.py`** buffers the latest depth image and, once a synced RGB
   frame arrives (and the static `base_link -> kinect_depth` TF has been
   resolved), hands both frames to `VisualSLAMCore.process_frame`.
2. **`slam_core.py`** detects ORB keypoints, sorts them by response, and
   keeps only one keypoint per `GRID_SIZE`-sized pixel cell whose depth lies
   between `MIN_DEPTH` and `MAX_DEPTH` — this spreads keypoints evenly over
   the image instead of clustering on strong corners.
3. Each surviving keypoint is back-projected to 3D in the base-link frame
   (`algorithms.calculate_local_cords_from_matches`) using the pinhole
   camera model (`CU`, `CV`, `F`).
4. The current descriptors are matched against the **previous** frame's
   descriptors with a Hamming `BFMatcher`; if there are enough matches
   (`MIN_MATCHES`), `algorithms.ransac_refinement` estimates a relative
   rotation/translation/heading delta between the two frames.
5. That delta is handed to every `Robot` instance (`robot.py`). Each
   instance independently: looks up which of *its own* map landmarks are
   currently visible (`algorithms.test_only_for_visible_landmarks`), matches
   them against the current frame, applies the delta pose plus Gaussian
   noise to its own position, checks RANSAC inliers, runs an `ExtKalman`
   update per matched inlier landmark, inserts unmatched keypoints as new
   landmarks, and prunes stale ones.
6. `slam_core.py` collects each instance's summed log-likelihood, converts
   it to a normalized weight (log-sum-exp trick for numerical stability),
   and picks the instance with the highest weight as the frame's result.
7. `slam_node.py` broadcasts the winning pose as a TF (`odom -> base_link`),
   publishes it as `nav_msgs/Odometry`, and publishes the winning instance's
   landmark map as a `PointCloud2`.

## Coordinate frames

```
odom
 ├─ base_link                (published every processed RGB frame)
 │   └─ kinect_depth         (static, looked up once via tf2 at startup)
 └─ virtual_robot_<id>       (one per robot instance, for debugging in RViz)
```

## Layout

```
imglistener/
├── config.py             configurations – all tunable parameters
├── data_types.py         Coordinate, RobotOdom2D dataclasses
├── algorithms.py         RANSAC/Kabsch, frame transforms, FOV/visibility check
├── extKalman_LM.py       ExtKalman – per-landmark Extended Kalman Filter
├── map_manager.py        Landmark dataclass + MapManager (map of one robot instance)
├── robot.py              Robot – one virtual robot instance
├── slam_core.py          VisualSLAMCore – ORB pipeline + instance orchestration
├── slam_node.py          SlamNode – ROS 2 I/O, TF, publishers
├── slam_project_launch.py  launch file: node + rosbag + RViz2
└── odometry_exporter.py  standalone node: CSV comparison of SLAM/wheel/IMU odometry
```

## Module notes

**`data_types.py`**
`Coordinate(x, y, z)` and `RobotOdom2D(x, y, theta)` are slotted dataclasses
(millimeters / radians) with `+`, `-`, `*`, `/` operators, used throughout
the rest of the package instead of raw tuples or arrays.

**`algorithms.py`**
Instantiated once per consumer and caches the relevant `config` values in
`__init__` to avoid repeated attribute lookups.
- `get_kapsch_2d(P, Q)` — closed-form 2D Kabsch: centers both point sets on
  their centroids, gets the rotation angle from an `atan2` over the cross-
  and dot-products of the centered points, and returns rotation matrix,
  translation and angle.
- `ransac_refinement(P, Q)` — draws random 3-point subsets, fits a
  hypothesis with `get_kapsch_2d`, scores it by counting points under
  `RANSAC_THRESHOLD` mm error, and refits the best hypothesis on all of its
  inliers. Returns `(None, None, 0)` if fewer than 5 point pairs are given.
- `matrix_from_local_robot_to_odom_coords` / `matrix_from_odom_to_local_robot_coords`
  — rotate a vector between the local robot frame and the odom frame using
  `+theta` / `-theta` respectively.
- `test_only_for_visible_landmarks` — vectorized FOV check: transforms all
  map landmarks odom → base → kinect in one matrix multiplication, keeps
  points with `0 < z < MAX_DEPTH` (note: `MIN_DEPTH` is **not** applied
  here, only in the keypoint-selection step in `slam_core.py`) whose pinhole
  projection falls inside the image bounds, and returns their descriptors,
  global 2D positions and map indices.

**`extKalman_LM.py`**
One `ExtKalman` per landmark. The state is treated as static (`predict_state`
just propagates covariance, the mean stays put). Measurement noise `R` is
built per update in `set_R`: `sigma_R_approximation` models the depth error
as a constant `A` plus a term quadratic in depth (`B`) and uses a fixed
lateral error `S_X`, and the resulting pixel-space covariance is pushed
through the projection Jacobian and the kinect→base rotation. `P` is
initialized to `R` on a landmark's first update. `update(...)` runs one
predict/update cycle and also returns a measurement log-likelihood from
`compute_measurement_likelihood` (falls back to
`PARTICLE_FILTER_FAIL_STANDARD_ERROR` if the innovation covariance is
singular).

**`map_manager.py`**
- `initialize_map(...)` seeds the map straight from local coordinates — only
  correct while the robot pose is still `(0, 0, 0)`.
- `add_new_landmarks(...)` rotates+translates unmatched local keypoints into
  the odom frame and appends them as new landmarks.
- `clean_map(frame_index)` keeps a landmark if `seen_count > SEEN_COUNT_THRESHOLD`
  **or** it was seen within `LAST_SEEN_THRESHOLD` frames — i.e. it is only
  dropped once it is both low-quality *and* stale.
- `get_all_points_for_msg()` returns landmark positions in meters (mm/1000)
  for the `PointCloud2` publisher.
- `clone()` calls `lm.clone()` on every landmark, but `Landmark` (a plain
  dataclass) has no `clone()` method defined — this will raise an
  `AttributeError` if ever called.

**`robot.py`**
`Robot.update_robot(...)` seeds the map on the very first call, fetches its
visible landmarks, matches them against the current frame, and — if enough
matches exist and a frame-to-frame delta was found — applies that delta plus
Gaussian pose noise (`SIGMA_X`, `SIGMA_Y`, `SIGMA_THETA`), then re-checks
each match against the RANSAC threshold: inliers get an `ExtKalman` update
and their `pt_glob` is replaced with the filtered result, outliers instead
add `PARTICLE_FILTER_FAIL_STANDARD_ERROR` to the log-likelihood. Note that
`seen_count`/`last_seen` on matched landmarks are updated for *all* matches,
not only RANSAC inliers. New landmarks are added and stale ones cleaned up
afterwards, and the whole update is invalidated (`log_robot_likelihood`
forced to the fallback value) if the delta exceeds
`RANSAC_MAX_DEVIATION_DELTA` / `RANSAC_MAX_DEVIATION_THETA`.

**`slam_core.py`**
`process_frame` returns `(pose_updated, best_pose, best_map_manager)`. Two
early-exit paths (no depth frame, or ORB producing no keypoints/descriptors)
return `self.map_manager` — a `MapManager` instance owned by
`VisualSLAMCore` itself that is created in `__init__` but never populated,
rather than one of the robot instances' actual maps; worth keeping in mind
if that return value is ever consumed downstream. Once a previous frame
exists, `pose_updated` is set to `True` unconditionally after running all
robot instances, even if every instance's update failed internally (they
would simply all carry the fallback log-likelihood).

**`slam_node.py`**
Skips `FRAME_COUNTER` frames after every processed frame for stability,
looks up the static `base_link -> kinect_depth` transform once (converting
translation to millimeters and caching both the matrix and its inverse),
and — only when `process_frame` reports `pose_updated` — publishes the TF,
one debug TF per robot instance (`virtual_robot_<id>`), the
`nav_msgs/Odometry` message (with a dummy `POSE_COVARIANCE` from
`config.py`, since the pipeline doesn't produce a real pose covariance),
and the landmark point cloud.

**`odometry_exporter.py`**
A separate node that logs SLAM, wheel and IMU odometry into one CSV file for
offline comparison. On the first wheel-odometry message after a SLAM pose
is available, it fixes the heading offset and start positions of both
systems and from then on rigidly rotates+translates every wheel-odometry
reading onto the SLAM frame before logging it.

**`slam_project_launch.py`**
Starts `slam_node`, `ros2 bag play` (0.4× speed, `--clock`, filtered to the
RGB/depth/`/tf_static`/wheel/filtered/IMU topics) and `rviz2` together. The
`odometry_exporter` node is present in the file but commented out.

## Configuration reference (`config.py`)

| Group | Parameters |
|---|---|
| ROS topics | `RGB_TOPIC`, `DEPTH_TOPIC`, `PCL_TOPIC`, `ODOM_TOPIC`, `PUFFER_SIZE` |
| Instance count | `NUM_ROBOTS` (≤ 0 falls back to exactly one instance) |
| ORB detector | `ORB_NFEATURES`, `ORB_PATCH_SIZE`, `ORB_EDGE_THRESHOLD`, `ORB_SCALE_FACTOR`, `ORB_NLEVELS`, `ORB_WTA_K`, `ORB_FIRST_LEVEL`, `ORB_FAST_THRESHOLD` |
| Camera intrinsics | `CU`, `CV`, `F`, `KINECT_WIDTH`, `KINECT_HEIGHT` |
| Depth filtering | `MIN_DEPTH`, `MAX_DEPTH` (mm) |
| Keypoint spacing | `GRID_SIZE` (px) |
| RANSAC | `RANSAC_ITERATIONS`, `RANSAC_THRESHOLD` (mm), `RANSAC_MAX_DEVIATION_DELTA` (mm), `RANSAC_MAX_DEVIATION_THETA` (rad) |
| Landmark lifecycle | `SEEN_COUNT_THRESHOLD`, `LAST_SEEN_THRESHOLD`, `MIN_MATCHES` |
| Frame skipping | `FRAME_COUNTER` |
| Pose noise (EKF sim.) | `SIGMA_X`, `SIGMA_Y`, `SIGMA_THETA` |
| Depth-noise model | `A`, `B`, `S_X` |
| Misc | `PARTICLE_FILTER_FAIL_STANDARD_ERROR`, `POSE_COVARIANCE` |

## ROS interfaces

| Node | Topic | Type | Direction |
|---|---|---|---|
| `slam_node` | `RGB_TOPIC` | `sensor_msgs/Image` | in |
| `slam_node` | `DEPTH_TOPIC` | `sensor_msgs/Image` | in |
| `slam_node` | `PCL_TOPIC` | `sensor_msgs/PointCloud2` | out — winning instance's landmark map |
| `slam_node` | `ODOM_TOPIC` | `nav_msgs/Odometry` | out — winning instance's pose |
| `slam_node` | `odom → base_link`, `odom → virtual_robot_<id>` | TF | out |
| `odometry_exporter` | `/serf01/odometry/project_slam` | `nav_msgs/Odometry` | in |
| `odometry_exporter` | `/serf01/odometry/wheel` | `nav_msgs/Odometry` | in |
| `odometry_exporter` | `/serf01/odometry/imu` | `sensor_msgs/Imu` | in |

## Dependencies

`rclpy`, `std_msgs`, `sensor_msgs`, `sensor_msgs_py`, `nav_msgs`,
`geometry_msgs`, `tf2_ros`, `cv_bridge`, `opencv-python`, `numpy`, `scipy`


## Build & run

```bash
cd ~/ros2_ws
colcon build --packages-select imglistener
source install/setup.bash
```

Full pipeline (node + rosbag playback + RViz2):

```bash
ros2 launch imglistener slam_project_launch.py
```

Individually:

```bash
ros2 run imglistener slam_node
ros2 run imglistener odometry_exporter   # optional, offline odometry comparison
```

> The rosbag path and RViz config path in `slam_project_launch.py` are built
> relative to the `imglistener` package share directory — adjust them if
> your workspace layout differs.


