# supereight Coordiname Frames

The following coordinate frames are used within supereight. Their units are
metres.

### World frame (W)

### Map frame (M)

Its origin is located at the outer corner of the octree's `[0 0 0]` voxel. The
axes are aligned with the octree so that the whole octree is located in the
positive octant of the frame (x, y and z coordinates are all positive).

### Camera frame (C)

It is located at the optical center of the sensor producing the depth images
that are integrated into the map. The x, y and z axes should point right, down
and forward respectively.

### Body frame (B)

Ground truth poses are expressed as transforms from the body frame to the world
frame. It is used to account for cases where the ground truth poses do not
refer to the sensor's optical center.



## Frame transforms

Homogeneous transforms between frames A and B are written as `Eigen::Matrix4f
T_AB`, translations as `Eigen::Vector4f t_AB` and rotations as `Eigen::Matrix3f
C_AB`.

- Constant transforms
  - `Eigen::Matrix4f T_MW` The world frame expressed in the map frame.
    Currently only translations are supported.
  - `Eigen::Matrix4f T_BC` The camera frame expressed in the body frame. Used
    to convert the ground truth poses `T_WB` to the camera frame. Set to
    identity by default assuming the camera and body frames are the same.
- Time-varying transforms
  - `Eigen::Matrix4f T_WB` The ground truth pose (position and orientation).

