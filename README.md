# Realsense-IsaacROS

This repository is the reproducible workspace for:

- Intel RealSense SDK (`librealsense`) on the host
- `realsense-ros` on the host
- Isaac ROS NITROS and image processing inside the Isaac ROS dev container
- The custom packages in this repo:
  - `realsense_benchmark`
  - `floor_box_perception`

The goal is simple: a new team member should be able to follow this document top to bottom, install everything once, then copy/paste the validation commands to confirm the stack is working.

## 1. Assumptions

This guide assumes:

- Ubuntu 22.04
- ROS 2 Humble already installed at `/opt/ros/humble`
- Docker is installed
- One Intel RealSense D405 and one Intel RealSense D435i are available
- The camera drivers run on the host
- Isaac ROS runs inside the Isaac ROS dev container

Important split:

- Host workspace: `~/ros2_ws`
- Isaac ROS workspace: `~/workspaces/isaac_ros-dev`
- Repository path after clone: `~/workspaces/isaac_ros-dev/src/Realsense-IsaacROS`

Throughout this README:

- "host terminal" means your normal machine terminal
- "container terminal" means a shell inside the Isaac ROS dev container

## 2. One-Time Host Setup

### 2.1 Set workspace variables

Run this in a host terminal:

```bash
export ROS2_WS=$HOME/ros2_ws
export ISAAC_ROS_WS=$HOME/workspaces/isaac_ros-dev
export REPO_ROOT=$ISAAC_ROS_WS/src/Realsense-IsaacROS
```

### 2.2 Install host dependencies

```bash
sudo apt update
sudo apt install -y \
  git cmake build-essential pkg-config \
  libssl-dev libusb-1.0-0-dev libgtk-3-dev \
  libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
  libudev-dev python3-dev python3-pip
```

### 2.3 Install the RealSense udev rule

```bash
cd ~
wget https://raw.githubusercontent.com/IntelRealSense/librealsense/v2.55.1/config/99-realsense-libusb.rules
sudo mv 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 2.4 Create the host ROS 2 workspace

```bash
mkdir -p ${ROS2_WS}/src
cd ${ROS2_WS}
```

### 2.5 Clone and build `librealsense` on the host

Version used in this repo: `v2.55.1`

```bash
cd ${ROS2_WS}
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.55.1
mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_EXAMPLES=true \
  -DBUILD_GRAPHICAL_EXAMPLES=true \
  -DBUILD_UNIT_TESTS=false \
  -DBUILD_PYTHON_BINDINGS=false
make -j"$(nproc)"
sudo make install
sudo ldconfig
```

Notes:

- `FORCE_RSUSB_BACKEND=ON` is required here.
- `BUILD_UNIT_TESTS=false` avoids unnecessary test build issues.

### 2.6 Clone and build `realsense-ros` on the host

Do not build `realsense-ros` inside the Isaac ROS container for this workflow.

```bash
cd ${ROS2_WS}/src
git clone -b ros2-master https://github.com/IntelRealSense/realsense-ros.git

cd ${ROS2_WS}
source /opt/ros/humble/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select realsense2_camera_msgs
source ${ROS2_WS}/install/setup.bash
colcon build --symlink-install --packages-select realsense2_camera
source ${ROS2_WS}/install/setup.bash
```

### 2.7 Confirm the host install works

Test the SDK:

```bash
realsense-viewer
```

Confirm the ROS 2 driver is visible:

```bash
source /opt/ros/humble/setup.bash
source ${ROS2_WS}/install/setup.bash
ros2 pkg list | grep realsense2_camera
```

## 3. One-Time Isaac ROS Workspace Setup

### 3.1 Prepare the Isaac ROS workspace

Run this on the host:

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
sudo apt-get install -y git-lfs
git lfs install --skip-repo

mkdir -p ${ISAAC_ROS_WS}/src
```

If Docker access is not already configured for your user:

```bash
sudo usermod -aG docker $USER
newgrp docker
groups
```

### 3.2 Clone this repository with submodules

Run this on the host:

```bash
cd ${ISAAC_ROS_WS}/src
git clone --recurse-submodules https://github.com/DhiaNeifar/Realsense-IsaacROS.git
cd Realsense-IsaacROS
git submodule update --init --recursive
```

This repo includes these Isaac ROS submodules:

- `isaac_ros_common`
- `isaac_ros_nitros`
- `isaac_ros_image_pipeline`

### 3.3 Configure the Isaac ROS dev container

Run this on the host:

```bash
export REPO_ROOT=${ISAAC_ROS_WS}/src/Realsense-IsaacROS

cd ${REPO_ROOT}/isaac_ros_common/scripts
touch .isaac_ros_common-config
echo CONFIG_IMAGE_KEY=ros2_humble.realsense > .isaac_ros_common-config
```

### 3.4 Start the Isaac ROS dev container

Run this on the host:

```bash
cd ${REPO_ROOT}/isaac_ros_common
./scripts/run_dev.sh -d ${ISAAC_ROS_WS}
```

Inside the container, the workspace root is:

```bash
/workspaces/isaac_ros-dev
```

### 3.5 Install dependencies inside the container

Run this in a container terminal:

```bash
export ISAAC_ROS_WS=/workspaces/isaac_ros-dev
export REPO_ROOT=${ISAAC_ROS_WS}/src/Realsense-IsaacROS

cd ${ISAAC_ROS_WS}
source /opt/ros/humble/setup.bash

sudo apt-get update
rosdep update
rosdep install --from-paths src --ignore-src -r -y
sudo apt-get install -y python3-opencv python3-numpy python3-matplotlib
```

If `apt` metadata is stale and `rosdep` fails:

```bash
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*
sudo apt-get update
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

### 3.6 Build the Isaac ROS and custom packages inside the container

Run this in a container terminal:

```bash
export ISAAC_ROS_WS=/workspaces/isaac_ros-dev
export REPO_ROOT=${ISAAC_ROS_WS}/src/Realsense-IsaacROS

cd ${ISAAC_ROS_WS}
source /opt/ros/humble/setup.bash

colcon build --packages-up-to isaac_ros_nitros --symlink-install --cmake-args -DBUILD_TESTING=OFF
source ${ISAAC_ROS_WS}/install/setup.bash

colcon build --packages-up-to isaac_ros_image_proc --symlink-install --cmake-args -DBUILD_TESTING=OFF
source ${ISAAC_ROS_WS}/install/setup.bash

colcon build --packages-select realsense_benchmark floor_box_perception --symlink-install
source ${ISAAC_ROS_WS}/install/setup.bash
```

Why the second Isaac ROS build matters:

- `RectifyNode` comes from `isaac_ros_image_proc`
- building only up to `isaac_ros_nitros` is not enough for the NITROS rectification test

Verify the packages are available:

```bash
ros2 pkg list | grep -E 'isaac_ros_image_proc|realsense_benchmark|floor_box_perception'
ros2 component types | grep -i Rectify
```

## 4. Export Camera Serials Instead of Hard-Coding Them

Do this in every host terminal where you will launch a RealSense camera:

```bash
export ROS2_WS=$HOME/ros2_ws
source /opt/ros/humble/setup.bash
source ${ROS2_WS}/install/setup.bash

eval "$(
  rs-enumerate-devices | awk -F': ' '
    /^[[:space:]]*Name[[:space:]]*:/ {
      name=$2
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
    }
    /^[[:space:]]*Serial Number[[:space:]]*:/ {
      serial=$2
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", serial)
      if (name ~ /D405/) print "export D405_SERIAL=_" serial
      if (name ~ /D435I|D435i/) print "export D435I_SERIAL=_" serial
    }
  '
)"

printf 'D405_SERIAL=%s\nD435I_SERIAL=%s\n' "$D405_SERIAL" "$D435I_SERIAL"
```

If either value is empty, inspect the raw device list and export the variable manually:

```bash
rs-enumerate-devices
```

## 5. Daily Runtime Commands

### 5.1 Start the Isaac ROS dev container

Host terminal:

```bash
export ISAAC_ROS_WS=$HOME/workspaces/isaac_ros-dev
export REPO_ROOT=$ISAAC_ROS_WS/src/Realsense-IsaacROS

cd ${REPO_ROOT}/isaac_ros_common
./scripts/run_dev.sh -d ${ISAAC_ROS_WS}
```

Container terminal setup:

```bash
export ISAAC_ROS_WS=/workspaces/isaac_ros-dev
export REPO_ROOT=${ISAAC_ROS_WS}/src/Realsense-IsaacROS

cd ${ISAAC_ROS_WS}
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### 5.2 Run both cameras and verify the streams in RViz2

Host terminal 1, D405:

```bash
export ROS2_WS=$HOME/ros2_ws
source /opt/ros/humble/setup.bash
source ${ROS2_WS}/install/setup.bash

eval "$(
  rs-enumerate-devices | awk -F': ' '
    /^[[:space:]]*Name[[:space:]]*:/ { name=$2; gsub(/^[[:space:]]+|[[:space:]]+$/, "", name) }
    /^[[:space:]]*Serial Number[[:space:]]*:/ {
      serial=$2
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", serial)
      if (name ~ /D405/) print "export D405_SERIAL=_" serial
      if (name ~ /D435I|D435i/) print "export D435I_SERIAL=_" serial
    }
  '
)"

ros2 run realsense2_camera realsense2_camera_node --ros-args \
  -r __ns:=/d405 \
  -p camera_name:=d405 \
  -p serial_no:=${D405_SERIAL} \
  -p enable_color:=true \
  -p enable_depth:=true \
  -p enable_infra1:=false \
  -p enable_infra2:=false \
  -p pointcloud.enable:=false \
  -p align_depth.enable:=false \
  -p rgb_camera.color_profile:="640x480x60" \
  -p depth_module.depth_profile:="640x480x60"
```

Host terminal 2, D435i:

```bash
export ROS2_WS=$HOME/ros2_ws
source /opt/ros/humble/setup.bash
source ${ROS2_WS}/install/setup.bash

eval "$(
  rs-enumerate-devices | awk -F': ' '
    /^[[:space:]]*Name[[:space:]]*:/ { name=$2; gsub(/^[[:space:]]+|[[:space:]]+$/, "", name) }
    /^[[:space:]]*Serial Number[[:space:]]*:/ {
      serial=$2
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", serial)
      if (name ~ /D405/) print "export D405_SERIAL=_" serial
      if (name ~ /D435I|D435i/) print "export D435I_SERIAL=_" serial
    }
  '
)"

ros2 run realsense2_camera realsense2_camera_node --ros-args \
  -r __ns:=/d435i \
  -p camera_name:=d435i \
  -p serial_no:=${D435I_SERIAL} \
  -p enable_color:=true \
  -p enable_depth:=true \
  -p enable_infra1:=false \
  -p enable_infra2:=false \
  -p enable_gyro:=false \
  -p enable_accel:=false \
  -p pointcloud.enable:=false \
  -p align_depth.enable:=false \
  -p rgb_camera.color_profile:="640x480x60" \
  -p depth_module.depth_profile:="640x480x60"
```

Host terminal 3, RViz2:

```bash
export ROS2_WS=$HOME/ros2_ws
source /opt/ros/humble/setup.bash
source ${ROS2_WS}/install/setup.bash
rviz2
```

Useful image topics for RViz2:

- `/d405/camera/color/image_raw`
- `/d405/camera/depth/image_rect_raw`
- `/d435i/camera/color/image_raw`
- `/d435i/camera/depth/image_rect_raw`

Quick topic check:

```bash
ros2 topic list | grep -E 'd405|d435i'
```

### 5.3 Run the benchmark nodes

These commands assume the D435i camera is already running on the host.

Container terminal:

```bash
export ISAAC_ROS_WS=/workspaces/isaac_ros-dev
export REPO_ROOT=${ISAAC_ROS_WS}/src/Realsense-IsaacROS

cd ${ISAAC_ROS_WS}
source /opt/ros/humble/setup.bash
source install/setup.bash
```

Live benchmark:

```bash
ros2 run realsense_benchmark live_benchmark_node --ros-args \
  -p depth_topic:=/d435i/camera/depth/image_rect_raw \
  -p cpu_loops:=0
```

Phase benchmark:

```bash
ros2 run realsense_benchmark phase_benchmark_node --ros-args \
  -p color_topic:=/d435i/camera/color/image_raw \
  -p depth_topic:=/d435i/camera/depth/image_rect_raw \
  -p baseline_duration_sec:=30.0 \
  -p stress_duration_sec:=30.0 \
  -p stress_cpu_loops:=8 \
  -p output_dir:=${REPO_ROOT}/realsense_benchmark/results \
  -p show_window:=true
```

Detection benchmark:

```bash
python3 -m pip install -U mediapipe --break-system-packages

ros2 run realsense_benchmark detection_benchmark_node --ros-args \
  -p color_topic:=/d435i/camera/color/image_raw \
  -p depth_topic:=/d435i/camera/depth/image_rect_raw \
  -p baseline_duration_sec:=30.0 \
  -p stress_duration_sec:=30.0 \
  -p output_dir:=${REPO_ROOT}/realsense_benchmark/results \
  -p show_window:=true
```

Note:

- Wait about 30 seconds after starting the detection benchmark to observe the facial-expression overlay.
- The one-command launch files in `realsense_benchmark/launch` start a RealSense camera in the same environment as the benchmark. In the default split documented here, the camera driver lives on the host and the benchmark runs in the container, so the separate host/container commands above are the reproducible path.

### 5.4 Verify NITROS with a rectified image topic

Host terminal, D405 color stream only:

```bash
export ROS2_WS=$HOME/ros2_ws
source /opt/ros/humble/setup.bash
source ${ROS2_WS}/install/setup.bash

eval "$(
  rs-enumerate-devices | awk -F': ' '
    /^[[:space:]]*Name[[:space:]]*:/ { name=$2; gsub(/^[[:space:]]+|[[:space:]]+$/, "", name) }
    /^[[:space:]]*Serial Number[[:space:]]*:/ {
      serial=$2
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", serial)
      if (name ~ /D405/) print "export D405_SERIAL=_" serial
    }
  '
)"

ros2 run realsense2_camera realsense2_camera_node --ros-args \
  -r __ns:=/d405 \
  -p camera_name:=d405 \
  -p serial_no:=${D405_SERIAL} \
  -p enable_color:=true \
  -p enable_depth:=false \
  -p enable_infra1:=false \
  -p enable_infra2:=false \
  -p rgb_camera.color_profile:="640x480x60"
```

Container terminal 1, fix NITROS temp permissions only if needed:

```bash
sudo rm -rf /tmp/isaac_ros_nitros
sudo mkdir -p /tmp/isaac_ros_nitros
sudo chmod -R 777 /tmp/isaac_ros_nitros
ls -ld /tmp /tmp/isaac_ros_nitros
```

Container terminal 2, start the component container:

```bash
export ISAAC_ROS_WS=/workspaces/isaac_ros-dev
cd ${ISAAC_ROS_WS}
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run rclcpp_components component_container_mt --ros-args -r __node:=nitros_rectify_container
```

Container terminal 3, load `RectifyNode`:

```bash
export ISAAC_ROS_WS=/workspaces/isaac_ros-dev
cd ${ISAAC_ROS_WS}
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 component load /nitros_rectify_container isaac_ros_image_proc nvidia::isaac_ros::image_proc::RectifyNode \
  --node-name d405_rectify \
  -p output_width:=640 \
  -p output_height:=480 \
  -r image_raw:=/d405/camera/color/image_raw \
  -r camera_info:=/d405/camera/color/camera_info \
  -r image_rect:=/d405/camera/color/image_rect \
  -r camera_info_rect:=/d405/camera/color/camera_info_rect
```

Host terminal, confirm the negotiated topic exists:

```bash
export ROS2_WS=$HOME/ros2_ws
source /opt/ros/humble/setup.bash
source ${ROS2_WS}/install/setup.bash

ros2 topic info -v /d405/camera/color/image_rect/nitros
```

Expected result:

```text
Type: negotiated_interfaces/msg/NegotiatedTopicsInfo
```

Optional visualization:

```bash
rviz2
```

Add this topic:

- `/d405/camera/color/image_rect`

### 5.5 Run the floor box perception demo

Host terminal, D435i with aligned depth enabled:

```bash
export ROS2_WS=$HOME/ros2_ws
source /opt/ros/humble/setup.bash
source ${ROS2_WS}/install/setup.bash

eval "$(
  rs-enumerate-devices | awk -F': ' '
    /^[[:space:]]*Name[[:space:]]*:/ { name=$2; gsub(/^[[:space:]]+|[[:space:]]+$/, "", name) }
    /^[[:space:]]*Serial Number[[:space:]]*:/ {
      serial=$2
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", serial)
      if (name ~ /D435I|D435i/) print "export D435I_SERIAL=_" serial
    }
  '
)"

ros2 run realsense2_camera realsense2_camera_node --ros-args \
  -r __ns:=/d435i \
  -p camera_name:=d435i \
  -p serial_no:=${D435I_SERIAL} \
  -p enable_color:=true \
  -p enable_depth:=true \
  -p enable_infra1:=false \
  -p enable_infra2:=false \
  -p enable_gyro:=false \
  -p enable_accel:=false \
  -p pointcloud.enable:=false \
  -p align_depth.enable:=true \
  -p rgb_camera.color_profile:="640x480x60" \
  -p depth_module.depth_profile:="640x480x60"
```

Container terminal, launch the detector:

```bash
export ISAAC_ROS_WS=/workspaces/isaac_ros-dev
cd ${ISAAC_ROS_WS}
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run floor_box_perception box_floor_detector --ros-args \
  -p color_topic:=/d435i/camera/color/image_raw \
  -p depth_topic:=/d435i/camera/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/d435i/camera/color/camera_info \
  -p show_debug_window:=true
```

Useful outputs:

- `/box_detection/debug_image`
- `/box_detection/mask`
- `/box_detection/distance`

## 6. Troubleshooting

### 6.1 `rs-enumerate-devices` works but the serial variables are empty

Run:

```bash
rs-enumerate-devices
```

Then export the values manually, keeping the leading underscore:

```bash
export D405_SERIAL=_YOUR_D405_SERIAL
export D435I_SERIAL=_YOUR_D435I_SERIAL
```

### 6.2 NITROS fails with:

```text
filesystem error: cannot create directories: Permission denied [/tmp/isaac_ros_nitros/...]
```

Fix it in the container:

```bash
sudo rm -rf /tmp/isaac_ros_nitros
sudo mkdir -p /tmp/isaac_ros_nitros
sudo chmod -R 777 /tmp/isaac_ros_nitros
```

### 6.3 `rosdep install` fails because package metadata is stale

Inside the container:

```bash
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*
sudo apt-get update
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

### 6.4 GUI windows do not open in the container

If you are running headless or without GUI forwarding, set:

```bash
-p show_window:=false
```

for the benchmark and floor-perception nodes.
