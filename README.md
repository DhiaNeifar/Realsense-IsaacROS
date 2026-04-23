# Realsense-IsaacROS

This repository documents a reproducible RealSense + Isaac ROS setup.

The workflow is split across two environments:

- Host machine:
  - `librealsense`
  - `realsense-ros`
  - direct access to the USB cameras
- Isaac ROS dev container:
  - `isaac_ros_common`
  - `isaac_ros_nitros`
  - `isaac_ros_image_pipeline`
  - `realsense_benchmark`
  - `floor_box_perception`

The goal is that a new team member can:

1. install the stack once by following this README from top to bottom
2. copy/paste the validation commands
3. confirm the cameras, Isaac ROS, NITROS, benchmarks, and demos are working

## 1. Assumptions

This guide assumes:

- Ubuntu 22.04
- ROS 2 Humble is already installed at `/opt/ros/humble`
- Docker is installed and working
- At least one Intel RealSense D405 or D435i is available

Two workspaces are used:

- Host RealSense workspace: `~/ros2_ws`
- Isaac ROS workspace: `~/workspaces/isaac_ros-dev`

Important workspace rule:

- `realsense-ros` is built on the host in `~/ros2_ws`
- Isaac ROS and the custom packages are built inside `~/workspaces/isaac_ros-dev`
- inside the Isaac ROS workspace, the repository contents are copied directly into `src`
- after setup, you should have directories such as `isaac_ros_common`, `isaac_ros_nitros`, `isaac_ros_image_pipeline`, `realsense_benchmark`, and `floor_box_perception` directly under `${ISAAC_ROS_WS}/src`
- there should not be an extra `${ISAAC_ROS_WS}/src/Realsense-IsaacROS` directory in the final Isaac ROS workspace

Throughout this README:

- "host terminal" means your normal machine terminal
- "container terminal" means a shell inside the Isaac ROS dev container

## 2. One-Time Host Setup

### 2.1 Export host workspace variables

Run this in a host terminal:

```bash
export ROS2_WS=$HOME/ros2_ws
export ISAAC_ROS_WS=$HOME/workspaces/isaac_ros-dev
export ISAAC_ROS_SRC=$ISAAC_ROS_WS/src
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

Version used here: `v2.55.1`

```bash
cd ~
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

- `FORCE_RSUSB_BACKEND=ON` is required for this setup.
- `BUILD_UNIT_TESTS=false` avoids unnecessary test build issues.

### 2.6 Clone and build `realsense-ros` on the host

`realsense-ros` must be built on the host inside `~/ros2_ws`, not inside the Isaac ROS container.

First clone the ROS wrapper:

```bash
cd ${ROS2_WS}/src
git clone -b ros2-master https://github.com/IntelRealSense/realsense-ros.git
cd ~/ros2_ws/src/realsense-ros
git fetch --tags
git checkout 4.55.1
```

Then build it on the host:

```bash
cd ${ROS2_WS}
source /opt/ros/humble/setup.bash
sudo apt-get install -y python3-rosdep

if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init
fi

rosdep update
rosdep install -i --from-paths src --rosdistro humble --skip-keys=librealsense2 -y
colcon build --symlink-install
source ${ROS2_WS}/install/setup.bash
```

If the build fails with a permissions error related to `Documents/librealsense2/presets`, fix it and rebuild:

```bash
sudo mkdir -p ${HOME}/Documents/librealsense2/presets
sudo chown -R ${USER}:${USER} ${HOME}/Documents
chmod -R u+rwX ${HOME}/Documents

cd ${ROS2_WS}
rm -rf build install log
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source ${ROS2_WS}/install/setup.bash
```

### 2.7 Confirm the host install works

Test the SDK:

```bash
realsense-viewer
```

Confirm that the ROS 2 driver is visible:

```bash
source /opt/ros/humble/setup.bash
source ${ROS2_WS}/install/setup.bash
ros2 pkg list | grep realsense2_camera
```

## 3. One-Time Isaac ROS Workspace Setup

### 3.1 Prepare the Isaac ROS workspace on the host

```bash
export ISAAC_ROS_WS=$HOME/workspaces/isaac_ros-dev
export ISAAC_ROS_SRC=$ISAAC_ROS_WS/src

sudo systemctl daemon-reload
sudo systemctl restart docker
sudo apt-get install -y git-lfs rsync
git lfs install --skip-repo

mkdir -p ${ISAAC_ROS_SRC}
```

Optional convenience step:

```bash
grep -qxF 'export ISAAC_ROS_WS=${HOME}/workspaces/isaac_ros-dev' ~/.bashrc || \
  echo 'export ISAAC_ROS_WS=${HOME}/workspaces/isaac_ros-dev' >> ~/.bashrc
source ~/.bashrc
```

If Docker access is not already configured for your user:

```bash
sudo usermod -aG docker $USER
newgrp docker
groups
```

### 3.2 Clone the repository into the Isaac ROS workspace

The final Isaac ROS workspace should be flat. That means the contents of this repository live directly under `${ISAAC_ROS_WS}/src`, not under a nested `Realsense-IsaacROS` folder.

The Isaac ROS helper script used in the next step expects `${ISAAC_ROS_WS}` itself to be a Git/LFS-enabled workspace root. Initialize that first, then copy the repository contents into `${ISAAC_ROS_WS}/src`.

Run this on the host:

```bash
export ISAAC_ROS_WS=$HOME/workspaces/isaac_ros-dev
export ISAAC_ROS_SRC=$ISAAC_ROS_WS/src

mkdir -p ${ISAAC_ROS_WS}
cd ${ISAAC_ROS_WS}
git init
git lfs install --local

cd ${ISAAC_ROS_SRC}
git clone --recurse-submodules https://github.com/DhiaNeifar/Realsense-IsaacROS.git temp_repo
git -C temp_repo submodule update --init --recursive
rsync -a --exclude .git temp_repo/ ./
rm -rf temp_repo
```

After the copy, these directories should exist directly under `${ISAAC_ROS_SRC}`:

- `isaac_ros_common`
- `isaac_ros_nitros`
- `isaac_ros_image_pipeline`
- `realsense_benchmark`
- `floor_box_perception`

If an extra `${ISAAC_ROS_SRC}/Realsense-IsaacROS` folder exists from an older setup, remove it:

```bash
cd ${ISAAC_ROS_SRC}
if [ -d Realsense-IsaacROS ]; then
  rm -rf Realsense-IsaacROS
fi
```

### 3.3 Configure and enter the Isaac ROS dev container

Create the Isaac ROS container config on the host:

```bash
export ISAAC_ROS_WS=$HOME/workspaces/isaac_ros-dev
export ISAAC_ROS_SRC=$ISAAC_ROS_WS/src

cd ${ISAAC_ROS_SRC}/isaac_ros_common/scripts
touch .isaac_ros_common-config
echo CONFIG_IMAGE_KEY=ros2_humble.realsense > .isaac_ros_common-config
```

Launch the dev container from the host:

The previous step initialized `${ISAAC_ROS_WS}` as a Git/LFS workspace root. Keep that `.git` directory in place. `isaac_ros_common/scripts/run_dev.sh` checks Git LFS from `${ISAAC_ROS_WS}` and may exit early if the workspace root is not a Git repository.

```bash
cd ${ISAAC_ROS_SRC}/isaac_ros_common
./scripts/run_dev.sh -d ${ISAAC_ROS_WS}
```

Inside the container, the workspace path is:

```bash
/workspaces/isaac_ros-dev
```

### 3.4 Install dependencies inside the Isaac ROS container

Run this in a container terminal:

```bash
export ISAAC_ROS_WS=/workspaces/isaac_ros-dev
export ISAAC_ROS_SRC=${ISAAC_ROS_WS}/src
export RESULTS_DIR=${ISAAC_ROS_SRC}/realsense_benchmark/results

cd ${ISAAC_ROS_WS}
source /opt/ros/humble/setup.bash

sudo apt-get update
rosdep update
rosdep install --from-paths src --ignore-src -r -y

sudo apt-get install -y \
  ros-humble-magic-enum \
  ros-humble-foxglove-msgs \
  python3-opencv \
  python3-numpy \
  python3-matplotlib

mkdir -p ${RESULTS_DIR}
```

If `apt` metadata is stale and dependency installation fails:

```bash
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*
sudo apt-get update
rosdep update
rosdep install --from-paths src --ignore-src -r -y

sudo apt-get install -y \
  ros-humble-magic-enum \
  ros-humble-foxglove-msgs \
  python3-opencv \
  python3-numpy \
  python3-matplotlib
```

### 3.5 Build Isaac ROS and the custom packages inside the container

Before building, make sure there is no duplicate `Realsense-IsaacROS` directory inside `src`:

```bash
export ISAAC_ROS_WS=/workspaces/isaac_ros-dev
export ISAAC_ROS_SRC=${ISAAC_ROS_WS}/src

cd ${ISAAC_ROS_SRC}
if [ -d Realsense-IsaacROS ]; then
  rm -rf Realsense-IsaacROS
fi
```

Then build the workspace:

```bash
cd ${ISAAC_ROS_WS}
rm -rf build install log
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

Verify that the packages are visible:

```bash
ros2 pkg list | grep -E 'isaac_ros_image_proc|realsense_benchmark|floor_box_perception'
ros2 component types | grep -i Rectify
```

## 4. Export Camera Serials Instead of Hard-Coding Them

Run this in the host terminal before launching either camera. Use `source`, not `./...`, so the exported variables stay in your current shell:

```bash
export ROS2_WS=$HOME/ros2_ws
source ./scripts/export_realsense_serials.sh
```

The script checks `/usr/local/bin/rs-enumerate-devices -s` before sourcing ROS, parses the serial number column from that command's table output, unsets old values first, exports raw serial numbers with no leading underscore, and prints a clear message when a D405 or D435i is not detected.

Add the underscore only when passing `serial_no` to ROS, for example:

```bash
-p serial_no:=_${D435I_SERIAL}
```

If automatic detection fails, set the values manually:

```bash
export D405_SERIAL=YOUR_D405_SERIAL
export D435I_SERIAL=YOUR_D435I_SERIAL
```

## 5. Daily Runtime Commands

### 5.1 Start the Isaac ROS dev container

Host terminal:

```bash
export ISAAC_ROS_WS=$HOME/workspaces/isaac_ros-dev
export ISAAC_ROS_SRC=$ISAAC_ROS_WS/src

cd ${ISAAC_ROS_SRC}/isaac_ros_common
./scripts/run_dev.sh -d ${ISAAC_ROS_WS}
```

Container terminal setup:

```bash
export ISAAC_ROS_WS=/workspaces/isaac_ros-dev
export ISAAC_ROS_SRC=${ISAAC_ROS_WS}/src
export RESULTS_DIR=${ISAAC_ROS_SRC}/realsense_benchmark/results

cd ${ISAAC_ROS_WS}
source /opt/ros/humble/setup.bash
source install/setup.bash
mkdir -p ${RESULTS_DIR}
```

### 5.2 Run both cameras and verify the streams in RViz2

In each host terminal below:

1. set `ROS2_WS`
2. source `./scripts/export_realsense_serials.sh`
3. launch the camera node with `-p serial_no:=_${...}`

Host terminal 1, D405:

```bash
export ROS2_WS=$HOME/ros2_ws
source ./scripts/export_realsense_serials.sh

ros2 run realsense2_camera realsense2_camera_node --ros-args \
  -r __ns:=/d405 \
  -p camera_name:=d405 \
  -p serial_no:=_${D405_SERIAL} \
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
source ./scripts/export_realsense_serials.sh

ros2 run realsense2_camera realsense2_camera_node --ros-args \
  -r __ns:=/d435i \
  -p camera_name:=d435i \
  -p serial_no:=_${D435I_SERIAL} \
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

Useful RViz2 image topics:

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
export ISAAC_ROS_SRC=${ISAAC_ROS_WS}/src
export RESULTS_DIR=${ISAAC_ROS_SRC}/realsense_benchmark/results

cd ${ISAAC_ROS_WS}
source /opt/ros/humble/setup.bash
source install/setup.bash
mkdir -p ${RESULTS_DIR}
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
  -p output_dir:=${RESULTS_DIR} \
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
  -p output_dir:=${RESULTS_DIR} \
  -p show_window:=true
```

Notes:

- Wait about 30 seconds after starting the detection benchmark to observe the facial-expression overlay.
- The launch files in `realsense_benchmark/launch` read `D435I_SERIAL` instead of a hard-coded serial number, but the default reproducible workflow here still keeps the camera driver on the host and the benchmark inside the container.

### 5.4 Verify NITROS with a rectified image topic

Host terminal, D405 color stream only:

```bash
export ROS2_WS=$HOME/ros2_ws
source ./scripts/export_realsense_serials.sh

ros2 run realsense2_camera realsense2_camera_node --ros-args \
  -r __ns:=/d405 \
  -p camera_name:=d405 \
  -p serial_no:=_${D405_SERIAL} \
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

Host terminal, confirm that the negotiated topic exists:

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

Add this image topic:

- `/d405/camera/color/image_rect`

### 5.5 Run the floor box perception demo

Host terminal, D435i with aligned depth enabled:

```bash
export ROS2_WS=$HOME/ros2_ws
source ./scripts/export_realsense_serials.sh

ros2 run realsense2_camera realsense2_camera_node --ros-args \
  -r __ns:=/d435i \
  -p camera_name:=d435i \
  -p serial_no:=_${D435I_SERIAL} \
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

Container terminal, run the detector:

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

### 6.1 Automatic detection fails but `rs-enumerate-devices -s` works

Export the raw serial numbers manually:

```bash
export D405_SERIAL=YOUR_D405_SERIAL
export D435I_SERIAL=YOUR_D435I_SERIAL
```

Add the underscore only when passing `serial_no` to ROS:

```bash
-p serial_no:=_${D405_SERIAL}
-p serial_no:=_${D435I_SERIAL}
```

### 6.2 `realsense-ros` fails to build because of `Documents/librealsense2/presets`

Fix the permissions and rebuild on the host:

```bash
export ROS2_WS=$HOME/ros2_ws

sudo mkdir -p ${HOME}/Documents/librealsense2/presets
sudo chown -R ${USER}:${USER} ${HOME}/Documents
chmod -R u+rwX ${HOME}/Documents

cd ${ROS2_WS}
rm -rf build install log
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source ${ROS2_WS}/install/setup.bash
```

### 6.3 `rosdep install` fails because package metadata is stale

Inside the container:

```bash
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*
sudo apt-get update
rosdep update
rosdep install --from-paths src --ignore-src -r -y

sudo apt-get install -y \
  ros-humble-magic-enum \
  ros-humble-foxglove-msgs \
  python3-opencv \
  python3-numpy \
  python3-matplotlib
```

### 6.4 Isaac ROS packages are not found because the workspace is nested

If you see a duplicate `${ISAAC_ROS_WS}/src/Realsense-IsaacROS` directory, remove it so the packages live directly under `src`:

```bash
export ISAAC_ROS_WS=$HOME/workspaces/isaac_ros-dev

cd ${ISAAC_ROS_WS}/src
rm -rf Realsense-IsaacROS
```

Then rebuild:

```bash
export ISAAC_ROS_WS=/workspaces/isaac_ros-dev

cd ${ISAAC_ROS_WS}
rm -rf build install log
source /opt/ros/humble/setup.bash
colcon build --packages-up-to isaac_ros_nitros --symlink-install --cmake-args -DBUILD_TESTING=OFF
source install/setup.bash
colcon build --packages-up-to isaac_ros_image_proc --symlink-install --cmake-args -DBUILD_TESTING=OFF
source install/setup.bash
colcon build --packages-select realsense_benchmark floor_box_perception --symlink-install
source install/setup.bash
```

### 6.5 NITROS fails with a permissions error under `/tmp/isaac_ros_nitros`

If you see:

```text
filesystem error: cannot create directories: Permission denied [/tmp/isaac_ros_nitros/...]
```

Fix it inside the container:

```bash
sudo rm -rf /tmp/isaac_ros_nitros
sudo mkdir -p /tmp/isaac_ros_nitros
sudo chmod -R 777 /tmp/isaac_ros_nitros
```

### 6.6 GUI windows do not open in the container

If you are running headless or without GUI forwarding, set:

```bash
-p show_window:=false
```

for the benchmark and floor-perception nodes.
