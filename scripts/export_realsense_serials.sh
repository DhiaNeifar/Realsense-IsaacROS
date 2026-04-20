#!/usr/bin/env bash

set -euo pipefail

export ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"

unset D405_SERIAL
unset D435I_SERIAL

if ! RS_OUTPUT=$(/usr/local/bin/rs-enumerate-devices -s 2>/dev/null); then
  echo "Failed to run /usr/local/bin/rs-enumerate-devices -s"
  echo "D405 not detected."
  echo "D435i not detected."
else
  eval "$(
    printf '%s\n' "$RS_OUTPUT" | awk -F': *' '
      BEGIN { IGNORECASE=1 }
      /^[[:space:]]*(Device )?Name[[:space:]]*:/ {
        name=$2
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
      }
      /^[[:space:]]*Serial Number[[:space:]]*:/ {
        serial=$2
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", serial)

        if (name ~ /D405/) {
          print "export D405_SERIAL=" serial
        } else if (name ~ /D435[iI]/) {
          print "export D435I_SERIAL=" serial
        }
      }
    '
  )"

  if [ -n "${D405_SERIAL:-}" ]; then
    echo "Detected D405 serial: ${D405_SERIAL}"
  else
    echo "D405 not detected."
  fi

  if [ -n "${D435I_SERIAL:-}" ]; then
    echo "Detected D435i serial: ${D435I_SERIAL}"
  else
    echo "D435i not detected."
  fi
fi

export D405_SERIAL="${D405_SERIAL:-}"
export D435I_SERIAL="${D435I_SERIAL:-}"

source /opt/ros/humble/setup.bash
source "${ROS2_WS}/install/setup.bash"
