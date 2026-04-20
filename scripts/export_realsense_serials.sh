#!/usr/bin/env bash

set -eo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced so the serial exports stay in your current shell:"
  echo "  source ./scripts/export_realsense_serials.sh"
  exit 1
fi

export ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"

unset D405_SERIAL
unset D435I_SERIAL

RS_OUTPUT=""
RS_STATUS=0
if ! RS_OUTPUT=$(/usr/local/bin/rs-enumerate-devices -s 2>&1); then
  RS_STATUS=$?
  echo "Failed to run /usr/local/bin/rs-enumerate-devices -s"
  if [ -n "$RS_OUTPUT" ]; then
    printf '%s\n' "$RS_OUTPUT"
  fi
  echo "D405 not detected."
  echo "D435i not detected."
else
  eval "$(
    printf '%s\n' "$RS_OUTPUT" | awk '
      BEGIN { IGNORECASE=1 }
      NR == 1 && $0 ~ /Device Name/ && $0 ~ /Serial Number/ { next }
      /^[[:space:]]*$/ { next }
      {
        serial = $(NF - 1)

        if ($0 ~ /D435I|D435i/) {
          print "export D435I_SERIAL=" serial
        } else if ($0 ~ /D405/) {
          print "export D405_SERIAL=" serial
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

set +u
source /opt/ros/humble/setup.bash
source "${ROS2_WS}/install/setup.bash"
