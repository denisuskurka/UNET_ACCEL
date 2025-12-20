#!/bin/bash
# File: vitis_ai/webapp-recycle/start_dma_engine.sh
# Author: Denis Kurka
# Year: 2025
# License: CC0


# Name of the driver process
DRIVER_NAME="dma_driver"

# 1) Check if the driver is running
pids=$(pgrep -f "$DRIVER_NAME")

if [ -n "$pids" ]; then
  echo "$DRIVER_NAME is running with PID(s): $pids"
  echo "Killing it..."
  sudo kill $pids
  echo "Done."
else
  echo "$DRIVER_NAME not running. Launching it with nohup..."
  nohup sudo ./$DRIVER_NAME > dma_driver.log 2>&1 &
  echo "Started $DRIVER_NAME in background (logs in dma_driver.log)."
fi
