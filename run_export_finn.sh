#!/bin/bash

# =============================================================================
# Script Name: run_export_finn.sh
# Description: Automates the process of exporting the UNetBrevitas model to FINN.
# Steps:
#   0) Delete /home/komaro/デスクトップ/Cermak/finn/workspace/ if it exists
#   1) Copy everything from /home/komaro/デスクトップ/Cermak/FZ5-UNET/model_construct
#      to /home/komaro/デスクトップ/Cermak/finn/workspace/
#   2) Source /home/komaro/デスクトップ/Cermak/FZ5-UNET/set_paths.sh
#   3) Run export_finn.py using run-docker.sh within the FINN Docker environment
# =============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# ------------------------------- #
#          Configuration          #
# ------------------------------- #

# Define absolute paths to avoid ambiguity
FIVN_ROOT="/home/komaro/デスクトップ/Cermak/finn"
MODEL_CONSTRUCT_DIR="/home/komaro/デスクトップ/Cermak/FZ5-UNET/model_construct"
WORKSPACE_DIR="$FIVN_ROOT/workspace"
SET_PATHS_SCRIPT="/home/komaro/デスクトップ/Cermak/FZ5-UNET/set_paths.sh"
RUN_DOCKER_SCRIPT="$FIVN_ROOT/run-docker.sh"
EXPORT_SCRIPT="export_finn.py"  # Ensure this script is located in model_construct

# ------------------------------- #
#           Functions             #
# ------------------------------- #

# Function to print informational messages
function echo_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

# Function to print error messages and exit
function echo_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1" >&2
    exit 1
}

# ------------------------------- #
#             Step 0               #
# Delete the workspace directory if it exists
# ------------------------------- #

echo_info "Checking if workspace directory exists at $WORKSPACE_DIR"

if [ -d "$WORKSPACE_DIR" ]; then
    echo_info "Workspace directory found. Deleting $WORKSPACE_DIR ..."
    rm -rf "$WORKSPACE_DIR"
    echo_info "Deleted $WORKSPACE_DIR successfully."
else
    echo_info "No existing workspace directory at $WORKSPACE_DIR. Skipping deletion."
fi

# ------------------------------- #
#             Step 1               #
# Copy contents from model_construct to finn/workspace
# ------------------------------- #

echo_info "Copying contents from $MODEL_CONSTRUCT_DIR to $WORKSPACE_DIR ..."

# Create the workspace directory
mkdir -p "$WORKSPACE_DIR"

# Copy all contents without folders, preserving attributes
cp -r "$MODEL_CONSTRUCT_DIR"/*.py "$WORKSPACE_DIR/"
cp -r "$MODEL_CONSTRUCT_DIR"/*.pth "$WORKSPACE_DIR/"

echo_info "Copied files to $WORKSPACE_DIR successfully."

# ------------------------------- #
#             Step 2               #
# Source the set_paths.sh script
# ------------------------------- #

echo_info "Sourcing the set_paths.sh script from $SET_PATHS_SCRIPT ..."

if [ -f "$SET_PATHS_SCRIPT" ]; then
    source "$SET_PATHS_SCRIPT"
    echo_info "Sourced set_paths.sh successfully."
else
    echo_error "set_paths.sh not found at $SET_PATHS_SCRIPT. Exiting."
fi

# ------------------------------- #
#             Step 3               #
# Run export_finn.py using run-docker.sh within Docker
# ------------------------------- #

echo_info "Preparing to run export_finn.py within FINN Docker environment."

# Verify that the export script exists in the workspace
EXPORT_SCRIPT_PATH="$WORKSPACE_DIR/$EXPORT_SCRIPT"

if [ ! -f "$EXPORT_SCRIPT_PATH" ]; then
    echo_error "Export script $EXPORT_SCRIPT_PATH not found. Please ensure it exists."
fi

echo_info "Export script found at $EXPORT_SCRIPT_PATH."

# Verify that run-docker.sh exists and is executable
if [ ! -x "$RUN_DOCKER_SCRIPT" ]; then
    echo_error "run-docker.sh not found or not executable at $RUN_DOCKER_SCRIPT."
fi

echo_info "Executing export_finn.py within Docker using run-docker.sh ..."

# Run the export script inside Docker
bash "$RUN_DOCKER_SCRIPT" python "./workspace/$EXPORT_SCRIPT"

echo_info "export_finn.py executed successfully within Docker."

# ------------------------------- #
#            Completion            #
# ------------------------------- #

echo_info "All steps completed successfully."
echo_info "The exported ONNX model should be located at $WORKSPACE_DIR/unet_finn.onnx"
