#!/bin/bash
# File: Variables
# Author: Denis Kurka
# Year: 2025
# License: CC0


# Variables
ZIP_FILE="deployment_package.zip"
REMOTE_USER="petalinux"
REMOTE_HOST="85.70.252.121"
REMOTE_PORT="8112"
REMOTE_DIR="/home/petalinux/"
SOURCE_DIR="./model_construct"

# Create the zip file
echo "Creating zip file: $ZIP_FILE"
zip -r $ZIP_FILE $SOURCE_DIR/*.py $SOURCE_DIR/test_data $SOURCE_DIR/best_unet_weights.pth $SOURCE_DIR/requirements.txt

# Check if the zip was created successfully
if [ $? -eq 0 ]; then
    echo "Zip file created successfully."
else
    echo "Failed to create zip file."
    exit 1
fi

# Send the zip file using SCP
echo "Transferring $ZIP_FILE to $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
scp -P $REMOTE_PORT $ZIP_FILE $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR

# Check if the SCP transfer was successful
if [ $? -eq 0 ]; then
    echo "File transferred successfully."
else
    echo "File transfer failed."
    exit 1
fi

# Cleanup: Remove the local zip file after transfer (optional)
echo "Cleaning up local zip file..."
rm -f $ZIP_FILE
echo "Done."

# COnnect to fz5 and setup env


