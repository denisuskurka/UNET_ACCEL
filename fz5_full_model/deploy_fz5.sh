#!/bin/bash
# File: fz5_full_model/deploy_fz5.sh
# Author: Denis Kurka
# Year: 2025
# License: CC0


# Variables
ZIP_FILE="deployment_package.zip"
REMOTE_USER="petalinux"
REMOTE_HOST="85.70.252.121"
REMOTE_PORT="8112"
REMOTE_DIR="/home/petalinux/"
SOURCE_DIR="./deployment"

# Ensure the deployment directory exists and is empty
if [ -d "$SOURCE_DIR" ]; then
    rm -rf "$SOURCE_DIR"
fi
mkdir -p "$SOURCE_DIR"

# Copy required files to the deployment directory
echo "Preparing deployment files..."
cp ./model.py "$SOURCE_DIR"
cp ./common.py "$SOURCE_DIR"
cp ./infer.py "$SOURCE_DIR"
cp ./testbench.py "$SOURCE_DIR"
cp ./best_unet_weights.pth "$SOURCE_DIR"
cp ./test_data/01.png "$SOURCE_DIR"
cp ./requirements_inference.txt "$SOURCE_DIR"
cp -r ./test_data "$SOURCE_DIR"

# Check if all required files were copied
if [ $? -ne 0 ]; then
    echo "Error: Failed to copy required files to $SOURCE_DIR."
    exit 1
fi

# Create the zip file
echo "Creating zip file: $ZIP_FILE"
zip -r "$ZIP_FILE" "$SOURCE_DIR"/* > /dev/null


# Verify if the zip file was created successfully
if [ $? -eq 0 ]; then
    echo "Zip file created successfully."
else
    echo "Error: Failed to create zip file."
    exit 1
fi

# Transfer the zip file using SCP
echo "Transferring $ZIP_FILE to $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
scp -P "$REMOTE_PORT" "$ZIP_FILE" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

# Verify if the SCP transfer was successful
if [ $? -eq 0 ]; then
    echo "File transferred successfully."
else
    echo "Error: File transfer failed."
    exit 1
fi

# Cleanup: Remove the local zip file after transfer
echo "Cleaning up local zip file..."
rm -f "$ZIP_FILE"

# Remote setup and testing
echo "Setting up the environment on the remote host..."
ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" <<EOF
    # Cleanup
    rm -rf ./deployment

    # Unzip the deployment package
    echo "Unzipping deployment package..."
    unzip -o $REMOTE_DIR$ZIP_FILE -d $REMOTE_DIR > /dev/null
    if [ $? -ne 0 ]; then
        echo "Error: Failed to unzip deployment package."
        exit 1
    fi
    rm -f $REMOTE_DIR$ZIP_FILE

    # Navigate to the deployment directory
    cd $REMOTE_DIR/deployment

    # Install Python dependencies
    echo "Installing dependencies..."
    if [ -f "requirements_inference.txt" ]; then
        pip3 install -r requirements_inference.txt > /dev/null
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install dependencies."
            exit 1
        fi
    else
        echo "Warning: requirements_inference.txt not found, skipping dependency installation."
    fi

    # Run a test inference
    echo "Testing the inference script..."
    python3 infer.py 01.png
    if [ $? -eq 0 ] && [ -f "01_mask.png" ]; then
        echo "Inference success! Environment is ready to use!"
    else
        echo "Error: Inference test failed or output mask file not created."
        exit 1
    fi
EOF

# Final cleanup and confirmation
echo "Deployment process completed successfully!"
#scp -P 8112 petalinux@85.70.252.121:/home/petalinux/deployment/01_mask.png ./01_mask_fz5_inferred.png
