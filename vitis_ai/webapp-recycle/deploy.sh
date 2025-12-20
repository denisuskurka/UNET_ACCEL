#!/usr/bin/env bash
# File: vitis_ai/webapp-recycle/deploy.sh
# Author: Denis Kurka
# Year: 2025
# License: CC0


# --------------------------------------------------
# Configuration
# --------------------------------------------------
ZIP_NAME="flask_app.zip"
REMOTE_USER="petalinux"
REMOTE_HOST="85.70.252.121"
REMOTE_PORT=8112
LOCAL_FILES="*.py dma_driver.c dummy.bin requirements.txt run_dma.sh start_dma_engine.sh stem_model.h5 ellipse_regresor.h5"

# --------------------------------------------------
# 1) Package the application into a zip
# --------------------------------------------------
echo "==> Creating zip package with $LOCAL_FILES"
zip -r "$ZIP_NAME" $LOCAL_FILES

# --------------------------------------------------
# 2) Send new app to server & clean old app
# --------------------------------------------------
echo "==> Uploading $ZIP_NAME to $REMOTE_HOST"
scp -P $REMOTE_PORT "$ZIP_NAME" "$REMOTE_USER@$REMOTE_HOST:/home/petalinux"

echo "==> Connecting via SSH to remove old app, check dependencies, and deploy new one"
ssh -p $REMOTE_PORT "$REMOTE_USER@$REMOTE_HOST" << 'EOF'
  # 2a) Check if pip3 is installed
  if ! command -v pip3 &>/dev/null; then
    echo "ERROR: pip3 is not installed."
    echo "Please install it using: sudo dnf install python3-pip"
    echo "Aborting deployment..."
    exit 1
  else
    echo "==> pip3 found."
  fi

  # 2b) Check if gcc is installed
  if ! command -v gcc &>/dev/null; then
    echo "ERROR: gcc is not installed."
    echo "Please install it using: sudo dnf install gcc"
    echo "Aborting deployment..."
    exit 1
  else
    echo "==> gcc found."
  fi

  # 2c) Remove old deployment
  echo "==> Removing old deployment (if present)"
  rm -rf "/home/petalinux/flask_app"

  # 2d) Create new deployment directory
  echo "==> Creating new deployment directory"
  mkdir "/home/petalinux/flask_app"
  cd "/home/petalinux/flask_app"
  mv "/home/petalinux/flask_app.zip" "/home/petalinux/flask_app/flask_app.zip"

  # 2e) Unzip the new application
  echo "==> Unzipping application"
  unzip "/home/petalinux/flask_app/flask_app.zip"

  # 2f) Install Python requirements
  echo "==> Installing requirements"
  pip3 install -r requirements.txt

  # 2g) Compile the DMA driver
  gcc dma_driver.c -o dma_driver
  echo "==> DMA driver compiled!"
EOF

echo "==> Deployment completed!"
echo "Now run DMA driver with:"
echo "   sudo ./start_dma_engine.sh"
echo "   Hint: you can kill it with the same cmd."
echo "Run the application with:"
echo "   python app.py"
