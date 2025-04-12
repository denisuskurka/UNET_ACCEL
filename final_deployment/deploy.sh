#!/usr/bin/env bash

# --------------------------------------------------
# Configuration
# --------------------------------------------------
ZIP_NAME="flask_app.zip"
REMOTE_USER="petalinux"
REMOTE_HOST="85.70.252.121"
REMOTE_PORT=8112
REMOTE_DIR="/home/petalinux/flask_app"
LOCAL_FILES="app.py requirements.txt"

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

echo "==> Connecting via SSH to remove old app and deploy new one"
ssh -p $REMOTE_PORT "$REMOTE_USER@$REMOTE_HOST" << EOF
  echo "==> Removing old deployment (if present)"
  rm -rf "$REMOTE_DIR"

  echo "==> Installing pip"
  sudo dnf install python3-pip

  echo "==> Creating new deployment directory"
  mkdir -p "$REMOTE_DIR"

  echo "==> Unzipping application"
  unzip "/home/petalinux/$ZIP_NAME" -d "$REMOTE_DIR"

  echo "==> Installing requirements"
  cd "$REMOTE_DIR"
  pip install -r requirements.txt

  echo "==> Starting Flask app on 0.0.0.0:5000"
  # Run in the background to keep it alive after logout
  nohup python app.py --host=0.0.0.0 > app.log 2>&1 &
  exit
EOF

echo "==> Deployment completed!"
