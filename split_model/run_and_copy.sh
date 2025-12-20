#!/bin/bash
# File: split_model/run_and_copy.sh
# Author: Denis Kurka
# Year: 2025
# License: CC0

python3 train.py
cp ./best_unet_weights.pth /home/komaro/デスクトップ/Cermak/finn/notebooks/unet/
cp ./common.py /home/komaro/デスクトップ/Cermak/finn/notebooks/unet/
cp ./model.py /home/komaro/デスクトップ/Cermak/finn/notebooks/unet/
