#!/bin/bash
python3 train.py
cp ./best_unet_weights.pth /home/komaro/デスクトップ/Cermak/finn/notebooks/unet/
cp ./common.py /home/komaro/デスクトップ/Cermak/finn/notebooks/unet/
cp ./model.py /home/komaro/デスクトップ/Cermak/finn/notebooks/unet/
