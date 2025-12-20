#!/bin/bash
# File: hls4ml/create.sh
# Author: Denis Kurka
# Year: 2025
# License: CC0

conda env create -f environment.yml
conda activate hls4ml
source /media/komaro/motomado/AMD/Vivado/2020.1/settings64.sh
pip install git+https://github.com/fastmachinelearning/hls4ml@main
