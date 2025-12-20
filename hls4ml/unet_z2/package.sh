#!/bin/bash
# File: hls4ml/unet_z2/package.sh
# Author: Denis Kurka
# Year: 2025
# License: CC0

mkdir -p pynq_deployment
cp quantized_pruned_cnn/myproject_vivado_accelerator/project_1.runs/impl_1/design_1_wrapper.bit pynq_deployment/hls4ml_nn.bit
cp quantized_pruned_cnn/myproject_vivado_accelerator/project_1.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh pynq_deployment/hls4ml_nn.hwh
cp quantized_pruned_cnn/axi_stream_driver.py pynq_deployment
cp X_test.npy pynq_deployment
tar -czvf pynq_deployment.tar.gz pynq_deployment

scp pynq_deployment.tar.gz xilinx@192.168.50.204:/home/xilinx/jupyter_notebooks/cnn/
ssh xilinx@192.168.50.204
#cd /home/xilinx/jupyter_notebooks/cnn/
#tar xvzf package.tar.gz