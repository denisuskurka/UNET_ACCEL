# File: archive/paddlepaddle-old/convert.sh
# Author: Denis Kurka
# Year: 2025
# License: CC0

# To convert paddlepaddle model to paddle-lite, firstly download paddle-lite from here
# https://github.com/PaddlePaddle/Paddle-Lite/releases
# Download opt_linux_x86 or similiar file based on your architecture
# Then train and export you model and run the command below.

./opt_linux_x86 --model_file=./resnet50_cifar10/model.pdmodel --param_file=./resnet50_cifar10/model.pdiparams --optimize_out=./resnet50_cifar10_opt --valid_targets=arm
