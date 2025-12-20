#!/bin/bash
# File: vitis_ai/ellipse_regressor/5_eval_quant.sh
# Author: Denis Kurka
# Year: 2025
# License: CC0


# Copyright 2020 Xilinx Inc.

# evaluate graph with test dataset
eval_graph() {
  dir_name=$1
  graph=$2
  python eval_graph.py \
    --graph        $dir_name/$graph \
    --input_node   ${INPUT_NODE} \
    --output_node  ${OUTPUT_NODE} \
    --height       ${INPUT_HEIGHT} \
    --width        ${INPUT_WIDTH} \
    --batchsize    100
}

echo "-----------------------------------------"
echo "EVALUATING THE QUANTIZED GRAPH.."
echo "-----------------------------------------"

# Note: The quantizer usually outputs 'quantize_eval_model.pb'
eval_graph ${QUANT} quantize_eval_model.pb 2>&1 | tee ${LOG}/${EVAL_Q_LOG}

echo "-----------------------------------------"
echo "EVALUATION COMPLETED"
echo "-----------------------------------------"
