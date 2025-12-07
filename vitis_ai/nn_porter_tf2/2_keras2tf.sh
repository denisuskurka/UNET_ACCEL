#!/bin/bash

# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


TF_CPP_MIN_LOG_LEVEL=3

# convert keras model to frozen graph
# Note: The updated python script now handles both Checkpoint creation AND Freezing
keras_2_tf() {
  python keras_2_tf.py \
    --keras_hdf5 ${KERAS}/${K_MODEL} \
    --tf_ckpt    ${TFCKPT_DIR}/${TFCKPT}  
}

echo "-----------------------------------------"
echo "CONVERTING KERAS MODEL TO TF CHECKPOINT & FREEZING.."
echo "-----------------------------------------"

# Prepare Checkpoint Directory
rm -rf ${TFCKPT_DIR}
mkdir -p ${TFCKPT_DIR}

# Run the Python conversion (which now also freezes the graph)
keras_2_tf 2>&1 | tee ${LOG}/${KERAS_LOG}

echo "-----------------------------------------"
echo "ORGANIZING OUTPUT FILES.."
echo "-----------------------------------------"

# Prepare Freeze Directory (Where the next scripts expect the .pb file)
rm -rf ${FREEZE}
mkdir -p ${FREEZE}

# The Python script creates 'frozen_graph.pb' inside TFCKPT_DIR.
# We move it to the defined ${FREEZE} directory and rename it to ${FROZEN_GRAPH}
# to maintain compatibility with 3_quantize.sh
if [ -f "${TFCKPT_DIR}/frozen_graph.pb" ]; then
    mv "${TFCKPT_DIR}/frozen_graph.pb" "${FREEZE}/${FROZEN_GRAPH}"
    echo "Success: Frozen graph moved to ${FREEZE}/${FROZEN_GRAPH}"
else
    echo "Error: frozen_graph.pb was not found in ${TFCKPT_DIR}. Check keras_2_tf.py output."
fi

echo "-----------------------------------------"
echo "PROCESS COMPLETED"
echo "-----------------------------------------"