#!/bin/bash
# File: convert keras model to frozen graph
# Author: Denis Kurka
# Year: 2025
# License: CC0


TF_CPP_MIN_LOG_LEVEL=3

# convert keras model to frozen graph
keras_2_tf() {
  python keras_2_tf.py \
    --keras_hdf5 ${KERAS}/${K_MODEL} \
    --tf_ckpt    ${TFCKPT_DIR}/${TFCKPT}  
}

echo "-----------------------------------------"
echo "CONVERTING KERAS MODEL TO FROZEN GRAPH.."
echo "-----------------------------------------"

rm -rf ${TFCKPT_DIR}
mkdir -p ${TFCKPT_DIR}
rm -rf ${FREEZE}
mkdir -p ${FREEZE}

# Run the python script
keras_2_tf 2>&1 | tee ${LOG}/${KERAS_LOG}

echo "-----------------------------------------"
echo "MOVING FROZEN GRAPH.."
echo "-----------------------------------------"

# The python script created 'frozen_graph.pb' inside TFCKPT_DIR.
# We move it to the ${FREEZE} folder and rename it if necessary.
if [ -f "${TFCKPT_DIR}/frozen_graph.pb" ]; then
    cp "${TFCKPT_DIR}/frozen_graph.pb" "${FREEZE}/${FROZEN_GRAPH}"
    echo "Graph moved to ${FREEZE}/${FROZEN_GRAPH}"
else
    echo "Error: frozen_graph.pb not found."
    exit 1
fi

echo "-----------------------------------------"
echo "PROCESS COMPLETED"
echo "-----------------------------------------"

