#!/bin/bash
BUILD_PATH=`pwd`
cd ..
source ./set_paths.sh
cd ..
cd finn
./run-docker.sh build_dataflow $BUILD_PATH
