#!/bin/bash
BUILD_PATH=`pwd`
cd /home/finnbox/Desktop
source ./set_paths.sh
cd finn
./run-docker.sh build_dataflow $BUILD_PATH
