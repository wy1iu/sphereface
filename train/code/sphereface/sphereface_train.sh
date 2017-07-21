#!/bin/bash
# Usage:
# ./code/sphereface/sphereface_train.sh GPU
#
# Example:
# ./code/sphereface/sphereface_train.sh 0,1

GPU_ID=$1
./../tools/caffe-sphereface/build/tools/caffe train -solver code/sphereface/sphereface_solver.prototxt -gpu ${GPU_ID} 2>&1 | tee result/sphereface/sphereface.log
