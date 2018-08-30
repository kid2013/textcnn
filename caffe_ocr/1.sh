#!/usr/bin/env sh
/home/yangxiao/Downloads/caffe-master/build/tools/caffe train \
  --solver examples/VDSR/VDSR_solver.prototxt \
  --gpu 1 
#  2>&1 | tee examples/VDSR/param/train_vdsr.log
