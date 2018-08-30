#!/bin/bash

for i in $@
do
    rm  -fr log$i snap_shot$i
    rm res$i.log* run_$i.sh solver_$i.prototxt Text_CNN_train_$i.prototxt
done




