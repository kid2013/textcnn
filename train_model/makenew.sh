#!/bin/bash

src=$1
shift
for i in $@
do
    mkdir -p log$i/201806 snap_shot$i
    cp run_$src.sh  run_$i.sh
    cp solver_$src.prototxt solver_$i.prototxt
    cp Text_CNN_train_$src.prototxt  Text_CNN_train_$i.prototxt

    sed -i "5s/$src/$i/;  s/log$src/log$i/; 5s/res$src/res$i/" run_$i.sh
    sed -i "1s/$src/$i/; /prefix/s/$src/$i/ " solver_$i.prototxt
    echo set gpu ord!
done
