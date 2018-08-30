#!/bin/bash

for i in $@
do
    j=snap_shot${i}
    if [ -e $j ]
        then
           rm -fr $j
           mkdir $j
    fi
done
