#!/bin/bash

# ./run_1.sh > $1.list  ;  ./run_2.sh > $2.list ; ./run_3.sh > $3.list  ; ./run_11.sh > $4.list ; ./run_12.sh > $5.list ; ./run_13.sh > $6.list ; ./run_14.sh > $7.list
for i in $@
do
    if [ -e run_$i.sh ]
    then
        ./run_$i.sh > $i.list &
        # wait
    fi
done

