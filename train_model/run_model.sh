#!/bin/bash

do
    if [ -e run_$i.sh ]
    then
        ./run_$i.sh > $i.list &
        # wait
    fi
done

