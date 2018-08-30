#!/bin/bash

# ./get_best_perf.sh $@
for num in $@
do
    i=res$num.log.test
    if [ -e $i ]
       then
           j=\"$i\"
           echo "
           set datafile sep ','
           set size 1, 1
           set title 'test acc for config $num'
           set term png size 1500, 1000
           set output  \"${num}_test_acc.png\"
           set yr [0.93 : 0.985]
           set xlabel 'iteration'
           set ylabel 'test_acc'
           plot $j  u 1 : 4 w line title ''   " | gnuplot
           # eog ${num}_test_acc.png &

    fi
done
