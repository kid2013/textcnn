#!/bin/bash

rm best_perf.list
for j in $@
do
    if  [ -e res$j.log ]
    then
        python /home/username/textcnn/caffe_ocr/tools/extra/parse_log.py res$j.log .
        echo res$j.log.test >> best_perf.list
        sort -t , -k 4 -n -r res$j.log.test | sed -n '1p'  >> best_perf.list
    fi
done
dos2unix best_perf.list


