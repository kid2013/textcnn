DATE=201806
CAFFEBIN=/home/wanwuming/textcnn/caffe_ocr/build/tools/caffe
GLOG_logtostderr=0
GLOG_log_dir=log3/$DATE
$CAFFEBIN train --solver=solver_3.prototxt --gpu 3  2>&1 | tee res3.log 
