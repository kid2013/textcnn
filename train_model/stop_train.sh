ps -ef | grep [t]extcnn | awk '{print $2 }' | xargs kill -9
