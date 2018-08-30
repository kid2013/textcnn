#coding: utf-8
"""
    The data_layer is provide data to the net
"""
### vehicle Attributes
#vehicle color : white:0 black:1 gray:2 red:3 yellow:4 blue:5 green:6 purple:7 brown:8 pink:9 orange:10 silver:11 unknow:12
#vehicle type  : car:0 SUV/MPV:1 Van:2  bus:3 mdeiumbus:4  truck:5 smalltruck:6 Tricycle:7 moto:8  unknow:9
#vehicle brand :


caffe_root_python = '/home/wanwuming/ocr1/caffe_cls/caffe_ocr/python'
import sys
sys.path.insert(0,caffe_root_python)

import caffe
import yaml
import os
import numpy as np
import cv2
import random
import chardet
import copy
from math import *

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

def UNICODE(str):
    if type(str).__name__ == 'unicode':
        # print str + 'unicode1'
        return str
    try:
        decode = str.decode('gbk')
        # print str + 'unicode2'
    except:
        try:
            decode = str.decode('utf-8')   
            # print str + "unicode3"
            
        except:
            print 'UNICOD: error.\n'
    return decode

# caffe.Layer
#class PythonDatalayer():
class PythonDatalayer(caffe.Layer):
    #load-batch-index
    _current = 0

    def _get_params(self):
        layer_param = yaml.load(self.param_str)
        #get the param
        if layer_param.has_key('batch_size'):
            self._batch_size = layer_param['batch_size']
        else:
            print 'batch_size must be initialized'
            assert 0

        if layer_param.has_key('channels'):
            self._channels = layer_param['channels']
        else:
            print 'channels must be initialized'
            assert 0

        if layer_param.has_key('height'):
            self._height = layer_param['height']
        else:
            print 'height must be initialized'
            assert 0

        if layer_param.has_key('width'):
            self._width = layer_param['width']
        else:
            print 'width must be initialized'
            assert 0

        if layer_param.has_key('root_folder'):
            self._root_folder = layer_param['root_folder']
        else:
            self._root_folder = ""


        if layer_param.has_key('source'):
            self._source = layer_param['source']
        else:
            print "must write the source field in data_layer"
            assert(0)

        print self._source


#setup load parameters,and load source-files
    def setup(self, bottom, top):
        self._get_params()
        self._blob_name_top_index = {}
        top[0].reshape(self._batch_size,self._channels,self._height,self._width)
        self._blob_name_top_index["data"] = 0
        top[1].reshape(self._batch_size, self._width);
        self._blob_name_top_index["label"] = 1

        #load source-file
        self._get_source_list()


    def reshape(self, bottom, top):
        pass


    def forward(self, bottom, top):
        blobs = self._load_next_batch()
        for blob_name, blob in blobs.iteritems():
            top_index = self._blob_name_top_index[blob_name]
            top[top_index].data[...] = blob

    def backward(self,bottom,top):
        pass
       

    #load source-anno-list-file,parse the linestr and get the imgname&&label
    def _get_source_list(self):
        file = open(self._source, 'r')
        file_line_strs = file.read()
        file.close()

        file_line_strs = file_line_strs.split('\n')
        self._list = []
        for item in file_line_strs:
            if item == '': continue
            tmp_full_img_name = os.path.join(self._root_folder, item)
            self._list.append(tmp_full_img_name)

        while self._batch_size > len(self._list):
            # print self._list
            # print '-------------'
            self._list.extend(self._list)
        random.shuffle(self._list)

    def _get_batch_list(self):
        listsize = len(self._list)
        if self._current + self._batch_size <= listsize:
            batch_list = copy.deepcopy(self._list[self._current:self._current + self._batch_size])
            self._current += self._batch_size
        else:
            batch_list = copy.deepcopy(self._list[self._current:])
            random.shuffle(self._list)
            self._current = self._batch_size - len(batch_list)
            batch_list.extend(copy.deepcopy(self._list[0:self._current]))
        if self._current >= listsize:
            random.shuffle(self._list)
            self._current = 0
        return batch_list




    def _load_next_batch(self):
        batchlist = self._get_batch_list()
        #load img to fill blobs
        blobs = {}
        blob_img_data = np.zeros((self._batch_size, self._height, self._width, self._channels),dtype=np.float32)
        blob_label_data = np.zeros((self._batch_size, self._width), dtype= np.float32)

        for i in range(self._batch_size):
            imgfile = batchlist[i]
            imgfile = imgfile.replace('\r', '')
            labelfile = imgfile.replace('.bmp', '.label')
            file = open(labelfile, 'r')
            labelstr = file.read()
            file.close()    
            for j in range(self._width):
                if labelstr[j] == '1': blob_label_data[i,j] = 1
            img = cv2.imread(imgfile, 0)#.encode('utf-8')

            #resize to fit the net
            img_res = cv2.resize(img,(self._width,self._height),interpolation = cv2.INTER_CUBIC)
            # cv2.imshow('dst',img)
            # cv2.imshow('src',img_res.astype(np.uint8))
            # cv2.waitKey()
            #continue        

            blob_img_data[i,:,:,0] = img_res.astype(np.float32)  
            #blob_label_data[i,:] = label

        channel_swap = (0,3,1,2)
        blob_img_data = blob_img_data.transpose(channel_swap)
        # blob_img_data -= 128.0
        blobs['data'] = blob_img_data
        blobs['label'] = blob_label_data

        return blobs



if __name__ == "__main__":
    param_str = "{'batch_size': 256, 'height':32, 'width':32, 'source':'/home/wanwuming/ocr1/data/ocr_idcard/list/name_address_label_id.txt', 'root_folder':''}"
    t = PythonDatalayer()
    t._source = '/home/wanwuming/ocr1/data/ocr_seg/imagelabel.list'
    t._root_folder = ''
    t._batch_size = 10 
    t._height = 5
    t._width = 512
    t._channels = 1
    t._get_source_list()
    for i in range(20):
        t._load_next_batch()
    
    