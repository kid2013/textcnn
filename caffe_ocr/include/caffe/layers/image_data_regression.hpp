#ifndef CAFFE_IMAGE_DATA_LAYER_HPP_
#define CAFFE_IMAGE_DATA_LAYER_HPP_
#include <pthread.h>
#include <string>
#include <utility>
#include <vector>
#include<iostream>
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/augmentation.hpp"
namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageDataRegressionLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageDataRegressionLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageDataRegressionLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  
  void load_trainlist(string source, int regression_num, std::vector<string>& vlabel)
  {
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    string line;
    size_t pos;
    size_t last_pos;
    size_t first_pos;

    while (std::getline(infile, line)) {
      vector<float> labels;
      pos = line.find_first_of(' ');
      float label;
      last_pos = line.find_first_of(' ');
      for (int i = 0; i < regression_num; ++i) {
        pos = line.find_first_of(' ', last_pos+1);
        if (0 == i) {
          string slabel = line.substr(last_pos+1, pos);
          //printf("%d\n", vlabel.size());
          //std::cout << slabel <<" ";
          if(vlabel.size() > 0)
          {
            label = list_find(vlabel, slabel);
            //printf("label: %f\n", label);
          }
          else
          {
            label = static_cast<float>(atoi(slabel.c_str()));
          }
        }
        else {
          label = atof(line.substr(last_pos+1, pos).c_str());
        }
        labels.push_back(label);
        last_pos = pos;
      }
      if(labels[0] < 0) continue;
      first_pos = line.find_first_of(' ');
      //printf("%s, %.0f\n", line.substr(0, first_pos).c_str(), labels[0]);
      lines_.push_back(std::make_pair(line.substr(0, first_pos), labels));
    }
  }
  
  vector<std::pair<std::string, std::vector<float> > > lines_;
  int lines_id_;
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
