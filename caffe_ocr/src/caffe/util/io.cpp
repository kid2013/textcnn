#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/util/augmentation.hpp"
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;
bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
#define PRINT printf("%s %d\n", __FILE__, __LINE__);
void RandRect(int& x1, int& x2, int& y1, int& y2, int maxr)
{
  int xr = (rand() % (maxr * 2 + 1)) - maxr;
  maxr += 2;
  int yr = (rand() % (maxr * 2 + 1)) - maxr;
  x1 += xr;
  x2 += xr;
  y1 += yr;
  y2 += yr;
}
void AddSLineNoise(cv::Mat& m)
{
  if(rand()%2) return;
  int n = 15;

  int k;
  //showimage(m);
  k = (rand() % n) + 1;
  for(int i = 0; i < k; i++)
  {
    int s = rand() % m.cols;
    int t = s;
    t = rand() % 2 == 0 ? s - 1: s+ 1;
    if(t < 0 || t >= m.cols)
      continue;
    int l2 = rand() % 5 + 1;
    int t2 = rand() % (m.rows - l2);

    cv::Mat tm = m.colRange(s, s+1).rowRange(t2, t2 + l2);
    tm.copyTo(m.colRange(t, t + 1).rowRange(t2, t2 + l2));
    if(rand() % 3 == 0)
      tm += (rand() % 128) - 64;
  }
  k = (rand() % n) + 1;
  for(int i = 0; i < k; i++)
  {
    int s = rand() % m.rows;
    int t = s;
    t = rand() % 2 == 0 ? s - 1: s+ 1;
    if(t < 0 || t >= m.rows)
      continue;
    int l2 = rand() % 5 + 1;
    int t2 = rand() % (m.cols - l2);

    cv::Mat tm = m.rowRange(s, s+1).colRange(t2, t2+l2);
    tm.copyTo(m.rowRange(t, t + 1).colRange(t2, t2+l2));
    if(rand() % 3 == 0)
      tm += (rand() % 128) - 64;
  }
  //showimage(m);
}
void RandResize(cv::Mat& m)
{
    float s = (rand() % 50 + 50) / 100.0f;
    int w = m.cols * s;
    int h = m.rows * s;
    cv::Mat sized;
    cv::resize(m, sized, cv::Size(w, h));
    cv::resize(sized, m, m.size());
}
void MotionBlur(cv::Mat& m)
{
  cv::Mat im = m.clone();
  int maxx = 5;
  int maxy = 5;
  int dx = rand() % (2 * maxx + 1) - maxx;
  int dy = rand() % (2 * maxy + 1) - maxy;
  if(dx < 0)
  {
    cv::Mat mv = im.colRange(0, dx + im.cols);
    mv.copyTo(im.colRange(-dx, im.cols));
  }
  else
  {
    cv::Mat mv = im.colRange(dx, im.cols);
    mv.copyTo(im.colRange(0, im.cols - dx));
  }
  if(dy < 0)
  {
    cv::Mat mv = im.rowRange(0, dy + im.rows);
    mv.copyTo(im.rowRange(-dy, im.rows));
  }
  else
  {
    cv::Mat mv = im.rowRange(dy, im.rows);
    mv.copyTo(im.rowRange(0, im.rows - dy));
  }

  float scale = rand() % 40 / 100.0;

  m = m * (1 - scale) + im * scale;
}


cv::Mat ReadDataAugToCVMat(const string& filename, const int x1_label, const int y1_label, 
    const int x2_label, const int y2_label,
    const int height, const int width, 
    const bool is_color, const bool is_resize_pad){
  int x1 = x1_label;
  int x2 = x2_label;
  int y1 = y1_label;
  int y2 = y2_label;

  int max_rotate_angle = 15;
  cv::RNG rng(cv::getTickCount());
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) 
  {
    std::cout<< "Could not open or find file " << filename;
    return cv_img_origin;
  }
  //printf("%s, %d, %d, %d, %d\n", filename.c_str(), x1, x2, y1, y2);
  RandResize(cv_img_origin);

  //printf("enter\n");
  // shape trans
  int trans_type = rng.next() % 3;
  int rotate_angle = (rng.next() % max_rotate_angle) * 2 - max_rotate_angle;
  switch(trans_type){
    case 0:
      if(x2 - x1 > 10 && y2 - y1 > 10) 
        AddRotateNoise(cv_img_origin, x1, y1, x2, y2, rotate_angle);
      break;
    default:
      break;
  }
  
  // AddWarpNoise(x1, x2, y1, y2, cv_img_origin);
  
  if(rand() % 2 < 10)
  {
    int max_rand = 3;
    RandRect(x1, x2, y1, y2, max_rand);
  }
  else
  {
    int roi_width = x2 - x1;
    int roi_height = y2 - y1;
    x1 +=  rng.uniform(-0.1,0.05) * roi_width;
    x2 +=  rng.uniform(-0.05,0.1) * roi_width;
    y1 +=  rng.uniform(-0.1,0.1) * roi_height;
    y2 +=  rng.uniform(-0.1,0.1) * roi_height;
  }
  x1 = std::max(0, x1);
  x2 = std::max(x1+5, x2);
  x2 = std::min(cv_img_origin.cols, x2);
  y1 = std::max(0, y1);
  y2 = std::max(y1+5, y2);
  y2 = std::min(cv_img_origin.rows, y2);
  //printf("%d, %d, %d, %d; %d, %d\n", x1, x2, y1, y2, cv_img_origin.cols, cv_img_origin.rows);
  cv::Rect rt(x1, y1, x2-x1, y2-y1);
  if(rt.width + rt.x > cv_img_origin.cols) rt.width = cv_img_origin.cols - rt.x;
  if(rt.height + rt.y > cv_img_origin.rows) rt.height = cv_img_origin.rows - rt.y;
  
  MotionBlur(cv_img_origin);
  cv::Mat cv_img_roi = cv_img_origin(rt);
 
 //  if(rand() % 2) rand() % 2 ? AddRandNoise(cv_img_roi, 64): AddLine2Noise(cv_img_roi, 64);
 // if(rand() % 2) rand() % 2 ? AddRandNoise(cv_img_roi, 64): AddLine2Noise(cv_img_roi, 64);
  //if(rand() % 2) AddRandNoise(cv_img_roi, 64);
  //if(rand() % 2) AddRandNoise(cv_img_roi, 64);
  int noise_type = rng.next() % 12;
  //noise_type = 1;
  int kernelGaussianBlur = (rng.next() % 3+1) * 2 + 1;
  double sigmaDefault = 0.3*((kernelGaussianBlur-1)*0.5 - 1) + 0.8;
  double sigmaX = rng.uniform(sigmaDefault/2, sigmaDefault);
  double sigmaY = sigmaX;
  double gaussianMean = rng.uniform(0.05,0.1);
  double gaussianSigma = rng.uniform(gaussianMean,0.2);
  switch(noise_type){
    case 0:
      AddGaussianNoise(cv_img_roi, gaussianMean, gaussianSigma);
      break;
    case 1:
      AddPepperSaltNoise(cv_img_roi, rng.uniform(0.05,0.1), 128);
      break;
    case 2:
      AddGaussianNoise(cv_img_roi, gaussianMean, gaussianSigma);
      cv::blur(cv_img_roi, cv_img_roi, cv::Size(3, 3));
      break;
    case 3:
      AddPepperSaltNoise(cv_img_roi, rng.uniform(0.05,0.1), 128);
      cv::blur(cv_img_roi, cv_img_roi, cv::Size(3, 3));
      break;
    case 4:
      AddGaussianNoise(cv_img_roi, gaussianMean, gaussianSigma);
      cv::GaussianBlur(cv_img_roi, cv_img_roi, cv::Size(kernelGaussianBlur, kernelGaussianBlur), sigmaX, sigmaY);
    break;
    case 5:
      AddPepperSaltNoise(cv_img_roi, rng.uniform(0.05, 0.05), 128);
      break;
    case 6:
      AddPepperSaltNoise(cv_img_roi, rng.uniform(0.05, 0.05), 128);
      cv::GaussianBlur(cv_img_roi, cv_img_roi, cv::Size(kernelGaussianBlur, kernelGaussianBlur), sigmaX, sigmaY);
      break;
    case 7:
      AddPepperSaltNoise(cv_img_roi, rng.uniform(0.01,0.05), 64);
      cv::GaussianBlur(cv_img_roi, cv_img_roi, cv::Size(kernelGaussianBlur, kernelGaussianBlur), sigmaX, sigmaY);
      break;
    case 8:
        cv::blur(cv_img_roi, cv_img_roi, cv::Size(5, 5));
        break;
    case 9:
        cv::GaussianBlur(cv_img_roi, cv_img_roi, cv::Size(kernelGaussianBlur, kernelGaussianBlur), sigmaX, sigmaY);
        break;
    default:
    break;
  }

  int color_type = rng.next() % 4;
  switch(color_type) {
    case 0:
      AddGammaNoise(cv_img_roi, rng.uniform(1.0,1.5));
      break;
    case 10:
      cv_img_roi.convertTo(cv_img_roi, CV_8UC1, rng.uniform(0.4,0.6), rng.next()%30 + 20);
      AddGammaNoise(cv_img_roi, 0.5);
     break;
    default:
      break;
  }
  int resize_type = rng.next() % 3;


  if (height > 0 && width > 0){
    cv::resize(cv_img_roi, cv_img, cv::Size(width, height));
  }
  // AddSLineNoise(cv_img);
//  AddWarp2Noise(cv_img);
  // norm_image(cv_img);
//  AddLightNoise(cv_img);
  // cv::imshow("", cv_img);
  // cv::waitKey(0);
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    std::cout << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  // norm_image(cv_img);
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}
#endif  // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}
#endif  // USE_OPENCV
}  // namespace caffe
