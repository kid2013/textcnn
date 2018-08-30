#include "caffe/util/augmentation.hpp"
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"


#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include <string>

#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#define ACCESS access
#define MKDIR(a) mkdir((a), 0755)

using namespace cv;
using namespace std;

using std::vector;

using cv::Mat;
using cv::Rect;
using cv::Point;
using cv::Point2f;


int create_dir_from_filename(std::string filename)
{return create_dir_from_filename(filename.c_str());}


void get_path(const char* filename, char* path)
{
	strcpy(path, filename);
	int len = strlen(path);
	while(len--)
	{
		if(path[len] == '\\' || path[len] == '/')
		{
			path[len] = '\0';
			break;
		}
	}
	if(len == -1)
		path[0] = '\0';
}
int create_dir_from_filename(const char* filename)
{
	char path[512];
	get_path(filename, path);
	int len = strlen(path);
	for(int i = 1; i < len; i++)
	{
		if (path[i] == '\\' || path[i] == '/')
		{
			path[i] = '\0';
			int iRet = ACCESS(path, 0); 
			if(iRet != 0)
			{
				printf("mkdir: %s\n", path);
				iRet = MKDIR(path);
				if (iRet != 0)
					return -1;
			}
			 path[i] = '/';
		}
	}
	return MKDIR(path);
}
void showimage(Mat m, int t)
{
	imshow("", m);
	imwrite("debug.bmp", m);
	cvWaitKey(t);
}
void imageshow(Mat m, int t)
{
	showimage(m, t);
}

std::string strip(std::string src)
{
	int len = src.length();
	char* buf = new char[len + 1];
	strcpy(buf, src.c_str());
	int start = 0;
	for(int i = len - 1; i >= 0; i--)
	{
		if(buf[i] != ' ' && buf[i] != '\t'  && buf[i] != '\r' && buf[i] != '\n')
		{
			buf[i+1] = '\0';
			break;
		}
	}
	for(int i = 0; i < len; i++)
	{
		if(buf[i] != ' ' && buf[i] != '\t' && buf[i] != '\r'  && buf[i] != '\n')
		{
			start = i;
			break;
		}
	}
	std::string dst = buf;
	delete[] buf;
	return dst;
}	
int txtread(const char* filename, std::vector<std::string>& listfile)
{
	FILE* fp = fopen(filename, "r");
	if(fp == NULL)
	{
		printf("txt file: %s is not EXIST! ERROR in FUNCTION: txtread\n", filename);
		return 1;
	}
	char buf[256];
	int i = 0;
	while(fp && fgets(buf, 255, fp))
	{
		if(i++ == 0)
		{
			if(strlen(buf) > 3)
			{
				if((buf[0] == -17) && (buf[1] == -69) && (buf[2] == -65))
				{
					strcpy(buf, buf+3);
				}
			}
		}

		std::string s = buf;
		if(s == "")
			continue;
		//if(buf[0] == '#')
		//	continue;

		int iend = strlen(buf)-1;
		if(iend >= 0 && buf[iend] == '\n')
			buf[iend] = 0;
		if(iend >= 1 && buf[iend-1] == '\r')
			buf[iend-1] = 0;
		s = buf;
		s = strip(s);
		if(s == "")
			continue;
		listfile.push_back(s);
	}
	fclose(fp);
	return 0;
}

void split(string s, vector<string>& slist, char tag)
{
	int k = 0;
	slist.clear();
	int len = s.length();
	char buf[256];
	for(int i = 0; i < len; i++)
	{
		if(s[i] != tag) buf[k++] = s[i];
		else
		{
			if(k > 0)
			{
				buf[k] = '\0';
				slist.push_back(buf);
				k =  0;
			}
		}
	}
	if(k > 0)
	{
		buf[k] = '\0';
		slist.push_back(buf);
	}
}
void norm_image(Mat& m)
{
    int hist[256] = {0};
    int total = m.size().area();
    uchar* d = m.data;
    for(int i = 0; i < total; i ++)
        hist[*d++]++; 
    float rato = 0.005;
    int minp = 0, maxp = 255;
    for(int i = 0, t = 0, cnt = 0; i < 255; i++)
    {
        cnt += hist[i];
        minp = i;
        if(cnt > total * rato)
        {
            if(t > 0)
                break;
            cnt = 0;
            t++;
        }
    }
    for(int i = 255, t = 0, cnt = 0; i >= 0; i--)
    {
        cnt += hist[i];
        maxp = i;
        if(cnt > total * rato)
        {
            if(t > 0)
                break;
            cnt = 0;
            t++;
        }
    }
    maxp = MAX(minp + 1, maxp);
    m = (m - uchar(minp)) * (255.0 / (maxp - minp));
}

int list_find(std::vector<std::string>& list, std::string s)
{
    // printf("list_find: %s\n", s.c_str());
    std::vector<std::string> vs;
    split(s, vs, ' ');
    s = strip(vs[0]);
    int rtn = -1;
	for(int i = 0; i < list.size(); i++)
	{
        // printf("%s, %s\n", list[i].c_str(), s.c_str());
        // printf("%d, %d\n", list[i].length(), s.length());
        // for(int k = 0; k < s.length(); k++)
        // {
        //     printf("k=%d, %d\n", k, s[k]);
        // }
		if(list[i] == s)
        {
            rtn = i;
            break;
        }
	}
	return rtn;
}

void fileparts(string filename, string& path, string& name, string& ext)
{
	int pos1 = filename.find_last_of("/");
	int pos2 = filename.find_last_of("\\");
	int pos_start = pos1 > pos2 ? pos1 : pos2;
	pos_start++;
	path = filename.substr(0, pos_start);

	int pos_end = filename.find_last_of('.');
	if(pos_end < pos_start)
	{
		name = filename.substr(pos_start, -1);
		ext = "";
	}
	else
	{
		name = filename.substr(pos_start, pos_end - pos_start);
		ext = filename.substr(pos_end, -1);
	}
}
string get_shortname(string filename)
{
	string path, name, ext;
	fileparts(filename, path, name, ext);
	return name;
}
void AddLineNoise(cv::Mat& img, const int minval, const int maxval, const int width, const int line_num) {
    cv::RNG rng;
    // float ave = cv::mean(img)[0];
    for (int i = 0; i < line_num; ++i) {
        int ystart = rng.next() % img.rows;
        int ystop = rng.next() % img.cols;
        int colorval = (rng.next()%(maxval-minval)+minval);
        cv::Mat tmp(img.size(), CV_8UC1, cv::Scalar(0));
        cv::line(tmp, cv::Point(0, ystart), cv::Point(img.cols-1, ystop), cv::Scalar(colorval), width);
        img += tmp;
    }
}

void AddDilateNoise(cv::Mat & img, const int kernelSize) {
  cv::Mat element(kernelSize, kernelSize, CV_8UC1, cv::Scalar(1,1,1));
  cv::dilate(img, img, element);
}

void AddErodeNoise(cv::Mat & img, const int kernelSize) {
    cv::Mat element(kernelSize, kernelSize, CV_8UC1, cv::Scalar(1,1,1));
    cv::erode(img, img, element);
}

 double GenerateGaussianNoise(double mu, double sigma)
{
  
    const double epsilon = std::numeric_limits<double>::min();
    static double z0;
    // static double z1;
    // static bool flag = false;
    // flag = !flag;
    // if (!flag)
    //     return z1 * sigma + mu;
    double u1, u2;
    do
    {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    } while (u1 <= epsilon);
    z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
    // z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);

    return z0*sigma + mu;
}

void AddGaussianNoise(cv::Mat& img, double mean, double normalized_sigma)
{
    if (img.empty()) {
        printf("image empty \n");
        return;
    }
    cv::Mat result;
    cv::Mat noise = Mat(img.size(), CV_64F);
    cv::normalize(img, result, 0.0, 1.0, CV_MINMAX, CV_64F);
    cv::randn(noise, mean, normalized_sigma);
    result = result + noise;
    cv::normalize(result, img, 0, 255, CV_MINMAX, CV_8UC1);
}


inline void ValConcat(int& val, const int min, const int max) {
    val = std::max(val, min);
    val = std::min(val, max);
}

void AddPepperSaltNoise(cv::Mat& img, const double percentage, const int maxnoiseval){  
    if (img.depth() != CV_8U){
        // LOG(ERROR) << "input img should be of type CV_8UC3 or CV_8UC1, pepper salt noise not added";
        return;
    }
    cv::RNG rng;
    int num_noise = static_cast<int>(img.rows*img.cols*percentage);
    for(int index_noise = 0; index_noise < num_noise; ++index_noise)  
    {  
        int i = rng.next() % img.cols;  
        int j = rng.next() % img.rows;  
        if(img.channels() == 1)  
        {  
            int pixval = img.at<uchar>(j,i) + rng.next() % (2*maxnoiseval)  - maxnoiseval;
            ValConcat(pixval, 0, 255);
            img.at<uchar>(j,i) = pixval;
        }else  
        {   
            int pixval = img.at<cv::Vec3b>(j,i)[0] + rng.next() % (2*maxnoiseval) - maxnoiseval;
            ValConcat(pixval, 0, 255);
            img.at<cv::Vec3b>(j,i)[0] = pixval;
            pixval = img.at<cv::Vec3b>(j,i)[1] + rng.next() % (2*maxnoiseval) - maxnoiseval;
            ValConcat(pixval, 0, 255);
            img.at<cv::Vec3b>(j,i)[1] = pixval;
            pixval = img.at<cv::Vec3b>(j,i)[2] + rng.next() % (2*maxnoiseval) - maxnoiseval;
            ValConcat(pixval, 0, 255);
            img.at<cv::Vec3b>(j,i)[2] = pixval;
        }  
    }  
}  

int AddWarpNoise(cv::Mat& m, vector<cv::Point2f>& spt)
{ 

    int w = spt[2].x - spt[0].x;
    int h = spt[1].y - spt[0].y;
    // vector<cv::Point2f> spt(4);  
    // spt[0] = Point2f(x1, y1);  
    // spt[1] = Point2f(x1, y2);  
    // spt[2] = Point2f(x2, y1);  
    // spt[3] = Point2f(x2, y2);  

    vector<cv::Point2f> dpt = spt;
    dpt[1].x += (rand() % w - 0.5 * w) * 0.1;
    dpt[1].y += (rand() % h - 0.5 * h) * 0.1;
    dpt[2].x += (rand() % w - 0.5 * w) * 0.1;
    dpt[2].y += (rand() % h - 0.5 * h) * 0.1;
    dpt[3].x += (rand() % w - 0.5 * w) * 0.1;
    dpt[3].y += (rand() % h - 0.5 * h) * 0.1;
    
    // if(dpt[1].x < 0 || dpt[2].x < 0 || dpt[3].x < 0 || dpt[1].y < 0 || dpt[2].y < 0 || dpt[3].y < 0)
    //     return;
    cv::Mat transform = cv::getPerspectiveTransform(spt, dpt);  
    cv::Mat img_trans;
    cv::warpPerspective(m, m, transform, m.size());
    spt = dpt;
    // x1 = std::min(dpt[0].x, dpt[1].x);
    // y1 = std::min(dpt[0].y, dpt[2].y);
    // x2 = std::min(dpt[2].x, dpt[3].x);
    // y2 = std::min(dpt[1].y, dpt[3].y);
    
    return 0;  
}

void AddRotateNoise(Mat &image, vector<cv::Point2f>& spt, const int angle){
    int rows = image.rows;
    int cols = image.cols;

    int w = spt[2].x - spt[0].x;
    int h = spt[1].y - spt[0].y;
    if (cols < int(w*1.8) || rows < int(h*1.8)){
        return;
    }

    Mat M = cv::getRotationMatrix2D(Point(cols/2, rows/2), angle, 1);
    cv::warpAffine(image, image, M, cv::Size(cols, rows));
    vector<Point2f> dpt;
    cv::transform(spt, dpt, M);
    spt = dpt;
 }
int AddWarpNoise(int& x1, int& x2, int& y1, int& y2, cv::Mat& m)
{ 
    int w = x2 - x1;
    int h = y2 - y1;
    vector<cv::Point2f> spt(4);  
    spt[0] = Point2f(x1, y1);  
    spt[1] = Point2f(x1, y2);  
    spt[2] = Point2f(x2, y1);  
    spt[3] = Point2f(x2, y2);  

    vector<cv::Point2f> dpt = spt;
    dpt[1].x += (rand() % w - 0.5 * w) * 0.1;
    dpt[1].y += (rand() % h - 0.5 * h) * 0.1;
    dpt[2].x += (rand() % w - 0.5 * w) * 0.1;
    dpt[2].y += (rand() % h - 0.5 * h) * 0.1;
    dpt[3].x += (rand() % w - 0.5 * w) * 0.1;
    dpt[3].y += (rand() % h - 0.5 * h) * 0.1;
  
    cv::Mat transform = cv::getPerspectiveTransform(spt, dpt);  
    cv::Mat img_trans;
    cv::warpPerspective(m, m, transform, m.size());
    x1 = std::min(dpt[0].x, dpt[1].x);
    y1 = std::min(dpt[0].y, dpt[2].y);
    x2 = std::min(dpt[2].x, dpt[3].x);
    y2 = std::min(dpt[1].y, dpt[3].y);
    
    return 0;  
}
void AddRotateNoise(Mat &image, int &x1, int &y1, int &x2, int& y2, const int angle){
    int rows = image.rows;
    int cols = image.cols;

    int width = x2 - x1 + 1;
    int height = y2 - y1 + 1;

    if (cols < int(width*1.8) || rows < int(height*1.8)){
        return;
    }

    Mat M = cv::getRotationMatrix2D(Point(cols/2, rows/2), angle, 1);
    
    vector<Point2f> srcPts;
    srcPts.push_back(Point2f(x1, y1));
    srcPts.push_back(Point2f(x2, y1));
    srcPts.push_back(Point2f(x1, y2));
    srcPts.push_back(Point2f(x2, y2));
    vector<Point2f> dstPts;
    int x1_rotated, x2_rotated, y1_rotated, y2_rotated;
    cv::transform(srcPts, dstPts, M);
    for (int i = 0; i < 4; ++i) {
        x1_rotated = std::min(x1, int(dstPts[i].x));
        x2_rotated = std::max(x2, int(dstPts[i].x));
    }
    for (int i = 0; i < 4; ++i) {
        y1_rotated = std::min(y1, int(dstPts[i].y));
        y2_rotated = std::max(y2, int(dstPts[i].y));
    }

    if (x1_rotated < 0 || x2_rotated >= cols || y1_rotated < 0 || y2_rotated >= rows) {
        return;
    }

    cv::warpAffine(image, image, M, cv::Size(cols, rows));
    x1 = x1_rotated;
    x2 = x2_rotated;
    y1 = y1_rotated;
    y2 = y2_rotated;
 }

void AddScaleNoise(Mat& image, int &x1, int &y1, int &x2, int& y2, const float rowscale,
    const float colscale) {
    cv::resize(image, image, cv::Size(int(image.cols*colscale), int(image.rows*rowscale)));
    x1 = int(x1 * colscale);
    y1 = int(y1 * rowscale);
    x2 = int(x2 * colscale);
    y2 = int(y2 * rowscale);
}

 void AddGammaNoise(cv::Mat& image, const float gamma)
{
    unsigned char lut[256];
 
    for (int i = 0; i < 256; i++)
    {
 
        lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
 
    }
    
    const int channels = image.channels();
    switch (channels)
    {
        case 1:
        {
             cv::MatIterator_<uchar> it, end;
            for (it = image.begin<uchar>(), end = image.end<uchar>(); it != end; it++)
            *it = lut[(*it)];
            break;
        }
        case 3:
        {
            cv::MatIterator_<cv::Vec3b> it, end;
            for (it = image.begin<cv::Vec3b>(), end = image.end<cv::Vec3b>(); it != end; it++)
            {
                (*it)[0] = lut[((*it)[0])];
                (*it)[1] = lut[((*it)[1])];
                (*it)[2] = lut[((*it)[2])];
            }
            break;
        }
    }
}
#define PRINT ;//printf("%s %d\n", __FILE__, __LINE__);
static void static_load_image(string listfile, vector<Mat>& mlist)
 {
    vector<string> vlist;
    txtread(listfile.c_str(), vlist);
    for(int i  = 0; i < vlist.size(); i++)
    {
    	Mat m = imread(vlist[i], 0);
    	if(m.data) mlist.push_back(m);
    }
 }
static Mat get_randmat(vector<Mat>& vmat, Rect rt)
{
	Mat tmp;
	Mat m = vmat[rand() % vmat.size()];
	if(m.cols - rt.width <= 0 || m.rows - rt.height <= 0)
	{
		resize(m, tmp, Size(rt.width, rt.height));
	}
	else
	{
		rt.x = rand() % (m.cols - rt.width);
		rt.y = rand() % (m.rows - rt.height);
		tmp = m(rt);
	}

	return tmp;
}
void AddRandNoise(cv::Mat& image, const int maxvalue)
{
    static vector<Mat> mlist;
    if(mlist.size() == 0)
    {
        string listfile = "/home/wanwuming/ocr1/caffe_cls/caffe_ocr/src/noise/rand.list";
        //string listfile = "/home/wanwuming/ocr1/caffe_cls/caffe_ocr/src/noise/idcardnoise.list";
        static_load_image(listfile, mlist);
    }
    Rect rt;
	rt.width = image.cols * random::randf(0.8, 1.2);
	rt.height = image.rows * random::randf(0.8, 1.2);
	Mat sized = get_randmat(mlist, rt);
    cv::resize(sized, sized, cv::Size(image.cols, image.rows));

	if(rand()%2)sized = 255-sized;
    double minp,maxp;
    minMaxIdx(sized, &minp, &maxp);  
    sized = (sized - uchar(minp)) * (rand() % maxvalue + maxvalue/4) / uchar(maxp - minp + 1);
    Scalar s = sum(sized);
    image += sized;
    image -= s[0] / sized.size().area();

	norm_image(sized);
	float fs = random::randf(4, 8);
	sized /= fs;
}

void AddLine2Noise(cv::Mat& image, const int maxvalue)
{
    static vector<Mat> mlist;
    if(mlist.size() == 0)
    {
        string listfile = "/home/wanwuming/ocr1/caffe_cls/caffe_ocr/src/noise/line.list";
        //string listfile = "/home/wanwuming/ocr1/caffe_cls/caffe_ocr/src/noise/idcardnoise.list";
        static_load_image(listfile, mlist);
    }
    Rect rt;
	rt.width = image.cols * random::randf(0.8, 1.2);
	rt.height = image.rows * random::randf(0.8, 1.2);
	Mat sized = get_randmat(mlist, rt);
    cv::resize(sized, sized, cv::Size(image.cols, image.rows));

	double minp,maxp;
    minMaxIdx(sized, &minp, &maxp);  

    sized = (sized - uchar(minp)) * (rand() % maxvalue + maxvalue/4) / uchar(maxp - minp + 1);
    Scalar s = mean(sized);
    image += sized;
    image -= s[0];
}

uchar setvalue(float x)
{
  if(x > 255) return 255;
  if(x < 0) return 0;
  return uchar(x);
}
void AddLight(cv::Mat m)
{
  float scale = ((rand() % 20) - 10) * 0.1;
  int w = m.cols / 2;
  int h = m.rows / 2;
  for(int j = 1; j < h; j++)
  {
    float fy = j * 1.0 / h;
    for(int i = 1; i < w; i++)
    {
      float fx = i * 1.0 / w;
      float rato = scale * fx * fy + 1;
      uchar uc1, uc2, uc3, uc4;
      uc1 = m.at<uchar>(j, i);
      uc2 = m.at<uchar>(m.rows - 1 - j,  i);
      uc3 = m.at<uchar>(j, m.cols - 1 -i);
      uc4 = m.at<uchar>(m.rows - 1 - j, m.cols - 1 - i);

      m.at<uchar>(j, i) = setvalue(uc1 * rato);
      m.at<uchar>(m.rows - 1 - j,  i) = setvalue(uc2 * rato);
      m.at<uchar>(j, m.cols - 1 -i) = setvalue(uc3 * rato);
      m.at<uchar>(m.rows - 1 - j, m.cols - 1 - i) = setvalue(uc4 * rato);
    }
  }
  
}
void AddLightNoise(cv::Mat& m)
{
  if(rand() % 2) return;
  int n =  rand() % 2 + 1;
  for(int i = 0; i < n; i++)
  {
    cv::Rect rt;
    rt.x = rand() % m.cols;
    rt.y = rand() % m.rows;
    rt.width = rand() % (m.cols - 5) + 5;
    rt.height = rand() % (m.rows - 5) + 5;
    rt.width &= 0xfe;
    rt.height &= 0xfe;
    if(rt.x + rt.width > m.cols) rt.x = m.cols - rt.width;
    if(rt.y + rt.height > m.rows) rt.y = m.rows - rt.height;
    AddLight(m(rt));
  }
  if(rand() % 2){
    Rect rt;
    rt.x = rand() % 2;
    rt.width = rand() % 2 + 2;
    rt.y = rand() % 8;
    rt.height = rand() % 24;
    //AddLight(m(rt));
  } 
}
void ResizeWithPad(const cv::Mat& img, cv::Mat& img_resize_padded, const int height, const int width, const int resize_type){
    float rowscale = static_cast<float>(height) / static_cast<float>(img.rows);
    float colscale = static_cast<float>(width) / static_cast<float>(img.cols);
    if (fabs(rowscale-colscale) <= 1e-6){
        // no need to pad
        cv::resize(img, img_resize_padded, cv::Size(width, height), 0, 0, rand() % 5);
    }
    else if (rowscale < colscale){ 
        //scale to height, pad width
        int tmp_width = static_cast<int>(img.cols * rowscale);
        if (tmp_width == width){
            // no need to pad
            cv::resize(img, img_resize_padded, cv::Size(width, height), 0, 0, resize_type);
        }
        else{
            cv::Mat cv_img_tmp_resize;
            cv::resize(img, cv_img_tmp_resize, cv::Size(tmp_width, height), 0, 0, resize_type);
            cv::Mat cv_img_tmp;
            // pad after scale
            cv::copyMakeBorder(cv_img_tmp_resize, cv_img_tmp, 
                (width-tmp_width)/2, (width-tmp_width)/2, 0, 0, cv::BORDER_REPLICATE);
            // resize again, make sure that cv_img.rows = height && cv_img.cols == height
            cv::resize(cv_img_tmp, img_resize_padded, cv::Size(width, height), 0, 0, resize_type);
        }
    }
    else{
        //scale to width, pad height
        int tmp_height = static_cast<int>(img.rows * colscale);
        if (tmp_height == height){
            // no need to pad
            cv::resize(img, img_resize_padded, cv::Size(width, height), 0, 0, resize_type);
        }
        else{
            cv::Mat cv_img_tmp_resize;
            cv::resize(img, cv_img_tmp_resize, cv::Size(width, tmp_height), 0, 0, resize_type);
            cv::Mat cv_img_tmp;
            // pad after scale
            cv::copyMakeBorder(cv_img_tmp_resize, cv_img_tmp, 
                0, 0, (height-tmp_height)/2, (height-tmp_height)/2, 0, cv::BORDER_REPLICATE);
            // resize again, make sure that cv_img.rows = height && cv_img.cols == height
            cv::resize(cv_img_tmp, img_resize_padded, cv::Size(width, height), 0, 0, resize_type);
        }
    }
}

class Distort
{
private:
    Mat mapx, mapy;
    bool rt_flag_;
public:
    void show(Mat& im, int t = 0)
    {
        imshow("", im);
        cvWaitKey(t);
    }
public:
    Distort(){init(1);}
    void init(bool rt_flag = false)
    {
        rt_flag_ = rt_flag;
    }
    int coss_product(Point2f p0, Point2f p1, Point2f p)
    {
        // 
        //Point2f p1 = pt1 - pt0;
        //Point2f p2 = pt2 - pt0;
        p0 -= p;
        p1 -= p;
        return p0.x * p1.y - p0.y * p1.x;
    }
    int is_inner_rect(Point2f p0, vector<Point2f>& pt)
    {
        if( coss_product(pt[0], pt[1], p0) >= 0 &&
            coss_product(pt[1], pt[3], p0) >= 0 && 
            coss_product(pt[3], pt[2], p0) >= 0 &&
            coss_product(pt[2], pt[0], p0) >= 0)
            return 1;
        return 0;
    }
    void get_control_point(int w, int h, int grid, vector<Point>& spt, vector<Point>& dpt)
    {
        spt.clear();
        dpt.clear();
        int xstep = w / grid;
        int ystep = h / grid;
        for(int j = 0; j < grid; j++)
        {
            for(int i = 0; i < grid; i++)
            {
                spt.push_back(Point(i * xstep, j * ystep));
            }
            spt.push_back(Point(w, j * ystep));
        }
        for(int i = 0; i < grid; i++)
        {
            spt.push_back(Point(i * xstep, h));
        }
        spt.push_back(Point(w, h));


        dpt = spt;
        for(int j = 0, k = 0; j <= grid; j++)
        {
            int dx = (rand() % 5 - 2) * (rand() % 2 ? 1 : 0.5);
            int dy = (rand() % 5 - 2) * (rand() % 2 ? 1 : 0.5);
            for(int i = 0; i <= grid; i++)
            {
                if(dpt[k].x != 0 && dpt[k].x != w)
                {
                    dpt[k].x += (rt_flag_ ? dx : (rand() % 5 - 2) * (rand() % 2 ? 1 : 0.5));
                }
                if(dpt[k].y != 0 && dpt[k].y != h)
                {
                    dpt[k].y +=  (rt_flag_ ? dy : (rand() % 5 - 2) * (rand() % 2 ? 1 : 0.5));
                }
                k++;
            }
        }

    }
    void get_pt_rt(int grid, int x, int y, vector<Point> pt, vector<Point2f>& pt_rt)
    {
        pt_rt.clear();
        pt_rt.push_back(pt[y * (grid + 1) + x]);
        pt_rt.push_back(pt[y * (grid + 1) + x + 1]);
        pt_rt.push_back(pt[y * (grid + 1) + grid + 1 + x]);
        pt_rt.push_back(pt[y * (grid + 1) + grid + 1 + x + 1]);
    }
    void make_rect_from_pt( vector<Point2f>& pt, Rect& rt)
    {
        rt.x = std::min(pt[0].x, pt[2].x);
        rt.y = std::min(pt[0].y, pt[1].y);
        rt.width = std::max(pt[1].x, pt[3].x) - rt.x;
        rt.height = std::max(pt[2].y, pt[3].y) - rt.y;

        int i = 0;
        pt[i].x -= rt.x; i++;
        pt[i].x -= rt.x; i++;
        pt[i].x -= rt.x; i++;
        pt[i].x -= rt.x; i++;

        i = 0;
        pt[i].y -= rt.y; i++;
        pt[i].y -= rt.y; i++;
        pt[i].y -= rt.y; i++;
        pt[i].y -= rt.y; i++;
    }
    void make_map(vector<Point2f>& spt, vector<Point2f>& dpt, Mat& mapx, Mat& mapy)
    {
        Rect srt, drt;
        make_rect_from_pt(spt, srt);
        make_rect_from_pt(dpt, drt);

        cv::Mat tf = cv::getPerspectiveTransform(dpt, spt);
        cv::Mat tmp(3, 1, CV_64FC1);
        Mat mx = mapx(drt);
        Mat my = mapy(drt);

        for(int y = 0; y < mx.rows; y++)
        {
            for(int x = 0; x < mx.cols; x++)
            {
                //printf("%d, %d\n", y,x );

                if(is_inner_rect(Point2f(x, y), dpt))
                {
                    tmp.at<double>(0,0) = x;
                    tmp.at<double>(1,0) = y;
                    tmp.at<double>(2,0) = 1;
                    cv::Mat d = tf * tmp;
                    mx.at<float>(y, x) = d.at<double>(0,0)/ d.at<double>(2,0) + srt.x;
                    my.at<float>(y, x) = d.at<double>(1,0)/ d.at<double>(2,0) + srt.y;
                }
            }
        }
    }
    void Jibian(cv::Mat& m, int grid, int rt_flag = 0)
    {
        init(rt_flag);
        mapx.create(m.size(), CV_32FC1);
        mapy.create(m.size(), CV_32FC1);
        mapx = 0;
        mapy = 0;

        vector<Point> spt;
        vector<Point> dpt;

        vector<Point2f> spt_rt;
        vector<Point2f> dpt_rt;

        get_control_point(m.cols, m.rows, grid, spt, dpt);

        for(int y = 0; y < grid; y++)
        {
            for(int x = 0; x < grid; x++)
            {
                //printf("%d, %d\n", y,x );
                get_pt_rt(grid, x, y, spt, spt_rt);
                get_pt_rt(grid, x, y, dpt, dpt_rt);
                make_map(spt_rt, dpt_rt, mapx, mapy);
            }
        }
        remap(m, m, mapx, mapy, 1);
        //show(m);
    }
    void ImagePrinter2(Mat& m, int step, int max_noise = 0)
    {
        int w = m.cols;
        int h = m.rows;

        for(int y = rand() % step; y <= h - step; y += step)
        {
            for(int x = rand() % step; x <= w - step; x += step)
            {
                Rect rt(x, y, step, step);
                Mat s = m(rt);
                Scalar sm = mean(s);
                s = sm[0] + (max_noise < 2 ? 0 : ((rand() % max_noise) - max_noise / 2));
            }
        }
    }
    void ImagePrinter(Mat& m, int step, int max_noise = 0)
    {
        int w = m.cols;
        int h = m.rows;

        for(int y = 0; y < h; y += step)
        {
            for(int x = 0; x < w; x += step)
            {
                Rect rt(x, y, step, step);
                if(y + step > h) rt.height = h - y;
                if(x + step > w) rt.width = w - x;
                Mat s = m(rt);
                Scalar sm = mean(s);
                s = sm[0] + (max_noise < 2 ? 0 : ((rand() % max_noise) - max_noise / 2));
            }
        }
    }
    // void test()
    // {
    //     Mat im = imread("t.jpg", 0);
    //     int i = 0;
    //     int grid[4] = {1,2,3,8};
    //     while(1)
    //     Jibian(im.clone(), 3);
    // }
};
class Demo_words{
	Distort distort;

	vector<Mat> lines_black;
	vector<Mat> lines_white;
	vector<Mat> noises;
public:

	void warp(cv::Mat &m)
	{
		if(m.rows < 30 || m.cols < 30) return;
		if(rand() % 2)return;
		if(rand() % 2)distort.ImagePrinter(m, 2, 16);
		if(rand() % 2)distort.Jibian(m, random::randi(4,8), 0);
		/*else if(rand() % 2)*/distort.Jibian(m, random::randi(4,8), 1);
	}


	void add_line(Mat& m)
	{
		Rect rt;
		Mat line;
		int n = 0;
		if(rand() % 2)
		{
			rt.width = m.cols * random::randf(0.7, 1.5);
			rt.height = m.rows * random::randf(1, 2);
			line = get_randmat(lines_black, rt);
			resize(line, line, m.size());
			m = m - (255 - line) * random::randf(0.5, 1);
			//imageshow(m);
		}
		if(rand() % 2)
		{
			rt.width = m.cols * random::randf(0.7, 1.5);
			rt.height = m.rows * random::randf(0.7, 1.5);
			line = get_randmat(lines_white, rt);
			resize(line, line, m.size());
			m = m + line * random::randf(0.5, 1);
			//imageshow(m);
		}

	}
	void add_noise(cv::Mat &m)
	{
		add_line(m);
		warp(m);
	}
	void addblur(Mat& word)
	{
		Mat buf;
		blur(word, buf, Size(5, 5));
		word = word * 0.5 + buf * 0.5;
	}

	float st_threshold_persent(Mat& m, int x, int y, uchar thval, Size sz)
	{
		int to_cnt = 0;
		int st_cnt = 0;

		for(int j = MAX(y-sz.height/2, 0); j <= MIN(sz.height/2 + y, m.rows-1); j++)
		{
			uchar* p = m.ptr<uchar>(j);
			for(int i = MAX(x-sz.width/2, 0); i <= MIN(sz.width/2 + x, m.cols-1); i++)
			{
				to_cnt++;
				if(p[i] < thval) st_cnt++;
			}
		}
		return st_cnt / (to_cnt + 0.1);
	}

	void merge_ground(Mat& m)
	{
		Rect rt;
		rt.width = m.cols * random::randf(0.8, 1.2);
		rt.height = m.rows * random::randf(0.8, 1.2);
		Mat ground = get_randmat(noises, rt);
		resize(ground, ground, m.size());
		merge(m, ground);
	}
	void merge(Mat& ocr, Mat& ground)
	{
		Mat ground_tmp = ground.clone();
		//showimage(ground_tmp);
		if(rand() % 2) ground_tmp = 255 - ground_tmp;

		norm_image(ground_tmp);
		float s = random::randf(3,4);
		ground_tmp = (ground_tmp / s) + 256 * (1 - 1 / s);

		Mat word = ocr.clone();
		norm_image(ocr);

		blur(word, word, Size(3,3));
		float scale = random::randf(0.6, 1);
		for(int j = 0; j < word.rows; j++)
		{
			uchar* gdata = ground_tmp.ptr<uchar>(j);
			uchar* wdata = word.ptr<uchar>(j);
			uchar* sdata = ocr.ptr<uchar>(j);
			for(int i = 0; i < word.cols; i++, gdata++, wdata++, sdata++)
			{
				float pst = st_threshold_persent(ocr, i, j, 250, Size(rand()%5+1,rand()%5+1));
				int v = *gdata - ( pst * (255 - *wdata)) * scale * random::randf(0.9, 1);
				if(v < 0) v = 0;
				if(v > 255) v = 255;
				//if(*wdata < 250) *wdata = v;
				//else *wdata = * gdata;
				*wdata = v;
			}
		}
		ocr = word;
		//showimage(word);
	}

	void createwords(Mat& src, Rect& rt)
	{
		add_noise(src);
		Mat m = Mat::zeros(src.rows + 31, src.cols + 31, src.type()) + 255;
		src.copyTo(m.colRange(15, m.cols - 16).rowRange(15, m.rows - 16));
		rt.x = rt.y = 13;
		rt.width = src.cols+5;
		rt.height = src.rows+5;
		addblur(m);
		if(rand()%2)merge_ground(m);
		src = m;
	}

	Demo_words()
	{
		static_load_image("/home/wanwuming/ocr1/caffe_cls/caffe_ocr/src/noise/rand.list", noises);
		static_load_image("/home/wanwuming/ocr1/caffe_cls/caffe_ocr/src/noise/texture.list", noises);
		//for(int i = 0; i < noises.size(); i++){ Mat n = noises[i]; norm_image(n); noises[i] = n / 4 + 192;}
		static_load_image("/home/wanwuming/ocr1/caffe_cls/caffe_ocr/src/noise/line_black.list", lines_black);
		for(int i = 0; i < lines_black.size(); i++) blur(lines_black[i], lines_black[i], Size(3,3));
		static_load_image("/home/wanwuming/ocr1/caffe_cls/caffe_ocr/src/noise/line_white.list", lines_white);
		for(int i = 0; i < lines_white.size(); i++) blur(lines_white[i], lines_white[i], Size(3,3));
	}

};

Distort distort;
Demo_words words;

void createword(cv::Mat& m, cv::Rect& rt)
{
	//printf("create word\n");
	words.createwords(m, rt);

	//printf("%d, %d, %d, %d; create word end\n",rt.x, rt.y, rt.width, rt.height);
}

void AddWarp2Noise(cv::Mat &m)
{
    if(rand() % 2)distort.Jibian(m, rand() % 1 + 2, 1);
    // if(rand() % 2)
    // {
    //     rand() % 2 ? 
    //     distort.ImagePrinter(m, 2, rand() % 16):
    //     distort.ImagePrinter2(m, 2, 0);
    // }
    if(rand() % 2)distort.Jibian(m, rand() % 1 + 2, 0);
}