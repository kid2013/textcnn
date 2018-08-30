#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>


int txtread(const char* filename, std::vector<std::string>& listfile);
std::string strip(std::string src);
void split(string s, vector<string>& slist, char tag);
int list_find(std::vector<std::string>& list, std::string s);
void fileparts(string filename, string& path, string& name, string& ext);
string get_shortname(string filename);
int is_exist_file(string filename);
int create_dir_from_filename(const char* filename);
int create_dir_from_filename(std::string filename);
void get_path(const char* filename, char* path);


void AddLineNoise(cv::Mat& img, const int minval, const int maxval, const int width, const int line_num) ;

void AddGaussianNoise(cv::Mat& img, double mu, double simga);

void AddRotateNoise(cv::Mat &image, int &x1, int &y1, int &x2, int& y2, const int angle);

void RotateNoise(cv::Mat &image, int &x1, int &y1, int &x2, int& y2, const int angle);
double GenerateGaussianNoise(double mu, double sigma);

void AddGaussianNoise(const cv::Mat& img, cv::Mat& img_gaussiannoised);

void ErodeImageByColumn(const cv::Mat& img, cv::Mat& img_eroded, const int kernelsize);

void AddPepperSaltNoise(const cv::Mat& img, cv::Mat& img_saltnoised, const double percentage);

void GammaCorrection(const cv::Mat& src, cv::Mat& dst, const float gamma);

void GetDilatedImage(const cv::Mat& img, cv::Mat& img_dilated, cv::Size ksize);

void AddAffineNoiseToImage(const cv::Mat& img, cv::Mat& img_affinenoised);

void AffineTransform(cv::Mat& img, cv::Mat& warped, cv::Size warpsize, 
                     const float colstartmax, const float colendmin, 
                     const float rowstartmax, const float rowendmin, 
                     int flags = cv::INTER_LINEAR, int borderMode = cv::BORDER_CONSTANT, 
                     const cv::Scalar& borderValue = cv::Scalar());

void PerspectiveTransform(cv::Mat& img, cv::Mat& warped, cv::Size warpsize, 
                     const float colstartmax, const float colendmin, 
                     const float rowstartmax, const float rowendmin, 
                     int flags = cv::INTER_LINEAR, int borderMode = cv::BORDER_CONSTANT, 
                     const cv::Scalar& borderValue = cv::Scalar());

void RotateTransform(cv::Mat& img, cv::Mat& rotated, cv::Size rotatedsize,
            double clockwise, double anticlockwise, double scale = 1.0,
            int flags = cv::INTER_LINEAR, int borderMode = cv::BORDER_CONSTANT, 
                     const cv::Scalar& borderValue = cv::Scalar());
void ScaleTransform(const cv::Mat& img, cv::Mat& scaled,
           const float minrowrate, const float maxrowrate,
           const float mincolrate, const float maxcolrate, int interpolation=cv::INTER_LINEAR);

void ResizeWithPad(const cv::Mat& img, cv::Mat& img_resize_padded, const int height, const int width, const int resize_type);


#endif  // USE_OPENCV
