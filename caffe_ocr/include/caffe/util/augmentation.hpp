#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
int txtread(const char* filename, std::vector<std::string>& listfile);
int list_find(std::vector<std::string>& list, std::string s);
void AddLineNoise(cv::Mat& img, const int minval, const int maxval, const int width, const int line_num) ;

void AddGaussianNoise(cv::Mat& img, double mean, double normalized_sigma);

void AddPepperSaltNoise(cv::Mat& img, const double percentage, const int maxnoiseval);

void AddErodeNoise(cv::Mat & img, const int kernelSize);

void AddDilateNoise(cv::Mat & img, const int kernelSize);

void AddRotateNoise(cv::Mat &image, int &x1, int &y1, int &x2, int& y2, const int angle);

void AddScaleNoise(cv::Mat& image, int &x1, int &y1, int &x2, int& y2, const float rowscale = 1.0,
    const float colscale = 1.0);

void AddGammaNoise(cv::Mat& image, const float gamma);

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
void AddLine2Noise(cv::Mat& image, const int maxvalue);
void AddRandNoise(cv::Mat& image, const int maxvalue);
int AddWarpNoise(int& x1, int& x2, int& y1, int& y2, cv::Mat& m);
void createword(cv::Mat& m, cv::Rect& rt);
void AddWarp2Noise(cv::Mat &m);
void AddLightNoise(cv::Mat& m);
void norm_image(cv::Mat& m);


int AddWarpNoise(cv::Mat& m, std::vector< cv::Point2f > & spt);
void AddRotateNoise(cv::Mat &image, std::vector< cv::Point2f > & spt, const int angle);



class random
{
public:
    static int randi(int mini, int maxi = 0)
    {
        return (int)randf(mini, maxi);
    }
    static float randf(float minf, float maxf = 0)
    {
        if(minf == maxf) return minf;
        if(minf > maxf)
        {
            float t = minf;
            minf = maxf;
            maxf = minf;
        }

        float f = (rand() % 1000) / 1000.0;
        return minf + (maxf - minf) * f;
    }
};
#endif  // USE_OPENCV
