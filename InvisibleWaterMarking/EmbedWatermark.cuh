#include <opencv2/opencv.hpp> 

using namespace cv;

Mat CUDA_Watermark(const Mat& image, const Mat& watermark, const Point& position, float alpha);