#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// DCT-based watermark extraction function
Mat extract_watermark(const Mat& watermarked_image, const Mat& original_image, const Point& watermark_position, double alpha) {
    // Convert images to float32
    Mat watermarked_image_float, original_image_float;
    watermarked_image.convertTo(watermarked_image_float, CV_32F);
    original_image.convertTo(original_image_float, CV_32F);

    // Split watermarked image into color channels
    vector<Mat> watermarked_image_channels;
    split(watermarked_image_float, watermarked_image_channels);

    // Compute DCT for each color channel of the watermarked image
    vector<Mat> watermarked_image_dct(watermarked_image_channels.size());
    for (size_t i = 0; i < watermarked_image_channels.size(); ++i) {
        dct(watermarked_image_channels[i], watermarked_image_dct[i]);
    }

    // Split original image into color channels
    vector<Mat> original_image_channels;
    split(original_image_float, original_image_channels);

    // Compute DCT for each color channel of the original image
    vector<Mat> original_image_dct(original_image_channels.size());
    for (size_t i = 0; i < original_image_channels.size(); ++i) {
        dct(original_image_channels[i], original_image_dct[i]);
    }

    // Extract watermark from DCT coefficients for each color channel
    vector<Mat> extracted_watermark_dct(watermarked_image_dct.size());
    for (size_t i = 0; i < watermarked_image_dct.size(); ++i) {
        extracted_watermark_dct[i] = (watermarked_image_dct[i] - original_image_dct[i]) / alpha;
    }

    // Inverse DCT to obtain extracted watermark for each color channel
    vector<Mat> extracted_watermark_channels(extracted_watermark_dct.size());
    for (size_t i = 0; i < extracted_watermark_dct.size(); ++i) {
        idct(extracted_watermark_dct[i], extracted_watermark_channels[i]);
    }

    // Merge color channels
    Mat extracted_watermark;
    merge(extracted_watermark_channels, extracted_watermark);

    //// Clip to ensure valid pixel values
    //extracted_watermark.convertTo(extracted_watermark, CV_8U);

    return extracted_watermark;
}

//int main() {
//    // Load the watermarked image
//    Mat watermarked_image = imread("C:/Users/60169/Desktop/PythonInvisible.png");
//    //Mat watermarked_image = imread("C:/Users/60169/Desktop/hayato.png");
//
//    // Load the original image (for extracting watermark)
//    Mat original_image = imread("C:/Users/60169/Desktop/hayato.png");
//
//    // Display the watermarked image
//    imshow("Watermarked Image", watermarked_image);
//    waitKey(0);
//
//    // Adjust alpha value to match the value used during embedding
//    double alpha = 0.01;
//
//     //Extract the watermark
//    Mat extracted_watermark = extract_watermark(watermarked_image, original_image, 
//    Point((original_image.cols - watermarked_image.cols) / 2,
//     (original_image.rows - watermarked_image.rows) / 2), alpha);
//
//    // Display the extracted watermark
//    imshow("Extracted Watermark", extracted_watermark);
//    waitKey(0);
//    destroyAllWindows();
//
//    return 0;
//}

