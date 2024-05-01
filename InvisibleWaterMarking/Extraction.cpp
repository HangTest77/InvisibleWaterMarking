#include <opencv2/opencv.hpp>

// DCT-based watermark extraction function
cv::Mat extract_watermark(const cv::Mat& watermarked_image, const cv::Mat& original_image, const cv::Point& watermark_position, double alpha) {
    // Convert images to float32
    cv::Mat watermarked_image_float, original_image_float;
    watermarked_image.convertTo(watermarked_image_float, CV_32F);
    original_image.convertTo(original_image_float, CV_32F);

    // Split watermarked image into color channels
    std::vector<cv::Mat> watermarked_image_channels;
    cv::split(watermarked_image_float, watermarked_image_channels);

    // Compute DCT for each color channel of the watermarked image
    std::vector<cv::Mat> watermarked_image_dct(watermarked_image_channels.size());
    for (size_t i = 0; i < watermarked_image_channels.size(); ++i) {
        cv::dct(watermarked_image_channels[i], watermarked_image_dct[i]);
    }

    // Split original image into color channels
    std::vector<cv::Mat> original_image_channels;
    cv::split(original_image_float, original_image_channels);

    // Compute DCT for each color channel of the original image
    std::vector<cv::Mat> original_image_dct(original_image_channels.size());
    for (size_t i = 0; i < original_image_channels.size(); ++i) {
        cv::dct(original_image_channels[i], original_image_dct[i]);
    }

    // Extract watermark from DCT coefficients for each color channel
    std::vector<cv::Mat> extracted_watermark_dct(watermarked_image_dct.size());
    for (size_t i = 0; i < watermarked_image_dct.size(); ++i) {
        extracted_watermark_dct[i] = (watermarked_image_dct[i] - original_image_dct[i]) / alpha;
    }

    // Inverse DCT to obtain extracted watermark for each color channel
    std::vector<cv::Mat> extracted_watermark_channels(extracted_watermark_dct.size());
    for (size_t i = 0; i < extracted_watermark_dct.size(); ++i) {
        cv::idct(extracted_watermark_dct[i], extracted_watermark_channels[i]);
    }

    // Merge color channels
    cv::Mat extracted_watermark;
    cv::merge(extracted_watermark_channels, extracted_watermark);

    return extracted_watermark;
}

int main() {
    // Load the watermarked image
    //cv::Mat watermarked_image = cv::imread("C:/Users/60169/Desktop/PythonInvisible.png");
    cv::Mat watermarked_image = cv::imread("C:/Users/60169/Desktop/hayatoOri_WA.jpeg");

    // Load the original image (for extracting watermark)
    cv::Mat original_image = cv::imread("C:/Users/60169/Desktop/hayato.png");

    // Display the watermarked image
    cv::imshow("Watermarked Image", watermarked_image);
    cv::waitKey(0);

    // Adjust alpha value to match the value used during embedding
    double alpha = 0.01;

    // Extract the watermark
    cv::Mat extracted_watermark = extract_watermark(watermarked_image, original_image,
        cv::Point((original_image.cols - watermarked_image.cols) / 2,
            (original_image.rows - watermarked_image.rows) / 2), alpha);

    // Display the extracted watermark
    cv::imshow("Extracted Watermark", extracted_watermark);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}


