#include <opencv2/opencv.hpp>

// Function to embed watermark into image
cv::Mat embedWatermark(const cv::Mat& image, const cv::Mat& watermark, const cv::Point& position, float alpha) {
    cv::Mat watermarkedImage = image.clone();

    // Convert image and watermark to float32
    cv::Mat imageFloat;
    image.convertTo(imageFloat, CV_32F);
    cv::Mat watermarkFloat;
    watermark.convertTo(watermarkFloat, CV_32F);

    // Split image channels
    std::vector<cv::Mat> imageChannels;
    cv::split(imageFloat, imageChannels);

    // Split watermark channels
    std::vector<cv::Mat> watermarkChannels;
    cv::split(watermarkFloat, watermarkChannels);

    // Embed watermark into image channels
    for (int i = 0; i < 3; ++i) {
        cv::Mat imageDCT, watermarkDCT;
        cv::dct(imageChannels[i], imageDCT);
        cv::dct(watermarkChannels[i], watermarkDCT);

        imageDCT += alpha * watermarkDCT;

        cv::idct(imageDCT, imageChannels[i]);
    }

    // Merge image channels
    cv::merge(imageChannels, watermarkedImage);

    // Clip pixel values to [0, 255]
    watermarkedImage = cv::abs(watermarkedImage);
    watermarkedImage.convertTo(watermarkedImage, CV_8U);

    return watermarkedImage;
}

// Function to create watermark with text
cv::Mat createTextWatermark(const cv::Size& size, const std::string& text) {
    cv::Mat watermarkImage(size, CV_8UC3, cv::Scalar(0, 0, 0)); // Transparent background

    // Define the step size for positioning the text watermark
    int stepX = 150; // Adjust this value as needed
    int stepY = 50; // Adjust this value as needed

    // Loop to place the text watermark at multiple positions
    for (int y = 30; y < size.height; y += stepY) {
        for (int x = 10; x < size.width; x += stepX) {
            cv::putText(watermarkImage, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255, 255), 2);
        }
    }

    return watermarkImage;
}




// Function to rotate image
cv::Mat rotateImage(const cv::Mat& image, double angle) {
    cv::Mat rotatedImage;
    cv::Point2f center(image.cols / 2.0, image.rows / 2.0);
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(image, rotatedImage, rotationMatrix, image.size());
    return rotatedImage;
}

int main() {
    // Load original image
    cv::Mat originalImage = cv::imread("C:/Users/60169/Desktop/hayato.png");

    // Create watermark with text "Stolen"
    cv::Mat watermarkImage = createTextWatermark(originalImage.size(), "Stolen");

    // Rotate watermark
    double angle = 0.0;
    cv::Mat rotatedWatermark = rotateImage(watermarkImage, angle);

    // Embed watermark into original image
    cv::Mat invisibleWatermarkedImage = embedWatermark(originalImage, rotatedWatermark, cv::Point(10, 10), 0.01);
    cv::Mat watermarkedImage = embedWatermark(originalImage, rotatedWatermark, cv::Point(10, 10), 100.01);

    // Save watermarked image
    cv::imwrite("C:/Users/60169/Desktop/PythonVisible.png", watermarkedImage);
    cv::imwrite("C:/Users/60169/Desktop/PythonInvisible.png", invisibleWatermarkedImage);

    // Display watermarked image
    cv::imshow("PythonVisible", watermarkedImage);
    cv::imshow("PythonInvisible", invisibleWatermarkedImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
