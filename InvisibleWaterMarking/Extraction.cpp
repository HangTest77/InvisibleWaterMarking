//#include <opencv2/opencv.hpp>
//#define M_PI 3.14159265358979323846
//
//using namespace cv;
//
//
//// Custom DCT implementation
//void custom_dct(const Mat& src, Mat& dst) {
//    int N = src.rows;
//    int M = src.cols;
//    dst.create(N, M, CV_32F);
//
//    for (int u = 0; u < N; u++) {
//        for (int v = 0; v < M; v++) {
//            float sum = 0.0;
//            for (int x = 0; x < N; x++) {
//                for (int y = 0; y < M; y++) {
//                    sum += src.at<float>(x, y) * cos((2 * x + 1) * u * M_PI / (2.0 * N)) *
//                        cos((2 * y + 1) * v * M_PI / (2.0 * M));
//                }
//            }
//            float Cu = (u == 0) ? 1 / sqrt(N) : sqrt(2.0 / N);
//            float Cv = (v == 0) ? 1 / sqrt(M) : sqrt(2.0 / M);
//            dst.at<float>(u, v) = Cu * Cv * sum;
//        }
//    }
//}
//
//// Custom IDCT implementation
//void custom_idct(const Mat& src, Mat& dst) {
//    int N = src.rows;
//    int M = src.cols;
//    dst.create(N, M, CV_32F);
//
//    for (int x = 0; x < N; x++) {
//        for (int y = 0; y < M; y++) {
//            float sum = 0.0;
//            for (int u = 0; u < N; u++) {
//                for (int v = 0; v < M; v++) {
//                    float Cu = (u == 0) ? 1 / sqrt(N) : sqrt(2.0 / N);
//                    float Cv = (v == 0) ? 1 / sqrt(M) : sqrt(2.0 / M);
//                    sum += Cu * Cv * src.at<float>(u, v) * cos((2 * x + 1) * u * M_PI / (2.0 * N)) *
//                        cos((2 * y + 1) * v * M_PI / (2.0 * M));
//                }
//            }
//            dst.at<float>(x, y) = sum;
//        }
//    }
//}
//
//
//// DCT-based watermark extraction function with resizing
//cv::Mat extract_watermark(const cv::Mat& watermarked_image, const cv::Mat& original_image, const cv::Point& watermark_position, double alpha, int targetWidth, int targetHeight) {
//    // Convert images to float32
//    cv::Mat watermarked_image_float, original_image_float;
//    watermarked_image.convertTo(watermarked_image_float, CV_32F);
//    original_image.convertTo(original_image_float, CV_32F);
//
//    // Resize images to target dimensions
//    cv::Mat watermarked_resized, original_resized;
//    cv::resize(watermarked_image_float, watermarked_resized, cv::Size(targetWidth, targetHeight));
//    cv::resize(original_image_float, original_resized, cv::Size(targetWidth, targetHeight));
//
//     //Split watermarked image into color channels
//    std::vector<cv::Mat> watermarked_image_channels;
//    cv::split(watermarked_resized, watermarked_image_channels);
//
//    // Compute DCT for each color channel of the watermarked image
//    std::vector<cv::Mat> watermarked_image_dct(watermarked_image_channels.size());
//    for (size_t i = 0; i < watermarked_image_channels.size(); ++i) {
//        dct(watermarked_image_channels[i], watermarked_image_dct[i]);
//    }
//
//     //Split original image into color channels
//    std::vector<cv::Mat> original_image_channels;
//    cv::split(original_resized, original_image_channels);
//
//     //Compute DCT for each color channel of the original image
//    std::vector<cv::Mat> original_image_dct(original_image_channels.size());
//    for (size_t i = 0; i < original_image_channels.size(); ++i) {
//        dct(original_image_channels[i], original_image_dct[i]);
//    }
//
//     //Extract watermark from DCT coefficients for each color channel
//    std::vector<cv::Mat> extracted_watermark_dct(watermarked_image_dct.size());
//    for (size_t i = 0; i < watermarked_image_dct.size(); ++i) {
//        extracted_watermark_dct[i] = (watermarked_image_dct[i] - original_image_dct[i]) / alpha;
//    }
//
//     //Inverse DCT to obtain extracted watermark for each color channel
//    std::vector<cv::Mat> extracted_watermark_channels(extracted_watermark_dct.size());
//    for (size_t i = 0; i < extracted_watermark_dct.size(); ++i) {
//        idct(extracted_watermark_dct[i], extracted_watermark_channels[i]);
//    }
//
//     //Merge color channels
//    cv::Mat extracted_watermark;
//    cv::merge(extracted_watermark_channels, extracted_watermark);
//
//    return extracted_watermark;
//}
//
//
//
//int main() {
//    // Load the watermarked image
//    cv::Mat watermarked_image = cv::imread("C:/Users/60169/Desktop/ImageInvisible.png");
//    if (watermarked_image.empty()) {
//        std::cerr << "Error: Unable to load watermarked image." << std::endl;
//        return -1;
//    }
//
//    // Load the original image
//    cv::Mat original_image = cv::imread("C:/Users/60169/Desktop/hayato.png");
//    if (original_image.empty()) {
//        std::cerr << "Error: Unable to load original image." << std::endl;
//        return -1;
//    }
//
//
//    // Display the watermarked image with maximum width and height
//    int maxDisplayWidth = 800; // Adjust as needed
//    int maxDisplayHeight = 600; // Adjust as needed
//
//    // Calculate scaling factor
//    double scaleWidth = 1.0, scaleHeight = 1.0;
//    if (watermarked_image.cols > maxDisplayWidth)
//        scaleWidth = static_cast<double>(maxDisplayWidth) / watermarked_image.cols;
//    if (watermarked_image.rows > maxDisplayHeight)
//        scaleHeight = static_cast<double>(maxDisplayHeight) / watermarked_image.rows;
//    double scaleFactor = std::min(scaleWidth, scaleHeight);
//
//    // Resize the image if necessary
//    if (scaleFactor < 1.0) {
//        cv::resize(watermarked_image, watermarked_image, cv::Size(), scaleFactor, scaleFactor);
//    }
//
//
//
//    // Display the watermarked image
//    cv::imshow("Watermarked Image", watermarked_image);
//    cv::waitKey(0);
//
//    // Adjust alpha value to match the value used during embedding
//    double alpha = 0.01;
//
//    // Target width and height for resizing
//    int targetWidth = 800;
//    int targetHeight = 600;
//
//    // Extract the watermark with resizing
//    cv::Mat extracted_watermark = extract_watermark(watermarked_image, original_image,
//        cv::Point((original_image.cols - watermarked_image.cols) / 2,
//            (original_image.rows - watermarked_image.rows) / 2), alpha, targetWidth, targetHeight);
//
//    cv::imwrite("C:/Users/60169/Desktop/extractedWatermark.png", extracted_watermark);
//
//    // Display the extracted watermark
//    cv::imshow("Extracted Watermark", extracted_watermark);
//    cv::waitKey(0);
//    cv::destroyAllWindows();
//
//    return 0;
//}
//
//
