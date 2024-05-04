#include <opencv2/opencv.hpp>
#include "omp.h"
#include <chrono>

using namespace cv;
using namespace std;

//    1 - Watermark Function

Mat embedWatermark(const Mat& image, const Mat& watermark, const Point& position, float alpha) {
    Mat watermarkedImage = image.clone();

    // Convert image and watermark to float32
    Mat imageFloat;
    image.convertTo(imageFloat, CV_32F);
    Mat watermarkFloat;
    watermark.convertTo(watermarkFloat, CV_32F);

    // Split image channels
    std::vector<Mat> imageChannels;
    split(imageFloat, imageChannels);

    // Split watermark channels 
    std::vector<Mat> watermarkChannels;
    split(watermarkFloat, watermarkChannels);

    // Embed watermark into image channels using element-wise operations
    for (int i = 0; i < 3; ++i) {
        // Normalize watermark channel values to [-1, 1] range
        Mat normalizedWatermark = watermarkChannels[i] / 255.0f;
        normalizedWatermark -= 0.5f;  // Shift to center around 0

        // Apply weighted addition with image channel
        imageChannels[i] = imageChannels[i] + alpha * normalizedWatermark;

        // Clip pixel values to [0, 1] after modification
        float minVal, maxVal;
        Point minLoc, maxLoc;

        // Find minimum and maximum values manually
        for (int y = 0; y < imageChannels[i].rows; ++y) {
            for (int x = 0; x < imageChannels[i].cols; ++x) {
                float value = imageChannels[i].at<float>(y, x);
                if (y == 0 && x == 0) {
                    minVal = maxVal = value;
                    minLoc = maxLoc = cv::Point(x, y);
                }
                else {
                    minVal = std::min(minVal, value);
                    maxVal = std::max(maxVal, value);
                    if (value == minVal) {
                        minLoc = cv::Point(x, y);
                    }
                    if (value == maxVal) {
                        maxLoc = cv::Point(x, y);
                    }
                }
            }
        }

        // Clip pixel values based on the minimum and maximum
        for (int y = 0; y < imageChannels[i].rows; ++y) {
            for (int x = 0; x < imageChannels[i].cols; ++x) {
                imageChannels[i].at<float>(y, x) = std::min(std::max(imageChannels[i].at<float>(y, x), minVal), maxVal);
            }
        }
    }

    // Merge modified channels back into a single image
    merge(imageChannels, watermarkedImage);

    // Convert back to original data type (8U)
    watermarkedImage.convertTo(watermarkedImage, CV_8U);

    return watermarkedImage;
}


Mat OpenMP_Watermark(const Mat& image, const Mat& watermark, const Point& position, float alpha) {
    Mat watermarkedImage = image.clone();

    // Convert image and watermark to float32
    Mat imageFloat;
    image.convertTo(imageFloat, CV_32F);
    Mat watermarkFloat;
    watermark.convertTo(watermarkFloat, CV_32F);

    // Split image channels
    vector<Mat> imageChannels;
    split(imageFloat, imageChannels);

    // Split watermark channels 
    vector<Mat> watermarkChannels;
    split(watermarkFloat, watermarkChannels);

    // Embed watermark into image channels using element-wise operations
#pragma omp parallel for
    for (int i = 0; i < 3; ++i) {
        // Normalize watermark channel values to [-1, 1] range
        Mat normalizedWatermark = watermarkChannels[i] / 255.0f;
        normalizedWatermark -= 0.5f;  // Shift to center around 0

        // Apply weighted addition with image channel
        imageChannels[i] = imageChannels[i] + alpha * normalizedWatermark;

        // Clip pixel values to [0, 1] after modification
        float minVal, maxVal;
        Point minLoc, maxLoc;

        // Find minimum and maximum values manually
        for (int y = 0; y < imageChannels[i].rows; ++y) {
            for (int x = 0; x < imageChannels[i].cols; ++x) {
                float value = imageChannels[i].at<float>(y, x);
                if (y == 0 && x == 0) {
                    minVal = maxVal = value;
                    minLoc = maxLoc = Point(x, y);
                }
                else {
                    minVal = min(minVal, value);
                    maxVal = max(maxVal, value);
                    if (value == minVal) {
                        minLoc = Point(x, y);
                    }
                    if (value == maxVal) {
                        maxLoc = Point(x, y);
                    }
                }
            }
        }

        // Clip pixel values based on the minimum and maximum
        for (int y = 0; y < imageChannels[i].rows; ++y) {
            for (int x = 0; x < imageChannels[i].cols; ++x) {
                imageChannels[i].at<float>(y, x) = min(max(imageChannels[i].at<float>(y, x), minVal), maxVal);
            }
        }
    }

    // Merge modified channels back into a single image
    merge(imageChannels, watermarkedImage);

    // Convert back to original data type (8U)
    watermarkedImage.convertTo(watermarkedImage, CV_8U);

    return watermarkedImage;
}


// Function to rotate image
Mat OpenMP_RotateImage(const Mat& image, double angle) {
    Mat rotatedImage;
    Point2f center(image.cols / 2.0, image.rows / 2.0);
    Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0);
    warpAffine(image, rotatedImage, rotationMatrix, image.size());
    return rotatedImage;
}




//    2 - Create Watermark

// Function to create watermark with text
Mat original_TextWatermark(const Size& size, const string& text) {
    Mat watermarkImage(size, CV_8UC3, Scalar(0, 0, 0, 0));

    // Define the step size for positioning the text watermark
    int stepX = 250; // Adjust this value as needed
    int stepY = 200; // Adjust this value as needed

    auto start = chrono::high_resolution_clock::now();
    // Loop to place the text watermark at multiple positions
    for (int y = 30; y < size.height; y += stepY) {
        for (int x = 10; x < size.width; x += stepX) {
            putText(watermarkImage, text, Point(x, y), FONT_HERSHEY_SIMPLEX, 2 , Scalar(0, 0, 255, 255), 2);
        }
    }
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

    cout << " " << endl;
    cout << " " << endl;
    cout << "Original version execution time: " << duration.count() << " milliseconds" << endl;
    cout << " " << endl;
    cout << " " << endl;

    return watermarkImage;
}


// OpenMP to create watermark with text
Mat OpenMP_TextWatermark(const Size& size, const string& text) {
    Mat watermarkImage(size, CV_8UC3, Scalar(0, 0, 0, 0));

    // Define the step size for positioning the text watermark
    int stepX = 250; // Adjust this value as needed
    int stepY = 200; // Adjust this value as needed
 
    auto start = chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(2)
    for (int y = 30; y < size.height; y += stepY) {
        for (int x = 10; x < size.width; x += stepX) {
            putText(watermarkImage, text, Point(x, y), FONT_HERSHEY_SIMPLEX, 2 , Scalar(0, 0, 255, 255), 2);
        }
    }
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

    cout << " " << endl;
    cout << " " << endl;
    cout << "OpenMP version execution time: " << duration.count() << " milliseconds" << endl;
    cout << " " << endl;
    cout << " " << endl;

    return watermarkImage;
}




//    3 - Extract Watermark





//    4 - Console

void goOriginal(Mat originalImage, Mat watermarkImage, Mat rotatedWatermark) {
    auto start = chrono::high_resolution_clock::now();
    Mat invisibleWatermarkedImage = embedWatermark(originalImage, rotatedWatermark, Point(10, 10), 10.01);
    Mat watermarkedImage = embedWatermark(originalImage, rotatedWatermark, Point(10, 10), 200.01);
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    // Save watermarked image
    imwrite("C:/Users/60169/Desktop/imageVisible.png", watermarkedImage);
    imwrite("C:/Users/60169/Desktop/imageInvisible.png", invisibleWatermarkedImage);


    const int maxWidth = 800; // Adjust as needed
    const int maxHeight = 600; // Adjust as needed


    cv::Size imageSize = watermarkedImage.size();
    if (imageSize.width > maxWidth || imageSize.height > maxHeight) {
        double scale = std::min((double)maxWidth / imageSize.width, (double)maxHeight / imageSize.height);
        cv::resize(watermarkedImage, watermarkedImage, cv::Size(), scale, scale);
        cv::resize(invisibleWatermarkedImage, invisibleWatermarkedImage, cv::Size(), scale, scale);
    }


    // Display watermarked image
    imshow("Image with Visible Watermark", watermarkedImage);
    imshow("Image with Invisible Watermark", invisibleWatermarkedImage);

    cout << " " << endl;
    cout << " " << endl;
    cout << "Execution time for Original watermark embedding: " << duration << " milliseconds" << endl;
    cout << " " << endl;
    cout << " " << endl;

    cout << "Prese any key to continue..." << endl;
    waitKey(0); 

    //char token;
    //cout << "Continue? (Y/N): " ;
    //cin >> token;

    //if (token == 'Y' || token == 'y') {
    //    //toUserSelection(originalImage, watermarkImage, rotatedWatermark);
    //}
    //else {
    //    destroyAllWindows();
    //}

    destroyAllWindows();
   
}


void goOpenMp(Mat originalImage, Mat watermarkImage, Mat rotatedWatermark) {
    // OpenMP Watermarking
    auto start = chrono::high_resolution_clock::now();
    Mat invisibleWatermarkedImage = OpenMP_Watermark(originalImage, rotatedWatermark, Point(10, 10), 10.01);
    Mat watermarkedImage = OpenMP_Watermark(originalImage, rotatedWatermark, Point(10, 10), 200.01);
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    // Save watermarked image
    imwrite("C:/Users/60169/Desktop/imageVisible.png", watermarkedImage);
    imwrite("C:/Users/60169/Desktop/imageInvisible.png", invisibleWatermarkedImage);


    const int maxWidth = 800; // Adjust as needed
    const int maxHeight = 600; // Adjust as needed


    cv::Size imageSize = watermarkedImage.size();
    if (imageSize.width > maxWidth || imageSize.height > maxHeight) {
        double scale = std::min((double)maxWidth / imageSize.width, (double)maxHeight / imageSize.height);
        cv::resize(watermarkedImage, watermarkedImage, cv::Size(), scale, scale);
        cv::resize(invisibleWatermarkedImage, invisibleWatermarkedImage, cv::Size(), scale, scale);
    }

    // Display watermarked image
    imshow("Image with Visible Watermark", watermarkedImage);
    imshow("Image with Invisible Watermark", invisibleWatermarkedImage);

    cout << " " << endl;
    cout << " " << endl;
    cout << "Execution time for OpenMp watermark embedding: " << duration << " milliseconds" << endl;
    cout << " " << endl;
    cout << " " << endl;
    cout << "Prese any key to continue..." << endl;

    waitKey(0);
    
    //char token;
    //cout << "Continue? (Y/N): ";
    //cin >> token;

    //if (token == 'Y' || token == 'y') {
    //    //toUserSelection(originalImage, watermarkImage, rotatedWatermark);
    //}
    //else {
    //    /*destroyAllWindows();*/
    //}

    destroyAllWindows();
}


void toUserSelection(Mat originalImage, Mat watermarkImage, Mat rotatedWatermark) {
    //Input - User Selection
    int option;
    char continueOption;

   
        cout << " " << endl;
        cout << "Which watermarking technique do you want to use? " << endl;
        cout << "Press 1 - Original Watermarking Technique " << endl;
        cout << "Press 2 - OpenMP Watermarking Technique " << endl;

        cout << " " << endl;
        cout << "Option: ";
        cin >> option;
        cout << " " << endl;
        cout << " " << endl;

        if (option == 1) 
            goOriginal(originalImage, watermarkImage, rotatedWatermark);

        else if (option == 2) 
            goOpenMp(originalImage, watermarkImage, rotatedWatermark);

}


void continueWatermarking(Mat originalImage) {
    //Input - Watermark Text
    string watermarkTexts;
    cout << " " << endl;
    cout << "Enter watermark text: " << endl;
    cout << "Text: ";
    cin.ignore(); 
    getline(cin, watermarkTexts);


    //Input - Create Watermark
    double options;
    cout << " " << endl;
    cout << "Select watermark creation technique: " << endl;
    cout << "1 - Original " << endl;
    cout << "2 - OpenMP " << endl;
    cout << "Option: ";
    cin >> options;

    if (options == 1) {
        Mat watermarkImage = original_TextWatermark(originalImage.size(), watermarkTexts);

        Mat rotatedWatermark = OpenMP_RotateImage(watermarkImage, 0.0);

        toUserSelection(originalImage, watermarkImage, rotatedWatermark);
    }
    else if (options ==2) {
        Mat watermarkImage = OpenMP_TextWatermark(originalImage.size(), watermarkTexts);

        Mat rotatedWatermark = OpenMP_RotateImage(watermarkImage, 0.0);

        toUserSelection(originalImage, watermarkImage, rotatedWatermark);
    }

    

}


void selectImagePath(int imagePathOptions) {

    if (imagePathOptions == 1) {
        Mat originalImage = imread("C:/Users/60169/Desktop/10MB.jpg");
        continueWatermarking(originalImage);

    }
    else if (imagePathOptions == 2) {
        Mat originalImage = imread("C:/Users/60169/Desktop/8MB.jpg");
        continueWatermarking(originalImage);

    }
    else if (imagePathOptions == 3) {
        string imagePath;
        cout << "Example of image path: C:/Users/60169/Desktop/10MB.jpg " << endl;
        cout << "Please input the image path: " << endl;
        cout << " " << endl;
        cout << "Image path: ";
        cin >> imagePath;
        Mat originalImage = imread(imagePath);

        if (!originalImage.empty()) {
            continueWatermarking(originalImage);
        }
        else {
            cout << "Invalid Image Path, please input again" << endl;
            selectImagePath(3);
        }


    }
}



int main() {

    //Input - Original Image
    int imagePathOptions;
    cout << " " << endl;
    cout << "Firstly, provide an image path" << endl;
    cout << "Input 1 - Light color image path " << endl;
    cout << "Input 2 - Dark color image path " << endl;
    cout << "Input 3 - Provide custom image path " << endl;
    cout << " " << endl;
    cout << "Option: ";
    cin >> imagePathOptions;

    selectImagePath(imagePathOptions);

    return 0;
}
