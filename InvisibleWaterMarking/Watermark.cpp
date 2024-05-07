#include <opencv2/opencv.hpp>
#include "omp.h"
#include <chrono>
#include <mpi.h>


using namespace cv;
using namespace std;


//  Original 1 - Split

void originalSplit(const cv::Mat& src, std::vector<cv::Mat>& channels) {
    channels.clear();
    for (int c = 0; c < src.channels(); ++c) {
        cv::Mat channel(src.rows, src.cols, CV_32F);
        for (int i = 0; i < src.rows; ++i) {
            for (int j = 0; j < src.cols; ++j) {
                channel.at<float>(i, j) = src.at<cv::Vec3f>(i, j)[c];
            }
        }
        channels.push_back(channel);
    }
}

//  Original 2 - Merge

cv::Mat originalMerge(const std::vector<cv::Mat>& channels, cv::Mat& output) {
    // Ensure input channels are of the same size
    CV_Assert(channels.size() == 3);

    // Create output image
    output.create(channels[0].size(), CV_32FC3); // Output image will be in float format

    // Merge channels
    for (int y = 0; y < output.rows; ++y) {
        for (int x = 0; x < output.cols; ++x) {
            cv::Vec3f& pixel = output.at<cv::Vec3f>(y, x); // Use Vec3f for floating-point values
            for (int c = 0; c < 3; ++c) {
                pixel[c] = channels[c].at<float>(y, x); // Directly copy pixel values
            }
        }
    }

    return output;
}





// OpenMP 1 - Split

void openMP_Split(const cv::Mat& src, std::vector<cv::Mat>& channels) {
    channels.clear();
    channels.resize(src.channels());

#pragma omp parallel for
    for (int c = 0; c < src.channels(); ++c) {
        cv::Mat& channel = channels[c];
        channel.create(src.rows, src.cols, CV_32F);
        for (int i = 0; i < src.rows; ++i) {
            for (int j = 0; j < src.cols; ++j) {
                channel.at<float>(i, j) = src.at<cv::Vec3f>(i, j)[c];
            }
        }
    }
}

// OpenMP 2 - Merge

cv::Mat openMP_Merge(const std::vector<cv::Mat>& channels, cv::Mat& output) {
    // Ensure input channels are of the same size
    CV_Assert(channels.size() == 3);

    // Create output image
    output.create(channels[0].size(), CV_32FC3); // Output image will be in float format

#pragma omp parallel for
    for (int y = 0; y < output.rows; ++y) {
        for (int x = 0; x < output.cols; ++x) {
            cv::Vec3f& pixel = output.at<cv::Vec3f>(y, x); // Use Vec3f for floating-point values
            for (int c = 0; c < 3; ++c) {
                pixel[c] = channels[c].at<float>(y, x); // Directly copy pixel values
            }
        }
    }

    return output;
}





// MPI 1 - Split

void MPISplit(const cv::Mat& src, std::vector<cv::Mat>& channels) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate the workload distribution
    int rowsPerProcess = (src.rows + size - 1) / size;
    int startRow = rank * rowsPerProcess;
    int endRow = std::min((rank + 1) * rowsPerProcess, src.rows);

    // Clear the channels vector
    channels.clear();

    // Split the image channels across processes
    for (int c = 0; c < 3; ++c) {
        // Create a sub-image for the current channel
        cv::Mat channel(endRow - startRow, src.cols, CV_32F);
        for (int i = startRow; i < endRow; ++i) {
            for (int j = 0; j < src.cols; ++j) {
                channel.at<float>(i - startRow, j) = src.at<cv::Vec3f>(i, j)[c];
            }
        }
        // Add the channel to the channels vector
        channels.push_back(channel);
    }
}

// MPI 2 - Merge

cv::Mat MPIMerge(const std::vector<cv::Mat>& channels) {
    // Ensure input channels are of the same size
    CV_Assert(channels.size() == 3);

    // Get process rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = channels[0].rows;
    int cols = channels[0].cols;

    // Create output image for all processes
    cv::Mat output(rows, cols, CV_32FC3);

    // Merge channels
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            cv::Vec3f& pixel = output.at<cv::Vec3f>(y, x);
            for (int c = 0; c < 3; ++c) {
                pixel[c] = channels[c].at<float>(y, x);
            }
        }
    }

    // Debug print statements
    std::cout << "Rank " << rank << ": After merging channels" << std::endl;

    // Prepare buffers for MPI_Gatherv
    std::vector<float> sendBuffer(output.rows * output.cols * 3);
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = ((i + 1) * rows / size - i * rows / size) * cols * 3;
        displs[i] = i * rows / size * cols * 3;
    }

    // Copy data from output matrix to send buffer
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int c = 0; c < 3; ++c) {
                sendBuffer[(i * cols + j) * 3 + c] = output.at<cv::Vec3f>(i, j)[c];
            }
        }
    }

    // Gather the results from all processes to process 0
    cv::Mat gatheredOutput(rows, cols, CV_32FC3);
    MPI_Gatherv(sendBuffer.data(), output.rows * output.cols * 3, MPI_FLOAT,
        gatheredOutput.data, recvcounts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Debug print statements
    std::cout << "Rank " << rank << ": After MPI_Gatherv" << std::endl;

    if (rank == 0) {
        return gatheredOutput;
    }
    else {
        return cv::Mat(); // Return an empty matrix for non-root processes
    }
}





//    1 - Watermark Function

Mat embedWatermark(const Mat& image, const Mat& watermark, const Point& position, float alpha) {
    Mat watermarkedImage = image.clone();

    // Convert image and watermark to float32
    Mat imageFloat;
    image.convertTo(imageFloat, CV_32F);
    Mat watermarkFloat;
    watermark.convertTo(watermarkFloat, CV_32F);


    // ***************************************************
    // Duration for Split 
    auto startSplitOri = chrono::high_resolution_clock::now();

     // Split image channels using custom function
    std::vector<cv::Mat> imageChannels;
    originalSplit(imageFloat, imageChannels);

    // Split watermark channels using custom function
    std::vector<cv::Mat> watermarkChannels;
    originalSplit(watermarkFloat, watermarkChannels);

    auto stopSplitOri = chrono::high_resolution_clock::now();
    auto durationSplitOri = chrono::duration_cast<chrono::milliseconds>(stopSplitOri - startSplitOri);



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

    // ***************************************************
// Duration for Merge
    auto startMergeOri = chrono::high_resolution_clock::now();
    // Merge modified channels back into a single image
    originalMerge(imageChannels, watermarkedImage);

    auto stopMergeOri = chrono::high_resolution_clock::now();
    auto durationMergeOri = chrono::duration_cast<chrono::milliseconds>(stopMergeOri - startMergeOri);


    // Convert back to original data type (8U)
    watermarkedImage.convertTo(watermarkedImage, CV_8U);

    cout << " " << endl;
    cout << " " << endl;
    cout << "Original Split execution time: " << durationSplitOri.count() << " milliseconds" << endl;
    cout << "Original Merge execution time: " << durationMergeOri.count() << " milliseconds" << endl;
    cout << " " << endl;
    cout << " " << endl;

    return watermarkedImage;
}


Mat OpenMP_Watermark(const Mat& image, const Mat& watermark, const Point& position, float alpha) {
    Mat watermarkedImage = image.clone();

    // Convert image and watermark to float32
    Mat imageFloat;
    image.convertTo(imageFloat, CV_32F);
    Mat watermarkFloat;
    watermark.convertTo(watermarkFloat, CV_32F);


    // ***************************************************
    // Duration for Split 
    auto startSplitOpenMP = chrono::high_resolution_clock::now();

    // Split image channels
    vector<Mat> imageChannels;
    openMP_Split(imageFloat, imageChannels);

    // Split watermark channels 
    vector<Mat> watermarkChannels;
    openMP_Split(watermarkFloat, watermarkChannels);

    auto stopSplitOpenMP = chrono::high_resolution_clock::now();
    auto durationSplitOpenMP = chrono::duration_cast<chrono::milliseconds>(stopSplitOpenMP - startSplitOpenMP);


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


    // ***************************************************
    // Duration for Merge 
    auto startMergeOpenMP = chrono::high_resolution_clock::now();

    // Merge modified channels back into a single image
    openMP_Merge(imageChannels, watermarkedImage);

    auto stopMergeOpenMP = chrono::high_resolution_clock::now();
    auto durationMergeOpenMP = chrono::duration_cast<chrono::milliseconds>(stopMergeOpenMP - startMergeOpenMP);


    // Convert back to original data type (8U)
    watermarkedImage.convertTo(watermarkedImage, CV_8U);

    cout << " " << endl;
    cout << " " << endl;
    cout << "OpenMP Split execution time: " << durationSplitOpenMP.count() << " milliseconds" << endl;
    cout << "OpenMP Merge execution time: " << durationMergeOpenMP.count() << " milliseconds" << endl;
    cout << " " << endl;
    cout << " " << endl;

    return watermarkedImage;
}


Mat MPI_Watermark(const Mat& image, const Mat& watermark, const Point& position, float alpha) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Convert image and watermark to float32
    Mat imageFloat, watermarkFloat;
    image.convertTo(imageFloat, CV_32F);
    watermark.convertTo(watermarkFloat, CV_32F);


    // ***************************************************
   // Duration for Split 
    auto startSplitMPI = chrono::high_resolution_clock::now();
    // Split image and watermark channels
    std::vector<Mat> imageChannels, watermarkChannels;
    MPISplit(imageFloat, imageChannels);
    MPISplit(watermarkFloat, watermarkChannels);

    auto stopSplitMPI = chrono::high_resolution_clock::now();
    auto durationSplitMPI = chrono::duration_cast<chrono::milliseconds>(stopSplitMPI - startSplitMPI);


    // Determine workload distribution
    int rowsPerProcess = image.rows / size;
    int remainderRows = image.rows % size;
    int startRow = rank * rowsPerProcess;
    int endRow = startRow + rowsPerProcess + (rank == size - 1 ? remainderRows : 0);

    // Apply watermark to image channels
    for (int i = 0; i < 3; ++i) {
        // Normalize watermark channel values to [-1, 1] range
        Mat normalizedWatermark = watermarkChannels[i] / 255.0f - 0.5f;

        // Apply weighted addition with image channel in parallel
        for (int y = startRow; y < endRow; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                imageChannels[i].at<float>(y, x) += alpha * normalizedWatermark.at<float>(y, x);
            }
        }
    }

    // ***************************************************
// Duration for Merge 
    auto startMergeMPI = chrono::high_resolution_clock::now();
    Mat mergedImage = MPIMerge(imageChannels);

    auto stopMergeMPI = chrono::high_resolution_clock::now();
    auto durationMergeMPI = chrono::duration_cast<chrono::milliseconds>(stopMergeMPI - startMergeMPI);


    // Convert back to original data type (8U)
    mergedImage.convertTo(mergedImage, CV_8U);

    return mergedImage;

    cout << " " << endl;
    cout << " " << endl;
    cout << "MPI Split execution time: " << durationSplitMPI.count() << " milliseconds" << endl;
    cout << "MPI Merge execution time: " << durationMergeMPI.count() << " milliseconds" << endl;
    cout << " " << endl;
    cout << " " << endl;
}


// Function to rotate image
Mat rotateImage(const Mat& image, double angle) {
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
    cout << "Original Watermark Image Creation Time: " << duration.count() << " milliseconds" << endl;
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
    cout << "OpenMP Watermark Image Creation Time: " << duration.count() << " milliseconds" << endl;
    cout << " " << endl;
    cout << " " << endl;

    return watermarkImage;
}


// MPI to create watermark with text
Mat MPI_TextWatermark(const Size& s, const string& text) {
    Mat watermarkImage(s, CV_8UC3, Scalar(0, 0, 0, 0));

    int rank, size_mpi;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size_mpi);

    int stepX = 250;
    int stepY = 200;

    auto start = chrono::high_resolution_clock::now();

    // Calculate the total number of positions
    int totalPositions = ((s.height - 30) / stepY) * ((s.width - 10) / stepX);

    // Calculate the workload distribution
    int positionsPerProcess = totalPositions / size_mpi;
    int remainder = totalPositions % size_mpi;

    // Determine the start and end index for the current process
    int startIdx = rank * positionsPerProcess;
    int endIdx = (rank + 1) * positionsPerProcess;

    // Adjust the end index for the last process to accommodate any remainder
    if (rank == size_mpi - 1) {
        endIdx += remainder;
    }

    // Parallelized watermark embedding loop
    parallel_for_(Range(startIdx, endIdx), [&](const Range& range) {
        for (int idx = range.start; idx < range.end; ++idx) {
            int y = 30 + idx / ((s.width - 10) / stepX) * stepY;
            int x = 10 + (idx % ((s.width - 10) / stepX)) * stepX;
            putText(watermarkImage, text, Point(x, y), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255, 255), 2);
        }
        });

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

    if (rank == 0) {
        cout << " " << endl;
        cout << " " << endl;
        cout << "MPI Watermark Image Creation Time: " << duration.count() << " milliseconds" << endl;
        cout << " " << endl;
        cout << " " << endl;
    }

    return watermarkImage;
}




//    3 - Extract Watermark






//    4 - Console

// ************************************************************************************
// Both

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


void goMPI(Mat originalImage, Mat watermarkImage, Mat rotatedWatermark) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto start = chrono::high_resolution_clock::now();

    // Determine workload distribution
    int rowsPerProcess = originalImage.rows / size;
    int remainderRows = originalImage.rows % size;
    int startRow = rank * rowsPerProcess;
    int endRow = startRow + rowsPerProcess + (rank == size - 1 ? remainderRows : 0);

    // Embed watermark into visible and invisible watermarked images in parallel
    Mat invisibleWatermarkedImage = MPI_Watermark(originalImage, rotatedWatermark, Point(10, 10), 10.01);
    Mat watermarkedImage = MPI_Watermark(originalImage, rotatedWatermark, Point(10, 10), 200.01);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    // Save watermarked images
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

    // Display watermarked images
    imshow("Image with Visible Watermark", watermarkedImage);
    imshow("Image with Invisible Watermark", invisibleWatermarkedImage);

    cout << " " << endl;
    cout << " " << endl;
    cout << "Execution time for MPI watermark embedding: " << duration << " milliseconds" << endl;
    cout << " " << endl;
    cout << " " << endl;

    cout << "Press any key to continue..." << endl;
    waitKey(0);

    destroyAllWindows();

}



// ************************************************************************************
// Invisible


void goOriginalInvisible(Mat originalImage, Mat watermarkImage, Mat rotatedWatermark) {
    auto start = chrono::high_resolution_clock::now();
    Mat invisibleWatermarkedImage = embedWatermark(originalImage, rotatedWatermark, Point(10, 10), 10.01);
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    // Save watermarked image
    imwrite("C:/Users/60169/Desktop/imageInvisible.png", invisibleWatermarkedImage);


    const int maxWidth = 800; // Adjust as needed
    const int maxHeight = 600; // Adjust as needed


    cv::Size imageSize = invisibleWatermarkedImage.size();
    if (imageSize.width > maxWidth || imageSize.height > maxHeight) {
        double scale = std::min((double)maxWidth / imageSize.width, (double)maxHeight / imageSize.height);
        cv::resize(invisibleWatermarkedImage, invisibleWatermarkedImage, cv::Size(), scale, scale);
    }


    // Display watermarked image
    imshow("Image with Invisible Watermark", invisibleWatermarkedImage);

    cout << " " << endl;
    cout << " " << endl;
    cout << "Execution time for Original watermark embedding: " << duration << " milliseconds" << endl;
    cout << " " << endl;
    cout << " " << endl;

    cout << "Prese any key to continue..." << endl;
    waitKey(0);


    destroyAllWindows();

}


void goOpenMpInvisible(Mat originalImage, Mat watermarkImage, Mat rotatedWatermark) {
    // OpenMP Watermarking
    auto start = chrono::high_resolution_clock::now();
    Mat invisibleWatermarkedImage = OpenMP_Watermark(originalImage, rotatedWatermark, Point(10, 10), 10.01);
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    // Save watermarked image
    imwrite("C:/Users/60169/Desktop/imageInvisible.png", invisibleWatermarkedImage);


    const int maxWidth = 800; // Adjust as needed
    const int maxHeight = 600; // Adjust as needed


    cv::Size imageSize = invisibleWatermarkedImage.size();
    if (imageSize.width > maxWidth || imageSize.height > maxHeight) {
        double scale = std::min((double)maxWidth / imageSize.width, (double)maxHeight / imageSize.height);
        cv::resize(invisibleWatermarkedImage, invisibleWatermarkedImage, cv::Size(), scale, scale);
    }

    // Display watermarked image
    imshow("Image with Invisible Watermark", invisibleWatermarkedImage);

    cout << " " << endl;
    cout << " " << endl;
    cout << "Execution time for OpenMp watermark embedding: " << duration << " milliseconds" << endl;
    cout << " " << endl;
    cout << " " << endl;
    cout << "Prese any key to continue..." << endl;

    waitKey(0);

    destroyAllWindows();
}


void goMPIInvisible(Mat originalImage, Mat watermarkImage, Mat rotatedWatermark) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto start = chrono::high_resolution_clock::now();

    // Determine workload distribution
    int rowsPerProcess = originalImage.rows / size;
    int remainderRows = originalImage.rows % size;
    int startRow = rank * rowsPerProcess;
    int endRow = startRow + rowsPerProcess + (rank == size - 1 ? remainderRows : 0);

    // Embed watermark into visible and invisible watermarked images in parallel
    Mat invisibleWatermarkedImage = MPI_Watermark(originalImage, rotatedWatermark, Point(10, 10), 10.01);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    // Save watermarked images
    imwrite("C:/Users/60169/Desktop/imageInvisible.png", invisibleWatermarkedImage);

    const int maxWidth = 800; // Adjust as needed
    const int maxHeight = 600; // Adjust as needed

    cv::Size imageSize = invisibleWatermarkedImage.size();
    if (imageSize.width > maxWidth || imageSize.height > maxHeight) {
        double scale = std::min((double)maxWidth / imageSize.width, (double)maxHeight / imageSize.height);
        cv::resize(invisibleWatermarkedImage, invisibleWatermarkedImage, cv::Size(), scale, scale);
    }

    // Display watermarked images
    imshow("Image with Invisible Watermark", invisibleWatermarkedImage);

    cout << " " << endl;
    cout << " " << endl;
    cout << "Execution time for MPI watermark embedding: " << duration << " milliseconds" << endl;
    cout << " " << endl;
    cout << " " << endl;

    cout << "Press any key to continue..." << endl;
    waitKey(0);

    destroyAllWindows();

}



// ************************************************************************************
// Visible


void goOriginalVisible(Mat originalImage, Mat watermarkImage, Mat rotatedWatermark) {
    auto start = chrono::high_resolution_clock::now();
    Mat invisibleWatermarkedImage = embedWatermark(originalImage, rotatedWatermark, Point(10, 10), 200.01);
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    // Save watermarked image
    imwrite("C:/Users/60169/Desktop/imageInvisible.png", invisibleWatermarkedImage);


    const int maxWidth = 800; // Adjust as needed
    const int maxHeight = 600; // Adjust as needed


    cv::Size imageSize = invisibleWatermarkedImage.size();
    if (imageSize.width > maxWidth || imageSize.height > maxHeight) {
        double scale = std::min((double)maxWidth / imageSize.width, (double)maxHeight / imageSize.height);
        cv::resize(invisibleWatermarkedImage, invisibleWatermarkedImage, cv::Size(), scale, scale);
    }


    // Display watermarked image
    imshow("Image with Invisible Watermark", invisibleWatermarkedImage);

    cout << " " << endl;
    cout << " " << endl;
    cout << "Execution time for Original watermark embedding: " << duration << " milliseconds" << endl;
    cout << " " << endl;
    cout << " " << endl;

    cout << "Prese any key to continue..." << endl;
    waitKey(0);


    destroyAllWindows();

}


void goOpenMpVisible(Mat originalImage, Mat watermarkImage, Mat rotatedWatermark) {
    // OpenMP Watermarking
    auto start = chrono::high_resolution_clock::now();
    Mat invisibleWatermarkedImage = OpenMP_Watermark(originalImage, rotatedWatermark, Point(10, 10), 200.01);
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    // Save watermarked image
    imwrite("C:/Users/60169/Desktop/imageInvisible.png", invisibleWatermarkedImage);


    const int maxWidth = 800; // Adjust as needed
    const int maxHeight = 600; // Adjust as needed


    cv::Size imageSize = invisibleWatermarkedImage.size();
    if (imageSize.width > maxWidth || imageSize.height > maxHeight) {
        double scale = std::min((double)maxWidth / imageSize.width, (double)maxHeight / imageSize.height);
        cv::resize(invisibleWatermarkedImage, invisibleWatermarkedImage, cv::Size(), scale, scale);
    }

    // Display watermarked image
    imshow("Image with Invisible Watermark", invisibleWatermarkedImage);

    cout << " " << endl;
    cout << " " << endl;
    cout << "Execution time for OpenMp watermark embedding: " << duration << " milliseconds" << endl;
    cout << " " << endl;
    cout << " " << endl;
    cout << "Prese any key to continue..." << endl;

    waitKey(0);

    destroyAllWindows();
}


void goMPIVisible(Mat originalImage, Mat watermarkImage, Mat rotatedWatermark) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto start = chrono::high_resolution_clock::now();

    // Determine workload distribution
    int rowsPerProcess = originalImage.rows / size;
    int remainderRows = originalImage.rows % size;
    int startRow = rank * rowsPerProcess;
    int endRow = startRow + rowsPerProcess + (rank == size - 1 ? remainderRows : 0);

    // Embed watermark into visible and invisible watermarked images in parallel
    Mat invisibleWatermarkedImage = MPI_Watermark(originalImage, rotatedWatermark, Point(10, 10), 200.01);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    // Save watermarked images
    imwrite("C:/Users/60169/Desktop/imageInvisible.png", invisibleWatermarkedImage);

    const int maxWidth = 800; // Adjust as needed
    const int maxHeight = 600; // Adjust as needed

    cv::Size imageSize = invisibleWatermarkedImage.size();
    if (imageSize.width > maxWidth || imageSize.height > maxHeight) {
        double scale = std::min((double)maxWidth / imageSize.width, (double)maxHeight / imageSize.height);
        cv::resize(invisibleWatermarkedImage, invisibleWatermarkedImage, cv::Size(), scale, scale);
    }

    // Display watermarked images
    imshow("Image with Invisible Watermark", invisibleWatermarkedImage);

    cout << " " << endl;
    cout << " " << endl;
    cout << "Execution time for MPI watermark embedding: " << duration << " milliseconds" << endl;
    cout << " " << endl;
    cout << " " << endl;

    cout << "Press any key to continue..." << endl;
    waitKey(0);

    destroyAllWindows();

}



// ************************************************************************************
// Console


void toUserSelection(Mat originalImage, Mat watermarkImage, Mat rotatedWatermark, int option) {
   

        if (option == 1) 
            goOriginal(originalImage, watermarkImage, rotatedWatermark);

        else if (option == 2) 
            goOpenMp(originalImage, watermarkImage, rotatedWatermark);

        else if (option == 3)
            goMPI(originalImage, watermarkImage, rotatedWatermark);

}

void toUserInvisible(Mat originalImage, Mat watermarkImage, Mat rotatedWatermark, int option) {


    if (option == 1)
        goOriginalInvisible(originalImage, watermarkImage, rotatedWatermark);

    else if (option == 2)
        goOpenMpInvisible(originalImage, watermarkImage, rotatedWatermark);

    else if (option == 3)
        goMPIInvisible(originalImage, watermarkImage, rotatedWatermark);

}

void toUserVisible(Mat originalImage, Mat watermarkImage, Mat rotatedWatermark, int option) {


    if (option == 1)
        goOriginalVisible(originalImage, watermarkImage, rotatedWatermark);

    else if (option == 2)
        goOpenMpVisible(originalImage, watermarkImage, rotatedWatermark);

    else if (option == 3)
        goMPIVisible(originalImage, watermarkImage, rotatedWatermark);

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
    cout << "Press 1 - Original Watermarking Technique " << endl;
    cout << "Press 2 - OpenMP Watermarking Technique " << endl;
    cout << "Press 3 - MPI Watermarking Technique " << endl;

    cout << "Option: ";
    cin >> options;

    if (options == 1) {
        Mat watermarkImage = original_TextWatermark(originalImage.size(), watermarkTexts);

        Mat rotatedWatermark = rotateImage(watermarkImage, 0.0);

        toUserSelection(originalImage, watermarkImage, rotatedWatermark, 1);
    }
    else if (options ==2) {
        Mat watermarkImage = OpenMP_TextWatermark(originalImage.size(), watermarkTexts);

        Mat rotatedWatermark = rotateImage(watermarkImage, 0.0);

        toUserSelection(originalImage, watermarkImage, rotatedWatermark, 2);
    }
    else if (options == 3) {
        Mat watermarkImage = MPI_TextWatermark(originalImage.size(), watermarkTexts);

        Mat rotatedWatermark = rotateImage(watermarkImage, 0.0);

        toUserSelection(originalImage, watermarkImage, rotatedWatermark, 3);
    }

    

}

void invisibleWatermarking(Mat originalImage) {
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
    cout << "Press 1 - Original Watermarking Technique " << endl;
    cout << "Press 2 - OpenMP Watermarking Technique " << endl;
    cout << "Press 3 - MPI Watermarking Technique " << endl;

    cout << "Option: ";
    cin >> options;

    if (options == 1) {
        Mat watermarkImage = original_TextWatermark(originalImage.size(), watermarkTexts);

        Mat rotatedWatermark = rotateImage(watermarkImage, 0.0);

        toUserInvisible(originalImage, watermarkImage, rotatedWatermark, 1);
    }
    else if (options == 2) {
        Mat watermarkImage = OpenMP_TextWatermark(originalImage.size(), watermarkTexts);

        Mat rotatedWatermark = rotateImage(watermarkImage, 0.0);

        toUserInvisible(originalImage, watermarkImage, rotatedWatermark, 2);
    }
    else if (options == 3) {
        Mat watermarkImage = MPI_TextWatermark(originalImage.size(), watermarkTexts);

        Mat rotatedWatermark = rotateImage(watermarkImage, 0.0);

        toUserInvisible(originalImage, watermarkImage, rotatedWatermark, 3);
    }



}

void visibleWatermarking(Mat originalImage) {
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
    cout << "Press 1 - Original Watermarking Technique " << endl;
    cout << "Press 2 - OpenMP Watermarking Technique " << endl;
    cout << "Press 3 - MPI Watermarking Technique " << endl;

    cout << "Option: ";
    cin >> options;

    if (options == 1) {
        Mat watermarkImage = original_TextWatermark(originalImage.size(), watermarkTexts);

        Mat rotatedWatermark = rotateImage(watermarkImage, 0.0);

        toUserVisible(originalImage, watermarkImage, rotatedWatermark, 1);
    }
    else if (options == 2) {
        Mat watermarkImage = OpenMP_TextWatermark(originalImage.size(), watermarkTexts);

        Mat rotatedWatermark = rotateImage(watermarkImage, 0.0);

        toUserVisible(originalImage, watermarkImage, rotatedWatermark, 2);
    }
    else if (options == 3) {
        Mat watermarkImage = MPI_TextWatermark(originalImage.size(), watermarkTexts);

        Mat rotatedWatermark = rotateImage(watermarkImage, 0.0);

        toUserVisible(originalImage, watermarkImage, rotatedWatermark, 3);
    }



}




void selectImagePath(int imagePathOptions) {

    if (imagePathOptions == 1) {
        //Mat originalImage = imread("C:/Users/60169/Desktop/10MB.jpg");
        Mat originalImage = imread("C:/Users/60169/Desktop/30MB.jpg");
        continueWatermarking(originalImage);

    }
    else if (imagePathOptions == 2) {
        Mat originalImage = imread("C:/Users/60169/Desktop/30MB.jpg");
        continueWatermarking(originalImage);

    }
    else if (imagePathOptions == 3) {
        string imagePath;
        cout << "Example of image path: C:/Users/60169/Desktop/30MB.jpg " << endl;
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


void selectVisiblity(int imagePathOptions) {

    
    //Input - Original Image
    int visiblityOptions;
    cout << " " << endl;
    cout << "Select Watermark Visibility" << endl;
    cout << "Input 1 - Invisible " << endl;
    cout << "Input 2 - Visible " << endl;
    cout << "Input 3 - Both " << endl;
    cout << " " << endl;
    cout << "Option: ";
    cin >> visiblityOptions;



    if (visiblityOptions == 3 && imagePathOptions == 1) {
        //Mat originalImage = imread("C:/Users/60169/Desktop/10MB.jpg");
        Mat originalImage = imread("C:/Users/60169/Desktop/30MB.jpg");
        continueWatermarking(originalImage);

    }
    else if (visiblityOptions == 3 && imagePathOptions == 2) {
        Mat originalImage = imread("C:/Users/60169/Desktop/30MB.jpg");
        continueWatermarking(originalImage);

    }
    else if (visiblityOptions == 3 && imagePathOptions == 3) {
        string imagePath;
        cout << "Example of image path: C:/Users/60169/Desktop/30MB.jpg " << endl;
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



    else if (visiblityOptions == 1 && imagePathOptions == 1) {
        //Mat originalImage = imread("C:/Users/60169/Desktop/10MB.jpg");
        Mat originalImage = imread("C:/Users/60169/Desktop/30MB.jpg");
        invisibleWatermarking(originalImage);

    }
    else if (visiblityOptions == 1 && imagePathOptions == 2) {
        Mat originalImage = imread("C:/Users/60169/Desktop/30MB.jpg");
        invisibleWatermarking(originalImage);

    }
    else if (visiblityOptions == 1 && imagePathOptions == 3) {
        string imagePath;
        cout << "Example of image path: C:/Users/60169/Desktop/30MB.jpg " << endl;
        cout << "Please input the image path: " << endl;
        cout << " " << endl;
        cout << "Image path: ";
        cin >> imagePath;
        Mat originalImage = imread(imagePath);

        if (!originalImage.empty()) {
            invisibleWatermarking(originalImage);
        }
        else {
            cout << "Invalid Image Path, please input again" << endl;
            selectImagePath(3);
        }


    }



    else if (visiblityOptions == 2 && imagePathOptions == 1) {
        //Mat originalImage = imread("C:/Users/60169/Desktop/10MB.jpg");
        Mat originalImage = imread("C:/Users/60169/Desktop/30MB.jpg");
        visibleWatermarking(originalImage);

    }
    else if (visiblityOptions == 2 && imagePathOptions == 2) {
        Mat originalImage = imread("C:/Users/60169/Desktop/30MB.jpg");
        visibleWatermarking(originalImage);

    }
    else if (visiblityOptions == 2 && imagePathOptions == 3) {
        string imagePath;
        cout << "Example of image path: C:/Users/60169/Desktop/30MB.jpg " << endl;
        cout << "Please input the image path: " << endl;
        cout << " " << endl;
        cout << "Image path: ";
        cin >> imagePath;
        Mat originalImage = imread(imagePath);

        if (!originalImage.empty()) {
            visibleWatermarking(originalImage);
        }
        else {
            cout << "Invalid Image Path, please input again" << endl;
            selectImagePath(3);
        }


    }



}








int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    //Input - Original Image
    int imagePathOptions;
    cout << " " << endl;
    cout << "Firstly, provide an image path" << endl;
    cout << "Input 1 - Light color image path (30MB) " << endl;
    cout << "Input 2 - Dark color image path (30MB) " << endl;
    cout << "Input 3 - Provide custom image path " << endl;
    cout << " " << endl;
    cout << "Option: ";
    cin >> imagePathOptions;

    selectVisiblity(imagePathOptions);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}

