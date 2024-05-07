#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "./EmbedWatermark.cuh" 
#include <opencv2/opencv.hpp> 

using namespace cv;
using namespace std; 



__global__ void split_kernel(const float* src, int cols, int rows, float* channel1, float* channel2, float* channel3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < cols && j < rows) {
        int index = j * cols + i;
        int pixelIndex = index * 3;
        channel1[index] = src[pixelIndex];
        channel2[index] = src[pixelIndex + 1];
        channel3[index] = src[pixelIndex + 2];
    }
}

void split_cuda(const cv::Mat& src, std::vector<cv::Mat>& channels) {
    channels.clear();

    // Allocate device memory for source image
     float* d_src;
    cudaMalloc(&d_src, sizeof(float) * src.rows * src.cols * src.channels());
    cudaMemcpy(d_src, src.ptr<float>(), sizeof(float) * src.rows * src.cols * src.channels(), cudaMemcpyHostToDevice);

    // Allocate device memory for each channel
    float* d_channel1, * d_channel2, * d_channel3;
    cudaMalloc((void**)&d_channel1, sizeof(float) * src.rows * src.cols);
    cudaMalloc((void**)&d_channel2, sizeof(float) * src.rows * src.cols);
    cudaMalloc((void**)&d_channel3, sizeof(float) * src.rows * src.cols);

    int threadsPerBlock = 16;
    dim3 threadPerBlock(16, 16);
    dim3 blocksPerGrid((src.cols + threadsPerBlock - 1) / threadsPerBlock, (src.rows + threadsPerBlock - 1) / threadsPerBlock);

    // Launch kernel to split all channels simultaneously
    split_kernel << <blocksPerGrid, threadPerBlock >> > (d_src, src.cols, src.rows, d_channel1, d_channel2, d_channel3);
    cudaDeviceSynchronize();  // Wait for kernel to finish before proceeding

    // Copy results back to host memory and create cv::Mat objects
    cv::Mat channel1(src.rows, src.cols, CV_32F);
    cv::Mat channel2(src.rows, src.cols, CV_32F);
    cv::Mat channel3(src.rows, src.cols, CV_32F);

    cudaMemcpy(channel1.ptr<float>(), d_channel1, sizeof(float) * src.rows * src.cols, cudaMemcpyDeviceToHost);
    cudaMemcpy(channel2.ptr<float>(), d_channel2, sizeof(float) * src.rows * src.cols, cudaMemcpyDeviceToHost);
    cudaMemcpy(channel3.ptr<float>(), d_channel3, sizeof(float) * src.rows * src.cols, cudaMemcpyDeviceToHost);

    channels.push_back(channel1);
    channels.push_back(channel2);
    channels.push_back(channel3);

    // Free device memory
    cudaFree(d_src);
    cudaFree(d_channel1);
    cudaFree(d_channel2);
    cudaFree(d_channel3);
}



__global__ void merge_kernel(const float* channel1, const float* channel2, const float* channel3, float* output, int cols, int rows) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int index = y * cols + x;
        int pixelIndex = index * 3;
        output[pixelIndex] = channel1[index];
        output[pixelIndex + 1] = channel2[index];
        output[pixelIndex + 2] = channel3[index];
    }
}

cv::Mat merge_cuda(const std::vector<cv::Mat>& channels, cv::Mat& output) {
    // Ensure input channels are of the same size
    CV_Assert(channels.size() == 3);

    // Create output image
    output.create(channels[0].size(), CV_32FC3); // Output image will be in float format

    // Allocate device memory for each channel
    float* d_channel1, * d_channel2, * d_channel3, * d_output;
    cudaMalloc((void**)& d_channel1, channels[0].rows * channels[0].cols * sizeof(float));
    cudaMalloc((void**)&d_channel2, channels[1].rows * channels[1].cols * sizeof(float));
    cudaMalloc((void**)&d_channel3, channels[2].rows * channels[2].cols * sizeof(float));
    cudaMalloc((void**)&d_output, channels[0].rows * channels[0].cols * sizeof(float) * 3); // Allocate memory for three channels

    Mat channel1;
    channels[0].convertTo(channel1, CV_32F);

    Mat channel2;
    channels[1].convertTo(channel2, CV_32F);

    Mat channel3;
    channels[2].convertTo(channel3, CV_32F);


    cudaMemcpy(d_channel1, channel1.data, channels[0].rows * channels[0].cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_channel2, channel2.data, channels[1].rows * channels[1].cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_channel3, channel3.data, channels[2].rows * channels[2].cols * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((channels[0].cols + blockDim.x - 1) / blockDim.x, (channels[0].rows + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    merge_kernel << <gridDim, blockDim >> > (d_channel1, d_channel2, d_channel3, d_output, channels[0].cols, channels[0].rows);

    // Copy output data from device
    cudaMemcpy(output.ptr<float>(), d_output, channels[0].rows * channels[0].cols * sizeof(float) * 3, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_channel1);
    cudaFree(d_channel2);
    cudaFree(d_channel3);
    cudaFree(d_output);

    return output;
}



__global__ void embedWatermarkKernel(float* imageChannels, float* watermarkChannels, int width, int height, float alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        for (int i = 0; i < 3; ++i) {
            int pixelIndex = index * 3 + i;
            imageChannels[pixelIndex] += alpha * (watermarkChannels[pixelIndex]) / 255.0f;
            //imageChannels[pixelIndex] = fminf(1.0f, fmaxf(0.0f, imageChannels[pixelIndex]));
        }
    }
}

Mat CUDA_Watermark(const Mat& image, const Mat& watermark, const Point& position, float alpha) {
    Mat watermarkedImage = image.clone();

    // Convert image and watermark to float32
    Mat imageFloat;
    image.convertTo(imageFloat, CV_32F);
    Mat watermarkFloat;
    watermark.convertTo(watermarkFloat, CV_32F);

    // Allocate memory on GPU
    float* d_image, * d_watermark;
    size_t imageSize = imageFloat.rows * imageFloat.cols * imageFloat.channels() * sizeof(float);
    size_t watermarkSize = watermarkFloat.rows * watermarkFloat.cols * watermarkFloat.channels() * sizeof(float);
    cudaMalloc((void**)&d_image, imageSize);
    cudaMalloc((void**)&d_watermark, watermarkSize);

    // Transfer data from CPU to GPU
    cudaMemcpy(d_image, imageFloat.data, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_watermark, watermarkFloat.data, watermarkSize, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((imageFloat.cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (imageFloat.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Call CUDA kernel
    embedWatermarkKernel << <numBlocks, threadsPerBlock >> > (d_image, d_watermark, imageFloat.cols, imageFloat.rows, alpha);

    // Transfer result back to CPU
    cudaMemcpy(imageFloat.data, d_image, imageSize, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_image);
    cudaFree(d_watermark); 

    // **************************************************
    // Duration for Split 
    auto startSplitOpenMP = chrono::high_resolution_clock::now();

    vector<Mat> modifiedChannels;
    split_cuda(imageFloat, modifiedChannels);

    auto stopSplitOpenMP = chrono::high_resolution_clock::now();
    auto durationSplitOpenMP = chrono::duration_cast<chrono::milliseconds>(stopSplitOpenMP - startSplitOpenMP);


    // Clipping after conversion to CV_8U
    for (int i = 0; i < modifiedChannels.size(); ++i) {
        modifiedChannels[i].convertTo(modifiedChannels[i], CV_8U);
    }


    // ***************************************************
    // Duration for Merge 
    auto startMergeOpenMP = chrono::high_resolution_clock::now();

    // Merge modified channels back into a single image
    merge_cuda(modifiedChannels, watermarkedImage);

    auto stopMergeOpenMP = chrono::high_resolution_clock::now();
    auto durationMergeOpenMP = chrono::duration_cast<chrono::milliseconds>(stopMergeOpenMP - startMergeOpenMP);


    // Convert back to original data type (8U)
    watermarkedImage.convertTo(watermarkedImage, CV_8U);


    cout << " " << endl;
    cout << " " << endl;
    cout << "CUDA Split execution time: " << durationSplitOpenMP.count() << " milliseconds" << endl;
    cout << "CUDA Merge execution time: " << durationMergeOpenMP.count() << " milliseconds" << endl;
    cout << " " << endl;
    cout << " " << endl;



    return watermarkedImage; 
}