#include "gradient.cuh"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel to generate gradient
// Each thread computes one pixel's color
__global__ void gradientKernel(unsigned char* image, int width, int height) {
    // Calculate pixel coordinates from thread/block indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // column (x)
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // row (y)

    // Check if thread is within image bounds
    if (i >= width || j >= height) return;

    // Calculate color based on pixel position
    // Creates a gradient from bottom-left (black) to top-right (yellow-ish)
    double r = double(i) / (width - 1);
    double g = double(j) / (height - 1);
    double b = 0.0;

    // Convert to [0, 255] range
    unsigned char ir = static_cast<unsigned char>(255.999 * r);
    unsigned char ig = static_cast<unsigned char>(255.999 * g);
    unsigned char ib = static_cast<unsigned char>(255.999 * b);

    // Calculate index in the 1D image array
    // Each pixel has 3 values (RGB)
    int pixelIndex = (j * width + i) * 3;

    // Write RGB values
    image[pixelIndex + 0] = ir;
    image[pixelIndex + 1] = ig;
    image[pixelIndex + 2] = ib;
}

void renderGradient() {
    // Image dimensions
    const int image_width = 1920;
    const int image_height = 1080;
    const int image_size = image_width * image_height * 3; // RGB for each pixel

    // Allocate host memory for the image
    unsigned char* h_image = new unsigned char[image_size];

    // Allocate device memory for the image
    unsigned char* d_image;
    cudaMalloc(&d_image, image_size * sizeof(unsigned char));

    // Define block and grid dimensions
    // Using 16x16 threads per block (common choice)
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (image_width + blockSize.x - 1) / blockSize.x,
        (image_height + blockSize.y - 1) / blockSize.y
    );

    std::cout << "Launching CUDA kernel with grid(" << gridSize.x << ", " << gridSize.y
              << ") and block(" << blockSize.x << ", " << blockSize.y << ")\n";

    // Start GPU timer
    auto gpu_start = std::chrono::high_resolution_clock::now();

    // Launch the kernel
    gradientKernel<<<gridSize, blockSize>>>(d_image, image_width, image_height);

    // Check for kernel launch errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_image);
        delete[] h_image;
        return;
    }

    // Wait for GPU to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_image);
        delete[] h_image;
        return;
    }

    // End GPU timer
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);

    std::cout << "CUDA kernel completed successfully!\n";
    std::cout << "GPU computation time: " << gpu_duration.count() << " ms\n";

    // Copy result from device to host
    cudaMemcpy(h_image, d_image, image_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);

    // Write to PPM file (binary format P6 for speed)
    std::ofstream outfile("../out_folder/gradient.ppm", std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not create output file!" << std::endl;
        delete[] h_image;
        return;
    }

    // Start file writing timer
    auto write_start = std::chrono::high_resolution_clock::now();

    // PPM Header (P6 = binary format)
    outfile << "P6\n" << image_width << ' ' << image_height << "\n255\n";

    // Write pixel data as raw binary (much faster!)
    outfile.write(reinterpret_cast<char*>(h_image), image_size);

    // End file writing timer
    auto write_end = std::chrono::high_resolution_clock::now();
    auto write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(write_end - write_start);

    outfile.close();

    // Free host memory
    delete[] h_image;

    std::cout << "File writing time: " << write_duration.count() << " ms\n";
    std::cout << "Image written to gradient.ppm\n";
}
