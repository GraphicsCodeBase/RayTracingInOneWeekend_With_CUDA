#include "gradient.cuh"
#include <iostream>
#include <fstream>
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
    const int image_width = 256;
    const int image_height = 256;
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

    std::cout << "CUDA kernel completed successfully!\n";

    // Copy result from device to host
    cudaMemcpy(h_image, d_image, image_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);

    // Write to PPM file
    std::ofstream outfile("../out_folder/gradient.ppm");
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not create output file!" << std::endl;
        delete[] h_image;
        return;
    }

    // PPM Header
    outfile << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    // Write pixel data
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            int pixelIndex = (j * image_width + i) * 3;
            outfile << static_cast<int>(h_image[pixelIndex + 0]) << ' '
                    << static_cast<int>(h_image[pixelIndex + 1]) << ' '
                    << static_cast<int>(h_image[pixelIndex + 2]) << '\n';
        }
    }

    outfile.close();

    // Free host memory
    delete[] h_image;

    std::cout << "Image written to gradient.ppm\n";
}
