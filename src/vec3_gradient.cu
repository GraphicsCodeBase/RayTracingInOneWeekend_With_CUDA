#include "vec3_gradient.cuh"
#include "vec3.cuh"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

// Helper function to convert color (vec3) to RGB bytes
__device__ void write_color(unsigned char* image, int index, color pixel_color) {
    // Clamp values to [0, 1] range
    double r = pixel_color.x();
    double g = pixel_color.y();
    double b = pixel_color.z();

    // Clamp to [0.0, 0.999]
    if (r < 0.0) r = 0.0;
    if (r > 0.999) r = 0.999;
    if (g < 0.0) g = 0.0;
    if (g > 0.999) g = 0.999;
    if (b < 0.0) b = 0.0;
    if (b > 0.999) b = 0.999;

    // Convert to [0, 255]
    image[index + 0] = static_cast<unsigned char>(256 * r);
    image[index + 1] = static_cast<unsigned char>(256 * g);
    image[index + 2] = static_cast<unsigned char>(256 * b);
}

// CUDA kernel using vec3 class
__global__ void vec3GradientKernel(unsigned char* image, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    // Create a color using vec3 - THIS IS THE KEY DIFFERENCE!
    // Instead of calculating RGB separately, we use vec3 operations
    color pixel_color(double(i) / (width - 1),
                      double(j) / (height - 1),
                      0.0);

    // You can also do vec3 math operations on the GPU!
    // For example, let's add a slight blue tint
    color blue_tint(0.0, 0.0, 0.2);
    pixel_color = pixel_color + blue_tint;

    // Or multiply colors (darken the image)
    pixel_color = pixel_color * 0.8;

    // Calculate pixel index
    int pixelIndex = (j * width + i) * 3;

    // Write the color to the image
    write_color(image, pixelIndex, pixel_color);
}

void renderVec3Gradient() {
    const int image_width = 1920;
    const int image_height = 1080;
    const int image_size = image_width * image_height * 3;

    std::cout << "\n=== Vec3 Gradient Demo ===\n";
    std::cout << "This demonstrates using the vec3 class ON THE GPU!\n\n";

    // Allocate host memory
    unsigned char* h_image = new unsigned char[image_size];

    // Allocate device memory
    unsigned char* d_image;
    cudaMalloc(&d_image, image_size * sizeof(unsigned char));

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (image_width + blockSize.x - 1) / blockSize.x,
        (image_height + blockSize.y - 1) / blockSize.y
    );

    std::cout << "Launching CUDA kernel with vec3 operations...\n";

    auto gpu_start = std::chrono::high_resolution_clock::now();

    // Launch kernel that uses vec3 class!
    vec3GradientKernel<<<gridSize, blockSize>>>(d_image, image_width, image_height);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_image);
        delete[] h_image;
        return;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_image);
        delete[] h_image;
        return;
    }

    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);

    std::cout << "GPU computation time: " << gpu_duration.count() << " ms\n";

    // Copy result back
    cudaMemcpy(h_image, d_image, image_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(d_image);

    // Write to file
    std::ofstream outfile("../out_folder/vec3_gradient.ppm", std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not create output file!" << std::endl;
        delete[] h_image;
        return;
    }

    auto write_start = std::chrono::high_resolution_clock::now();

    outfile << "P6\n" << image_width << ' ' << image_height << "\n255\n";
    outfile.write(reinterpret_cast<char*>(h_image), image_size);

    auto write_end = std::chrono::high_resolution_clock::now();
    auto write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(write_end - write_start);

    outfile.close();
    delete[] h_image;

    std::cout << "File writing time: " << write_duration.count() << " ms\n";
    std::cout << "Image written to vec3_gradient.ppm\n";
    std::cout << "\nNotice: The gradient has a blue tint and is darker!\n";
    std::cout << "This is because we used vec3 math operations in the kernel.\n";
}
