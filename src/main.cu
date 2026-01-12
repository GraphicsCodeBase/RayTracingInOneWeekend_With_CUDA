#include <iostream>
#include <cuda_runtime.h>

__global__ void helloKernel() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    std::cout << "Ray Tracing CUDA Project" << std::endl;

    // Query CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    // Launch a simple kernel
    helloKernel<<<1, 5>>>();
    cudaDeviceSynchronize();

    std::cout << "Program completed successfully!" << std::endl;

    return 0;
}
