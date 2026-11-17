#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * height) {
        float tmp = 0;
        tmp += input[3 * i] * 0.299;
        tmp += input[3 * i + 1] * 0.587;
        tmp += input[3 * i + 2] * 0.114;
        output[i] = tmp;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
