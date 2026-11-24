#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        float mx = input[0];
        float sum = 0;
        // exp(i) / sum{exp(j)} = exp(i - MAX) / sum{exp(j - MAX)} [max trick](softmax)
        for(int j=1; j<N; ++j) if(mx < input[j]) mx = input[j];
        for(int j=0; j<N; ++j) sum += expf(input[j] - mx);
        output[i] = expf(input[i] - mx) / sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
