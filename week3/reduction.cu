#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    // put input to GPU array
    float* h_input = new float[N];
    cudaMemcpy(h_input, input, N * sizeof(float), cudaMemcpyDeviceToHost);

    double sum = 0;
    for(int i = 0; i < N; ++i){
        sum += h_input[i];
    }
    // type cast & put output
    float result = static_cast<float>(sum);
    cudaMemcpy(output, &result, sizeof(float), cudaMemcpyHostToDevice);
    
    delete[] h_input;
}
