#include <iostream>
#include <cuda_runtime.h>

// Device kernel
__global__ void multiplyByConstant(float *d_out, float *d_in, float constant, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        d_out[idx] = d_in[idx] * constant;
    }
}

int main() {
    const int ARRAY_SIZE = 512;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
    float h_in[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = float(i);
    }
    float h_out[ARRAY_SIZE];

    float *d_in;
    float *d_out;

    cudaMalloc((void**)&d_in, ARRAY_BYTES);
    cudaMalloc((void**)&d_out, ARRAY_BYTES);

    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 128;
    const int GRID_SIZE = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    multiplyByConstant<<<GRID_SIZE, BLOCK_SIZE>>>(d_out, d_in, 2.0f, ARRAY_SIZE);

    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    for (int i = 0; i < ARRAY_SIZE; i++) {
        std::cout << h_in[i] << " * 2 = " << h_out[i] << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
