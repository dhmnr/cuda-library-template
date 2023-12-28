#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include "cuda_starter/utils.h"

__global__ void add_vectors_kernel(double *a, double *b, double *c, size_t N) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < N) c[id] = a[id] + b[id];
}

void add_vectors(double *d_a, double *d_b, double *d_c, size_t N) {
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((N - 1) / blockSize.x + 1, 1, 1);

    add_vectors_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    cudaError_t err = cudaDeviceSynchronize();
    cudaErrorCheck(err, "Failed to execute add_vectors_kernel");
}

void add_vectors_with_copy(double *h_A, double *h_B, double *h_C, size_t N) {
    cudaError_t err;
    size_t size = N * sizeof(double);

    double *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);
    cudaErrorCheck(err, "Failed to allocate device vector A");

    double *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);
    cudaErrorCheck(err, "Failed to allocate device vector B");

    double *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);
    cudaErrorCheck(err, "Failed to allocate device vector C");

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaErrorCheck(err, "Failed to copy vector A from host to device");

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaErrorCheck(err, "Failed to copy vector B from host to device");

    add_vectors(d_A, d_B, d_C, N);

    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaErrorCheck(err, "Failed to copy vector C from device to host");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
