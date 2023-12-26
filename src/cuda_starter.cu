#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void add_vectors_kernel(double *a, double *b, double *c, size_t N) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < N) c[id] = a[id] + b[id];
}

void add_vectors(double *d_a, double *d_b, double *d_c, size_t N) {
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((N - 1) / blockSize.x + 1, 1, 1);

    add_vectors_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
}