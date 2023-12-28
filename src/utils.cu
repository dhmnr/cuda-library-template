#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

int cudaErrorCheck(cudaError_t err, const char *errString, bool isFatal = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s \n\tError %s : %s\n", errString, cudaGetErrorName(err),
                cudaGetErrorString(err));
        if (isFatal)
            exit(EXIT_FAILURE);
        else
            return 1;
    }
    return 0;
}