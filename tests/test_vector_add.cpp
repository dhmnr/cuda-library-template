#include "acutest.h"
#include "cuda_starter/cuda_starter.h"
#include "math.h"

void test_vector_add(size_t N) {
    size_t size = N * sizeof(double);

    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);

    TEST_ASSERT_(!(h_A == NULL || h_B == NULL || h_C == NULL),
                 "Failed to allocate host vectors!\n");

    for (int i = 0; i < N; ++i) {
        h_A[i] = rand() / (double)RAND_MAX;
        h_B[i] = rand() / (double)RAND_MAX;
    };

    add_vectors_with_copy(h_A, h_B, h_C, N);

    for (int i = 0; i < N; ++i) {
        TEST_CHECK_(fabs(h_A[i] + h_B[i] - h_C[i]) <= 1e-5,
                    "Result verification failed at element %d!\n", i);
    }
    free(h_A);
    free(h_B);
    free(h_C);
}

void test_vector_add_10K() { test_vector_add(1e4); }

void test_vector_add_100K() { test_vector_add(1e5); }

void test_vector_add_1M() { test_vector_add(1e6); }

TEST_LIST = {
    {"add_vector with 10 thousand elements", test_vector_add_10K},
    {"add_vector with 100 thousand elements", test_vector_add_100K},
    {"add_vector with 1 million elements", test_vector_add_1M},
    {NULL, NULL} /* zeroed record marking the end of the list */
};