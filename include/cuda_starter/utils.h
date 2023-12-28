#pragma once

int cudaErrorCheck(cudaError_t err, const char *errString, bool isFatal = true);
