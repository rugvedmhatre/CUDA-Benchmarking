// vecAddKernel01.cu
// For ECE-GY 9143 - High Performance Computing for Machine Learning
// Instructor: Parijat Dubey

// This Kernel adds two Vectors A and B in C on GPU
// using coalesced memory access.

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    // Calculate the base index for this thread
    int baseIndex = (blockIdx.x * blockDim.x + threadIdx.x) * N;

    for (int i = 0; i < N; ++i) {
        C[baseIndex + i] = A[baseIndex + i] + B[baseIndex + i];
    }
}
