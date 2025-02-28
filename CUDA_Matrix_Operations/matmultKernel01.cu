#include "matmultKernel.h"

#define FOOTPRINT_SIZE 32  // Updated footprint size for four values per thread

// Define a GPU kernel to perform matrix multiplication of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {

  // Matrix block descriptors
  float *Asub, *Bsub, *Csub;
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Each THREAD BLOCK computes one submatrix Csub of C
  // Each thread creates its own matrix descriptor Csub
  Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * block_col];

  // Each thread computes four elements of Csub
  float Cvalue[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // Store four results per thread

  // Loop over all sub-matrices of A and B that are required to compute Csub
  for (int m = 0; m < (A.width / FOOTPRINT_SIZE); ++m) {
    // Get Asub and Bsub descriptors
    Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * m];
    Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_col];

    // Shared memory sub-matrices for Asub and Bsub
    __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
    __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

    // Each thread loads four elements of shared_A and shared_B
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int row_offset = i / 2;
      int col_offset = i % 2;
      shared_A[thread_row * 2 + row_offset][thread_col * 2 + col_offset] = Asub[(thread_row * 2 + row_offset) * A.stride + thread_col * 2 + col_offset];
      shared_B[thread_row * 2 + row_offset][thread_col * 2 + col_offset] = Bsub[(thread_row * 2 + row_offset) * B.stride + thread_col * 2 + col_offset];
    }

    // Synchronize to ensure all elements are loaded
    __syncthreads();

    // Perform multiplication using unrolled loop to accumulate results
    #pragma unroll
    for (int e = 0; e < FOOTPRINT_SIZE; ++e) {
      for (int i = 0; i < 4; ++i) {
        int row_offset = i / 2;
        int col_offset = i % 2;
        Cvalue[i] += shared_A[thread_row * 2 + row_offset][e] * shared_B[e][thread_col * 2 + col_offset];
      }
    }

    // Synchronize to ensure all threads have completed this stage
    __syncthreads();
  }

  // Write results to global memory
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    int row_offset = i / 2;
    int col_offset = i % 2;
    Csub[(thread_row * 2 + row_offset) * C.stride + thread_col * 2 + col_offset] = Cvalue[i];
  }
}

