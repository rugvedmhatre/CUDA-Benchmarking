// matmultKernel01.cu
// For ECE-GY 9143 - High Performance Computing for Machine Learning
// Instructor: Parijat Dubey

// Multiplies two matrices using CUDA: A x B = C
#include "matmultKernel.h"

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
    // Compute offsets in the output matrix for coalesced memory access.
    int outputRowOffset = BLOCK_SIZE * B.width / FOOTPRINT_SIZE;
    int outputColOffset = BLOCK_SIZE * A.height / FOOTPRINT_SIZE;

    // Calculate row and column indices for the current thread within the output matrix.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Variables to accumulate the products for the output matrix C.
    float c1sum = 0;
    float c2sum = 0;
    float c3sum = 0;
    float c4sum = 0;

    // Loop over all elements in the row of A and column of B to compute product contributions to C.
    for (int k = 0; k < A.width; ++k) {
        c1sum += A.elements[row * A.width + k] * B.elements[k * B.width + col];
        c2sum += A.elements[(row + outputRowOffset) * A.width + k] * B.elements[k * B.width + col];
        c3sum += A.elements[row * A.width + k] * B.elements[k * B.width + (col + outputColOffset)];
        c4sum += A.elements[(row + outputRowOffset) * A.width + k] * B.elements[k * B.width + (col + outputColOffset)];
    }   
    // Store the results in the appropriate elements of matrix C.
    C.elements[row * C.width + col] = c1sum;
    C.elements[(row + outputRowOffset) * C.width + col] = c2sum;
    C.elements[row * C.width + (col + outputColOffset)] = c3sum;
    C.elements[(row + outputRowOffset) * C.width + (col + outputColOffset)] = c4sum;
}
