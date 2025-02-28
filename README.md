# CUDA-Benchmarking
This repository contains CUDA implementations for Matrix Multiplication, Unified Memory, and Convolution. It is designed for benchmarking GPU performance, measuring execution time, memory throughput, and computational efficiency. Ideal for analyzing CUDA optimizations and GPU acceleration strategies.

---

## CUDA Matrix Operations

### Vector Addition

To understand the effects of memory coalescing, we test a microbenchmark, first without any memory coalescing and, later, with memory coalescing. We run this microbenchmark for different values per thread - 500, 1000, and 2000. 

#### Results

##### Vector Addition (without Memory Coalescing)

The below figure is the output of the provided program `vecadd00` (using the kernel `vecaddKernel00`) for the values per thread - 500, 1000, 2000 on A100 GPU. We can see that as input size increases, the execution time also increases.

![vector addition without memory coalescing](https://github.com/rugvedmhatre/CUDA-Benchmarking/blob/main/images/a_q1.png?raw=true)

##### Vector Addition (with Memory Coalescing)

The below figure is the output of the provided program `vecadd01` (using the kernel `vecaddKernel01`) for the values per thread - 500, 1000, 2000 on A100 GPU. We can see that as input size increases, the execution time also increases. The execution times on this one are a bit lower than the one in the previous figure because of coalesced memory reads. As a result, we are getting higher GFlopsS and GByteS in this case.

![vector addition with memory coalescing](https://github.com/rugvedmhatre/CUDA-Benchmarking/blob/main/images/a_q2.png?raw=true)

### Matrix Multiplication

In this experiment we investigate how each of the following factors influences performance of matrix multiplication in CUDA:

1. The size of matrices to be multiplied
2. The size of the block computed by each thread block

#### Results

##### Matrix Multiplication (without Memory Coalescing)

The below figure is the output of the provided program `matmult00` (using the kernel `matmultKernel00`) for square matrices of size - 256, 512, 1024 on A100 GPU.

![matrix multiplication](https://github.com/rugvedmhatre/CUDA-Benchmarking/blob/main/images/a_q3.png?raw=true)

##### Matrix Multiplication (with Memory Coalescing and Loop Unrolling)

We modify the kernel so that each thread calculates four values, and we are able to utilize coalesced memory reads for lower execution times. We also unroll loops to increase parallelism.

The below figure is the output of the provided program `matmult01` (using the kernel `matmultKernel01`) for square matrices of size - 256, 512, 1024 on A100 GPU. We can see that as input size increases, the execution time also increases. But the execution times for large input sizes are significantly lower than the one in the previous figure, due to the optimized kernel.

![matrix multiplication optimized](https://github.com/rugvedmhatre/CUDA-Benchmarking/blob/main/images/a_q4.png?raw=true)

### Conclusion

Based on the experiments we have conducted above, the following methods can help us extract maximum
performance with CUDA:

- Using coalesced memory accesses, we can reduce the memory latency and maximize bandwidth uti-
lization.
- By using the appropriate values for the number of blocks and the number of grids, we are able to utilize
the underlying hardware to the maximum, and extract the maximum throughput.
- We can use unified memory to minimize data transfers between device and host, and get higher per-
formance.
- We can also utilize the shared memory to minimize global memory accesses and improve the perfor-
mance.

## CUDA Unified Memory

In this experiments, we compare vector operations executed on host vs. GPU to quantify the speedup.

### Vector Addition (on CPU)

We write a C++ program that adds the elements of two arrays with a K million elements each. Here K is a parameter of the program. We profile and get the execution times for K = 1, 5, 10, 50, 100. The below figure shows the results.

![vector addition on cpu](https://github.com/rugvedmhatre/CUDA-Benchmarking/blob/main/images/b_q1.png?raw=true)

### Vector Addition (on GPU without Shared Memory)

Next, using CUDA, we execute the vector add operation as a kernel on GPU. Use `cudaMalloc()` to allocate memory on GPU for storing the arrays and `cudaMemcpy()` to copy data to and from the GPUs. Note that the host and GPU memory is not shared in this case. We test three scenarios for this part:

1. Using one block with 1 thread
2. Using one block with 256 threads
3. Using multiple blocks with 256 threads per block with the total number of threads across all the blocks equal to the size of arrays.

Again, we profile and get the time to execute this program for K = 1, 5, 10, 50, 100.

#### Results 

The below figure is the output of a CUDA kernel using 1 block with 1 thread to add elements of two vectors with K million elements. We run the code for K - 1, 5, 10, 50, 100 on A100 GPU.

![vector addition on gpu 2a](https://github.com/rugvedmhatre/CUDA-Benchmarking/blob/main/images/b_q2a.png?raw=true)

The below figure is the output of a CUDA kernel using 1 block with 256 threads to add elements of two vectors with K million elements. We run the code for K - 1, 5, 10, 50, 100 on A100 GPU.

![vector addition on gpu 2b](https://github.com/rugvedmhatre/CUDA-Benchmarking/blob/main/images/b_q2b.png?raw=true)

The below figure is the output of a CUDA kernel using multiple blocks with 256 threads per block with the total number of threads across all the blocks equal to the size of the vector. This kernel is used to add elements of two vectors with K million elements for K - 1, 5, 10, 50, 100 on A100 GPU.

![vector addition on gpu 2c](https://github.com/rugvedmhatre/CUDA-Benchmarking/blob/main/images/b_q2c.png?raw=true)

### Vector Addition (on GPU with Shared Memory)

In this experiment we repeat the above three scenarios using CUDA Unified Memory. Instead of `cudaMalloc()` we use `cudaMallocManaged()` to allocate data in unified memory. 

#### Results 

The below figure is the output of a CUDA kernel with Unified Memory using 1 block with 1 thread to add elements of two vectors with K million elements. We run the code for K - 1, 5, 10, 50, 100 on A100 GPU.

![vector addition on gpu 3a](https://github.com/rugvedmhatre/CUDA-Benchmarking/blob/main/images/b_q3a.png?raw=true)

The below figure is the output of a CUDA kernel with Unified Memory using 1 block with 256 threads to add
elements of two vectors with K million elements. We run the code for K - 1, 5, 10, 50, 100 on A100 GPU.

![vector addition on gpu 3b](https://github.com/rugvedmhatre/CUDA-Benchmarking/blob/main/images/b_q3b.png?raw=true)

The below figure is the output of a CUDA kernel with Unified Memory using multiple blocks with 256 threads per block with the total number of threads across all the blocks equal to the size of the vector. This kernel is used to add elements of two vectors with K million elements for K - 1, 5, 10, 50, 100 on A100 GPU.

![vector addition on gpu 3c](https://github.com/rugvedmhatre/CUDA-Benchmarking/blob/main/images/b_q3c.png?raw=true)

### Conclusion

![graph 1](https://github.com/rugvedmhatre/CUDA-Benchmarking/blob/main/images/bq2-graph.png?raw=true)

![graph 2](https://github.com/rugvedmhatre/CUDA-Benchmarking/blob/main/images/bq3-graph.png?raw=true)

We can see that using Unified memory improves our overall execution time. But the CPU overall time is
lower, this might be because we don’t have to copy between memories. The one with unified memory for
large blocks and large K is very close to the CPU time.

If we take a look at the execution times, the exeuction time on the GPU is very low, but the memory
allocation and memory copy time is high. This is what makes it slower than the CPU.

## Convolution in CUDA

We implement a convolution of an image using a set of filters. Consider the following:

- An input tensor $I$ with dimensions: $C$, $H$, $W$. Each element of $I$ is generated as follows:
$$I[c, x, y] = c \cdot (x + y)$$
- A set of convolution filters with dimensions: $K$, $C$, $FH$, $FW$. Each element of the filter $F$ is generated as follows:
$$F [k, c, i, j] = (c + k) \cdot (i + j)$$
- Dimensions are: $H = 1024$, $W = 1024$, $C = 3$, $FW = 3$, $FH = 3$, $K = 64$.
- All tensors have double elements (double precision)
- A tensor $I_0$ of sizes $C$, $W+2P$, $H+2P$ where $P=1$. $I_0$ is obtained from the tensor $I$ adding the padding rows and columns with the elements set to zero.
- The output tensor $O$ with dimensions: $K$, $W$, $H$. Each pixel of the output tensor $O[k, x, y]$ is obtained as:

$$O[k, x, y] = \sum^{C-1}_{c=0} \sum^{FH-1}_{j=0} \sum^{FW-1}_{i=0} F[k, c, F W− 1− i, F H− 1− j] \cdot I_0[c, x + i, y + j]$$

Note that we use the transpose of the filter in order to compute a convolution rather than a cross-correlation. 

We implement a simple convolution algorithm in CUDA without tiling and without shared memory. We print the checksum as the total sum of the elements of $O$ along all its dimensions.

Next, we implement the convolution using shared memory and tiling.

Finally we implement the convolution using `cuDNN` with the `CUDNN_CONVOLUTION_FWD_PREFER_FASTEST` algorithm selection.

### Results

The below figure is the output of the convolution CUDA kernel. We see that `cuDNN` implementation is the fastest, followed by the implementation with shared memory and tiling.

![cuda convolution](https://github.com/rugvedmhatre/CUDA-Benchmarking/blob/main/images/c.png?raw=true)


