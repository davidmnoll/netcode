#include <stdio.h>

// GPU kernel function
__global__ void helloFromGPU()
{
    printf("Hello World from GPU!\n");
}

extern "C" void launch_hello_kernel()
{
    // Launch GPU kernel with 1 block containing 10 threads
    helloFromGPU<<<1, 10>>>();
    cudaDeviceSynchronize();
}
