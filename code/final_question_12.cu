#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "compare.h"
#include "gputimer.h"

__global__ void reduce(unsigned int *d_out_shared, const unsigned int *d_in)
{
    extern __shared__ unsigned int s[];
    int t = threadIdx.x;
    
    if (t % 2 == 0)
    {
       s[t] = d_in[t] + d_in[t + 1];
    }

    if (t % 4 == 0)
    {
       s[t] = s[t] + s[t + 2];
    }

    if (t % 8 == 0)
    {
       s[t] = s[t] + s[t + 4];
    }

    if (t % 16 == 0)
    {
       s[t] = s[t] + s[t + 8];
    }

    if (t == 0)
    {
       *d_out_shared = s[0] + s[16];
    }
}

int main(int argc, char **argv)
{
    const int ARRAY_SIZE = 32;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(unsigned int);

    // generate the input array on the host
    unsigned int h_in[ARRAY_SIZE];
    unsigned int sum = 0;

    // generate random float in [0, 1]
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        h_in[i] = (float) random() / (float) RAND_MAX > 0.5f ? 1 : 0;
        sum += h_in[i];
    }

    // declare GPU memory pointers
    unsigned int * d_in, * d_out_shared;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out_shared, sizeof(unsigned int));

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 

    // launch the kernel
    GpuTimer timer;
    timer.Start();
    reduce<<<1, ARRAY_SIZE, ARRAY_SIZE * sizeof(unsigned int)>>>(d_out_shared, d_in);
    timer.Stop();

    printf("Your code executed in %g ms\n", timer.Elapsed());

    // copy back the sum from GPU
    unsigned int h_out_shared;
    cudaMemcpy(&h_out_shared, d_out_shared, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // compare your resulst against the sum
    compare(h_out_shared, sum);

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out_shared);
}
