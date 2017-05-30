#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "compare.h"
#include "gputimer.h"

__global__ void smooth(float * v_new, const float * v) 
{
    int myIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * gridDim.x;
    int myLeftIdx = (myIdx == 0) ? 0 : myIdx - 1;
    int myRightIdx = (myIdx == (numThreads - 1)) ? numThreads - 1 : myIdx + 1;
    float myElt = v[myIdx];
    float myLeftElt = v[myLeftIdx];
    float myRightElt = v[myRightIdx];
    v_new[myIdx] = 0.25f * myLeftElt + 0.5f * myElt + 0.25f * myRightElt;
}

__global__ void smooth_shared(float * v_new, const float * v) 
{
    extern __shared__ float s[];
    
    int myIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int numElements = blockDim.x * gridDim.x;

    s[threadIdx.x + 1] = v[myIdx];

    if (myIdx == 0)
    {
        s[0] = v[0];
    }
    else if (threadIdx.x == 0)
    {
        s[0] = v[myIdx - 1];
    }

    if (myIdx == numElements - 1)
    {
        s[257] = v[numElements - 1];
    }
    else if (threadIdx.x == blockDim.x - 1)
    {
        s[257] = v[myIdx + 1];
    }
    __syncthreads();

    float myElt = s[threadIdx.x + 1];
    float myLeftElt = s[threadIdx.x];
    float myRightElt = s[threadIdx.x + 2];
    v_new[myIdx] = 0.25f * myLeftElt + 0.5f * myElt + 0.25f * myRightElt;
}

int main(int argc, char **argv)
{
    const int ARRAY_SIZE = 4096;
    const int BLOCK_SIZE = 256;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
    const int BLOCK_BYTES = (BLOCK_SIZE + 2) * sizeof(float);

    // generate the input array on the host
    float h_in[ARRAY_SIZE];
    float h_cmp[ARRAY_SIZE];
    float h_out[ARRAY_SIZE];
    float h_out_shared[ARRAY_SIZE];
    
    for (int i = 0; i < ARRAY_SIZE; i++) 
    {
        // generate random float in [0, 1]
        h_in[i] = (float) random() / (float) RAND_MAX;
    }
    
    for (int i = 0; i < ARRAY_SIZE; i++) 
    {
        h_cmp[i] = (0.25f * h_in[(i == 0) ? 0 : i - 1] + 0.50f * h_in[i] +
                    0.25f * h_in[(i == (ARRAY_SIZE - 1)) ? ARRAY_SIZE - 1 : i + 1]);
    }

    // declare GPU memory pointers
    float * d_in, * d_out, * d_out_shared;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);
    cudaMalloc((void **) &d_out_shared, ARRAY_BYTES);

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 

    // launch the reference kernel
    GpuTimer timer;
    timer.Start();
    smooth<<<ARRAY_SIZE / BLOCK_SIZE, BLOCK_SIZE>>>(d_out, d_in);
    timer.Stop();

    printf("Reference code executed in %g ms\n", timer.Elapsed());  

    // launch the student kernel
    timer.Start();
    //smooth<<<ARRAY_SIZE / BLOCK_SIZE, BLOCK_SIZE>>>(d_out_shared, d_in);
    smooth_shared<<<ARRAY_SIZE / BLOCK_SIZE, BLOCK_SIZE, BLOCK_BYTES>>>(d_out_shared, d_in);
    timer.Stop();

    printf("Your code executed in %g ms\n", timer.Elapsed());  

    // copy back the result from GPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_shared, d_out_shared, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // testing for correctness
    compare(h_in, h_out, h_out_shared, h_cmp, ARRAY_SIZE);

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out_shared);
}
