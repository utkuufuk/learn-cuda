#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#include "utils.h"

const int THREADS_PER_BLOCK = 1024;
const int NUM_EXAMPLES = 100;
const int MIN = 0;
const int MAX = 1;

__global__
void histogram(unsigned int* d_bins, 
               const float* d_outBuffer, 
               const int binCount, 
               const float lumMin, 
               const float lumMax, 
               const int bufferSize) 
{  
    int bufferIndex = threadIdx.x + blockDim.x * blockIdx.x;
    
    // handle out of bounds situations
    if (bufferIndex >= bufferSize)
    {
        return;
    }
    float range = lumMax - lumMin;
    int binIndex = ((d_outBuffer[bufferIndex] - lumMin) / range) * binCount;
    atomicAdd(&d_bins[binIndex], 1);
}

__global__ 
void scan(unsigned int* d_bins, int bufferSize) 
{
    int bufferIndex = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (bufferIndex >= bufferSize)
    {
        return;
    }
    
    for (int s = 1; s <= bufferSize; s *= 2) 
    {
        int spot = bufferIndex - s; 
        unsigned int val = 0;

        if (spot >= 0)
        {
          val = d_bins[spot];
        }
        __syncthreads();

        if (spot >= 0)
        {
          d_bins[bufferIndex] += val;
        }
        __syncthreads();
    }
}

__global__
void reduce(const float* const d_inBuffer, float* d_outBuffer, const size_t bufferSize, int minOrMax) 
{
    extern __shared__ float sharedBuf[];
    
    int blockSize = blockDim.x;
    int blockIndex = blockIdx.x;
    int threadIndex = threadIdx.x; 
    int bufferIndex = threadIndex + blockSize * blockIndex;
    
    // copy this block's portion of the input buffer into the shared buffer
    if (bufferIndex < bufferSize) 
    {
        sharedBuf[threadIndex] = d_inBuffer[bufferIndex];
    } 
    else
    {
        // handle out-of-bounds situations
        if (minOrMax == MIN)
        {
            sharedBuf[threadIndex] = FLT_MAX;
        }
        else
        {
            sharedBuf[threadIndex] = -FLT_MAX;
        }
    }
    __syncthreads();
       
    // min & max reduce
    for (unsigned int s = blockSize / 2; s > 0; s /= 2) 
    {
        if (threadIndex < s) 
        {
            if (minOrMax == MIN) 
            {
                sharedBuf[threadIndex] = min(sharedBuf[threadIndex], sharedBuf[threadIndex + s]);
            } 
            else 
            {
                sharedBuf[threadIndex] = max(sharedBuf[threadIndex], sharedBuf[threadIndex + s]);
            }
        }
        __syncthreads();
    }
    
    // write the min/max value of this block's portion into the output buffer
    if (threadIndex == 0) 
    {
        d_outBuffer[blockIndex] = sharedBuf[0];
    }
}

static int calculateNumBlocks(int bufferSize) 
{
    return (int) ceil((float) bufferSize / (float) THREADS_PER_BLOCK) + 1;
}

static float reduceMinMax(const float* const d_outBuffer, const size_t bufferSize, int minOrMax) 
{
    size_t currentBufferSize = bufferSize;
    float* d_inTemp;
    float* d_outTemp;
    
    // copy the input buffer to a temporary buffer
    size_t bufferMemSize = sizeof(float) * bufferSize; 
    checkCudaErrors(cudaMalloc(&d_inTemp, bufferMemSize));    
    checkCudaErrors(cudaMemcpy(d_inTemp, d_outBuffer, bufferMemSize, cudaMemcpyDeviceToDevice));
    
    size_t sharedMemorySize = sizeof(float) * THREADS_PER_BLOCK;
    
    // keep reducing until the entire thing fits into a single block
    while (1) 
    {
        int numBlocks = calculateNumBlocks(currentBufferSize);
        checkCudaErrors(cudaMalloc(&d_outTemp, sizeof(float) * numBlocks));
        reduce<<<numBlocks, THREADS_PER_BLOCK, sharedMemorySize>>>
              (d_inTemp, d_outTemp, currentBufferSize, minOrMax);
        cudaDeviceSynchronize(); 
        checkCudaErrors(cudaGetLastError());
            
        // move the current input to the output, and clear the last input if necessary
        checkCudaErrors(cudaFree(d_inTemp));
        d_inTemp = d_outTemp;
        
        if (currentBufferSize < THREADS_PER_BLOCK) 
        {
            break;
        }
        else
        {
            // update current buffer size
            currentBufferSize = calculateNumBlocks(currentBufferSize);
        }
    }
    
    // transfer the min/max value to host memory and return
    float h_result;
    cudaMemcpy(&h_result, d_outTemp, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_outTemp);
    return h_result;
}

void your_histogram_and_prefixsum(const float* const d_logLum,
                                  unsigned int* const d_cdf,
                                  float &minLogLum,
                                  float &maxLogLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
    // REDUCE
    const size_t imageSize = numRows * numCols;
    minLogLum = reduceMinMax(d_logLum, imageSize, 0);
    maxLogLum = reduceMinMax(d_logLum, imageSize, 1);
    
    printf("Min: %f\n", minLogLum);
    printf("Max: %f\n", maxLogLum);
    printf("Number of Bins: %d\n", numBins);
    
    // HISTOGRAM
    unsigned int* d_bins;
    size_t histogramSize = sizeof(unsigned int) * numBins;
    checkCudaErrors(cudaMalloc(&d_bins, histogramSize));    
    checkCudaErrors(cudaMemset(d_bins, 0, histogramSize));
    dim3 numBlocksHistogram(calculateNumBlocks(imageSize));
    histogram<<<numBlocksHistogram, THREADS_PER_BLOCK>>>
                (d_bins, d_logLum, numBins, minLogLum, maxLogLum, imageSize);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());

    // display some examples of histogram
    unsigned int h_outBuffer[NUM_EXAMPLES];
    cudaMemcpy(&h_outBuffer, d_bins, sizeof(unsigned int) * NUM_EXAMPLES, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < NUM_EXAMPLES; i++)
    {
        printf("hist out %d\n", h_outBuffer[i]);
    }

    // SCAN
    dim3 numBlocksScan(calculateNumBlocks(numBins));
    scan<<<numBlocksScan, THREADS_PER_BLOCK>>>(d_bins, numBins);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());
    
    // display some examples of CDF
    cudaMemcpy(&h_outBuffer, d_bins, sizeof(unsigned int) * NUM_EXAMPLES, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < NUM_EXAMPLES; i++)
    {
        printf("cdf out %d\n", h_outBuffer[i]);
    }
    cudaMemcpy(d_cdf, d_bins, histogramSize, cudaMemcpyDeviceToDevice);
    checkCudaErrors(cudaFree(d_bins));
}
