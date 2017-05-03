#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>

const int BLOCK_ROWS = 32;
const int BLOCK_COLS = 32;
const int MAX_THREADS_PER_BLOCK = 1024;

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

__global__
void gaussianBlur(const unsigned char* const inputChannel,
                  unsigned char* const outputChannel,
                  int numRows, 
                  int numCols,
                  int numThreadMatrices,
                  const float* const filter, 
                  const int filterWidth)
{       
    // each 9x9 thread matrix in a thread block will write
    // its output into the corresponding location in this buffer
    extern __shared__ float sharedBuffer[];
     
    // calculate the center pixel location of the thread 
    int centerColIndex = blockIdx.x * numThreadMatrices + threadIdx.x;
    int centerRowIndex = blockIdx.y;
    int centerPixelIndex = centerRowIndex * numCols + centerColIndex;

    // return if the center index is out of bounds
    if (centerColIndex >= numCols || centerRowIndex >= numRows)
    {
        return;
    }

    // let the first thread in each thread matrix initialize the shared buffer
    if (threadIdx.y == 0 && threadIdx.z == 0)
    {
        sharedBuffer[threadIdx.x] = 0;
    }
    __syncthreads();

    // calculate the corresponding filter coefficient index for this thread
    int rowOffset = (-filterWidth / 2) + threadIdx.z; 
    int colOffset = (-filterWidth / 2) + threadIdx.y;
    int filterIndex = threadIdx.z * filterWidth + threadIdx.y;

    // calculate the mapped rows and columns of each thread 
    int threadRowIndex = min(max(centerRowIndex + rowOffset, 0), static_cast<int>(numRows - 1));
    int threadColIndex = min(max(centerColIndex + colOffset, 0), static_cast<int>(numCols - 1));
    int threadPixelIndex = threadRowIndex * numCols + threadColIndex;
    
    // atomically update the weighted sum for the center pixetl
    atomicAdd(&sharedBuffer[threadIdx.x], inputChannel[threadPixelIndex] * filter[filterIndex]);
    __syncthreads();
    
    // let the first thread in each thread matrix write the result to the global output buffer
    if (threadIdx.y == 0 && threadIdx.z == 0)
    {
        outputChannel[centerPixelIndex] = sharedBuffer[threadIdx.x];
    }
}

__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{   
    const int2 threadIndex2D = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                         blockIdx.y * blockDim.y + threadIdx.y);

    const int index = threadIndex2D.y * numCols + threadIndex2D.x;

    // avoid accessing the memory outside the image by having any threads mapped there return early
    if (threadIndex2D.x >= numCols || threadIndex2D.y >= numRows)
    { 
        return;
    }
    uchar4 rgba = inputImageRGBA[index];                
    redChannel[index] = rgba.x;
    greenChannel[index] = rgba.y;
    blueChannel[index] = rgba.z;
}

__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
    const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, 
                                         blockIdx.y * blockDim.y + threadIdx.y);
    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    // avoid accessing the memory outside the image by having any threads mapped there return early
    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    {
        return;
    }

    unsigned char red   = redChannel[thread_1D_pos];
    unsigned char green = greenChannel[thread_1D_pos];
    unsigned char blue  = blueChannel[thread_1D_pos];

    //Alpha should be 255 for no transparency
    uchar4 outputPixel = make_uchar4(red, green, blue, 255);
    outputImageRGBA[thread_1D_pos] = outputPixel;
}

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{
    checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

    size_t filterMemSize = filterWidth * filterWidth * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_filter, filterMemSize));
    checkCudaErrors(cudaMemcpy(d_filter, h_filter, filterMemSize, cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4* const h_inputRGBA, 
                        uchar4* const d_inputRGBA,
                        uchar4* const d_outputRGBA, 
                        const size_t numRows, 
                        const size_t numCols,
                        unsigned char *d_redBlur, 
                        unsigned char *d_greenBlur, 
                        unsigned char *d_blueBlur,
                        const int filterWidth)
{
    // set the thread and block sizes for kernels that seperate and recombine the channels
    const dim3 channelThreads(BLOCK_COLS, BLOCK_ROWS);
    const dim3 channelBlocks(1 + (numCols / channelThreads.x), 1 + (numRows / channelThreads.y));
    
    // set the thread and block sizes for the blurring kernel
    int threadsPerBlurBlock = MAX_THREADS_PER_BLOCK / (filterWidth * filterWidth);
    const dim3 blurThreads(threadsPerBlurBlock, filterWidth, filterWidth);
    const dim3 blurBlocks((numCols / threadsPerBlurBlock) + 1, numRows);

    // print useful information
    printf("Image size: %dx%d\n", numCols, numRows);
    printf("Threads for channel kernels: %dx%d\nBlocks for channel kernels: %dx%d\n",
            channelThreads.x, channelThreads.y, channelBlocks.x, channelBlocks.y);
    printf("Threads for blurring a channel: %dx%dx%d\nBlocks for blurring a channel: %dx%d\n",
            blurThreads.x, blurThreads.y, blurThreads.z, blurBlocks.x, blurBlocks.y);

    // separate the color channels
    separateChannels<<<channelBlocks, channelThreads>>>
                    (d_inputRGBA, numRows, numCols, d_red, d_green, d_blue);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());
    
    // blur the red channel 
    gaussianBlur<<<blurBlocks, blurThreads, threadsPerBlurBlock>>>
                (d_red, d_redBlur, numRows, numCols, threadsPerBlurBlock, d_filter, filterWidth);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());

    // blur the green channel
    gaussianBlur<<<blurBlocks, blurThreads, threadsPerBlurBlock>>>
                (d_green, d_greenBlur, numRows, numCols, threadsPerBlurBlock, d_filter, filterWidth);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());

    // blur the blue channel
    gaussianBlur<<<blurBlocks, blurThreads, threadsPerBlurBlock>>>
                (d_blue, d_blueBlur, numRows, numCols, threadsPerBlurBlock, d_filter, filterWidth);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());
    
    // recombine the blurred channels
    recombineChannels<<<channelBlocks, channelThreads>>>
                     (d_redBlur, d_greenBlur, d_blueBlur, d_outputRGBA, numRows, numCols);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());
}

void cleanup() 
{
    checkCudaErrors(cudaFree(d_red));
    checkCudaErrors(cudaFree(d_green));
    checkCudaErrors(cudaFree(d_blue));
    checkCudaErrors(cudaFree(d_filter));
}
