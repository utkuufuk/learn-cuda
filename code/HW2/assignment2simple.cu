#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, 
		           int numCols,
                   const float* const filter, 
        		   const int filterWidth)
{	
	const int2 threadIndex2D = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                         blockIdx.y * blockDim.y + threadIdx.y);
    const int index = threadIndex2D.y * numCols + threadIndex2D.x;
    
    if (threadIndex2D.x >= numCols || threadIndex2D.y >= numRows)
    {
        return;
    }

    float color = 0.0f;
    
    for (int filterRowIndex = 0; filterRowIndex < filterWidth; filterRowIndex++) 
    {
        for (int filterColIndex = 0; filterColIndex < filterWidth; filterColIndex++) 
        {
            int colIndex = threadIndex2D.x + filterColIndex - filterWidth/2;
            int rowIndex = threadIndex2D.y + filterRowIndex - filterWidth/2;
            colIndex = min(max(colIndex, 0), numCols - 1);
            rowIndex = min(max(rowIndex, 0), numRows - 1);
            float filter_value = filter[filterRowIndex*filterWidth + filterColIndex];
            color += filter_value * static_cast<float>(inputChannel[rowIndex * numCols + colIndex]);
        }
    }
    outputChannel[index] = color;
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

    //make sure we don't try and access memory outside the image by having any threads mapped there return early
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

    checkCudaErrors(cudaMalloc(&d_filter,  sizeof(float) * filterWidth * filterWidth));
    checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4* const h_inputImageRGBA, 
	            		uchar4* const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, 
               			const size_t numRows, 
			            const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
    const dim3 blockSize(32, 32);
    const dim3 gridSize(numCols / blockSize.x + 1, numRows / blockSize.y + 1);

    separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());

    gaussian_blur<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());

    gaussian_blur<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());

    gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());

    recombineChannels<<<gridSize, blockSize>>>(d_redBlurred, d_greenBlurred, d_blueBlurred, d_outputImageRGBA, numRows, numCols);
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
