#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>

const int THREADS = 1024;

__global__
void brickSort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               const size_t numElems,
               bool toggle)
{
    int index = (threadIdx.x + blockDim.x * blockIdx.x) * 2;

    if (toggle)
    {
        if (index + 2 >= numElems)
        {
            return; 
        }

        if (d_inputVals[index + 1] > d_inputVals[index + 2])
        {
            unsigned int tempPos = d_inputPos[index + 1];
            d_inputPos[index + 1] = d_inputPos[index + 2];
            d_inputPos[index + 2] = tempPos;

            unsigned int tempVal = d_inputVals[index + 1];
            d_inputVals[index + 1] = d_inputVals[index + 2];
            d_inputVals[index + 2] = tempVal;
        }
    }
    else
    {
        if (index + 1 >= numElems)
        {
            return; 
        }

        if (d_inputVals[index] > d_inputVals[index + 1])
        {
            unsigned int tempPos = d_inputPos[index];
            d_inputPos[index] = d_inputPos[index + 1];
            d_inputPos[index + 1] = tempPos;

            unsigned int tempVal = d_inputVals[index];
            d_inputVals[index] = d_inputVals[index + 1];
            d_inputVals[index + 1] = tempVal;
        }
   }
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
    const dim3 blocks((numElems / (2 * THREADS)) + 1);
    
    // iterate until the buffers are fully sorted
    for (int i = 0; i < numElems; i++)
    {
        bool toggle = (i % 2) == 1;
        brickSort<<<blocks, THREADS>>>(d_inputVals, d_inputPos, numElems, toggle); 
    }

    // copy the input buffer to the output buffer
    size_t memSize = sizeof(unsigned int) * numElems;
    checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, memSize, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, memSize, cudaMemcpyDeviceToDevice));
}
