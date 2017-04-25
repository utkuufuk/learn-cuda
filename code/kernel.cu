#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void cube(float * d_out, float * d_in)
{
	int index = threadýdx.x;
	float value = d_in[index];
	d_out[index] = value * value * value;
}

int main(int argc, char ** argv)
{
	const int number_of_blocks = 1;
	const int array_sýze = 1024;
	const int array_bytes = array_sýze * sizeof(float);

	// generate the input array on the host
	float h_in[array_sýze];
	for (int i = 0; i < array_sýze; i++)
	{
		h_in[i] = float(i);
	}
	float h_out[array_sýze];

	// declare gpu memory pointers
	float * d_in;
	float * d_out;

	// allocate gpu memory
	cudamalloc((void**)&d_in, array_bytes);
	cudamalloc((void**)&d_out, array_bytes);

	// transfer the array to the gpu
	cudamemcpy(d_in, h_in, array_bytes, cudamemcpyhosttodevice);

	// launch the kernel with 64 threads in 1 block.
	cube <<<number_of_blocks, array_sýze >>>(d_out, d_in);

	// copy back the result array to the cpu
	cudamemcpy(h_out, d_out, array_bytes, cudamemcpydevicetohost);

	// print out the resulting array
	for (int i = 0; i < array_sýze; i++)
	{
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}

	cudafree(d_in);
	cudafree(d_out);

	getchar();
	return 0;
}