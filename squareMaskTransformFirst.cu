#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include <stdio.h>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

// Input size
int const BATCH = 1; //Must be 1 in this program
int const DEPTH = 3;
int const WIDTH = 2048;
int const LENGTH = 2048;
// Kernel characteristics
int const ZPADX = 0;
int const ZPADY = 0;
int const STRIDEX = 1;
int const STRIDEY = 1;
int const CONV_RECP_SIZEX = 3;
int const CONV_RECP_SIZEY = 3;
int const NUM_OF_KERNELS = 1;
// Convolution output characteristics
int const convLayerSizeX = ((WIDTH - CONV_RECP_SIZEX + 2 * ZPADX) / STRIDEX + 1);
int const convLayerSizeY = ((LENGTH - CONV_RECP_SIZEY + 2 * ZPADY) / STRIDEY + 1);
// transformation matrix characteristics
int const transformSizeY = convLayerSizeY * convLayerSizeX;
int const transformSizeX = CONV_RECP_SIZEX * CONV_RECP_SIZEY * DEPTH;
int const KERNEL_2D_SIZE = CONV_RECP_SIZEX * CONV_RECP_SIZEY;
int const MAT_SIZE_ONE_CHANNEL = transformSizeY * CONV_RECP_SIZEX * CONV_RECP_SIZEY;
int const NUM_ELEMENTS = transformSizeY * transformSizeX;
int const KERNEL_LIMIT = transformSizeY * DEPTH;

#define COUT_input if (0) std::cout
#define COUT_result if (0) std::cout

__global__
void transformToMul(float* inputMatrix, float* reducedMatrix)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < KERNEL_LIMIT)
	{
		unsigned int w_out = idx % convLayerSizeX;
		unsigned int indx = idx / convLayerSizeX;
		unsigned int h_out = indx % convLayerSizeY;
		unsigned int h_in = h_out * STRIDEY - ZPADY;
		unsigned int w_in = w_out * STRIDEX - ZPADX;
		if (((h_out & 1) == 0 && (w_out & 1) == 0) || ((h_out & 1) == 1 && (w_out & 1) == 1))
		{
			if ((h_out & 1) == 0)
			{
				w_out = (w_out >> 1);
			}
			else
			{
				w_out = (w_out >> 1) + (convLayerSizeX >> 1);
			}
			h_out = (h_out >> 1);
			
			unsigned int channel_in = indx / convLayerSizeY;
			unsigned int channel_out = channel_in * KERNEL_2D_SIZE;
			reducedMatrix += (channel_out * convLayerSizeY + h_out) * convLayerSizeX + w_out; // here
			inputMatrix += (channel_in * LENGTH + h_in) * WIDTH + w_in;
#pragma unroll
			for (int i = 0; i < CONV_RECP_SIZEY; ++i)
			{
				for (int j = 0; j < CONV_RECP_SIZEX; ++j)
				{
					unsigned int h = h_in + i;
					unsigned int w = w_in + j;
					*reducedMatrix = (h >= 0 && w >= 0 && h < LENGTH && w < WIDTH) ?
						inputMatrix[i * WIDTH + j] : 0;
					reducedMatrix += transformSizeY;
				}
			}
		}
	}
}

void generateFlat4DData(float* matrix, int x, int y, int z, int d, double type, double jump)
{
	double w = jump;
	for (int b = 0; b < d; b++)
	{
		for (int c = 0; c < z; c++)
		{
			COUT_input << "slice: " << c + 1 << "\n";
			for (int j = 0; j < y; j++)
			{
				for (int i = 0; i < x; i++)
				{
					if (type == -1)
					{
						matrix[((b * z + c) * y + j) * x + i] = rand() % 10;
					}
					else if (type == 0)
					{
						matrix[((b * z + c) * y + j) * x + i] = jump;
					}
					else
					{
						matrix[((b * z + c) * y + j) * x + i] = w;
						w += jump;
					}

					COUT_input << std::setprecision(1) << std::fixed << matrix[((b * z + c) * y + j) * x + i] << " , ";
				}
				COUT_input << "\n";
			}
			COUT_input << "\n";
		}
		COUT_input << "\n";
	}
}

int main()
{

	// Performance test variables
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaError_t cudaStatus;

	// Initialize Host data, kernel and output
	float* hostInputMatrix = new float[BATCH * DEPTH * LENGTH * WIDTH];
	float* hostTransformedInput = new float[transformSizeY * transformSizeX]();

	// GENERATING INPUT
	COUT_input << "Inputs:\n";
	generateFlat4DData(hostInputMatrix, WIDTH, LENGTH, DEPTH, BATCH, 1, 0.1);

	// Initializing and allocating Device data, kernels and output
	float* deviceInputMatrix;
	float* deviceTransformedInput;

	cudaStatus = cudaMalloc((void **)&deviceInputMatrix, (DEPTH * LENGTH * WIDTH) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void **)&deviceTransformedInput, (transformSizeY * transformSizeX) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(deviceInputMatrix, hostInputMatrix, (DEPTH * LENGTH * WIDTH) * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Initializing sizes of grid and block of threads 
	dim3 threadsPerBlock(1024);
	dim3 blocksPerGrid(1);
	if (transformSizeY * DEPTH > 1024) {
		threadsPerBlock.x = 1024;
		blocksPerGrid.x = ceil(double(transformSizeY * DEPTH) / double(threadsPerBlock.x));
	}

	// Run the kernel function and meassure time
	cudaEventRecord(start, 0);

	transformToMul << < blocksPerGrid, threadsPerBlock >> > (deviceInputMatrix, deviceTransformedInput);

	cudaStatus = cudaEventRecord(stop, NULL);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "EventRecord failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaEventSynchronize(stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "EventSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaEventElapsedTime(&time, start, stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "ElapsedTime failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "DeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Get the results from device
	cudaStatus = cudaMemcpy(hostTransformedInput, deviceTransformedInput, (transformSizeX * transformSizeY) * sizeof(float), cudaMemcpyDeviceToHost); // Not relevant to this program
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// PRINTING RESULTS
	COUT_result << "Transformed matrix:\n";
	for (int k = 0; k < transformSizeX; k++)
	{
		for (int j = 0; j < transformSizeY; j++)
		{

			COUT_result << std::setprecision(1) << std::fixed << hostTransformedInput[k * transformSizeY + j] << " ";

		}
		COUT_result << "\n";
	}

	// CLEAN UP
	printf("Transform time: %f msec.\n", time);
Error:
	cudaFree(deviceInputMatrix);
	cudaFree(deviceTransformedInput);

	return 0;
}
