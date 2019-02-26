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
int const WIDTH = 1024;
int const LENGTH = 1024;
// Kernel characteristics
int const ZPADX = 0;
int const ZPADY = 0;
int const STRIDEX = 1;
int const STRIDEY = 1;
int const CONV_RECP_SIZEX = 3;
int const CONV_RECP_SIZEY = 3;
int const NUM_OF_KERNELS = 128;
// Convolution output characteristics
int const convLayerSizeX = ((WIDTH - CONV_RECP_SIZEX + 2 * ZPADX) / STRIDEX + 1);
int const convLayerSizeY = ((LENGTH - CONV_RECP_SIZEY + 2 * ZPADY) / STRIDEY + 1);
// transformation matrix characteristics
int const transformSizeY = convLayerSizeY * convLayerSizeX;
int const transformSizeX = CONV_RECP_SIZEX * CONV_RECP_SIZEY * DEPTH;

#define COUT_input if (0) std::cout
#define COUT_result if (0) std::cout

__global__
void transformToMul(float* inputMatrix, float* reducedMatrix)
{

	int Y = blockIdx.y * blockDim.y + threadIdx.y;
	int X = blockIdx.x * blockDim.x + threadIdx.x;

	if (Y < transformSizeY)
	{
		int inputX = (Y % convLayerSizeX) * STRIDEX + X % CONV_RECP_SIZEY;
		int inputY = (Y / convLayerSizeX) * STRIDEY + (X % (CONV_RECP_SIZEX * CONV_RECP_SIZEY)) / CONV_RECP_SIZEX;
		int inputZ = X / (CONV_RECP_SIZEX * CONV_RECP_SIZEY);
		if ((inputX >= ZPADX && inputX <= (ZPADX + WIDTH - 1)) && (inputY >= ZPADY && inputY <= (ZPADY + LENGTH - 1)))
		{

			reducedMatrix[(Y * transformSizeX) + X] = inputMatrix[(inputZ * LENGTH + inputY - ZPADY) * WIDTH + inputX - ZPADX];
		}
		else
		{
			reducedMatrix[(Y * transformSizeX) + X] = 0;
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
	dim3 threadsPerBlock(transformSizeX, transformSizeY);
	dim3 blocksPerGrid(1, 1);
	if (transformSizeY * transformSizeX > 1024) {
		threadsPerBlock.x = transformSizeX;
		threadsPerBlock.y = 1024 / transformSizeX;
		blocksPerGrid.x = ceil(double(transformSizeX) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(transformSizeY) / double(threadsPerBlock.y));
	}

	// Run the kernel function and meassure time
	cudaEventRecord(start, 0);

	transformToMul << < blocksPerGrid, threadsPerBlock >> > (deviceInputMatrix, deviceTransformedInput);
	cudaStatus = cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaEventRecord(stop, 0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "EventRecord failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaEventSynchronize(stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "EventSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
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
	for (int k = 0; k < transformSizeY; k++)
	{
		for (int j = 0; j < transformSizeX; j++)
		{

			COUT_result << std::setprecision(1) << std::fixed << hostTransformedInput[k * transformSizeX + j] << " ";

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
