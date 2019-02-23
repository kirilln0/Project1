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
int const DEPTH = 1;
int const WIDTH = 4;
int const LENGTH = 4;
// Kernel characteristics
int const ZPADX = 0;
int const ZPADY = 0;
int const STRIDEX = 1;
int const STRIDEY = 1;
int const CONV_RECP_SIZEX = 2;
int const CONV_RECP_SIZEY = 2;
int const NUM_OF_KERNELS = 2;
// Convolution output characteristics
int const convLayerSizeX = ((WIDTH - CONV_RECP_SIZEX + 2 * ZPADX) / STRIDEX + 1);
int const convLayerSizeY = ((LENGTH - CONV_RECP_SIZEY + 2 * ZPADY) / STRIDEY + 1);
// transformation matrix characteristics
int const transformSizeY = convLayerSizeY * convLayerSizeX;
int const transformSizeX = CONV_RECP_SIZEX * CONV_RECP_SIZEY * DEPTH;
int const transformSizeX_nodepth = CONV_RECP_SIZEX * CONV_RECP_SIZEY;
int const CONV_FINAL_SIZE = convLayerSizeX * convLayerSizeY * NUM_OF_KERNELS;

//MASK :2X2
int const MASKX = 3;
int const MASKY = 3;
int const MASKPADX = convLayerSizeX % MASKX;
int const MASKPADY = convLayerSizeY % MASKY;
int const MASK_SIZE = MASKX*MASKY;
int const pseudo_transformSizeY = (convLayerSizeY + MASKPADY) * (convLayerSizeX + MASKPADX);
int const MASKS_IN_Y = convLayerSizeY / (MASKY + MASKPADY);
int const MASKS_IN_X = convLayerSizeX / (MASKX + MASKPADX);

__global__
void Convolution(float* inputMatrix, float* reducedMatrix, float* weights, float* result)
{
	//__shared__ int flag;
	int Y = blockIdx.x  * MASK_SIZE + (threadIdx.x / transformSizeX);
	int X = threadIdx.x % transformSizeX;
	int maskX_offset = Y % MASKX;
	int maskY_offset = (Y % MASK_SIZE) / MASKX;
	if (!(maskX_offset == maskY_offset))
	{
		__syncthreads();
	}
	if ((maskX_offset == maskY_offset) || 0)
	{
		int MaskNum = (Y / MASK_SIZE);
		int convXNew = (MaskNum % MASKS_IN_X) * MASKX + maskX_offset;
		int convYNew = (MaskNum / MASKS_IN_X) * MASKY + maskY_offset;
		int inputX = convXNew * STRIDEX + X % CONV_RECP_SIZEY;
		int inputY = convYNew * STRIDEY + (X % (transformSizeX_nodepth)) / CONV_RECP_SIZEX;
		int inputZ = X / (transformSizeX_nodepth);
		if (convXNew >= convLayerSizeX || convYNew >= convLayerSizeY)
		{
			reducedMatrix[(Y * transformSizeX) + X] = 0;
		}
		else if ((inputX >= ZPADX && inputX <= (ZPADX + WIDTH - 1)) && (inputY >= ZPADY && inputY <= (ZPADY + LENGTH - 1)))
		{

			reducedMatrix[(Y * transformSizeX) + X] = inputMatrix[(inputZ * LENGTH + inputY - ZPADY) * WIDTH + inputX - ZPADX];
		}
		else
		{
			reducedMatrix[(Y * transformSizeX) + X] = 0;
		}
		float reduced_value = reducedMatrix[Y * transformSizeX + X];
		for (int j = 0; j < NUM_OF_KERNELS; j++)
		{
			atomicAdd(result + (j * convLayerSizeX * convLayerSizeY + convYNew * convLayerSizeX + convXNew), reduced_value * weights[j * transformSizeX + X]);
		}
		if (maskX_offset == maskY_offset)
		{
			__syncthreads();
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
			//std::cout << "slice: " << c + 1 << "\n";
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

					//std::cout << std::setprecision(1) << std::fixed << matrix[((b * z + c) * y + j) * x + i] << " , ";
				}
				//std::cout << "\n";
			}
			//std::cout << "\n";
		}
		//std::cout << "\n";
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
	float* hostTransformedInput = new float[pseudo_transformSizeY * transformSizeX]();
	float* hostConvResult = new float[CONV_FINAL_SIZE]();
	float* hostConvLayerWeights = new float[NUM_OF_KERNELS * DEPTH * CONV_RECP_SIZEY * CONV_RECP_SIZEX];

	// GENERATING INPUT
	std::cout << "Inputs:\n";
	generateFlat4DData(hostInputMatrix, WIDTH, LENGTH, DEPTH, BATCH, 1, 0.1);
	generateFlat4DData(hostConvLayerWeights, CONV_RECP_SIZEX, CONV_RECP_SIZEY, DEPTH, NUM_OF_KERNELS, 1, 0.1);

	// Initializing and allocating Device data, kernels and output
	float* deviceInputMatrix;
	float* deviceTransformedInput;
	float* deviceConvLayerWeights;
	float* deviceConvResult;

	cudaStatus = cudaMalloc((void **)&deviceConvLayerWeights, (CONV_RECP_SIZEX * CONV_RECP_SIZEY * DEPTH * NUM_OF_KERNELS) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void **)&deviceInputMatrix, (DEPTH * LENGTH * WIDTH) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void **)&deviceTransformedInput, (pseudo_transformSizeY * transformSizeX) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void **)&deviceConvResult, (CONV_FINAL_SIZE) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(deviceInputMatrix, hostInputMatrix, (DEPTH * LENGTH * WIDTH) * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(deviceConvLayerWeights, hostConvLayerWeights, (CONV_RECP_SIZEX * CONV_RECP_SIZEY * DEPTH * NUM_OF_KERNELS) * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Initializing sizes of grid and block of threads 
	dim3 threadsPerBlock(transformSizeX * MASK_SIZE, 1);
	dim3 blocksPerGrid(ceil(double(pseudo_transformSizeY) / double(MASK_SIZE)) , 1);

	/*
	if (transformSizeY * transformSizeX > 1024) {
		threadsPerBlock.x = transformSizeX;
		threadsPerBlock.y = 1;//1024 / transformSizeX;
		blocksPerGrid.x = ceil(double(transformSizeX) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(transformSizeY) / double(threadsPerBlock.y));
	}
	*/
	// Run the kernel function and meassure time
	cudaEventRecord(start, 0);

	Convolution << < blocksPerGrid, threadsPerBlock >> > (deviceInputMatrix, deviceTransformedInput, deviceConvLayerWeights, deviceConvResult);
	cudaStatus = cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Transform addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
	time = time * 1000;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "DeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}


	// Get the results from device
	cudaStatus = cudaMemcpy(hostTransformedInput, deviceTransformedInput, (transformSizeX * pseudo_transformSizeY) * sizeof(float), cudaMemcpyDeviceToHost); // Not relevant to this program
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(hostConvResult, deviceConvResult, (CONV_FINAL_SIZE) * sizeof(float), cudaMemcpyDeviceToHost); // Not relevant to this program
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	// PRINTING RESULTS
	std::cout << "Transformed matrix:\n";
	for (int k = 0; k < pseudo_transformSizeY; k++)
	{
		for (int j = 0; j < transformSizeX; j++)
		{

			std::cout << std::setprecision(1) << std::fixed << hostTransformedInput[k * transformSizeX + j] << " ";

		}
		std::cout << "\n";
	}
	std::cout << "Convolution result:\n";
	for (int k = 0; k < CONV_FINAL_SIZE; k++)
	{
		if (k % convLayerSizeX == 0)
		{
			printf("\n");
		}
		if (k % (convLayerSizeX * convLayerSizeY) == 0)
		{
			printf("Depth=%d\n", k / (convLayerSizeX * convLayerSizeY));
		}
		std::cout << std::setprecision(1) << std::fixed << hostConvResult[k] << " ";


	}
	printf("\n\n");

	// CLEAN UP
	printf("Time for Convolution: %f us\n", time);
Error:
	cudaFree(deviceInputMatrix);
	cudaFree(deviceTransformedInput);
	cudaFree(deviceConvLayerWeights);

	return 0;
}
