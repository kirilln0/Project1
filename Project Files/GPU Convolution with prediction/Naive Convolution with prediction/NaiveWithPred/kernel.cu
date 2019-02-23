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
int const WIDTH = 128;
int const LENGTH = 128;
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
int const transformSizeX_nodepth = CONV_RECP_SIZEX * CONV_RECP_SIZEY;
int const CONV_FINAL_SIZE = convLayerSizeX * convLayerSizeY * NUM_OF_KERNELS;

//MASK :2X2
int const MASKX = 3;
int const MASKY = 3;
int const MASKPADX = convLayerSizeX % MASKX;
int const MASKPADY = convLayerSizeY % MASKY;
int const MASK_SIZE = MASKX * MASKY;
int const pseudo_transformSizeY = (convLayerSizeY + MASKPADY) * (convLayerSizeX + MASKPADX);
int const MASKS_IN_Y = convLayerSizeY / (MASKY + MASKPADY);
int const MASKS_IN_X = convLayerSizeX / (MASKX + MASKPADX);

#define COUT_input if (1) std::cout
#define COUT_result if (1) std::cout

__global__
void Convolution(float* inputMatrix, float* weights, float* result)
{
	int x1 = 0; //(blockIdx.x * MASK_SIZE) % (convLayerSizeX);
	int y1 = 0;// ((blockIdx.x * MASK_SIZE) / (convLayerSizeX)) % convLayerSizeY;
	int z1 = 0;//(blockIdx.x * MASK_SIZE) / transformSizeY;
	int x2 = 1;//(blockIdx.x * MASK_SIZE + 4) % (convLayerSizeX);
	int y2 = 1;//((blockIdx.x * MASK_SIZE + 4) / (convLayerSizeX)) % convLayerSizeY;
	int z2 = 1;//(blockIdx.x * MASK_SIZE + 4) / transformSizeY;
	int x3 = 2;//(blockIdx.x * MASK_SIZE + 8) % (convLayerSizeX);
	int y3 = 2;//((blockIdx.x * MASK_SIZE + 8) / (convLayerSizeX)) % convLayerSizeY;
	int z3 = 2;//(blockIdx.x * MASK_SIZE + 8) / transformSizeY;

	int X = (blockIdx.x * MASK_SIZE + threadIdx.x) % (convLayerSizeX);
	int Y = ((blockIdx.x * MASK_SIZE + threadIdx.x) / (convLayerSizeX)) % convLayerSizeY;
	int Z = (blockIdx.x * MASK_SIZE + threadIdx.x) / transformSizeY;
	int maskX_offset = X % MASKX;
	int maskY_offset = Y % MASKY;

	if (maskX_offset == maskY_offset)
	{
		for (int i = 0; i < DEPTH; i++)
		{
			for (int j = 0; j < CONV_RECP_SIZEY; j++)
			{
				for (int l = 0; l < CONV_RECP_SIZEX; l++)
				{
					result[Z * transformSizeY + Y * convLayerSizeX + X] += inputMatrix[i * WIDTH * LENGTH + (j + Y * STRIDEY) * WIDTH + (l + X * STRIDEX)] * weights[Z * transformSizeX + i * transformSizeX_nodepth + j * CONV_RECP_SIZEX + l];
				}
			}
		}
	}
	__syncthreads();
	if (!(maskX_offset == maskY_offset) &&
		((int)result[z1 * transformSizeY + y1 * convLayerSizeX + x1] != 0 ||
		(int)result[z2 * transformSizeY + y2 * convLayerSizeX + x2] != 0 ||
		(int)result[z3 * transformSizeY + y3 * convLayerSizeX + x3] != 0)&& 0)
		
	{
		for (int i = 0; i < DEPTH; i++)
		{
			for (int j = 0; j < CONV_RECP_SIZEY; j++)
			{
				for (int l = 0; l < CONV_RECP_SIZEX; l++)
				{
					result[Z * transformSizeY + Y * convLayerSizeX + X] += inputMatrix[i * WIDTH * LENGTH + (j + Y * STRIDEY) * WIDTH + (l + X * STRIDEX)] * weights[Z * transformSizeX + i * transformSizeX_nodepth + j * CONV_RECP_SIZEX + l];
				}
			}
		}
	}
	
	/*
	int MaskNum = (X / MASKX) + (Y / MASKY) * MASKS_IN_X;
	int masksX = (MaskNum % MASKS_IN_X) * MASKX;
	int masksY = (MaskNum / MASKS_IN_X) * MASKY;
	int convXNew = masksX + maskX_offset;
	int convYNew = masksY + maskY_offset;
	int inputX = convXNew * STRIDEX + X % CONV_RECP_SIZEY;
	int inputY = convYNew * STRIDEY + (X % (transformSizeX_nodepth)) / CONV_RECP_SIZEX;
	int inputZ = X / (transformSizeX_nodepth);
	result[Z * transformSizeY + Y * convLayerSizeX + X] = 1;
	*/
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
	float* hostConvResult = new float[CONV_FINAL_SIZE]();
	float* hostConvLayerWeights = new float[NUM_OF_KERNELS * DEPTH * CONV_RECP_SIZEY * CONV_RECP_SIZEX];

	// GENERATING INPUT
	std::cout << "Inputs:\n";
	generateFlat4DData(hostInputMatrix, WIDTH, LENGTH, DEPTH, BATCH, 1, 0.1);
	generateFlat4DData(hostConvLayerWeights, CONV_RECP_SIZEX, CONV_RECP_SIZEY, DEPTH, NUM_OF_KERNELS, 1, 0.1);

	// Initializing and allocating Device data, kernels and output
	float* deviceInputMatrix;
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
	dim3 threadsPerBlock(MASK_SIZE, 1);
	dim3 blocksPerGrid(ceil(double(CONV_FINAL_SIZE) / double(MASK_SIZE)), 1);

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

	Convolution << < blocksPerGrid, threadsPerBlock >> > (deviceInputMatrix, deviceConvLayerWeights, deviceConvResult);
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
	cudaStatus = cudaMemcpy(hostConvResult, deviceConvResult, (CONV_FINAL_SIZE) * sizeof(float), cudaMemcpyDeviceToHost); // Not relevant to this program
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	// PRINTING RESULTS
	COUT_result << "Convolution result:\n";
	for (int k = 0; k < CONV_FINAL_SIZE; k++)
	{
		if (k % convLayerSizeX == 0)
		{
			COUT_result << "\n";
		}
		if (k % (convLayerSizeX * convLayerSizeY) == 0)
		{
			COUT_result << "Depth = " << k / (convLayerSizeX * convLayerSizeY) << "\n";
		}
		COUT_result << std::setprecision(1) << std::fixed << hostConvResult[k] << " ";


	}
	COUT_result << "\n\n";

	// CLEAN UP
	printf("Time for Convolution: %f us\n", time);
Error:
	cudaFree(deviceInputMatrix);
	cudaFree(deviceConvLayerWeights);

	return 0;
}
