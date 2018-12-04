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
int const WIDTH = 32;
int const LENGTH = 32;
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
int const CONV_FINAL_SIZE = convLayerSizeX * convLayerSizeY * NUM_OF_KERNELS;

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

__global__
void rowMul(float* A, float* B, float* C)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int N = X % transformSizeY;
	int n = X / transformSizeY;
	float sum = 0;
	if (X < CONV_FINAL_SIZE)
	{
		for (int i = 0; i < transformSizeX; i++)
		{
			sum += A[n * transformSizeX + i] * B[N * transformSizeX + i];

		}
		C[X] = sum;
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
	cudaEvent_t start1, stop1, start2, stop2;
	float time1,time2;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaError_t cudaStatus;

	// Initialize Host data, kernel and output
	float* hostInputMatrix = new float[BATCH * DEPTH * LENGTH * WIDTH];
	float* hostTransformedInput = new float[transformSizeY * transformSizeX]();
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
	cudaStatus = cudaMalloc((void **)&deviceTransformedInput, (transformSizeY * transformSizeX) * sizeof(float));
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
	dim3 threadsPerBlock(transformSizeX, transformSizeY);
	dim3 blocksPerGrid(1, 1);
	if (transformSizeY * transformSizeX > 1024) {
		threadsPerBlock.x = transformSizeX;
		threadsPerBlock.y = 1024 / transformSizeX;
		blocksPerGrid.x = ceil(double(transformSizeX) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(transformSizeY) / double(threadsPerBlock.y));
	}

	// Run the kernel function and meassure time
	cudaEventRecord(start1, 0);

	transformToMul << < blocksPerGrid, threadsPerBlock >> > (deviceInputMatrix, deviceTransformedInput);
	cudaStatus = cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Transform addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaEventRecord(stop1, 0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "EventRecord failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaEventSynchronize(stop1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "EventSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaEventElapsedTime(&time1, start1, stop1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "ElapsedTime failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	time1 = time1 * 1000;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "DeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	dim3 threadsPerBlockMul(CONV_FINAL_SIZE);
	dim3 blocksPerGridMul(1);
	if (CONV_FINAL_SIZE > 1024) {
		threadsPerBlockMul.x = 1024;
		blocksPerGridMul.x = ceil(double(CONV_FINAL_SIZE) / double(threadsPerBlock.x));

	}

	// Run the kernel function and meassure time
	cudaEventRecord(start2, 0);

	rowMul << < blocksPerGridMul, threadsPerBlockMul >> > (deviceConvLayerWeights, deviceTransformedInput, deviceConvResult);
	cudaStatus = cudaEventRecord(stop2, 0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "EventRecord failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Mul addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaEventSynchronize(stop2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "EventSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaEventElapsedTime(&time2, start2, stop2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "ElapsedTime failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	time2 = time2 * 1000;
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
	cudaStatus = cudaMemcpy(hostConvResult, deviceConvResult, (CONV_FINAL_SIZE) * sizeof(float), cudaMemcpyDeviceToHost); // Not relevant to this program
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	// PRINTING RESULTS
	std::cout << "Transformed matrix:\n";
	for (int k = 0; k < transformSizeY; k++)
	{
		//for (int j = 0; j < transformSizeX; j++)
		//{

			//std::cout << std::setprecision(1) << std::fixed << hostTransformedInput[k * transformSizeX + j] << " ";

		//}
		//std::cout << "\n";
	}
	std::cout << "Convolution result:\n";
	for (int k = 0; k < CONV_FINAL_SIZE; k++)
	{
		if (k % convLayerSizeX == 0)
		{
			//printf("\n");
		}
		if (k % (convLayerSizeX * convLayerSizeY) == 0)
		{
			//printf("Depth=%d\n", k / (convLayerSizeX * convLayerSizeY));
		}
		//std::cout << std::setprecision(1) << std::fixed << hostConvResult[k] << " ";


	}
	printf( "\n\n");

	// CLEAN UP
	printf("Time for the kernel transform: %f us\n", time1);
	printf("Time for the kernel mul: %f us\n", time2);
Error:
	cudaFree(deviceInputMatrix);
	cudaFree(deviceTransformedInput);
	cudaFree(deviceConvLayerWeights);
	
	return 0;
}
