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
int const WIDTH = 3*3*3;
int const LENGTHA = 3;
int const LENGTHB = 126*126*3;
// transformation matrix characteristics
int const OUTPUT_SIZEY = LENGTHA * LENGTHB;


__global__
void rowMul(float* A, float* B, float* C)
{
	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int N = X % LENGTHB;
	int n = X / LENGTHB;
	float sum = 0;
	if (X < OUTPUT_SIZEY)
	{
		for (int i = 0; i < WIDTH; i++)
		{
			sum += A[n * WIDTH + i] * B[N * WIDTH + i];
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
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaError_t cudaStatus;

	// Initialize Host data, kernel and output
	float* hostInputMatrixA = new float[LENGTHA * WIDTH];
	float* hostInputMatrixB = new float[LENGTHB * WIDTH];
	float* hostResult = new float[OUTPUT_SIZEY]();

	// GENERATING INPUT
	std::cout << "Inputs:\n";
	generateFlat4DData(hostInputMatrixA, WIDTH, LENGTHA, 1, 1, 1, 0.1);
	generateFlat4DData(hostInputMatrixB, WIDTH, LENGTHB, 1, 1, 1, 0.1);

	// Initializing and allocating Device data, kernels and output
	float* deviceInputMatrixA;
	float* deviceInputMatrixB;
	float* deviceResult;

	cudaStatus = cudaMalloc((void **)&deviceInputMatrixA, (LENGTHA * WIDTH) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void **)&deviceInputMatrixB, (LENGTHB * WIDTH) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void **)&deviceResult, (OUTPUT_SIZEY) * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(deviceInputMatrixA, hostInputMatrixA, (LENGTHA * WIDTH) * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(deviceInputMatrixB, hostInputMatrixB, (LENGTHA * WIDTH) * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Initializing sizes of grid and block of threads 
	dim3 threadsPerBlock(OUTPUT_SIZEY);
	dim3 blocksPerGrid(1);
	if (OUTPUT_SIZEY > 1024) {
		threadsPerBlock.x = 1024;
		blocksPerGrid.x = ceil(double(OUTPUT_SIZEY) / double(threadsPerBlock.x));
	}

	// Run the kernel function and meassure time
	cudaEventRecord(start, 0);

	rowMul << < blocksPerGrid, threadsPerBlock >> > (deviceInputMatrixA, deviceInputMatrixB, deviceResult);
	cudaStatus = cudaEventRecord(stop, 0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "EventRecord failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
	cudaStatus = cudaMemcpy(hostResult, deviceResult, (OUTPUT_SIZEY) * sizeof(float), cudaMemcpyDeviceToHost); // Not relevant to this program
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	// PRINTING RESULTS
	std::cout << "Transformed matrix:\n";
	for (int k = 0; k < OUTPUT_SIZEY; k++)
	{
			//std::cout << std::setprecision(1) << std::fixed << hostResult[k] << " , ";
	}
	printf("\n");
	// CLEAN UP
	printf("Time for the kernel: %f us\n", time);
Error:
	cudaFree(deviceInputMatrixA);
	cudaFree(deviceInputMatrixB);
	cudaFree(deviceResult);

	return 0;
}
