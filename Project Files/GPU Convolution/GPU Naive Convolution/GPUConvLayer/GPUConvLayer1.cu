
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
// Input size
int const BATCH = 1;
int const DEPTH = 2;
int const WIDTH = 3;
int const LENGTH = 3;
// Kernel characteristics
int const ZPADX = 0;
int const ZPADY = 0;
int const STRIDEX = 1;
int const STRIDEY = 1;
int const CONV_RECP_SIZEX = 2;
int const CONV_RECP_SIZEY = 2;
int const NUM_OF_KERNELS = 2;
int const convLayerSizeX = ((WIDTH - CONV_RECP_SIZEX + 2 * ZPADX) / STRIDEX + 1);
int const convLayerSizeY = ((LENGTH - CONV_RECP_SIZEY + 2 * ZPADY) / STRIDEY + 1);
#define COUT_weights if (1) std::cout
#define COUT_input if (1) std::cout
#define COUT_result if (1) std::cout


__global__
void conv(float* inputm, float* weights, float* outputm)
{
	int ROW = (blockIdx.y * blockDim.y + threadIdx.y) % (convLayerSizeX * convLayerSizeY);
	int COL = blockIdx.x * blockDim.x + threadIdx.x;
	int DEP = (blockIdx.y * blockDim.y + threadIdx.y) / (convLayerSizeX * convLayerSizeY);
	for (int i = 0; i < DEPTH; i++)
	{
		for (int j = 0; j < CONV_RECP_SIZEY; j++)
		{
			for (int l = 0; l < CONV_RECP_SIZEX; l++)
			{
				outputm[DEP * convLayerSizeY * convLayerSizeX + ROW * convLayerSizeX + COL] += inputm[i * WIDTH * LENGTH + (j + ROW * STRIDEY) * WIDTH + (l + COL * STRIDEX)] * weights[DEP * DEPTH * CONV_RECP_SIZEX * CONV_RECP_SIZEY + i * CONV_RECP_SIZEX * CONV_RECP_SIZEY + j * CONV_RECP_SIZEX + l];
			}
		}
	}

}

int main()
{

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaError_t cudaStatus;

	float* hinputMatrix = new float[BATCH * DEPTH * LENGTH * WIDTH];
	float* hconvLayer = new float[NUM_OF_KERNELS * convLayerSizeY * convLayerSizeX];
	float* hconvLayerWeights = new float[NUM_OF_KERNELS * DEPTH * CONV_RECP_SIZEY * CONV_RECP_SIZEX];
	//GENERATING INPUT
	COUT_input << "Inputs:\n";
	float x = 0;
	for (int b = 0; b < BATCH; b++)
	{
		for (int c = 0; c < DEPTH; c++)
		{
			COUT_input << "slice: " << c + 1 << "\n";
			for (int j = 0; j < LENGTH; j++)
			{
				for (int i = 0; i < WIDTH; i++)
				{
					hinputMatrix[b * DEPTH * LENGTH *WIDTH + c * LENGTH * WIDTH + j * WIDTH + i] = x;
					x += 0.5;
					COUT_input << std::setprecision(1) << std::fixed << hinputMatrix[b * DEPTH * LENGTH *WIDTH + c * LENGTH * WIDTH + j * WIDTH + i]<<" ";//<< " ("<< b * DEPTH * LENGTH *WIDTH + c * LENGTH * WIDTH + j * WIDTH + i<<") ";
				}
				COUT_input << "\n";
			}
			COUT_input << "\n";
		}
		COUT_input << "\n";
	}
	COUT_input << "Weights:\n";
	float w = 0;
	for (int d = 0; d < NUM_OF_KERNELS; d++)
	{
		COUT_weights << "kernel: " << d + 1 << "\n";
		for (int c = 0; c < DEPTH; c++)
		{
			COUT_weights << "slice: " << c + 1 << "\n";
			for (int j = 0; j < CONV_RECP_SIZEY; j++)
			{
				for (int i = 0; i < CONV_RECP_SIZEX; i++)
				{
					hconvLayerWeights[d * DEPTH * CONV_RECP_SIZEX * CONV_RECP_SIZEY + c * CONV_RECP_SIZEX * CONV_RECP_SIZEY + j * CONV_RECP_SIZEX + i] = w;
					w += 0.1;
					COUT_weights << std::setprecision(1) << std::fixed << hconvLayerWeights[d * DEPTH * CONV_RECP_SIZEX * CONV_RECP_SIZEY + c * CONV_RECP_SIZEX * CONV_RECP_SIZEY + j * CONV_RECP_SIZEX + i] << " ";// " (" << d * DEPTH * CONV_RECP_SIZEX * CONV_RECP_SIZEY + c * CONV_RECP_SIZEX * CONV_RECP_SIZEY + j * CONV_RECP_SIZEX + i << ") ";
				}
				COUT_weights << "\n";
			}
			COUT_weights << "\n";
		}
		COUT_weights << "\n";
	}
	//
	float* dinputMatrix;
	float* dconvLayerWeights;
	float* dconvLayer;

	cudaMalloc((void **)&dconvLayer, (NUM_OF_KERNELS * convLayerSizeY * convLayerSizeX) * sizeof(float));
	cudaMalloc((void **)&dconvLayerWeights, (NUM_OF_KERNELS * DEPTH * CONV_RECP_SIZEY * CONV_RECP_SIZEX) * sizeof(float));
	cudaMalloc((void **)&dinputMatrix, (DEPTH * LENGTH * WIDTH) * sizeof(float));


	cudaMemcpy(dinputMatrix, hinputMatrix, (DEPTH * LENGTH * WIDTH) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dconvLayerWeights, hconvLayerWeights, (NUM_OF_KERNELS * DEPTH * CONV_RECP_SIZEY * CONV_RECP_SIZEX) * sizeof(float), cudaMemcpyHostToDevice);


	dim3 threadsPerBlock(convLayerSizeX, convLayerSizeY *  NUM_OF_KERNELS);
	dim3 blocksPerGrid(1, 1);
	if (NUM_OF_KERNELS * convLayerSizeY * convLayerSizeX > 1024) {
		threadsPerBlock.x = 32;
		threadsPerBlock.y = 32;
		blocksPerGrid.x = ceil(double(convLayerSizeX) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(convLayerSizeY *  NUM_OF_KERNELS) / double(threadsPerBlock.y));
	}
	cudaEventRecord(start, 0);
	conv << < blocksPerGrid, threadsPerBlock >> > (dinputMatrix, dconvLayerWeights, dconvLayer);
	cudaStatus = cudaEventRecord(stop, 0);
	cudaStatus = cudaEventSynchronize(stop);
	cudaDeviceSynchronize();
	cudaStatus = cudaEventElapsedTime(&time, start, stop);
	time = time * 1000;

	cudaMemcpy(hconvLayer, dconvLayer, (NUM_OF_KERNELS * convLayerSizeY * convLayerSizeX) * sizeof(float), cudaMemcpyDeviceToHost);
	// PRINTING RESULTS
	COUT_result << "Results:\n" << "Convolution matrix:\n";
	for (int k = 0; k < NUM_OF_KERNELS; k++)
	{
		COUT_result << "slice: " << k + 1 << "\n";
		for (int j = 0; j < convLayerSizeY; j++)
		{
			for (int i = 0; i < convLayerSizeX; i++)
			{
				COUT_result << std::setprecision(2) << std::fixed << hconvLayer[k * convLayerSizeY * convLayerSizeX + j * convLayerSizeX + i] << " ";
			}
			COUT_result << "\n";
		}
		COUT_result << "\n";
	}
	// CLEAN UP
	printf("Time for the kernel: %f us\n", time);

	cudaFree(dconvLayerWeights);
	cudaFree(dinputMatrix);
	cudaFree(dconvLayer);
	return 0;
}
