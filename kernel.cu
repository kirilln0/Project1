
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
// Input size
int const BATCH = 1;
int const DEPTH = 3;
int const WIDTH = 100;
int const LENGTH = 100;
// Kernel characteristics
int const ZPADX = 0;
int const ZPADY = 0;
int const STRIDEX = 1;
int const STRIDEY = 1;
int const CONV_RECP_SIZEX = 3;
int const CONV_RECP_SIZEY = 3;
int const NUM_OF_KERNELS = 1;
int const convLayerSizeX = ((WIDTH - CONV_RECP_SIZEX + 2 * ZPADX) / STRIDEX + 1);
int const convLayerSizeY = ((LENGTH - CONV_RECP_SIZEY + 2 * ZPADY) / STRIDEY + 1);

float**** init4DArray(int dim4, int depth, int height, int length)
{
	float**** ptr = new float***[dim4];
	for (int d = 0; d < dim4; d++)
	{
		ptr[d] = new float**[depth];
		for (int i = 0; i < depth; i++)
		{
			ptr[d][i] = new float*[height];
			for (int j = 0; j < height; j++)
			{
				ptr[d][i][j] = new float[length];
			}
		}
	}

	return ptr;
}
float*** init3DArray(int depth, int height, int length)
{
	float*** ptr = new float**[depth];
	for (int i = 0; i < depth; i++)
	{
		ptr[i] = new float*[height];
		for (int j = 0; j < height; j++)
		{
			ptr[i][j] = new float[length];
		}
	}
	return ptr;
}
void delete4DArray(float**** ptr, int dim4, int depth, int height)
{
	for (int d = 0; d < dim4; ++d)
	{
		for (int i = 0; i < depth; ++i)
		{
			for (int j = 0; j < height; ++j)
			{
				delete[] ptr[d][i][j];
			}
			delete[] ptr[d][i];
		}
		delete[] ptr[d];
	}
	delete[] ptr;
}
void delete3DArray(float*** ptr, int depth, int height)
{
	for (int i = 0; i < depth; ++i)
	{
		for (int j = 0; j < height; ++j)
		{
			delete[] ptr[i][j];
		}
		delete[] ptr[i];
	}
	delete[] ptr;
}


__global__
void conv(float* inputm, float* weights, float* outputm ) 
{
	int ROW = blockIdx.y * blockDim.y + threadIdx.y;
	int COL = blockIdx.x * blockDim.x + threadIdx.x;
	int DEP = blockIdx.z * blockDim.z + threadIdx.z;
	outputm[DEP * convLayerSizeY * convLayerSizeX + ROW * convLayerSizeX + COL] = 1;

	for (int i = 0; i < DEPTH; i++)
	{
		for (int j = 0; j < CONV_RECP_SIZEY; j++)
		{
			for (int l = 0; l < CONV_RECP_SIZEX; l++)
			{
				//printf("output=%d,input=%d,weiht=%d\n", DEP * convLayerSizeY * convLayerSizeX + ROW * convLayerSizeX + COL, i * WIDTH * LENGTH + (j + ROW * STRIDEY) * WIDTH + (l + COL * STRIDEX), DEP * DEPTH * CONV_RECP_SIZEX * CONV_RECP_SIZEY + i * CONV_RECP_SIZEX * CONV_RECP_SIZEY + j * CONV_RECP_SIZEX + l);
				outputm[DEP * convLayerSizeY * convLayerSizeX + ROW * convLayerSizeX + COL] += inputm[i * WIDTH * LENGTH + (j + ROW * STRIDEY) * WIDTH + (l + COL * STRIDEX)] * weights[DEP * DEPTH * CONV_RECP_SIZEX * CONV_RECP_SIZEY + i * CONV_RECP_SIZEX * CONV_RECP_SIZEY + j * CONV_RECP_SIZEX + l];
			}
		}
	}
	
}

int main()
{

	float hinputMatrix[BATCH * DEPTH * LENGTH * WIDTH];
	float hconvLayer[NUM_OF_KERNELS * convLayerSizeY * convLayerSizeX] = {1,2,3,4};
	float hconvLayerWeights[NUM_OF_KERNELS * DEPTH * CONV_RECP_SIZEY * CONV_RECP_SIZEX];
	//GENERATING INPUT
	std::cout << "Inputs:\n";
	float x = 0;
	for (int b = 0; b < BATCH; b++)
	{
		for (int c = 0; c < DEPTH; c++)
		{
			std::cout << "slice: " << c + 1 << "\n";
			for (int j = 0; j < LENGTH; j++)
			{
				for (int i = 0; i < WIDTH; i++)
				{
					hinputMatrix[b * DEPTH * LENGTH *WIDTH + c * LENGTH * WIDTH + j * WIDTH + i] = x;
					x += 0.5;
					//std::cout << std::setprecision(1) << std::fixed << hinputMatrix[b * DEPTH * LENGTH *WIDTH + c * LENGTH * WIDTH + j * WIDTH + i]<<" ";//<< " ("<< b * DEPTH * LENGTH *WIDTH + c * LENGTH * WIDTH + j * WIDTH + i<<") ";
				}
				std::cout << "\n";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}
	std::cout << "Weights:\n";
	float w = 0;
	for (int d = 0; d < NUM_OF_KERNELS; d++)
	{
		std::cout << "kernel: " << d + 1 << "\n";
		for (int c = 0; c < DEPTH; c++)
		{
			std::cout << "slice: " << c + 1 << "\n";
			for (int j = 0; j < CONV_RECP_SIZEY; j++)
			{
				for (int i = 0; i < CONV_RECP_SIZEX; i++)
				{
					hconvLayerWeights[d * DEPTH * CONV_RECP_SIZEX * CONV_RECP_SIZEY + c * CONV_RECP_SIZEX * CONV_RECP_SIZEY + j * CONV_RECP_SIZEX + i] = w;
					w += 0.1;
					//std::cout << std::setprecision(1) << std::fixed << hconvLayerWeights[d * DEPTH * CONV_RECP_SIZEX * CONV_RECP_SIZEY + c * CONV_RECP_SIZEX * CONV_RECP_SIZEY + j * CONV_RECP_SIZEX + i] << " ";// " (" << d * DEPTH * CONV_RECP_SIZEX * CONV_RECP_SIZEY + c * CONV_RECP_SIZEX * CONV_RECP_SIZEY + j * CONV_RECP_SIZEX + i << ") ";
				}
				std::cout << "\n";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
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
	

	dim3 threadsPerBlock(convLayerSizeX, convLayerSizeY, NUM_OF_KERNELS);
	dim3 blocksPerGrid(1, 1, 1);
	if (NUM_OF_KERNELS * convLayerSizeY * convLayerSizeX> 512) {
		threadsPerBlock.x = 8;
		threadsPerBlock.y = 8;
		threadsPerBlock.z = 8;
		blocksPerGrid.x = ceil(double(convLayerSizeX) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(convLayerSizeY) / double(threadsPerBlock.y));
		blocksPerGrid.z = ceil(double(NUM_OF_KERNELS) / double(threadsPerBlock.z));
	}
	std::cout << "--------------"<< threadsPerBlock.x<<" "<< threadsPerBlock.y << " "<< threadsPerBlock.z << " "<< blocksPerGrid.x << " "<< blocksPerGrid.y << " "<< blocksPerGrid.z << " " <<"----------------------------";
	conv<<< blocksPerGrid, threadsPerBlock >>> (dinputMatrix, dconvLayerWeights, dconvLayer);
	cudaDeviceSynchronize();
	cudaMemcpy(hconvLayer, dconvLayer, (NUM_OF_KERNELS * convLayerSizeY * convLayerSizeX) * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	// PRINTING RESULTS
	std::cout << "Results:\n" << "Convolution matrix:\n";
	for (int k = 0; k < NUM_OF_KERNELS; k++)
	{
		std::cout << "slice: " << k + 1 << "\n";
		for (int j = 0; j < convLayerSizeY; j++)
		{
			for (int i = 0; i < convLayerSizeX; i++)
			{
				std::cout << std::setprecision(1) << std::fixed << hconvLayer[k * convLayerSizeY * convLayerSizeX + j * convLayerSizeX + i] << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}
	// CLEAN UP
	

	cudaFree(dconvLayerWeights);
	cudaFree(dinputMatrix);
	cudaFree(dconvLayer);
return 0;
}
