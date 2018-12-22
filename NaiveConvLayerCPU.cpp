
#include "pch.h"
#include <iostream>
#include <iomanip>
#include <cmath>

// Input size
int const DEPTH = 1;
int const WIDTH = 3;
int const LENGTH = 3;
// Kernel characteristics
int const STRIDE = 1;
int const CONV_RECP_SIZEX = 3;
int const CONV_RECP_SIZEY = 3;
int const NUM_OF_KERNELS = 1;

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



int main()
{
	// Initializing input, convolution and output layers.
	int convLayerSizeX = ceil(WIDTH / STRIDE);
	int convLayerSizeY = ceil(LENGTH / STRIDE);
	float*** inputMatrix = init3DArray(DEPTH, LENGTH, WIDTH);
	float*** convLayer = init3DArray(NUM_OF_KERNELS, convLayerSizeY, convLayerSizeX);
	// Initializing convolution layer weights and output layer weights.
	float**** convLayerWeights = init4DArray(NUM_OF_KERNELS ,DEPTH, CONV_RECP_SIZEY, CONV_RECP_SIZEX);
	//GENERATING INPUT
	std::cout << "Inputs:\n";
	float x = 0;
	for (int c = 0; c < DEPTH; c++)
	{
		std::cout << "slice: " << c + 1 << "\n";
		for (int j = 0; j < LENGTH; j++)
		{
			for (int i = 0; i < WIDTH; i++)
			{
				inputMatrix[c][j][i] = x;
				x += 0.5;
				std::cout << std::setprecision(1) << std::fixed << inputMatrix[c][j][i] << " ";
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
					convLayerWeights[d][c][j][i] = w;
					w += 0.1;
					std::cout << std::setprecision(1) << std::fixed << convLayerWeights[d][c][j][i] << " ";
				}
				std::cout << "\n";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}
	// CONVOLUTION
	int offsetY = 0;
	int offsetX = 0;
	int inputIdxX = 0;
	int inputIdxY = 0;
	for (int k = 0; k < NUM_OF_KERNELS; k++)
	{
		for (int i = 0; i < convLayerSizeX; i++)// Iterating over the width of Conv layer
		{
			for (int j = 0; j < convLayerSizeY; j++)// Iterating over the height of Conv layer
			{
				convLayer[k][j][i] = 0;// Initiating each value to 0. can be changed to different value that will represent the Bias
				for (int p = 0; p < CONV_RECP_SIZEX; p++)// Iterating over the width of the receptive field of the conv kernel
				{
					for (int q = 0; q < CONV_RECP_SIZEY; q++)// Iterating over the height of the receptive field of the conv kernel
					{
						offsetY = q - CONV_RECP_SIZEY / 2 ;
						offsetX = p - CONV_RECP_SIZEX / 2 ;
						inputIdxY = STRIDE * j;
						inputIdxX = STRIDE * i;
						if ((inputIdxY + offsetY) >= 0 && (inputIdxX + offsetX) >= 0 && (inputIdxY + offsetY) < LENGTH  && (inputIdxX + offsetX) < WIDTH )
						{
							for (int d = 0; d < DEPTH; d++)// Iterating over the depth of the receptive field of the conv kernel
							{
								convLayer[k][j][i] += convLayerWeights[k][d][q][p] * inputMatrix[d][inputIdxY + offsetY][inputIdxX + offsetX];
							}
						}
					}
				}
			}
		}
	}
	
	// PRINTING RESULTS
	std::cout << "Results:\n" << "Convolution matrix:\n";
	for (int k = 0; k < NUM_OF_KERNELS; k++)
	{
		std::cout << "slice: " << k+1 << "\n";
		for (int j = 0; j < convLayerSizeY; j++)
		{
			for (int i = 0; i < convLayerSizeX; i++)
			{
				std::cout << std::setprecision(1) << std::fixed << convLayer[k][j][i] << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}
	
	// CLEAN UP
	delete3DArray(inputMatrix ,DEPTH, LENGTH);
	delete4DArray(convLayerWeights ,NUM_OF_KERNELS, DEPTH, CONV_RECP_SIZEY);
	delete3DArray(convLayer ,NUM_OF_KERNELS , convLayerSizeY);
	return 0;
}
