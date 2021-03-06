
#include "pch.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;
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
#define COUT_weights if (1) std::cout
#define COUT_input if (1) std::cout
#define COUT_result if (1) std::cout

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
	int convLayerSizeX = ( (WIDTH - CONV_RECP_SIZEX + 2 * ZPADX) / STRIDEX + 1);
	int convLayerSizeY = ((LENGTH - CONV_RECP_SIZEY + 2 * ZPADY) / STRIDEY + 1);
	float**** inputMatrix = init4DArray(BATCH ,DEPTH, LENGTH, WIDTH);
	float**** convLayer = init4DArray(BATCH, NUM_OF_KERNELS, convLayerSizeY, convLayerSizeX);
	// Initializing convolution layer weights and output layer weights.
	float**** convLayerWeights = init4DArray(NUM_OF_KERNELS ,DEPTH, CONV_RECP_SIZEY, CONV_RECP_SIZEX);
	//GENERATING INPUT
	float test[1][1][3][3] = { 0 };

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
					inputMatrix[b][c][j][i] = x;
					x += 0.5;
					COUT_input << std::setprecision(1) << std::fixed << inputMatrix[b][c][j][i] << " ";
				}
				COUT_input << "\n";
			}
			COUT_input << "\n";
		}
		COUT_input << "\n";
	}

	COUT_weights << "Weights:\n";
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
					convLayerWeights[d][c][j][i] = w;
					w += 0.1;
					COUT_weights << std::setprecision(1) << std::fixed << convLayerWeights[d][c][j][i] << " ";
				}
				COUT_weights << "\n";
			}
			COUT_weights << "\n";
		}
		COUT_weights << "\n";
	}
	// CONVOLUTION
	auto t1 = Clock::now();

	//int offsetY = 0;
	//int offsetX = 0;
	int inputIdxX = 0;
	int inputIdxY = 0;
	for (int b = 0; b < BATCH; b++)
	{
		for (int k = 0; k < NUM_OF_KERNELS; k++)
		{
			for (int i = 0; i < convLayerSizeX; i++)// Iterating over the width of Conv layer
			{
				for (int j = 0; j < convLayerSizeY; j++)// Iterating over the height of Conv layer
				{
					convLayer[b][k][j][i] = 0;// Initiating each value to 0. can be changed to different value that will represent the Bias
					for (int p = 0; p < CONV_RECP_SIZEX; p++)// Iterating over the width of the receptive field of the conv kernel
					{
						for (int q = 0; q < CONV_RECP_SIZEY; q++)// Iterating over the height of the receptive field of the conv kernel
						{
							//offsetY = q - CONV_RECP_SIZEY / 2;
							//offsetX = p - CONV_RECP_SIZEX / 2;
							inputIdxY = STRIDEY * j;
							inputIdxX = STRIDEX * i;
							if ((inputIdxY + q) >= 0 && (inputIdxX + p) >= 0 && (inputIdxY + q) < LENGTH && (inputIdxX + p) < WIDTH)
							{
								for (int d = 0; d < DEPTH; d++)// Iterating over the depth of the receptive field of the conv kernel
								{
									convLayer[b][k][j][i] += convLayerWeights[k][d][q][p] * inputMatrix[b][d][inputIdxY + q][inputIdxX + p];
								}
							}
						}
					}
				}
			}
		}
	}
	
	auto t2 = Clock::now();

	// PRINTING RESULTS
	COUT_result << "Results:\n" << "Convolution matrix:\n";
	for (int b = 0; b < BATCH; b++)
	{
		for (int k = 0; k < NUM_OF_KERNELS; k++)
		{
			COUT_result << "slice: " << k + 1 << "\n";
			for (int j = 0; j < convLayerSizeY; j++)
			{
				for (int i = 0; i < convLayerSizeX; i++)
				{
					COUT_result << std::setprecision(1) << std::fixed << convLayer[b][k][j][i] << " ";
				}
				COUT_result << "\n";
			}
			COUT_result << "\n";
		}
	}
	std::cout << "Calculation time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " usec" << std::endl;
	// CLEAN UP
	delete4DArray(inputMatrix ,BATCH ,DEPTH, LENGTH);
	delete4DArray(convLayerWeights ,NUM_OF_KERNELS, DEPTH, CONV_RECP_SIZEY);
	delete4DArray(convLayer ,BATCH , NUM_OF_KERNELS , convLayerSizeY);
	return 0;
}
