#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<time.h>

#define TILE_WIDTH_M 40
#define TILE_WIDTH_N 16
#define K TILE_WIDTH_M/TILE_WIDTH_N

__global__ void matrix_multiply_kernel(float *A, float *B, float *C,
				int numARows, int numAColumns,
				int numBRows, int numBColumns,
				int numCRows, int numCColumns);
void matrix_multiply(float *A, float *B, float *C, int numARows,
				int numAColumns, int numBRows, int numBColumns,
				int numCRows, int numCColumns);
