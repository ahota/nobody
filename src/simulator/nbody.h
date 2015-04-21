#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/time.h>
#include<unistd.h>
//#include<cuda.h>
//#include<vector_types.h>

#define CMAX       1496         //+-1 AU * 10e-5
#define CMIN      -1496
#define MMAX       9e2         //approximately mass of Ceres
#define MMIN       14e1         //approximately mass of Bennu
#define AMAX       1000
#define AMIN      -1000
#define VC         299792458    //speed of light

#define EPSILON2   0.5f         //softener used to prevent r^2 -> 0

#define NUM_BODIES 256
#define G          1.0f//6.673e-11f   //gravitational constant
#define TIMESTEP   0.1f
#define NUM_STEPS  10000

float rand_acceleration();
float rand_coordinate();
float rand_mass();
__global__ void main_nbody_kernel(float4 *dev_pos_mass, float3 *dev_acc,
        float3 *dev_output, int cur_step);
__device__ void tile_nbody_kernel(float4 *my_pos_mass, float3 *my_acc);
__device__ void force_kernel(float4 *body_i, float4 *body_j,
        float3 *acc_i);
