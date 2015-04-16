#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/time.h>
#include<unistd.h>
//#include<cuda.h>
//#include<vector_types.h>

#define CMAX       1496         //+-1 AU * 10e-5
#define CMIN      -1496
#define MMAX       1.898e22     //mass of Jupiter * 10e-5
#define MMIN       328.5e16     //mass of Mercury * 10e-5
#define AMAX       1000
#define AMIN      -1000
#define VC         299792458    //speed of light

#define EPSILON2   0.5          //softener used to prevent r^2 -> 0

#define NUM_BODIES 4
#define G          6.673e-11    //gravitational constant
#define TIMESTEP   1
#define NUM_STEPS  100

float rand_acceleration();
float rand_coordinate();
float rand_mass();
__global__ void main_nbody_kernel(float4 *dev_pos_mass, float3 *dev_acc,
        float3 *dev_output, int cur_step);
__device__ void tile_nbody_kernel(float4 *my_pos_mass, float3 *my_acc);
__device__ void force_kernel(float4 *body_i, float4 *body_j,
        float3 *acc_i);
