#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/time.h>
#include<unistd.h>

#define CMAX       15000000
#define CMIN      -15000000
#define MMAX       1.898e27     //mass of Jupiter
#define MMIN       328.5e21     //mass of Mercury
#define VMAX       1000
#define VMIN      -1000
#define VC         299792458    //speed of light

#define NUM_BODIES 3
#define G          6.673e-11
#define TIMESTEP   1
#define NUM_STEPS  100

typedef struct fdim3 {
    float x;
    float y;
    float z;
} fdim3;

float rand_velocity();
float rand_coordinate();
float rand_mass();
__global__ void nbody_kernel(fdim3 *pos, fdim3 *vel, float *mass, int num_body,
        float gravity, float timestep, int num_steps, fdim3 *out_position);
