#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>
#include<math.h>
#include<sys/time.h>
#include<time.h>

#define CMAX       1496         //+-1 AU * 10e-5
#define CMIN      -1496
#define MMAX       9e2         //approximately mass of Ceres
#define MMIN       14e1         //approximately mass of Bennu
#define AMAX       1000
#define AMIN      -1000
#define VC         299792458    //speed of light

#define EPSILON2   0.5f         //softener used to prevent r^2 -> 0

#define DEF_BODIES 256
#define G          1.0f//6.673e-11f   //gravitational constant
#define DEF_DELTA  0.1f
#define DEF_STEPS  10000

//Lazy programming
int NUM_BODIES, NUM_STEPS;
float DELTA_T;

//tools
int parse_args(int argc, char **argv);
int output = 1;

//parameter functions
float rand_acceleration();
float rand_coordinate();
float rand_mass();

//timing functions and struct
typedef struct {
    struct timeval start;
    struct timeval end;
} timer;
void start_timer(timer *t);
void stop_timer(timer *t);
float elapsed_time(timer *t);
