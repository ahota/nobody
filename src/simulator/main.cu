#include "nbody.h"
#include<cuda.h>

//CUDA-specific vars and functions
__device__ int NBODIES;
__device__ float DT;
__global__ void main_nbody_kernel(float4 *dev_pos_mass, float3 *dev_acc,
        float3 *dev_output, int cur_step);
__device__ void tile_nbody_kernel(float4 *my_pos_mass, float3 *my_acc);
__device__ void force_kernel(float4 *body_i, float4 *body_j,
        float3 *acc_i);

int main(int argc, char **argv) {
    //Get parameters, if any, from user
    NUM_BODIES = DEF_BODIES;
    NUM_STEPS  = DEF_STEPS;
    DELTA_T    = DEF_DELTA;
    int status = parse_args(argc, argv);
    if(status)
        return status;
    cudaMemcpyToSymbol(NBODIES, &NUM_BODIES, sizeof(int), 0, 
            cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(DT, &DELTA_T, sizeof(int), 0, 
            cudaMemcpyHostToDevice);

    int i;
    srand(time(NULL));
    timer perf_timer;
    timer total_timer;
    
	printf("Creating bodies...\n");

    float4 *host_pos_mass, *dev_pos_mass;
    float3 *host_acc, *dev_acc;
    float3 *host_output, *dev_output;

	printf("Allocating host memory...\n");

    start_timer(&total_timer);
    host_pos_mass = (float4 *)malloc(NUM_BODIES * sizeof(float4));
    host_acc      = (float3 *)malloc(NUM_BODIES * sizeof(float3));
	host_output   = (float3 *)malloc(NUM_BODIES * NUM_STEPS * sizeof(float3));

	printf("Allocating device memory...\n");

	cudaMalloc((void **)&dev_pos_mass, NUM_BODIES * sizeof(float4));
    cudaMalloc((void **)&dev_acc, NUM_BODIES * sizeof(float3));
    cudaMalloc((void **)&dev_output, NUM_BODIES * NUM_STEPS * sizeof(float3));

	printf("Initializing bodies...\n");

	for(i = 0; i < NUM_BODIES; i++) {
        host_pos_mass[i].x = rand_coordinate();
        host_pos_mass[i].y = rand_coordinate();
        host_pos_mass[i].z = rand_coordinate();
        host_pos_mass[i].w = rand_mass();
	}

    printf("SIMULATION SETTINGS:\n");
    printf("  bodies  = %d\n", NUM_BODIES);
    printf("  steps   = %d\n", NUM_STEPS);
    printf("  delta t = %f\n", DELTA_T);

    /*
    printf("Initial positions and masses:\n");
    for(i = 0; i < NUM_BODIES; i++) {
        printf("%d:\t%f\t%f\t%f\n", i, host_pos_mass[i].x, host_pos_mass[i].y,
                host_pos_mass[i].z, host_pos_mass[i].w);
    }
    */

	printf("Copying to device...\n");

	cudaMemcpy(dev_pos_mass, host_pos_mass, NUM_BODIES * sizeof(float3),
					cudaMemcpyHostToDevice);
	cudaMemcpy(dev_acc, host_acc, NUM_BODIES * sizeof(float3),
					cudaMemcpyHostToDevice);
    cudaMemcpy(dev_output, host_output, NUM_BODIES * NUM_STEPS * sizeof(float3),
                    cudaMemcpyHostToDevice);

	printf("Running kernel...\n");

    start_timer(&perf_timer);
    int block_size = (NUM_BODIES < 16) ? 4 : (NUM_BODIES < 256) ? 16 : 32;
    int grid_size  = NUM_BODIES / block_size;
    int mem_size = (block_size+1) * sizeof(float4);
    printf("KERNEL SETTINGS:\n");
    printf("  bodies    = %d\n", NUM_BODIES);
    printf("  tile size = %d\n", block_size);
    printf("  grid size = %d\n", grid_size);
    for(i = 0; i < NUM_STEPS; i++) {
        main_nbody_kernel<<<grid_size, block_size, mem_size>>>(dev_pos_mass,
                dev_acc, dev_output, i);
    }
    stop_timer(&perf_timer);

    printf("Simulation runtime:\t%f s\n", elapsed_time(&perf_timer));

    printf("Copying to host...\n");

    cudaMemcpy(host_output, dev_output, NUM_BODIES * NUM_STEPS * sizeof(float3),
                    cudaMemcpyDeviceToHost);
    cudaFree(dev_pos_mass);
    cudaFree(dev_acc);
    cudaFree(dev_output);
    stop_timer(&total_timer);

    printf("Total runtime:\t%f s\n", elapsed_time(&total_timer));

    if(output) {
        time_t raw_time;
        struct tm *current_time;
        time(&raw_time);
        current_time = localtime(&raw_time);
        char *filename = (char *)malloc(64);
        sprintf(filename, "cuda_%02d%02d%02d_%02d%02d%02d.nbd", 
                current_time->tm_year%100, current_time->tm_mon,
                current_time->tm_mday, current_time->tm_hour,
                current_time->tm_min, current_time->tm_sec);

        printf("Saving to %s...\n", filename);

        FILE *outfile = fopen(filename, "w");
        if(outfile == NULL)
            fprintf(stderr, "Error opening file\n");
        else {
            //printf("%f\n", host_output[0].x);
            fprintf(outfile, "%d,%d\n", NUM_BODIES, NUM_STEPS);
            for(i = 0; i < NUM_BODIES * NUM_STEPS; i++) {
                fprintf(outfile, "%f,%f,%f\n", host_output[i].x, 
                        host_output[i].y, host_output[i].z);
            }
            fclose(outfile);
        }
    }

	printf("Done.\n");
	return 0;
}

__global__ void main_nbody_kernel(float4 *dev_pos_mass, float3 *dev_acc,
        float3 *dev_output, int cur_step) {
    //index into global arrays for this thread's body
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    //local copies of this body's position, mass, and acceleration
    float4 my_pos_mass = dev_pos_mass[global_id];
    float3 my_acc = dev_acc[global_id];

    //copy of position and mass for bodies in the current tile
    extern __shared__ float4 tile_pos_mass[]; 

    //iterate over all tiles and update position and acceleration
    //each iteration loads one tile's worth of data from global memory
    //these reads should be coalesced
    int i, tile;
    for(i = 0, tile = 0; i < NBODIES; i += blockDim.x, tile++) {
        //index into global for this thread's body *for this tile*
        int tile_id = tile * blockDim.x + threadIdx.x;

        //threads collaborate to load from global for this tile
        tile_pos_mass[threadIdx.x] = dev_pos_mass[tile_id];
        __syncthreads();

        //update acceleration for this thread's body for this tile
        tile_nbody_kernel(&my_pos_mass, &my_acc);
        __syncthreads();
    }

    //update position for this body
    my_pos_mass.x += my_acc.x;
    my_pos_mass.y += my_acc.y;
    my_pos_mass.z += my_acc.z;

    //update global position array
    dev_pos_mass[global_id] = my_pos_mass;
	dev_acc[global_id] = my_acc;

    //update global output
    dev_output[cur_step * NBODIES + global_id].x = my_pos_mass.x;
    dev_output[cur_step * NBODIES + global_id].y = my_pos_mass.y;
    dev_output[cur_step * NBODIES + global_id].z = my_pos_mass.z;
}

__device__ void tile_nbody_kernel(float4 *my_pos_mass, float3 *my_acc) {
    //tile position array from the outer kernel
    //pre-loaded with this tile's positions and masses
    extern __shared__ float4 tile_pos_mass[];

    //iterate over each body in the tile and calculate its effect on
    //this thread's body
    int i;
    for(i = 0; i < blockDim.x; i++) {
        force_kernel(my_pos_mass, &tile_pos_mass[i], my_acc);
    }
}

__device__ void force_kernel(float4 *body_i, float4 *body_j, float3 *acc_i) {
    //calculate distance components
    float3 d;
    d.x = body_j->x - body_i->x;
    d.y = body_j->y - body_i->y;
    d.z = body_j->z - body_i->z;

    //use episilon softener
    //  r^2 + epsilon^2
    float denominator = d.x * d.x + d.y * d.y + d.z * d.z + EPSILON2;
    //cube and sqrt to get (r^2 + epsilon^2)^(3/2)
    denominator = sqrt( denominator * denominator * denominator );

    float acc = G * body_j->w / denominator;

    //update acceleration
    acc_i->x += acc * d.x * DT;
    acc_i->y += acc * d.y * DT;
    acc_i->z += acc * d.z * DT;
}

int parse_args(int argc, char **argv) {
	int i;
    for(i = 1; i + 1 <= argc; i += 2) {
        if(strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [-b num_bodies] [-s num_steps] [-t delta_t] \
                    [-o]\n", argv[0]);
            return 1;
        }
        else if(strcmp(argv[i], "-b") == 0) {
            NUM_BODIES = atoi(argv[i + 1]);
        }
        else if(strcmp(argv[i], "-s") == 0) {
            NUM_STEPS  = atoi(argv[i + 1]);
        }
        else if(strcmp(argv[i], "-t") == 0) {
            DELTA_T    = atof(argv[i + 1]);
        }
        else if(strcmp(argv[i], "-o") == 0) {
            output     = 0;
        }
        else {
            fprintf(stderr, "Error: unsupported flag %s\n", argv[i]);
            return -1;
        }
    }
    return 0;
}

float rand_coordinate() {
    return ((float)rand() / (float)RAND_MAX) * (CMAX - CMIN) + CMIN;
}

float rand_acceleration() {
    return ((float)rand() / (float)RAND_MAX) * (AMAX - AMIN) + AMIN;
}

float rand_mass() { 
    return ((float)rand() / (float)RAND_MAX) * (MMAX - MMIN) + MMIN;
}

void start_timer(timer *t) {
    gettimeofday( &(t->start), NULL);
}

void stop_timer(timer *t) {
    gettimeofday( &(t->end), NULL);
}

float elapsed_time(timer *t) {
    return (float) (t->end.tv_sec  - t->start.tv_sec)  + 
                   (t->end.tv_usec - t->start.tv_usec) /
                   1000000.0;
}
