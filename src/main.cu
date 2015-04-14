#include "nbody.h"

int main(int argc, char **argv) {
    srand(time(NULL));

	printf("Creating bodies...\n");

	fdim3 *host_position, *device_position;
	fdim3 *host_velocity, *device_velocity;
	float *host_mass, *device_mass;
    fdim3 *host_output, *device_output;

	printf("Allocating host memory...\n");

	host_position = (fdim3 *)malloc(NUM_BODIES * sizeof(fdim3));
	host_velocity = (fdim3 *)malloc(NUM_BODIES * sizeof(fdim3));
	host_mass     = (float *)malloc(NUM_BODIES * sizeof(float));
	host_output   = (fdim3 *)malloc(NUM_BODIES * NUM_STEPS * sizeof(fdim3));

	printf("Allocating device memory...\n");

	cudaMalloc((void **)&device_position, NUM_BODIES * sizeof(fdim3));
	cudaMalloc((void **)&device_velocity, NUM_BODIES * sizeof(fdim3));
	cudaMalloc((void **)&device_mass,     NUM_BODIES * sizeof(float));
    cudaMalloc((void **)&device_output, NUM_BODIES * NUM_STEPS * sizeof(fdim3));

	printf("Initializing bodies...\n");

	int i;
	for(i = 0; i < NUM_BODIES; i++) {
		host_position[i].x = rand_coordinate() * (i + 1);
		host_position[i].y = rand_coordinate() * (i + 1);
		host_position[i].z = rand_coordinate() * (i + 1);
		host_velocity[i].x = rand_velocity()   * (i + 1);
		host_velocity[i].y = rand_velocity()   * (i + 1);
		host_velocity[i].z = rand_velocity()   * (i + 1);
		host_mass[i]       = rand_mass() * (i + 1);
	}
    for(i = 0; i < NUM_BODIES * NUM_STEPS; i++) {
        host_output[i].x = 0; host_output[i].y = 0; host_output[i].z = 0;
    }

	printf("Copying to device...\n");

	cudaMemcpy(device_position, host_position, NUM_BODIES * sizeof(fdim3),
					cudaMemcpyHostToDevice);
	cudaMemcpy(device_velocity, host_velocity, NUM_BODIES * sizeof(fdim3),
					cudaMemcpyHostToDevice);
	cudaMemcpy(device_mass, host_mass, NUM_BODIES * sizeof(float),
					cudaMemcpyHostToDevice);
    cudaMemcpy(device_output, host_output, 
            NUM_BODIES * NUM_STEPS * sizeof(fdim3), cudaMemcpyHostToDevice);

	printf("Running kernel...\n");

	printf("\t%d iterations\n", NUM_STEPS);
    nbody_kernel<<<1, NUM_BODIES>>>(device_position, device_velocity,
                    device_mass, NUM_BODIES, G, TIMESTEP, NUM_STEPS,
                    device_output);

    printf("Copying to host...\n");

    cudaMemcpy(host_output, device_output, 
            NUM_BODIES * NUM_STEPS * sizeof(fdim3), cudaMemcpyDeviceToHost);
    cudaFree(device_position);
    cudaFree(device_velocity);
    cudaFree(device_mass);
    cudaFree(device_output);

    time_t raw_time;
    struct tm *current_time;
    time(&raw_time);
    current_time = localtime(&raw_time);
    char *filename = (char *)malloc(64);
    sprintf(filename, "%d%d%d_%d%d%d.nbd", current_time->tm_year,
            current_time->tm_mon, current_time->tm_mday, current_time->tm_hour,
            current_time->tm_min, current_time->tm_sec);

    printf("Saving to %s...\n", filename);

    FILE *outfile = fopen(filename, "w");
    if(outfile == NULL)
        fprintf(stderr, "Error opening file\n");
    else {
        fprintf(outfile, "%d %d\n", NUM_BODIES, NUM_STEPS);
        for(i = 0; i < NUM_BODIES * NUM_STEPS; i++) {
            fprintf(outfile, "%f,%f,%f\n", host_output[i].x, host_output[i].y,
                    host_output[i].z);
        }
        fclose(outfile);
    }

	printf("Done.\n");
	return 0;
}

float rand_coordinate() {
    return ((float)rand() / (float)RAND_MAX) * (CMAX - CMIN) + CMIN;
}

float rand_velocity() {
    return ((float)rand() / (float)RAND_MAX) * (VMAX - VMIN) + VMIN;
}

float rand_mass() { 
    return ((float)rand() / (float)RAND_MAX) * (MMAX - MMIN) + MMIN;
}

__global__ void nbody_kernel(fdim3 *pos, fdim3 *vel, float *mass, int num_body, 
        float gravity, float timestep, int num_steps, fdim3 *out_position) {
    int id = threadIdx.x;
    //optimizations:
    //  - use __shared__ array of pos and mass for faster access
    //    - this leaves room in shared memory for about 1024 bodies
    extern __shared__ fdim3 positions[];
    extern __shared__ float masses[];

    //Fill shared arrays
    positions[id] = pos[id];
    masses[id]    = mass[id];
    fdim3 d, a;
    fdim3 my_vel = vel[id];
    fdim3 my_pos = positions[id];
    
    int i, t;
    for(t = 0; t < num_steps; t++) {
        for(i = 0; i < num_body; i++) {
            //This causes divergence
            if(i != id) {
                //distances
                d.x = my_pos.x - positions[i].x;
                d.y = my_pos.y - positions[i].y;
                d.z = my_pos.z - positions[i].z;
                //accelerations
                a.x = (masses[i] * gravity) / (d.x * d.x);
                a.y = (masses[i] * gravity) / (d.y * d.y);
                a.z = (masses[i] * gravity) / (d.z * d.z);
                //update velocities
                my_vel.x += a.x * timestep;
                my_vel.y += a.y * timestep;
                my_vel.z += a.z * timestep;
            }
        }
        my_pos.x += my_vel.x;
        my_pos.y += my_vel.y;
        my_pos.z += my_vel.z;
        positions[id] = my_pos;
        out_position[t * num_body + id].x = my_pos.x;
        out_position[t * num_body + id].y = my_pos.y;
        out_position[t * num_body + id].z = my_pos.z;
    }
}
