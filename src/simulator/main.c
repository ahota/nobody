#include "nbody.h"

//CPU-specific structs and functions
typedef struct {
    float x;
    float y;
    float z;
} float3;

typedef struct {
    float x;
    float y;
    float z;
    float w;
} float4;
void interact(float4 *body_i, float4 *body_j, float3 *acc_i, float4 *inter_i);

int main(int argc, char **argv) {
    //Get parameters, if any, from user
    NUM_BODIES = DEF_BODIES;
    NUM_STEPS  = DEF_STEPS;
    DELTA_T    = DEF_DELTA;
    int status = parse_args(argc, argv);
    if(status)
        return status;

	int i, j, t;
    srand(time(NULL));
    timer perf_timer;
    timer total_timer;

	printf("Creating bodies...\n");

    float4 *pos_mass;
    float4 *intermediate;
    float3 *acc;

	printf("Allocating memory...\n");

    start_timer(&total_timer);
    pos_mass     = (float4 *)malloc(NUM_BODIES * sizeof(float4));
    intermediate = (float4 *)malloc(NUM_BODIES * sizeof(float4));
    acc          = (float3 *)malloc(NUM_BODIES * sizeof(float3));

	printf("Initializing bodies...\n");

	for(i = 0; i < NUM_BODIES; i++) {
        pos_mass[i].x = rand_coordinate();
        pos_mass[i].y = rand_coordinate();
        pos_mass[i].z = rand_coordinate();
        pos_mass[i].w = rand_mass();
	}

    printf("SIMULATION SETTINGS:\n");
    printf("  bodies  = %d\n", NUM_BODIES);
    printf("  steps   = %d\n", NUM_STEPS);
    printf("  delta t = %f\n", DELTA_T);

    printf("Running simulation...\n");
    
    start_timer(&perf_timer);
    for(t = 0; t < NUM_STEPS; t++) {
        for(i = 0; i < NUM_BODIES; i++) {
            for(j = 0; j < NUM_BODIES, j != i; j++) {
                interact(&pos_mass[i], &pos_mass[j], &acc[i], &intermediate[i]);
            }
        }
        //Update positions
        for(i = 0; i < NUM_BODIES; i++) {
            pos_mass[i] = intermediate[i];
        }
    }
    stop_timer(&perf_timer);
    stop_timer(&total_timer);

    printf("Simulation runtime:\t%f s\n", elapsed_time(&perf_timer));
    printf("Total runtime:\t%f s\n", elapsed_time(&total_timer));

    if(output) {
        time_t raw_time;
        struct tm *current_time;
        time(&raw_time);
        current_time = localtime(&raw_time);
        char *filename = (char *)malloc(64);
        sprintf(filename, "cpu_%02d%02d%02d_%02d%02d%02d.nbd", 
                current_time->tm_year%100, current_time->tm_mon,
                current_time->tm_mday, current_time->tm_hour,
                current_time->tm_min, current_time->tm_sec);

        printf("Saving to %s...\n", filename);

        FILE *outfile = fopen(filename, "w");
        if(outfile == NULL)
            fprintf(stderr, "Error opening file\n");
        else {
            fprintf(outfile, "Final output:\n");
            
            fprintf(outfile, "i\tx\t\ty\t\tz\n");
            fprintf(outfile,
                    "----------------------------------------------------\n");
            for(i = 0; i < NUM_BODIES; i++) {
                fprintf(outfile, "%d\t%f\t%f\t%f\n", i, pos_mass[i].x,
                        pos_mass[i].y, pos_mass[i].z);
            }
            fclose(outfile);
        }
    }

    printf("Done.\n");
    
    return 0;
}

void interact(float4 *body_i, float4 *body_j, float3 *acc_i, float4 *inter_i) {
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
    acc_i->x += acc * d.x * DELTA_T;
    acc_i->y += acc * d.y * DELTA_T;
    acc_i->z += acc * d.z * DELTA_T;

    //update position of body i
    inter_i->x = body_i->x + acc_i->x;
    inter_i->y = body_i->y + acc_i->y;
    inter_i->z = body_i->z + acc_i->z;
    inter_i->w = body_i->w;
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
