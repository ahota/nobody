#include "nbody.h"

int main() {
    srand(time(NULL));

	printf("Creating bodies...\n");

    float4 *pos_mass;
    float4 *intermediate;
    float3 *acc;

	printf("Allocating memory...\n");

    pos_mass     = (float4 *)malloc(NUM_BODIES * sizeof(float4));
    intermediate = (float4 *)malloc(NUM_BODIES * sizeof(float4));
    acc          = (float3 *)malloc(NUM_BODIES * sizeof(float3));

	printf("Initializing bodies...\n");

	int i, j, t;
	for(i = 0; i < NUM_BODIES; i++) {
        pos_mass[i].x = rand_coordinate();
        pos_mass[i].y = rand_coordinate();
        pos_mass[i].z = rand_coordinate();
        pos_mass[i].w = rand_mass();
	}

    printf("Running simulation...\n");
    
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

    printf("Final output:\n");
    
    printf("i\tx\t\ty\t\tz\n");
    printf("----------------------------------------------------\n");
    for(i = 0; i < NUM_BODIES; i++) {
        printf("%d\t%f\t%f\t%f\n", i, pos_mass[i].x, pos_mass[i].y,
                pos_mass[i].z);
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
    acc_i->x += acc * d.x * TIMESTEP;
    acc_i->y += acc * d.y * TIMESTEP;
    acc_i->z += acc * d.z * TIMESTEP;

    //update position of body i
    inter_i->x = body_i->x + acc_i->x;
    inter_i->y = body_i->y + acc_i->y;
    inter_i->z = body_i->z + acc_i->z;
    inter_i->w = body_i->w;
}
