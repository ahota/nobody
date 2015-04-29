#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    struct timeval start;
    struct timeval end;
} timer;

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

int main(int argc, char **argv) {
	float *A; // The A matrix
	float *B; // The B matrix
	float *C; // The output C matrix
	int numARows;    // number of rows in the matrix A
	int numAColumns; // number of columns in the matrix A
	int numBRows;    // number of rows in the matrix B
	int numBColumns; // number of columns in the matrix B
	int numCRows;    // number of rows in the matrix C (you have to set this)
	int numCColumns; // number of columns in the matrix C (you have to set this)
    char pathA[8] = "A.txt";
    char pathB[8] = "B.txt";
    char pathC[8] = "C.txt";
    timer perf_timer;
    timer total_timer;
	
    printf("Loading matrices...\n");

    start_timer(&total_timer);

    //load A and B
    FILE *file = fopen(pathA, "r");
    fscanf(file, "%d %d", &numARows, &numAColumns);
    A = (float *)malloc(numARows * numAColumns * sizeof(float));
    int r, c;
    for(r = 0; r < numARows; r++) {
        for(c = 0; c < numAColumns; c++) {
            fscanf(file, "%f", &A[r*numARows + c]);
        }
    }
    fclose(file);
    file = fopen(pathB, "r");
    fscanf(file, "%d %d", &numBRows, &numBColumns);
    B = (float *)malloc(numBRows * numBColumns * sizeof(float));
    for(r = 0; r < numBRows; r++) {
        for(c = 0; c < numBColumns; c++) {
            fscanf(file, "%f", &B[r*numBRows + c]);
        }
    }
    fclose(file);

    printf("Allocating memory...\n");

	numCRows = numAColumns; //Had to switch this because A is transposed
	numCColumns = numBColumns;
	C = ( float * )malloc(sizeof(float) * numCRows * numCColumns);

    printf("Calculating...\n");
    start_timer(&perf_timer);
    //multiply!
    int i;
    for(r = numAColumns - 1; r >= 0; r--) {
        for(c = 0; c < numBColumns; c++) {
            float sum = 0;
            for(i = 0; i < numARows; i++) {
                sum += A[r*numAColumns + i] * B[i*numBColumns + c];
            }
            C[r * numCRows + c] = sum;
        }
    }
    stop_timer(&perf_timer);

    printf("Saving result...\n");

    FILE *output = fopen(pathC, "w");
    fprintf(output, "%d %d\n", numCRows, numCColumns);
    for(r = 0; r < numCRows; r++) {
        for(c = 0; c < numCColumns; c++) {
            fprintf(output, "%.5f ", C[r*numCRows + c]);
        }
        fprintf(output, "\n");
    }
    fclose(output);
    stop_timer(&total_timer);

    printf("Calculation runtime = %f s\n", elapsed_time(&perf_timer));
    printf("Total runtime       = %f s\n", elapsed_time(&total_timer));


    return 0;

}
