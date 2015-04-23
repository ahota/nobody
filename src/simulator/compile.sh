#!/bin/bash

gcc main.c -lm -o nbody_cpu
nvcc main.cu -o nbody_cuda
