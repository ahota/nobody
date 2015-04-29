#!/bin/bash

nvcc sgemm.cu -o sgemm_cuda
gcc sgemm.c -o sgemm_cpu
