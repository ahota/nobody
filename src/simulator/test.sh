#!/bin/bash

BODIES="64 128 256 512 1024"
STEPS="1000"
DELTA="0.1"

echo $(date) > cuda_test.txt;
echo $(date) > cpu_test.txt;

for b in $BODIES; do
    echo $b
    ./nbody_cpu  -b $b -s $STEPS -t $DELTA -o >> cpu_test.txt;
    ./nbody_cuda -b $b -s $STEPS -t $DELTA -o >> cuda_test.txt;
done
