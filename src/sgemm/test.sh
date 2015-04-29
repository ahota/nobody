#!/bin/bash

N="64 128 256 512 1024 2048"

echo $(date) > cuda_test.txt;
echo $(date) > cpu_test.txt;

for n in $N; do
    echo $n
    ./gen_matrix.py $n A.txt -t
    ./gen_matrix.py $n B.txt -r
    ./sgemm_cpu >> cpu_test.txt;
    ./sgemm_cuda >> cuda_test.txt;
done
