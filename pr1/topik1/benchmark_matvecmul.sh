#!/bin/bash

procs=(2 4 8 16 32)

mpicc -o mvm matrix_vector_multiplication.c
chmod +x mvm

echo "np,N,M,valid,job_time,fanout_time,fanin_time,"
for i in "${procs[@]}"
do
    for j in {1..5}
    do
        mpirun --hostfile $(pwd)/../hostfile -np "$(($i + 1))" $(pwd)/mvm
    done
done
echo ""