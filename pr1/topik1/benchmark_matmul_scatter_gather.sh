#!/bin/bash

procs=(2 4 8 16 32)

mpicc -o mmsg matrix_multiplication_scatter_gather.c
chmod +x mmsg

echo "np,N,valid,job_time,fanout_time,fanin_time,"
for i in "${procs[@]}"
do
    for j in {1..5}
    do
        mpirun --hostfile $(pwd)/../hostfile -np "$i" $(pwd)/mmsg 
    done
done
echo ""