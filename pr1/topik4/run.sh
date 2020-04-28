#!/usr/bin/env sh

echo $2 $3 $4 > header # numprocs train test
cat header input > in
mpic++ knn.cpp -std=c++14 -g && mpirun --hostfile hostfile -n $1 `pwd`/a.out
rm header in
