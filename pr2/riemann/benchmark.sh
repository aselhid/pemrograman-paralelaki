#!/bin/bash

g++ -o riemann -fopenmp riemann.cpp

threads=(2 4 8 16)
for i in "${threads[@]}"
do
  for j in {1..10}
  do
    OMP_NUM_THREADS=$i ./riemann 
  done
done