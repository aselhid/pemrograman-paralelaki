#!/usr/bin/env bash

TIMEFORMAT="%E"
i="256"

while [ $i -le 8192 ]
    do
        sum="0"
        for t in {1..5}
        do
            res=`(time $1 $i) 2>&1 | tail -n 1`
            sum=`echo "$res + $sum" | bc -l`
        done
        avg=`echo "scale=3; $sum / 5" | bc -l  | awk '{printf "%f", $0}'`
        echo -e "$i\t$avg"
        i=$(($i*2))
    done
