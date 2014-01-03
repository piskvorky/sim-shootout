#!/bin/bash

EXPECTED_ARGS=1
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
  # when run without params, print help and exit
  echo "first argument must be data directory from prepare_shootout"
  exit $E_BADARGS
fi

datadir=$1
shift 1

ks="1 10 50 100 1000"
methods="gensim flann annoy"
accs="low high"

# run all gensim tests
for method in $methods; do
    for acc in $accs; do
        for k in $ks; do
            echo "running $method $k $acc"
            ./shootout_${method}.py $datadir $k $acc &> ./log_${method}_${k}_${acc}.txt
        done
    done
done
