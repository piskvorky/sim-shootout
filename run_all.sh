#!/bin/bash

EXPECTED_ARGS=1
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
  # when run without params, print help and exit
  echo "first argument must be a directory where data & indexes will be stored; make sure you have at least 100gb free space there"
  exit $E_BADARGS
fi

datadir=$1
shift 1

# first, download the raw wiki dump and convert it to LSI vectors, if not already present
articles=$datadir/lsi_vectors.mm.gz
if [ ! -e $articles ]; then
	wiki_file=$datadir/enwiki-latest-pages-articles.xml.bz2
	if [ ! -e $wiki_file ]; then
		echo "downloading wiki dump to $wiki_file"
		wget -O $wiki_file 'http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2'
	fi
	./prepare_shootout.py $wiki_file $datadir 2>&1 | tee ./log_prepare.txt
	gzip -v $datadir/lsi_vectors.mm
fi

run_combinations () {
	for method in $1; do
	    for acc in $2; do
	        for k in $ks; do
	            echo "running $method k=$k acc=$acc"
	            ./shootout_${method}.py $datadir $k $acc &> ./log_${method}_${k}_${acc}.txt
	        done
	    done
	done
}

# then create indexes for the various libraries & take accuracy measurements
ks="1 10 100 1000"
OPENBLAS_NUM_THREADS=1 run_combinations "gensim" "exact" $ks
run_combinations "annoy" "1 10 50 100 500" $ks
run_combinations "flann" "7 9 99" $ks
run_combinations "kgraph" "default" $ks
