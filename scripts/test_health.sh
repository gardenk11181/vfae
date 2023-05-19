#!/bin/bash
dataset="health"
version=$1

> logs/$dataset/$version.log

for i in 1 2 3 4 5; do 
	echo "${i}th dataset" >> logs/$dataset/$version.log
	python src/eval.py +experiment=$dataset/$version \
	domain=$i | egrep 'acc_y' >> logs/$dataset/$version.log
done
