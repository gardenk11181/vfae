#!/bin/bash
dataset="health"
version=$1

for i in 1 2 3 4 5; do 
	python src/train.py experiment=$dataset/$version \
	trainer.max_epochs=100 domain=$i
done
