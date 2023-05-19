#!/bin/bash
dataset="german"
version=$1

rm -f logs/$dataset/$version.log

for domain in 1 2 3 4 5; do
	echo "remove ${domain}th dataset of ${version} version model"
	rm -rf logs/$dataset/$domain/$version
	rm -rf logs/tensorboard/$dataset/$domain/$version
done
