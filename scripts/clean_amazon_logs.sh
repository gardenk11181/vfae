#!/bin/bash
dataset="amazon"
domains="
books
dvd
electronics
kitchen
"

version=$1

rm -f logs/$dataset/$version.log

for source in $domains; do 
	for target in $domains; do
		if [ $source != $target ]
		then
			echo "remove $source to $target"
			rm -rf logs/$dataset/${source}-${target}/$version
			rm -rf logs/tensorboard/$dataset/${source}-${target}/$version
		fi
	done
done
