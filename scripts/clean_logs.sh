#!/bin/bash
dataset=$1
version=$2

echo "remove log files for $version model of $dataset"

if [ dataset != 'amazon' ]
then 

	rm -f logs/$dataset/${version}.log
	rm -rf logs/$dataset/1/$version
	rm -rf logs/tensorboard/$dataset/1/$version
else
	domains="
	books
	dvd
	electronics
	kitchen
	"

	rm -f logs/$dataset/$version.log
	rm -f logs/$dataset/${version}_da.log

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
fi
