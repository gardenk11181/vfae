#!/bin/bash
dataset=$1
version=$2

if [ $dataset != 'amazon' ]
then
	domain=1
	python src/train.py experiment=$dataset/$version trainer.max_epochs=100 domain=$domain
else
	domains="
	books
	dvd
	electronics
	kitchen
	"

	for source in $domains; do 
		for target in $domains; do
			if [ $source != $target ]
			then
				python src/train.py experiment=$dataset/$version \
				data.source=$source data.target=$target trainer.max_epochs=100 \
				domain=$source-$target 
			fi
		done
	done
fi
