#!/bin/bash
dataset="amazon"
domains="
books
dvd
electronics
kitchen
"

version=$1

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
