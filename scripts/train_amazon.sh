#!/bin/bash
dataset="amazon"
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
			python src/train.py experiment=$dataset/default \
			data.source=$source data.target=$target trainer.max_epochs=100 \
			domain=$source-$target 
		fi
	done
done
