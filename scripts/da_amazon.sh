#!/bin/bash
dataset="amazon"
domains="
books
dvd
electronics
kitchen
"

version=$1

> logs/$dataset/${version}_da.log

for source in $domains; do 
	for target in $domains; do
		if [ $source != $target ]
		then
			echo "----------------------------"
			echo "$source to $target" >> logs/$dataset/${version}_da.log
			python src/da.py +experiment=$dataset/$version \
			data.source=$source data.target=$target \
			domain=$source-$target \
			| egrep 'acc_y|lr_score|rf_score' >> logs/$dataset/${version}_da.log
		fi
	done
done
