#!/bin/bash
dataset="amazon"
domains="
books
dvd
electronics
kitchen
"

version=$1

> logs/$dataset/$version.log

for source in $domains; do 
	for target in $domains; do
		if [ $source != $target ]
		then
			echo "----------------------------"
			echo "$source to $target" >> logs/$dataset/$version.log
			python src/eval.py +experiment=$dataset/$version \
			data.source=$source data.target=$target \
			domain=$source-$target \
			| egrep 'acc_y' >> logs/$dataset/$version.log
		fi
	done
done
