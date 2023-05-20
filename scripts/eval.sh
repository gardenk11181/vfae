#!/bin/bash
dataset=$1
version=$2

if [ $dataset != 'amazon' ]
then
	domain=1
	> logs/$dataset/${version}.log

	metric='acc_y|lr_score|rf_score|disc|disc_prob|lr_y_score'

	echo "adult dataset" >> logs/$dataset/${version}.log
	python src/fair_rep.py +experiment=$dataset/$version domain=$domain \
	| egrep $metric >> logs/$dataset/${version}.log
else
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
				python src/domain_adapt.py +experiment=$dataset/$version \
				data.source=$source data.target=$target \
				domain=$source-$target \
				| egrep 'acc_y|lr_score|rf_score' >> logs/$dataset/${version}_da.log
			fi
		done
	done
fi
