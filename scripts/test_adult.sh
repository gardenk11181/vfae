#!/bin/bash
dataset="adult"
version=$1

> logs/$dataset/$version.log

echo "$version" >> logs/$dataset/$version.log
python src/eval.py +experiment=$dataset/$version \
| egrep 'acc_y' >> logs/$dataset/$version.log
