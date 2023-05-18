#!/bin/bash
dataset="adult"
version=$1

rm -f logs/$dataset/$version.log

echo "remove $version"
rm -rf logs/$dataset/1/$version
rm -rf logs/tensorboard/$dataset/1/$version
