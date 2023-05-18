#!/bin/bash
dataset="adult"
version=$1

python src/train.py experiment=$dataset/$version trainer.max_epochs=100
