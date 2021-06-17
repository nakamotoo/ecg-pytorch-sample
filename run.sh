#!/bin/bash
for outcome in AR
do
echo $outcome
python run.py Model_1DRes adam $outcome
python run.py Model_2D sgd $outcome
done