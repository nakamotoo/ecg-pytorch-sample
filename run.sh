#!/bin/bash
for outcome in MR
do
echo $outcome
python run.py Model_1DRes adam $outcome
python run.py Model_2D sgd $outcome
done