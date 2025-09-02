#!/bin/bash

python instance_generator.py --problem least_squares_2
python instance_generator.py --problem consensus

python instance_generator.py --problem least_squares_2 --test
python instance_generator.py --problem consensus --test
