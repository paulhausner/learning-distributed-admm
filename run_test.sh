#!/bin/bash

# test consensus problem
python test.py --problem consensus --name baseline
python test.py --problem consensus --learnalpha --name alpha
python test.py --problem consensus --meanalpha --name mean
python test.py --problem consensus --learnedges --name edges
python test.py --problem consensus --learnalpha --learnedges --name combined

# test least squares problem 
python test.py --problem least_squares_2 --name baseline
python test.py --problem least_squares_2 --learnalpha --name alpha
python test.py --problem least_squares_2 --meanalpha --name mean
python test.py --problem least_squares_2 --learnedges --name edges
python test.py --problem least_squares_2 --learnalpha --learnedges --name combined
