#!/bin/bash

# conda activate admm
wandb enabled

# different alpha vector training schemes (4 different ones...)
python train.py --problem consensus --learnalpha --name alpha
python train.py --problem consensus --meanalpha --name mean
python train.py --problem consensus --learnedges --name edges
python train.py --problem consensus --learnedges --meanalpha --name combined

python train.py --problem least_squares_2 --learnalpha --name alpha
python train.py --problem least_squares_2 --meanalpha --name mean
python train.py --problem least_squares_2 --learnedges --name edges
python train.py --problem least_squares_2 --learnedges --meanalpha --name combined
