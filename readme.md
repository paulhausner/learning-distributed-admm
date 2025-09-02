# Learning to accelerate distributed ADMM using graph neural networks

This repository contains the code for the paper **Learning to accelerate distributed ADMM using graph neural networks** by Henri Doerks, Paul Häusner, Daniel Hernández Escobar, and Jens Sjölund.

## Dependencies

The experiments have been executed with the following pacakges:

- python 12.9
- jax with support for CUDA 12
- flax with linen API 0.11
- networkx
- jaxopt
- jraph

Optional packages for plotting and logging

- ipdb
- wandb
- matplotlib

## Data generation

In order to generate the problem instances for training and testing run

 ```bash
./generate_data.sh
```

## Experiments

To train and test the models run

```bash
./run_training.sh
./run_test.sh
```

## Reference

coming soon

## Contact

If you have any questions or comments please reach out to henri.doerks@math.uu.se and paul.hausner@it.uu.se
