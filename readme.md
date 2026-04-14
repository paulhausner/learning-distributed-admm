# Learning to accelerate distributed ADMM using graph neural networks

This repository contains the code for the paper [Learning to accelerate distributed ADMM using graph neural networks](https://arxiv.org/abs/2509.05288) by [Henri Doerks](https://www.uu.se/en/contact-and-organisation/staff?query=N24-2206), [Paul Häusner](https://paulhausner.github.io/), [Daniel Hernández Escobar](https://www.uu.se/kontakt-och-organisation/personal?query=N23-2641), and [Jens Sjölund](https://jsjol.github.io/).

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

```
@inproceedings{doerks2026learning,
  title={Learning to accelerate distributed {ADMM} using graph neural networks},
  author={Henri Doerks and Paul H{\"a}usner and Daniel Hern{\'a}ndez Escobar and Jens Sj{\"o}lund},
  booktitle={8th Annual Learning for Dynamics and Control Conference},
  year={2026},
  url={https://openreview.net/forum?id=VK0NIqNqzE}
}
```

## Contact

If you have any questions or comments please reach out to henri.doerks@math.uu.se and paul.hausner@it.uu.se
