import argparse
import os

from admm.gnn import ADMM_GNN
from admm.utils import get_test_data

import jax.numpy as jnp
from flax import nnx

import jax

import numpy as np

import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager

import ipdb


def argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--problem", type=str, default="least_squares_1")
    parser.add_argument("--iterations", type=int, default=10)
    
    # solver parameters
    parser.add_argument("--iterative", action="store_true", default=False)
    
    # step size parameters
    parser.add_argument("--learnalpha", action="store_true", default=False)
    parser.add_argument("--meanalpha", action="store_true", default=False)
    
    parser.add_argument("--shared", action="store_true", default=False)
    
    # edges
    parser.add_argument("--learnedges", action="store_true", default=False)
    
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--steps", type=int, default=20)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()

    os.makedirs("./results", exist_ok=True)
    
    network = ADMM_GNN(args.iterations, rngs=nnx.Rngs(0), learned_steps=(args.learnalpha or args.meanalpha),
                       learned_edges=args.learnedges, shared=args.shared, problem=args.problem,
                       iterative=args.iterative, meanalpha=args.meanalpha, test=True, epsilon=10e-5)
    
    if args.learnalpha or args.meanalpha or args.learnedges:
        
        # load checkpoint
        ckpt_dir = f'/home/pauha615/learning_distributed_admm/model/{args.problem}_{args.name}/'
        options = ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_time_interval=None,
            create=True
        )
        checkpoint_manager = CheckpointManager(ckpt_dir, options=options)
        latest_step = checkpoint_manager.latest_step()
        
        state = nnx.state(network)
        
        restored_state = checkpoint_manager.restore(
            latest_step,
            args=ocp.args.StandardRestore(state)
        )
        
        # merge the two states
        nnx.update(network, restored_state)
    
    # get the data
    test_data = get_test_data(args.problem)
    
    # run some extra iterations for the network
    network.admm_steps = args.steps
    
    distance_to_sol = []
    node_consensus = []
    function_value = []
    residuals = []
    
    for i in range(len(test_data)):
        [g, sol, naive_sol] = test_data[i]

        # run the network
        output, iterates = network(g, sol)
        
        # compute metrics
        distance_to_sol.append([jnp.sum(jnp.square(iterate - sol)) / iterate.shape[0] for iterate in iterates])
        node_consensus.append([jnp.sum(jnp.square(iterate - jnp.mean(iterate, axis=0))) / iterate.shape[0] for iterate in iterates])
        
        expanded = sol[None, :]
        tiled = jnp.repeat(expanded, repeats=8, axis=0)
        f_star = network.f(g, iterate=tiled)
        
        function_value.append([np.abs(network.f(g, iterate=iterate) - f_star) / np.abs(f_star) for iterate in iterates])
    
    # save the metrics...
    np.savez(f"./results/{args.problem}_{args.name}.npz",
            distance=np.array(distance_to_sol),
            consensus=np.array(node_consensus),
            objective=np.array(function_value))
    