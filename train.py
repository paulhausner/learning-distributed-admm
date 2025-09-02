import jax
import optax
import time
import argparse

import numpy as np
import jax.numpy as jnp

import orbax.checkpoint as ocp
from orbax.checkpoint import CheckpointManager

from flax import nnx

from admm.gnn import ADMM_GNN
from admm.utils import get_data

import wandb
import os


# set some flags for XLA
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

# initalize some logging (optimal)
run = wandb.init(project="ADMM")


def compute_loss(model, g, sol, naive_sol, loss="l2"):
    
    def loss_fn(model):
        # pass to neural network
        # sol, output = model(g, L, B)
        output, _ = model(g)
        
        # err = jnp.linalg.norm(sol - output.nodes["x"])
        
        m = output.nodes["x"].shape[0]
        
        if loss == "l2":
            
            # take the l2 norm of the error
            err = jnp.sum(jnp.square(sol - output.nodes["x"])) / m
        
        elif loss == "normalized":
            
            # divide by the norm of the solution
            err = jnp.sum(jnp.square(sol - output.nodes["x"]))
            err = err / (m * max(jnp.linalg.norm(sol), 1e-6))

        elif loss == "normalized_v2":
            
            naive_sol_mean = jnp.mean(naive_sol, axis=0)
            
            # divide by the norm of the naive error
            err = jnp.sum(jnp.square(sol - output.nodes["x"]))
            err = err / (m * max(jnp.sum(jnp.square(sol - naive_sol_mean)), 1e-6))
        
        elif loss == "normalized_v3":
            res = jnp.sum(jnp.square(sol - output.nodes["x"]))
            nor = max(jnp.sum(jnp.square(sol - naive_sol)), 1e-6)
            err = res / nor
            
        elif loss == "geometric":
            # take the geometric mean of the error
            err = jnp.square(sol - output.nodes["x"])
            err = jnp.sum(jnp.log(err + 1e-6))
        
        elif loss == "function":
            # directly minimize the objective 
            err = model.f(g, iterate=output.nodes["x"])
        
        elif loss == "function_normalized":
            err = model.f(g, iterate=output.nodes["x"]) / model.f(g, iterate=naive_sol)
        
        return err

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    
    return loss, grads
    
    
def finite_difference(model, g):
    alpha = 1
    epsilon = 0.01
    
    # ! finite difference approximation
    # f(x + eps)
    # model.learned_steps[1].network = alpha + epsilon
    model.learned_steps[1].network.layers[0].kernel[0, 0] += epsilon
    
    sol1, output1 = model(g)
    err1 = jnp.linalg.norm(1 / sol1.shape[0] * (sol1 - output1.nodes["x"]))
    
    # f(eps - eps)
    # model.learned_steps[1].alpha = alpha - epsilon
    model.learned_steps[1].network.layers[0].kernel[0, 0] -= epsilon
    sol2, output2 = model(g)
    err2 = jnp.linalg.norm(1 / sol2.shape[0] * (sol2 - output2.nodes["x"]))
    
    print("Finite difference", (err1 - err2) / (2 * epsilon))
    
    # optimizer.update(grads)  # inplace updates
    # return loss


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--iterations", type=int, default=10)
    
    parser.add_argument("--problem", type=str, default="least_squares_1")
    parser.add_argument("--loss_fn", type=str, default="normalized_v3")
    
    # solver parameters
    parser.add_argument("--iterative", action="store_true", default=False)
    
    # step size parameters
    parser.add_argument("--learnalpha", action="store_true", default=False)
    parser.add_argument("--meanalpha", action="store_true", default=False)
    
    parser.add_argument("--shared", action="store_true", default=False)
    
    # edges
    parser.add_argument("--learnedges", action="store_true", default=False)
    
    # others
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--name", type=str, default="")
    
    return parser.parse_args()


def train():

    args = parse_args()
    
    assert (args.learnalpha or args.meanalpha or args.learnedges), "must include a learned component..."
    
    # reandom seeding
    seed = 4
    rng = jax.random.key(seed)
    rng, init_rng = jax.random.split(rng)
    
    # initalize the network
    num_iters = args.iterations
    network = ADMM_GNN(num_iters, rngs=nnx.Rngs(0), learned_steps=(args.learnalpha or args.meanalpha),
                       learned_edges=args.learnedges, shared=args.shared, problem=args.problem,
                       iterative=args.iterative, meanalpha=args.meanalpha)

    # count model parameters
    params = nnx.state(network, nnx.Param)
    total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print(f"Total number of parameters {total_params}")
    
    # initialize the optimizer
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=1e-4)
    )

    optimizer = nnx.Optimizer(network, opt)
    
    # 100 update steps per epoch
    num_epochs = 100
    batch_size = 5
    
    if args.debug:
        train_data, val_data = get_data(args.problem, train_samples=5, val_samples=2)
        num_epochs = 2
        
        print("WARNING: using debug mode...")
        print()
    
    else:
        train_data, val_data = get_data(args.problem)
        
    
    train_samples = len(train_data)
    val_samples = len(val_data)
    
    start = time.time()

    run.config.update({
        "problem": args.problem,
        "iterations": num_iters,
        "iterative": args.iterative,
        "mean_alpha": args.meanalpha,
        "learned_alpha": args.learnalpha,
        "shared_weights": args.shared,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "loss_fn": args.loss_fn,
    })

    steps = 0
    
    for epoch in range(num_epochs):
        
        acc_loss = 0
        batch_loss = 0
        grads_accum = None
        
        indices = np.random.permutation(train_samples)
        
        for instance_id in indices:
            
            steps += 1
            
            [g, sol, naive_sol] = train_data[instance_id]
            loss, grad = compute_loss(network, g, sol, naive_sol, loss=args.loss_fn)
            
            # log the loss
            acc_loss += jnp.sqrt(loss)
            batch_loss += jnp.sqrt(loss)
            
            # dummy fallback for nan losses...
            if jnp.isnan(loss):
                print("loss is nan, leading to problems?")
                grads_accum = None
                break
            
            # accumulate gradient
            if grads_accum is None:
                grads_accum = grad
            else:
                grads_accum = jax.tree_util.tree_map(lambda g1, g2: g1 + g2, grads_accum, grad)
            
            # do update
            if steps % batch_size == 0:
                grads_accum = jax.tree_util.tree_map(lambda g: g / batch_size, grads_accum)
                
                optimizer.update(grads_accum)  # in-place updates
                grads_accum = None # reset gradient accumulator
                
                # log the batched loss
                run.log({"loss": batch_loss / batch_size, "step": steps // batch_size})
                batch_loss = 0
        
        print("Epoch time:", time.time() - start, f"\tLoss for epoch {epoch}: {acc_loss / train_samples:.4f}\t")
        
        start = time.time()
        
        # run validation after every epoch
        acc_loss = 0
        
        for instance_id in range(val_samples):
            [g, sol, naive_sol] = val_data[instance_id]
            loss, _ = compute_loss(network, g, sol, naive_sol, loss=args.loss_fn)
            acc_loss += jnp.sqrt(loss)

        run.log({"val_loss": acc_loss / val_samples, "epoch": epoch})
        print("Average validation loss:", acc_loss / val_samples)
        print()
    
    
    # trying to save the model
    ckpt_dir = f'/home/pauha615/learning_distributed_admm/model/{args.problem}_{args.name}/'
    options = ocp.CheckpointManagerOptions(
        max_to_keep=1,
        keep_time_interval=None,
        create=True
    )
    checkpoint_manager = CheckpointManager(ckpt_dir, options=options)
    
    state = nnx.state(network)
    checkpoint_manager.save(0, args=ocp.args.StandardSave(state), metrics={})
    
    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()


if __name__ == "__main__":
    train()
