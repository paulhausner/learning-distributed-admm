import jax.numpy as jnp
import numpy as np
import scipy.stats as scs

import pickle
import argparse

from admm.gnn import ADMM_GNN
from admm.graph import sample_connected_graph, convert_gnn_graph

from flax import nnx


def sample_square_matrix(n, eps=1e-1):
    # sample B (n x n) matrix
    while(True):
        B=1*np.random.rand(n,n)
        B=jnp.array(B)

        #check that ev of B are not to close to 0
        ev, _ = jnp.linalg.eig(B)      #maybe better B^TB directly?
        if (jnp.all(jnp.abs(ev)>eps)):
            break
    
    return B


def network_consensus_problem(n=2, m=8):
    
    bi=10*scs.norm.rvs(size=m*n)
    bi=bi.reshape((m,n))
    bi=jnp.array(bi)
    
    sol = jnp.mean(bi, axis=0, keepdims=True)
    
    return [{"fi": [bi]}, None, sol]
    

def least_squares_problem_1(n=2, m=8):
    
    def unique_solution(B, bi):
        # ! observe that this might not be correct
        m=bi.shape[0]
        rhs=1/m*B.T@jnp.sum(bi,axis=0)
        result=jnp.linalg.solve(B.T@B,rhs)
        
        return result
    
    B = sample_square_matrix(n)
    
    # sample bi's (n x m)
    bi = 10*scs.norm.rvs(size=m*n)
    bi = bi.reshape((m,n))
    bi = jnp.array(bi)
    
    # compute solution
    sol = unique_solution(B, bi)
    
    return [{"fi": [bi]}, {"B": B}, sol]


def least_squares_problem_2(n=2, m=8):
    
    # generate matrices Bi
    Bi = jnp.stack([sample_square_matrix(n) for _ in range(m)])
    
    # sample linear terms
    bi = 10*scs.norm.rvs(size=m*n)
    bi = bi.reshape((m,n))
    bi = jnp.array(bi)
    
    # compute the solution
    rhs = jnp.sum(jnp.einsum("bij,bj->bi", Bi.transpose(0, 2, 1), bi), axis=0)
    B = jnp.sum(jnp.matmul(Bi.transpose(0, 2, 1), Bi), axis=0)
    sol = jnp.linalg.solve(B, rhs)
    
    return [{"fi": [bi, Bi]}, None, sol]


#########################################
######  save instances generated   ######
#########################################
def save_instance(n=2, m_min=8, m_max=8, p=0.4, num_graphs=1_000, problem="least_squares_1", test=False):
    
    # save the instances in a list
    instances_s = []

    init = 1 if test else 0
    
    # before instance creation, make sure that steps are deterministic
    # (adapt stepsize to learned stepsize!!)
    network = ADMM_GNN(10, rngs=nnx.Rngs(init), learned_steps=False, problem=problem)
    
    for i in range (num_graphs):
        
        # sample a number of agents
        m = np.random.randint(m_min, m_max+1)
        
        G = sample_connected_graph(m, p)
        
        # generate problem from specific problem class
        if problem == "least_squares_1":
            node, graph, solution = least_squares_problem_1(n, m)
        
        elif problem == "least_squares_2":
            node, graph, solution = least_squares_problem_2(n, m)
        
        elif problem == "consensus":
            node, graph, solution = network_consensus_problem(n, m)
        
        # transform to graph
        G_tup = convert_gnn_graph(G, (m, n), node, graph)

        # maybe different method than just taking mean? -> consensus measure
        G_out_naive, _ = network(G_tup)
        # naive_sol = jnp.mean(G_out_naive.nodes["x"], axis=0)
        naive_sol = G_out_naive.nodes["x"]
        
        instances_s.append((G_tup, solution, naive_sol))
    
    # saving instances
    problem = problem + "_test" if test else problem
    with open(f'data/instances_{problem}.pkl', 'wb') as f:
        pickle.dump(instances_s, f)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="least_squares_1")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--test", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()

    # use to create instance set once:
    save_instance(problem=args.problem, num_graphs=args.samples, test=args.test)
