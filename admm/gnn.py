from typing import Sequence

import jax
import jax.numpy as jnp
import jaxopt.linear_solve
import jraph
import jax.tree_util as tree

from flax import linen as nn
from flax import nnx

import jaxopt
import ipdb

from admm.utils import symmetrize_edge_weights_sum, local_degree_profile, normalize_per_sample


class ADMM_Layer(nnx.Module):
    
    # ! each ADMM layer consists of two message passing steps.
    def __init__(self, learned, rngs: nnx.Rngs, mlp_struct=None,
                 problem="least_squares_1", iterative=False, meanalpha=True):
        super().__init__()
        
        self.learned = learned # learn the current step
        self.mean_alpha = meanalpha # aggregate the alphas
        
        self.iterative = iterative # use iterative solver for x-update if closed form solution exists
        
        self.aggregation_function = jraph.segment_sum
        
        # the x-update depends on the problem class
        if problem == "least_squares_1":
            num_node_params = 1 # number of arrays describing f_i
            solve_x_subproblem = self.update_x_least_squares_1
        
        elif problem == "least_squares_2":
            num_node_params = 2 # here we both have the rhs and the matrix per node
            solve_x_subproblem = self.update_x_least_squares_2
        
        elif problem == "consensus":
            num_node_params = 1
            solve_x_subproblem = self.update_x_consensus
        
        else:
            raise NotImplementedError(f"the x-subproblem for {problem} is not implemented")
        
        # wrapper around the solver for parallelization to solve the subproblem at each node
        in_axes = tuple(0 for _ in range(8 + num_node_params)) + (None,)
        self.update_x = jax.vmap(solve_x_subproblem, in_axes=in_axes)
        
        if self.learned:
            # we need a neural network to predict the step-size
            # input layer needs to adapted to input size
            self.network = MLP([11, 32, 1], activation="softplus", rngs=nnx.Rngs(0), input_norm=True)
    
    # line 4 in Algorithm 3
    def update_edge_1(self, edges, sender):
        # how to include A_ij, possible to access index of receiver + sender node?
        # at every edge we take the first 3n features as messages...
        efs = {"lam": - edges * sender["lam"], "y": - edges * sender["y"],
                "deg": edges, "deg_sq": jnp.pow(edges, 2)}
        
        return efs
    
    # line 13 in Algorithm 3
    def update_edge_2(self, edges, sender):
        efs = {"x": - edges * sender["x"], "deg": jnp.ones_like(edges)}
        return efs
    
    # (y, lam)-update
    def update_y_lam(self, alpha, deg, w_deg, node_x, nodes_lam, messages_x):
        
        acc =  messages_x + (w_deg[:, None] * node_x)
        y_new = (1. / (1 + deg))[:, None] * acc
        
        # alpha = nodes["alpha_vec"]
        lam_new = nodes_lam + alpha * y_new
        
        return y_new, lam_new
    
    # x-update: least squares 1
    def update_x_least_squares_1(self, alpha, x, y, y_agg, lam, lam_agg, degree, sq_degree, bi, globals_):
        
        # ! only weighted degrees here...
        
        # extract the problem information
        B = globals_["B"]
        
        # construct the right-hand side
        Mii = (degree**2 + sq_degree) * jnp.identity(B.shape[0])
        Aii = degree * jnp.identity(B.shape[0])
        rhs = 2 * B.T@bi - lam_agg - Aii @ lam + alpha * (Mii @ x - y_agg - Aii @ y)
        
        if self.iterative:
            def matvec(u):
                return 2 * jnp.dot(B.T, jnp.dot(B, u)) + alpha * jnp.dot(Mii, u)
            
            x_new = jaxopt.linear_solve.solve_cg(matvec=matvec, b=rhs, ridge=0, init=x, maxiter=10)
        
        else:
            B_inv = jnp.linalg.inv(2*B.T@B + alpha * Mii)
            x_new = B_inv @ rhs
        
        return x_new
    
    # x-update: least-squares 2
    def update_x_least_squares_2(self, alpha, x, y, y_agg, lam, lam_agg, degree, sq_degree, bi, Bi, globals_):
        
        # construct the right-hand side
        Mii = (degree**2 + sq_degree) * jnp.identity(Bi.shape[0])
        Aii = degree * jnp.identity(Bi.shape[0])
        rhs = 2 * Bi.T@bi - lam_agg - Aii @ lam + alpha * (Mii @ x - y_agg - Aii @ y)
        
        if self.iterative:
            def matvec(u):
                return 2 * jnp.dot(Bi.T, jnp.dot(Bi, u)) + alpha * jnp.dot(Mii, u)
            
            x_new = jaxopt.linear_solve.solve_cg(matvec=matvec, b=rhs, ridge=0, init=x, maxiter=10)
        
        else:
            B_inv = jnp.linalg.inv(2*Bi.T@Bi + alpha * Mii)
            x_new = B_inv @ rhs
        
        return x_new
    
    # x-update: consensus
    def update_x_consensus(self, alpha, x, y, y_agg, lam, lam_agg, degree, sq_degree, bi, globals_):
        
        assert not self.iterative, "consens problem does not work with iterative solvers"
        
        mii = (degree**2 + sq_degree)
        rhs = 2 * bi - lam_agg - degree * lam + alpha * (mii * x - y_agg - degree * y)
        
        x_new = rhs / (2 + alpha * mii)
        
        return x_new
    
    # run two message passing steps
    def __call__(self, g):
        
        # hyper-parameters
        # deg = jnp.diag(A)
        
        # extract information from the graph
        nodes, edges, receivers, senders, globals_, n_node, n_edge = g
        
        sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
        # sum_n_edge = senders.shape[0]
        
        # compute edge embedding (messages)
        sent_attributes = tree.tree_map(lambda n: n[senders], nodes)
        message = self.update_edge_1(edges, sent_attributes)
        
        # aggregate the messages
        received_attributes = tree.tree_map(lambda e: self.aggregation_function(e, receivers, sum_n_node), message)
        
        # we can extract the degree from the messages
        w_deg = received_attributes["deg"].squeeze()
        deg_sq = received_attributes["deg_sq"].squeeze()
        
        if self.learned:
            
            inp = jnp.stack([nodes["x"],
                             nodes["y"],
                             received_attributes["y"],
                             nodes["lam"],
                             received_attributes["lam"]])
            
            inp = jnp.transpose(inp, (1, 0, 2)).reshape(nodes["x"].shape[0], -1)
            inp = jnp.concatenate([inp, w_deg[:, jnp.newaxis]], axis=1)
            
            # ! for batching compute the mean only over the graphs
            # probably easier to work with a alpha vector here instead (node_size)
            alpha = self.network(inp)
            
            if self.mean_alpha:
                alpha_vec = jnp.mean(alpha) * jnp.ones_like(nodes["x"])
            
            else:
                alpha_vec = jnp.repeat(alpha, repeats=2, axis=1)
            
        else:
            alpha_vec = jnp.ones_like(nodes["x"])
        
        # solve the x-update
        new_x = self.update_x(alpha_vec,
                              nodes["x"],
                              nodes["y"], received_attributes["y"],
                              nodes["lam"], received_attributes["lam"],
                              w_deg,
                              deg_sq,
                              *nodes["fi"], # this contains the node specific objective
                              globals_)     # this contains more information on the objective
        
        # start of the second step
        # second message passing step (always deterministic)
        
        # compute the edge embeddings (messages)
        sent_attributes = tree.tree_map(lambda n: n[senders], {"x": new_x})
        message = self.update_edge_2(edges, sent_attributes)
        
        # aggregate the messages
        received_attributes = tree.tree_map(lambda e: self.aggregation_function(e, receivers, sum_n_node), message)
        
        # node update 2 (update y and lambda)
        # line 17 in Algorithm 3
        
        deg = received_attributes["deg"].squeeze()
        
        # node_x, nodes_lam, messages_x
        new_y, new_lam = self.update_y_lam(alpha_vec, deg, w_deg, new_x, nodes["lam"], received_attributes["x"])
        
        # generate new output graph data
        node_values = {"x": new_x, "y": new_y, "lam": new_lam,
                       "fi": nodes["fi"]}
        
        graph = jraph.GraphsTuple(
            nodes=node_values,
            edges=edges,
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=n_edge,
            globals=globals_
        )
        
        return graph
        

class ADMM_GNN(nnx.Module):
    
    def __init__(self, admm_steps: int, rngs: nnx.Rngs,
                 problem="least_squares_1",
                 learned_steps=True, # use learning for some iterations 
                 skip_first=True,     # use default step size for the first step
                 learned_edges=False,# use learning for the edge connectivity 
                 shared=False,       # share the parameters accross iterations 
                 test=False,         # log test metrics 
                 epsilon=10e-2,      # stopping criterion 
                 **kwargs
                 ):
        
        super().__init__()
        
        self.admm_steps = admm_steps  # number of admm steps to do
        
        # ! currently only deterministic steps
        # ?? maybe we can make a list with indices of ADMM steps that are learned
        
        # default values (will be overwritten)
        self.learned_step_index = []
        self.edge_network = None
        
        if learned_steps:
            start = 1 if skip_first else 0
            self.learned_step_index = list(range(start, admm_steps))
            
        if learned_edges:
            # input consists of two local degree profiles for the nodes
            self.edge_network = MLP([10, 32, 1], "softplus", rngs)
        
        self.problem = problem
        self.shared = shared
        self.test = test
        
        # moderate convergence
        self.epsilon = epsilon
        self.deterministic_step = ADMM_Layer(learned=False, rngs=rngs, problem=problem, **kwargs)
        
        if shared:
            self.learned_steps = ADMM_Layer(learned=True, rngs=rngs, problem=problem, **kwargs)
        
        else:
            self.learned_steps = {}
            for i in self.learned_step_index:
                self.learned_steps[i] = ADMM_Layer(learned=True, rngs=rngs, problem=problem, **kwargs)
    
    def f(self, g, iterate=None):
        
        # choose default iterate if not given
        if iterate is None:
            iterate = g.nodes["x"]
        
        # compute function value for each problem class
        if self.problem == "consensus":
            bi = g.nodes["fi"][0]
            obj = jnp.sum(jnp.square(iterate - bi))
        
        elif self.problem == "least_squares_1":
            pass
        
        elif self.problem == "least_squares_2":
            b = g.nodes["fi"][0]
            Bi = g.nodes["fi"][1]
            
            C = jnp.einsum('bij,bi->bj', Bi, iterate)
            obj = jnp.sum(jnp.square(C - b))
        
        return obj
    
    def get_step(self, step):
        if self.shared:
            return self.learned_steps
        elif step not in self.learned_step_index:
            return self.deterministic_step
        else:
            return self.learned_steps[step]
    
    def warm_start(self, g):
        
        if self.warm_starter is not None:
            
            nodes, edges, receivers, senders, globals_, n_node, n_edge = g
            sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
            
            # intialize nodes
            node_init = jnp.ones_like(nodes["x"])
            
            # communicate to neighbours
            sent_attributes = tree.tree_map(lambda n: n[senders], {"x": node_init})
            message = {"x": - edges * sent_attributes["x"], "deg": 1, "w_deg": edges}
            
            # aggregate the messages
            received_attributes = tree.tree_map(lambda e: jraph.segment_sum(e, receivers, sum_n_node), message)
            deg, w_deg = received_attributes["deg"].squeeze(), received_attributes["w_deg"].squeeze()
            x_acc = received_attributes["x"]
            
            # initialize y
            y_init = 1 / (deg[:, None] + 1) * (x_acc + w_deg[:, None] * node_init)
            
            # obs: directly uses in place updates
            nodes["x"] = node_init
            nodes["y"] = y_init
    
    def __call__(self, g, sol=None):
        
        # save the first iterate in the first step
        iterates = [g.nodes["x"].copy()] if self.test else None
        
        if self.edge_network is not None:
            
            # compute local degree profile
            degree_profile = local_degree_profile(g)
            
            send_feats = degree_profile[g.senders] 
            recv_feats = degree_profile[g.receivers]
            
            features = jnp.concatenate([send_feats, recv_feats], axis=1)
            
            # compute the predictions
            new_edges = 0.5 * self.edge_network(features)
            
            # use the new edges and symmetrize...
            g = symmetrize_edge_weights_sum(g, edges=new_edges)
        
        for i in range(self.admm_steps):
            
            # check for cnovergence
            if sol is not None:
                # just check if the first one converges
                err = jnp.sum(jnp.square(g.nodes["x"] - sol)) / g.nodes["x"].shape[0]
                
                if err < self.epsilon:
                    print(f"Converged after {i} ADMM steps")
                    break
            
            # compute the new iterates
            g = self.get_step(i)(g)
            
            # save the iterates during testing
            if iterates is not None:
                iterates.append(g.nodes["x"].copy())
        
        return g, iterates


class MLP(nnx.Module):
    
    def __init__(self, feature_sizes, activation, rngs: nnx.Rngs, input_norm=False):
        
        super().__init__()
        
        self.layers = []
        self.normalizations = []
        
        self.input_norm = input_norm
        
        for j in range(len(feature_sizes)-2):
            # self.normalizations.append(nnx.LayerNorm(feature_sizes[j], rngs=rngs))
            self.layers.append(nnx.Linear(feature_sizes[j], feature_sizes[j+1], rngs=rngs))
        
        self.linear_out = nnx.Linear(feature_sizes[-2], feature_sizes[-1], rngs=rngs)
        
        self.scale = 1.0
        self.activation = activation
    
    def __call__(self, x):
        
        if self.input_norm:
            x = normalize_per_sample(x)
        
        for i in range(len(self.layers)):
            x = nnx.relu(self.layers[i](x))
        
        x = self.linear_out(x)
        
        # make sure that the output is positive
        if self.activation == "softplus":
            # try using the softplus function instead
            x = jax.nn.softplus(x)
        
        elif self.activation == "exponential":
            x = jnp.exp(x)
        
        elif self.activation == "sigmoid":
            x = self.scale * nnx.sigmoid(x)
        
        elif self.activation == "relu":
            x = nnx.relu(x)
        
        return x
