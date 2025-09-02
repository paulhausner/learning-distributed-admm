import pickle
import jax
import jax.numpy as jnp


def get_data(problem, train_samples=0, val_samples=0):
    
    # open problem sets from saved class
    with open(f'data/instances_{problem}.pkl', 'rb') as file:
        my_data = pickle.load(file)

    t_data_size = int(len(my_data)*9/10)
    
    train_samples = t_data_size if train_samples == 0 else train_samples
    val_samples = len(my_data) if val_samples == 0 else t_data_size + val_samples
    
    train_data = my_data[:train_samples]
    val_data = my_data[t_data_size:val_samples]
    
    return train_data, val_data


def get_test_data(problem):
    with open(f'data/instances_{problem}_test.pkl', 'rb') as file:
        my_data = pickle.load(file)
        
    return my_data
    

def symmetrize_edge_weights_sum(graph, edges=None):
    
    senders, receivers = graph.senders, graph.receivers
    if edges is None:
        edges = graph.edges

    # Get undirected representation (so (i,j) and (j,i) collapse to one)
    undirected_pairs = jnp.stack([jnp.minimum(senders, receivers),
                                  jnp.maximum(senders, receivers)], axis=1)

    # Unique undirected edges and mapping back to directed ones
    unique_pairs, inv_idx = jnp.unique(undirected_pairs, axis=0, return_inverse=True)

    # Sum edge features for both directions
    # shape: (num_unique, feature_dim)
    summed = jax.ops.segment_sum(edges, inv_idx, num_segments=unique_pairs.shape[0])

    # Assign summed values back to both directed edges
    sym_edges = summed[inv_idx]

    return graph._replace(edges=sym_edges)


def local_degree_profile(graph):
    # Number of nodes (assuming single graph)
    num_nodes = graph.n_node[0]

    # Compute node degrees
    degree = jnp.zeros(num_nodes)
    degree = degree.at[graph.senders].add(1)

    # Initialize accumulators for neighbor degree stats
    # We'll track sum, min, max, and count of neighbor degrees
    neighbor_sum = jnp.zeros(num_nodes)
    neighbor_sq_sum = jnp.zeros(num_nodes)   # for variance
    neighbor_min = jnp.full((num_nodes,), jnp.inf)
    neighbor_max = jnp.full((num_nodes,), -jnp.inf)
    neighbor_count = jnp.zeros(num_nodes)

    # For each edge (u -> v), v gets info about u's degree
    for u, v in zip(graph.senders, graph.receivers):
        d = degree[u]
        neighbor_sum = neighbor_sum.at[v].add(d)
        neighbor_sq_sum = neighbor_sq_sum.at[v].add(d * d)
        neighbor_min = neighbor_min.at[v].min(d)
        neighbor_max = neighbor_max.at[v].max(d)
        neighbor_count = neighbor_count.at[v].add(1)

    # Avoid division by zero for isolated nodes
    neighbor_mean = jnp.where(neighbor_count > 0,
                              neighbor_sum / neighbor_count,
                              0.0)

    # Variance = E[d^2] - (E[d])^2
    neighbor_var = jnp.where(
        neighbor_count > 0,
        neighbor_sq_sum / neighbor_count - neighbor_mean ** 2,
        0.0
    )
    neighbor_std = jnp.sqrt(jnp.maximum(neighbor_var, 0.0))  # avoid negatives
    
    # Replace infinities in min/max for isolated nodes
    neighbor_min = jnp.where(neighbor_count > 0, neighbor_min, 0.0)
    neighbor_max = jnp.where(neighbor_count > 0, neighbor_max, 0.0)

    # Concatenate degree profile: [degree, min, max, mean]
    updated_node_features = jnp.stack(
        [degree, neighbor_min, neighbor_max, neighbor_mean, neighbor_std], axis=1
    )

    return updated_node_features


def normalize_per_sample(x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)
