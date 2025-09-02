import networkx as nx
import jax.numpy as jnp

import jraph


def sample_connected_graph(m=8, p=0.4):
    while(True):
        
        G=nx.erdos_renyi_graph(m, p)
        
        if (nx.is_connected(G)):
            break

    return G


def convert_gnn_graph(G, shape, local_data, global_data):
    # converts a networkx graph into a Graphtuple object for Jraph 
    # with three n-dim iterates (x,y,lam) and both directed edges
    
    m, n = shape
    
    # initialize admm variables
    x_vec=jnp.zeros((m,n))
    y_vec=jnp.zeros((m,n))
    lam_vec=jnp.zeros((m,n))
    
    node_init = {
        "x": x_vec,
        "y": y_vec,
        "lam": lam_vec,
    }
    
    # add local objective information to the node features
    node_init = node_init | local_data
    
    # init anpassen according to algo, schauen ob warum zeros auch geht
    edges_G=jnp.array(G.edges)
    senders = jnp.concatenate((edges_G.at[:,0].get(),edges_G.at[:,1].get()))
    receivers =jnp.concatenate((edges_G.at[:,1].get(),edges_G.at[:,0].get()))
    
    #edge_features=np.zeros((2*len(G.edges),n))
    edge_init=jnp.ones((2*len(G.edges), 1))
    
    n_node=jnp.array([len(G.nodes)])
    n_edge=jnp.array([2*len(G.edges)])
    
    graph = jraph.GraphsTuple(
        nodes=node_init,
        edges=edge_init,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=global_data
    )

    return graph
