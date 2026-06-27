import numpy as np
import networkx as nx
import ot
import random

from EDRep_main.EDRep import NodeEmbedding


def node_embedding(graph_a, partition_A, dim_embedding,seed=42):
    g_a = nx.to_scipy_sparse_array(graph_a, format='csr')
    # save the seed state
    rdn_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    embedding_a = NodeEmbedding(g_a, dim=dim_embedding, k=1, verbose=False)
    X_total = embedding_a.X
    X_list = [X_total[list(nodes), :] for nodes in partition_A]
    np.random.set_state(rdn_state)
    return X_list


def block_dists(X_list):
    m = len(X_list)
    blocks = {}
    for i in range(m):
        for j in range(i, m):
            A = X_list[i] @ X_list[j].T
            if i == j:
                # intra-class: upper triangle only (symmetric, diagonal excluded)
                if A.shape[0] > 1:
                    vals = A[np.triu_indices_from(A, k=1)]
                else:
                    continue  # single-node class: no pairs to compare
            else:
                # inter-class: all pairwise products
                vals = A.flatten()
            blocks[(i, j)] = np.sort(vals)
    return blocks

def graph_distance(bd_a, bd_b):
    D = []
    for blk, va in bd_a.items():
        vb = bd_b.get(blk)
        if vb is not None:
            if len(va) == len(vb):
                D.append(float(np.sqrt(np.mean((va - vb) ** 2))))
            else:
                D.append(ot.wasserstein_1d(va, vb, p=2) ** 0.5)
    return np.linalg.norm(D)


