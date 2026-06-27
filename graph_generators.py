import itertools
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF



def DCSBM(args):
    C, c, l_nodes, theta, symmetric, make_connected = args
    k = len(np.unique(l_nodes))
    n = len(theta)
    c_v = C[l_nodes].T
    fs, ss = [], []
    if symmetric:
        first = np.random.choice(n, int(n * c / 2), p=theta / n)
    else:
        first = np.random.choice(n, int(n * c), p=theta / n)
    for i in range(k):
        v = theta * c_v[i]
        first_sel = first[l_nodes[first] == i]
        fs.append(first_sel.tolist())
        second_sel = np.random.choice(n, len(first_sel), p=v / np.sum(v))
        ss.append(second_sel.tolist())
    fs = np.array(list(itertools.chain(*fs)))
    ss = np.array(list(itertools.chain(*ss)))
    if make_connected:
        idx = np.arange(n)[~np.isin(np.arange(n), fs)]
        fs = np.concatenate([fs, idx])
        ss = np.concatenate([ss, np.argmax(theta) * np.ones(len(idx), dtype=int)])
    edge_list = np.unique(np.column_stack((fs, ss)), axis=0)
    mask = edge_list[:, 0] == edge_list[:, 1]
    edge_list[:, 1][mask] += 1
    edge_list[:, 1][edge_list[:, 1] == n] = 0
    if symmetric:
        M = np.maximum(edge_list[:, 0], edge_list[:, 1])
        m = np.minimum(edge_list[:, 0], edge_list[:, 1])
        df = pd.DataFrame({'i': np.concatenate([M, m]), 'j': np.concatenate([m, M])})
    else:
        df = pd.DataFrame({'i': edge_list[:, 0], 'j': edge_list[:, 1]})
    return df


def GeometricModel(args):
    X, d, beta = args
    n = np.shape(X)[0]
    no = (X ** 2) @ np.ones(2)
    D = np.zeros((n, n))
    for i in range(n):
        D[i] += no
        D[:, i] += no
    D -= 2 * X @ X.T
    D = np.sqrt(np.abs(D))
    P = np.exp(-beta * D)
    P = P - np.diag(np.diag(P))
    p = P @ np.ones(n)
    P = np.diag(p ** (-1)).dot(P)
    idx1 = np.concatenate([np.ones(int(d / 2)) * i for i in range(n)])
    idx2 = np.concatenate([
        np.random.choice(np.arange(n), int(d / 2), p=P[i], replace=False)
        for i in range(n)
    ])
    df = pd.DataFrame(columns=['i', 'j'])
    df.i = np.concatenate([idx1, idx2])
    df.j = np.concatenate([idx2, idx1])
    return df


def gen_cm_via_dcsbm(args):
    n, c, alpha, symmetric, make_connected = args
    theta = np.random.uniform(3, 10, n) ** alpha
    theta = theta / np.mean(theta)
    ell = np.zeros(n, dtype=int)
    C = np.array([[c]])
    return DCSBM((C, c, ell, theta, symmetric, make_connected))

def dcsbm_to_nx(df):
    n = int(max(df['i'].max(), df['j'].max())) + 1
    G = nx.Graph()
    G.add_nodes_from(range(n))  # enforce label order 0..n-1 in G.nodes()
    G.add_edges_from(zip(df['i'].astype(int), df['j'].astype(int)))
    return G


def compute_cin_cout(c, k, alpha):
    c_out = c - alpha * np.sqrt(c)
    c_in = k*c - (k-1)*c_out
    return c_in, c_out


def df_to_adj(df):
    G = nx.from_pandas_edgelist(df, 'i', 'j')
    return nx.adjacency_matrix(G).astype(float)



def NMF_kmeans(M, k):
    n, _ = M.shape
    Mt = M + np.eye(n)*np.mean(M[M.nonzero()])
    Mt = Mt / np.mean(Mt)
    Y = NMF(n_components=k, max_iter=2000).fit(Mt).components_
    kmeans = KMeans(n_clusters=k, n_init=10).fit(Y.T)
    return kmeans.labels_, np.abs(kmeans.score(Y.T))


def ClusterNMF(M, k):
    est_l, score = NMF_kmeans(M, k)
    for _ in range(20):
        est_l_, score_ = NMF_kmeans(M, k)
        if score_ < score:
            score = score_; est_l = est_l_
    return est_l