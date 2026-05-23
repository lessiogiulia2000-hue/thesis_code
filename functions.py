import numpy as np
import networkx as nx
import ot
import matplotlib.pyplot as plt
import seaborn as sns
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


def w2_distance_classes(X_list,Y_list):
    # Validate the number of partitions
    m = len(X_list)
    if len(Y_list) != m:
        raise ValueError("The two graphs must have the same number of classes.")
    D = []
    distribution_data = {}

    # For cycles for the upper triangular part and diagonal blocks
    for i in range(m):
        for j in range(i, m):
            # Compute the dot product between class i and class j
            A_block = X_list[i] @ X_list[j].T
            B_block = Y_list[i] @ Y_list[j].T

            if i == j:
                if len(A_block)>1:
                    # If we are on a diagonal block, we take only the upper triangular part, excluding the main diagonal fixing k=1
                    upper_tri_indices_A = np.triu_indices_from(A_block, k=1)
                    upper_tri_indices_B= np.triu_indices_from(B_block, k=1)
                    vals_A = A_block[upper_tri_indices_A]
                    vals_B = B_block[upper_tri_indices_B]
                else:
                    continue
            else:
                # For blocks outside the diagonal we take all the elements
                vals_A = A_block.flatten()
                vals_B = B_block.flatten()

             # Store the distributions for later visualization
            distribution_data[(i, j)] = (vals_A, vals_B)

            # Calculate the 1D Wasserstein distance between the two distributions
            wd2 = ot.wasserstein_1d(vals_A, vals_B, p=2)**(1/2)
            D.append(wd2)
            
    # Convert the list of block distances into a NumPy array
    D_vector = np.array(D)

    # Calculate the final global distance as the Euclidean norm of the block distances vector
    finale_distance = np.linalg.norm(D_vector)

    return finale_distance, D_vector, distribution_data


def plot_wasserstein_heatmap(distance_vector, m):

    distances_matrix = np.zeros((m, m))

    # 1. Reconstructing the symmetric matrix from the distance vector
    idx = 0
    for i in range(m):
        for j in range(i, m):
            distances_matrix[i, j] = distance_vector[idx]
            distances_matrix[j, i] = distance_vector[idx] 
            idx += 1

    # 2. Generating the heatmap
    plt.figure(figsize=(8, 6))

    sns.heatmap(distances_matrix,
                annot=True,
                cmap="YlOrRd",
                fmt=".7f",
                linewidths=.5,
                xticklabels=[f"Class {i + 1}" for i in range(m)],
                yticklabels=[f"Class {i + 1}" for i in range(m)])

    plt.title("Heatmap of Wasserstein Distances Between Classes", pad=20)
    plt.xlabel("Graph B (Classes)")
    plt.ylabel("Graph A (Classes)")
    plt.tight_layout()
    plt.show()
    
    
def plot_dist_grid(distribution_data, m):
    # Create an m × m grid
    fig, axes = plt.subplots(m, m, figsize=(12, 12))

    for i in range(m):
        for j in range(m):
            ax = axes[i, j] # type: ignore

            #We draw only the upper triangular part

            if j >= i:
                dist_A, dist_B = distribution_data[(i, j)]

                # KDE plot of both graphs
                sns.kdeplot(dist_A, fill=True, ax=ax, color="blue", alpha=0.5, label="Grafo A")
                sns.kdeplot(dist_B, fill=True, ax=ax, color="orange", alpha=0.5, label="Grafo B")

                ax.set_title(f"Block ({i + 1} vs {j + 1})")
                ax.set_xlabel("Scalar product")

                if j == i:
                    ax.set_ylabel("Densità")
                else:
                    ax.set_ylabel("")

                if i == 0 and j == 0:
                    ax.legend()
            else:
                ax.axis('off')

    plt.suptitle("Complete Comparison of Distributions for Each Block", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
    