# Graph Similarity via Wasserstein Distance and Node Embeddings

Thesis project comparing **Wasserstein distance** and **unmatched distance** as metrics for graph similarity, using node embeddings learned with [EDRep](https://openreview.net/forum?id=9M4NKMZOPu).

The core idea: embed graph nodes into a vector space using EDRep, then compare graphs by measuring the Wasserstein-2 distance between the distributions of scalar products `X @ Xᵀ`, computed block-by-block over graph partitions (community structure).

---

## Project Structure

```
.
├── data/
│   ├── abide_data/          # ABIDE brain connectivity dataset
│   ├── data_wol/            # Web of Life bipartite networks
│   ├── socio_patterns/      # High school contact networks (2011, 2012)
│   └── synthetic_graphs/    # Synthetic ER, DCSBM, geometric, configuration model graphs
├── notebooks/
│   ├── experiments/         # Main experiment notebooks
│   │   ├── distance_notebook.ipynb         # Wasserstein distance experiments
│   │   ├── configuration_model.ipynb       # Configuration model experiments
│   │   ├── comparison_unmatched.ipynb      # W2 vs unmatched distance comparison
│   │   └── confront_unmatched_distances.ipynb
│   ├── real_world/          # Real-world dataset experiments
│   │   ├── test_ABIDE.ipynb
│   │   ├── test_socio_patterns.ipynb
│   │   └── test_wol.ipynb
│   └── drafts/              # Exploratory notebooks
├── results/figures/         # Output figures from experiments
├── src/                     # Core utilities (submodule → thesis_code)
│   └── functions.py         # node_embedding(), w2_distance_classes(), plot utilities
├── EDRep_main/              # EDRep library source
└── functions.py             # Root-level copy of utility functions
```

---

## Setup

### Option 1 — pip + venv (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
pip install -e EDRep_main/
```

### Option 2 — Conda

```bash
conda env create -f EDRep_main/EDRep_env.yml
conda activate EDRep
pip install -e EDRep_main/
```

---

## Usage

Open any notebook in `notebooks/experiments/` with Jupyter:

```bash
jupyter notebook
```

The main workflow in each notebook:
1. Generate or load a graph
2. Compute node embeddings with `node_embedding(G, partition, dim)`
3. Compare two graphs with `w2_distance_classes(emb_A, emb_B)`
4. Visualize with `plot_wasserstein_heatmap()` or `plot_dist_grid()`

---

## Key Functions (`functions.py`)

| Function | Description |
|---|---|
| `node_embedding(G, partition, dim)` | Learns EDRep embeddings from adjacency matrix, returns list of embedding matrices per class |
| `w2_distance_classes(X_list, Y_list)` | Computes block-wise Wasserstein-2 distance between two partitioned graphs |
| `plot_wasserstein_heatmap(D_vec, m)` | Heatmap of inter-class Wasserstein distances |
| `plot_dist_grid(dist_data, m)` | Grid of KDE plots comparing scalar product distributions per block |

---

## Reference

```bibtex
@article{dall'amico2025learning,
  title={Learning distributed representations with efficient SoftMax normalization},
  author={Lorenzo Dall'Amico and Enrico Maria Belliardo},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=9M4NKMZOPu}
}
```
