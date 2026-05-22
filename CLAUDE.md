# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**EDRep** — a Python research library for learning node embeddings from graphs using variational SoftMax normalization. Based on the paper "Learning distributed representations with efficient SoftMax normalization" (Dall'Amico & Belliardo, 2025, TMLR). The broader repo compares Wasserstein distance vs unmatched distance metrics for graph similarity and runs experiments on synthetic and real-world datasets.

## Setup

```bash
# Recommended: Conda environment
conda env create -f EDRep_main/EDRep_env.yml
conda activate EDRep

# Or pip
pip install -r EDRep_main/requirements.txt
```

Requires Python >= 3.9. Key dependencies: `faiss-cpu`, `networkx`, `scipy`, `POT` (optimal transport), `scikit-learn`, `matplotlib`, `seaborn`.

## Architecture

### Core Package: `EDRep_main/EDRep/`

- **[EDRep.py](EDRep_main/EDRep/EDRep.py)** — optimization engine. `CreateEmbedding(Pv, dim, n_epochs, k, eta, sym)` learns embedding matrix `X ∈ R^(n×d)` from a list of sparse probability matrices. `k > 1` enables mixture-of-Gaussians clustering; `sym=False` produces separate `X` and `Y` matrices. Returns an `EDRep_class` with `.X`, `.Y`, `.ℓ` (cluster labels).
- **[node_embedding.py](EDRep_main/EDRep/node_embedding.py)** — high-level wrapper. `NodeEmbedding(A, dim, walk_length=5)` builds `P = D⁻¹A` from a sparse CSR adjacency matrix and calls `CreateEmbedding`. `walk_length` controls how many powers of `P` are stacked for multi-hop context.
- **[__init__.py](EDRep_main/EDRep/__init__.py)** — exports `CreateEmbedding`, `computeZest`, `NodeEmbedding`.

### Root-level

- **[functions.py](src/functions.py)** — research utilities: `node_embedding()`, `w2_distance_classes()` (Wasserstein-2 between partitioned embeddings), `plot_wasserstein_heatmap()`, `plot_dist_grid()`.
- **Notebooks** (`.ipynb` files in root) — experiments. Each notebook is self-contained: generates/loads graphs → embeds → computes distances → visualizes. Key ones: `distance_notebook.ipynb`, `configuration_model.ipynb`, `comparison_unmatched.ipynb`, `test_ABIDE.ipynb`.

### Algorithm Flow

1. Build row-normalized probability matrix `P = D⁻¹A` from adjacency matrix.
2. `CreateEmbedding` minimizes a variational loss with softmax normalization to learn `X`.
3. Distance between graphs: compare scalar-product distributions `X @ Xᵀ` using Wasserstein-2 (via POT) or eigenvalue-based unmatched distance.
4. NMI (Normalized Mutual Information) scores evaluate clustering quality.

## Key Implementation Details

- Adjacency matrices must be **CSR format**: `nx.to_scipy_sparse_array(G, format='csr')`.
- All matrix entries must be non-negative.
- Notebooks preserve random state with `np.random.get_state()` / `np.random.set_state()` for reproducibility.
- Output artifacts for configuration model experiments are pre-computed in `output_configuration_model/alpha_*/`.

## Reference

Dall'Amico, L., & Belliardo, E. M. (2025). *Learning distributed representations with efficient SoftMax normalization.* TMLR. https://openreview.net/forum?id=9M4NKMZOPu
