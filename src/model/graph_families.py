"""
Generate the 4 graph families for measurement.
"""
import numpy as np
import networkx as nx
from typing import List, Tuple
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # sibling modules
from hyper_utils import hyperedges_to_adj, adjacency_to_csr


def family_1d_cycle(n: int, r: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    1D cycle with 3-node hyperedges.
    Returns: (indptr, indices, beta)
    """
    hyperedges = []
    for i in range(n):
        e = [i, (i+1) % n, (i+2) % n]
        hyperedges.append(e)

    adj = hyperedges_to_adj(n, hyperedges)
    indptr, indices, degrees = adjacency_to_csr(adj)
    beta = min(n, 4 * r + 1)
    return indptr, indices, beta


def family_2d_grid(L: int, r: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    2D hypergrid with 3x3 hyperedges.
    n = L^2
    Returns: (indptr, indices, beta)
    """
    n = L * L
    hyperedges = []

    for i in range(L):
        for j in range(L):
            e = []
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni = (i + di) % L
                    nj = (j + dj) % L
                    e.append(ni * L + nj)
            hyperedges.append(e)

    adj = hyperedges_to_adj(n, hyperedges)
    indptr, indices, degrees = adjacency_to_csr(adj)

    # Compute beta from first ball
    if len(indptr) > 1:
        beta = int(indptr[1] - indptr[0])
    else:
        beta = 0

    return indptr, indices, beta


def family_config_model(n: int, k: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Configuration model with degree k.
    Returns: (indptr, indices, beta)
    """
    rng = np.random.default_rng(seed)
    degrees = np.full(n, k, dtype=int)
    G = nx.configuration_model(degrees, seed=seed)
    G = nx.Graph(G)  # Remove parallel edges
    G.remove_edges_from(nx.selfloop_edges(G))

    adj = [set() for _ in range(n)]
    for u, v in G.edges():
        adj[u].add(v)
        adj[v].add(u)

    indptr, indices, degrees_arr = adjacency_to_csr(adj)
    # For config model, beta depends on r, but we'll compute it later
    beta = 0  # Placeholder
    return indptr, indices, beta


def family_watts_strogatz(n: int, k: int, p: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Watts-Strogatz small-world graph.
    Returns: (indptr, indices, beta)
    """
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)

    adj = [set() for _ in range(n)]
    for u, v in G.edges():
        adj[u].add(v)
        adj[v].add(u)

    indptr, indices, degrees = adjacency_to_csr(adj)
    beta = 0  # Placeholder
    return indptr, indices, beta


def get_family(family_name: str, n: int, r: int, **kwargs) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Get graph family by name.
    """
    if family_name == "1D":
        return family_1d_cycle(n, r)
    elif family_name == "2D":
        L = int(np.sqrt(n))
        assert L * L == n, "n must be perfect square for 2D"
        return family_2d_grid(L, r)
    elif family_name == "config":
        k = kwargs.get("k", 6)
        seed = kwargs.get("seed", 42)
        return family_config_model(n, k, seed)
    elif family_name == "WS":
        k = kwargs.get("k", 6)
        p = kwargs.get("p", 0.1)
        seed = kwargs.get("seed", 42)
        return family_watts_strogatz(n, k, p, seed)
    else:
        raise ValueError(f"Unknown family: {family_name}")