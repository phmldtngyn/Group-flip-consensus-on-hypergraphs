"""
Common utilities for hypergraph analysis.
"""
import numpy as np
from typing import List, Tuple, Dict, Any
import json
import os


def ensure_dir(path: str):
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict[str, Any], path: str):
    """Save dict to JSON."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Dict[str, Any]:
    """Load dict from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def adjacency_to_csr(adj: List[set]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert adjacency list (list of sets) to CSR format.
    Returns: (indptr, indices, degrees)
    """
    n = len(adj)
    degrees = np.array([len(a) for a in adj], dtype=np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(degrees)
    indices = np.zeros(indptr[-1], dtype=np.int64)
    for i in range(n):
        indices[indptr[i]:indptr[i+1]] = sorted(adj[i])
    return indptr, indices, degrees


def hyperedges_to_adj(n: int, hyperedges: List[List[int]]) -> List[set]:
    """
    Convert hyperedges to clique-expanded adjacency list.
    """
    adj = [set() for _ in range(n)]
    for e in hyperedges:
        if len(e) <= 1:
            continue
        for u in e:
            for v in e:
                if u != v:
                    adj[u].add(v)
    return adj


def compute_beta_from_balls(balls_masks: np.ndarray) -> int:
    """
    Compute ball size from bitmask array (assumes uniform).
    """
    if len(balls_masks) == 0:
        return 0
    return int(balls_masks[0].bit_count())


def compute_beta_from_csr(indptr: np.ndarray, balls_csr: np.ndarray) -> int:
    """
    Compute ball size from CSR representation.
    """
    if len(indptr) <= 1:
        return 0
    return int(indptr[1] - indptr[0])