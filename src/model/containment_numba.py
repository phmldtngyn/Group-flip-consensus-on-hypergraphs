"""
BFS and containment computation using Numba for speed.
"""
import numpy as np
from numba import njit, prange
from typing import Tuple


@njit(cache=True)
def bfs_ball_csr(indptr: np.ndarray, indices: np.ndarray, source: int, r: int, n: int) -> np.ndarray:
    """
    BFS from source up to radius r. Returns bitmask of visited nodes.
    """
    dist = np.full(n, -1, dtype=np.int32)
    dist[source] = 0
    queue = np.zeros(n, dtype=np.int32)
    head = 0
    tail = 0
    queue[tail] = source
    tail += 1
    mask = np.uint64(1) << np.uint64(source)

    while head < tail:
        u = queue[head]
        head += 1
        if dist[u] >= r:
            continue
        for idx in range(indptr[u], indptr[u+1]):
            v = indices[idx]
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                mask |= np.uint64(1) << np.uint64(v)
                queue[tail] = v
                tail += 1
    return mask


@njit(cache=True, parallel=True)
def compute_all_balls_csr(indptr: np.ndarray, indices: np.ndarray, r: int, n: int) -> np.ndarray:
    """
    Compute BFS balls for all nodes. Returns array of bitmasks.
    Only works for n <= 64 (uses uint64).
    """
    balls = np.zeros(n, dtype=np.uint64)
    for i in prange(n):
        balls[i] = bfs_ball_csr(indptr, indices, i, r, n)
    return balls


@njit(cache=True)
def bfs_ball_csr_large(indptr: np.ndarray, indices: np.ndarray, source: int, r: int, n: int) -> np.ndarray:
    """
    BFS from source up to radius r. Returns boolean array of visited nodes.
    For n > 64.
    """
    dist = np.full(n, -1, dtype=np.int32)
    dist[source] = 0
    queue = np.zeros(n, dtype=np.int32)
    head = 0
    tail = 0
    queue[tail] = source
    tail += 1
    visited = np.zeros(n, dtype=np.bool_)
    visited[source] = True

    while head < tail:
        u = queue[head]
        head += 1
        if dist[u] >= r:
            continue
        for idx in range(indptr[u], indptr[u+1]):
            v = indices[idx]
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                visited[v] = True
                queue[tail] = v
                tail += 1
    return visited


@njit(cache=True, parallel=True)
def compute_all_balls_large(indptr: np.ndarray, indices: np.ndarray, r: int, n: int) -> np.ndarray:
    """
    Compute BFS balls for all nodes. Returns boolean 2D array.
    For n > 64.
    """
    balls = np.zeros((n, n), dtype=np.bool_)
    for i in prange(n):
        balls[i] = bfs_ball_csr_large(indptr, indices, i, r, n)
    return balls


@njit(cache=True, parallel=True)
def containment_bitmask(balls: np.ndarray, n: int) -> Tuple[float, float]:
    """
    Compute c_G and c_tilde using bitmask representation.
    Only for n <= 64.
    """
    full = np.uint64((1 << n) - 1)
    cG_count = np.int64(0)
    ct_num = np.int64(0)
    ct_den = np.int64(0)

    for i in prange(n):
        Bi = balls[i]
        R = full ^ Bi
        residual_size = np.int64(0)
        for bit in range(n):
            if (R >> bit) & np.uint64(1):
                residual_size += 1

        if R == np.uint64(0):
            cG_count += n
            ct_num += 1
            ct_den += 1
            continue

        cnt_all = np.int64(0)
        cnt_residual = np.int64(0)

        for j in range(n):
            Bj = balls[j]
            # Check if R is subset of Bj: (R & ~Bj) == 0
            if (R & (full ^ Bj)) == np.uint64(0):
                cnt_all += 1
                if (R >> j) & np.uint64(1):
                    cnt_residual += 1

        cG_count += cnt_all
        ct_num += cnt_residual
        ct_den += residual_size

    cG = float(cG_count) / float(n * n)
    ctilde = float(ct_num) / float(ct_den) if ct_den > 0 else 1.0
    return cG, ctilde


@njit(cache=True, parallel=True)
def containment_large(balls: np.ndarray, n: int) -> Tuple[float, float]:
    """
    Compute c_G and c_tilde using boolean array representation.
    For n > 64.
    """
    cG_count = np.int64(0)
    ct_num = np.int64(0)
    ct_den = np.int64(0)

    for i in prange(n):
        Bi = balls[i]
        R = ~Bi
        residual_size = np.int64(0)
        for v in range(n):
            if R[v]:
                residual_size += 1

        if residual_size == 0:
            cG_count += n
            ct_num += 1
            ct_den += 1
            continue

        cnt_all = np.int64(0)
        cnt_residual = np.int64(0)

        for j in range(n):
            Bj = balls[j]
            # Check if R is subset of Bj
            is_subset = True
            for v in range(n):
                if R[v] and not Bj[v]:
                    is_subset = False
                    break
            if is_subset:
                cnt_all += 1
                if R[j]:
                    cnt_residual += 1

        cG_count += cnt_all
        ct_num += cnt_residual
        ct_den += residual_size

    cG = float(cG_count) / float(n * n)
    ctilde = float(ct_num) / float(ct_den) if ct_den > 0 else 1.0
    return cG, ctilde


@njit(cache=True, parallel=True)
def containment_popcount_equal_balls(balls: np.ndarray, n: int, beta: int) -> float:
    """
    Compute c_G using popcount intersection for equal-size balls.
    Only for n <= 64.
    """
    target = 2 * beta - n
    if target < 0:
        return 0.0

    count = np.int64(0)
    for i in prange(n):
        Bi = balls[i]
        for j in range(n):
            Bj = balls[j]
            intersection = Bi & Bj
            # Count bits
            bits = np.int64(0)
            while intersection:
                intersection &= intersection - np.uint64(1)
                bits += 1
            if bits == target:
                count += 1

    return float(count) / float(n * n)


def compute_containment(indptr: np.ndarray, indices: np.ndarray, r: int, n: int) -> Tuple[float, float]:
    """
    High-level function to compute c_G and c_tilde.
    Automatically chooses bitmask or large representation.
    """
    if n <= 64:
        balls = compute_all_balls_csr(indptr, indices, r, n)
        return containment_bitmask(balls, n)
    else:
        balls = compute_all_balls_large(indptr, indices, r, n)
        return containment_large(balls, n)