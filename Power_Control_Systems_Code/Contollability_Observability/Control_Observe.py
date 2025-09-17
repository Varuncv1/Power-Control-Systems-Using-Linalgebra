# Let's create a Python module that builds grid Laplacians, computes eigenpairs,
# checks PBH controllability/observability for chosen actuator/sensor nodes,
# and implements the Notarstefano–Parlangeli simple-grid test (Theorem 3.6 & Prop. 3.7).
#
# We'll also include a small demo at the bottom that users can edit.
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable, Set

def path_laplacian(n: int) -> np.ndarray:
    """Laplacian of a path graph P_n (n>=2)."""
    if n < 2:
        raise ValueError("n must be >= 2")
    L = np.zeros((n, n), dtype=float)
    for i in range(n):
        deg = 0
        if i-1 >= 0:
            L[i, i-1] = -1
            deg += 1
        if i+1 < n:
            L[i, i+1] = -1
            deg += 1
        L[i, i] = deg
    return L

def grid_laplacian(n1: int, n2: int) -> np.ndarray:
    """Laplacian of an n1 x n2 grid via the Kronecker sum (Lemma 3.1)."""
    L1 = path_laplacian(n1)
    L2 = path_laplacian(n2)
    return np.kron(L1, np.eye(n2)) + np.kron(np.eye(n1), L2)

def grid_index_1based_to_0based(n1: int, n2: int, i: int, j: int) -> int:
    """Map grid node [i,j] (1-based per paper) to Python 0-based flat index."""
    if not (1 <= i <= n1 and 1 <= j <= n2):
        raise ValueError("grid index out of bounds")
    # Paper stacks i as the slow axis (first path), j as the fast axis
    # Flatten index = (i-1)*n2 + (j-1)
    return (i - 1) * n2 + (j - 1)

def is_simple_grid(n1: int, n2: int, tol: float = 1e-12) -> bool:
    """Check Definition 3.2: all sums of distinct path eigenvalues are distinct."""
    # Path Laplacian eigenvalues: λ_k = 2 - 2 cos(k π / n), k=0..n-1 (for path)
    # Note: For a path, indices are 0..n-1 with λ_0=0, λ_{k} distinct.
    k1 = np.arange(n1)
    k2 = np.arange(n2)
    lam1 = 2 - 2*np.cos(np.pi * k1 / n1)
    lam2 = 2 - 2*np.cos(np.pi * k2 / n2)
    sums = []
    for a in lam1:
        for b in lam2:
            sums.append(a+b)
    sums = np.array(sums)
    sums_sorted = np.sort(sums)
    diffs = np.diff(sums_sorted)
    return np.all(diffs > tol)

def eigh_with_vectors(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenvectors of symmetric M (ascending)."""
    w, V = np.linalg.eigh(M)
    return w, V

@dataclass(frozen=True)
class Node:
    i: int  # row (1..n1)
    j: int  # col (1..n2)

def pbh_uncontrollable_eigs(L: np.ndarray, actuator_indices_0based: Iterable[int], tol: float = 1e-10) -> List[int]:
    """Return indices k of eigenpairs (λ_k, v_k) s.t. all B^T v_k == 0 (Lemma 2.1)."""
    w, V = eigh_with_vectors(L)
    idxs = []
    act = list(set(actuator_indices_0based))
    for k in range(V.shape[1]):
        vk = V[:, k]
        if all(abs(vk[a]) <= tol for a in act):
            idxs.append(k)
    return idxs

def pbh_unobservable_eigs(L: np.ndarray, sensor_indices_0based: Iterable[int], tol: float = 1e-10) -> List[int]:
    """Return indices k of eigenpairs (λ_k, v_k) s.t. C v_k == 0 (Lemma 2.1)."""
    return pbh_uncontrollable_eigs(L, sensor_indices_0based, tol)

def prime_factors_odd(n: int) -> List[int]:
    """Odd prime factors of n (with multiplicities removed)."""
    facs = []
    d = 2
    while d*d <= n:
        if n % d == 0:
            facs.append(d)
            while n % d == 0:
                n //= d
        d += 1 if d == 2 else 2  # 2 then odd numbers
    if n > 1:
        facs.append(n)
    # keep only odd primes
    return [p for p in facs if p % 2 == 1]

def theorem36_condition_fail(n: int, idx: int, p: int) -> bool:
    """Implements condition (3): (n - i) mod p == (i - 1) for path length n & node i (1-based)."""
    return ((n - idx) % p) == ((idx - 1) % p)

def simple_grid_uncontrollable_nodes(n1: int, n2: int) -> Dict[Tuple[int,int], Set[Tuple[int,int]]]:
    """
    For a simple grid, build for each node the set of (p1, p2) pairs that make it uncontrollable per Prop. 3.7.
    Each pair corresponds to a block of uncontrollable eigenvalues λ = (2-2 cos((2ν1-1)π/p1)) + (2-2 cos((2ν2-1)π/p2)).
    Returns: dict (i,j) -> set of (p1,p2) where p1|n1 and/or p2|n2 and condition (3) holds in that direction.
    """
    P1 = prime_factors_odd(n1)
    P2 = prime_factors_odd(n2)
    result: Dict[Tuple[int,int], Set[Tuple[int,int]]] = {}
    for i in range(1, n1+1):
        for j in range(1, n2+1):
            bad_pairs: Set[Tuple[int,int]] = set()
            # direction 1
            for p1 in P1:
                if theorem36_condition_fail(n1, i, p1):
                    # pairs with any p2 factor (including a placeholder None to indicate "only dir 1"?)
                    if P2:
                        for p2 in P2:
                            bad_pairs.add((p1, p2))
                    else:
                        bad_pairs.add((p1, 1))
            # direction 2
            for p2 in P2:
                if theorem36_condition_fail(n2, j, p2):
                    if P1:
                        for p1 in P1:
                            bad_pairs.add((p1, p2))
                    else:
                        bad_pairs.add((1, p2))
            if bad_pairs:
                result[(i,j)] = bad_pairs
    return result

def prop37_is_controllable_from_set(n1: int, n2: int, nodes: List[Node]) -> bool:
    """
    Proposition 3.7: On simple grids, controllable from a set iff intersection of O_alpha is empty.
    We'll compute O_alpha for each node and check intersection.
    """
    if not is_simple_grid(n1, n2):
        raise ValueError("Grid is not simple; Proposition 3.7 applies to simple grids only.")
    node_pairs = simple_grid_uncontrollable_nodes(n1, n2)
    # Build O_alpha for each requested node
    O_sets: List[Set[Tuple[int,int]]] = []
    for nd in nodes:
        O_sets.append(set(node_pairs.get((nd.i, nd.j), set())))
    if not O_sets:
        return False
    # Empty O for some node -> trivially controllable
    if any(len(S)==0 for S in O_sets):
        return True
    inter = set.intersection(*O_sets) if O_sets else set()
    return len(inter) == 0

def uncontrollable_eigs_from_pairs(n1: int, n2: int, pairs: Set[Tuple[int,int]]) -> List[float]:
    """
    Given a set of (p1,p2) prime pairs for a node, generate the associated uncontrollable eigenvalues (Theorem 3.6 eq. (4)).
    """
    vals = []
    for p1, p2 in pairs:
        if p1 <= 1 and p2 <= 1:
            continue
        # handle placeholders 1 by skipping that direction sum (edge case when one dimension has no odd prime factors)
        spec1 = [0.0] if p1 <= 1 else [2 - 2*np.cos((2*nu-1)*np.pi/p1) for nu in range(1, (p1-1)//2 + 1)]
        spec2 = [0.0] if p2 <= 1 else [2 - 2*np.cos((2*nu-1)*np.pi/p2) for nu in range(1, (p2-1)//2 + 1)]
        for a in spec1:
            for b in spec2:
                vals.append(a+b)
    # dedupe within tolerance
    vals = np.array(vals)
    vals = np.unique(np.round(vals, 12))
    return vals.tolist()

@dataclass
class PBHReport:
    controllable: bool
    uncontrollable_eigs: List[Tuple[float, int]]  # (eigenvalue, multiplicity)
    tol: float

def pbh_report_controllability(L: np.ndarray, actuators_0: Iterable[int], tol: float = 1e-10) -> PBHReport:
    w, V = eigh_with_vectors(L)
    idxs = pbh_uncontrollable_eigs(L, actuators_0, tol)
    uncont_vals = [w[k] for k in idxs]
    # multiplicities via rounding
    if uncont_vals:
        arr = np.round(np.array(uncont_vals), 12)
        uniq, counts = np.unique(arr, return_counts=True)
        unordered = list(zip(uniq.tolist(), counts.tolist()))
    else:
        unordered = []
    return PBHReport(controllable=(len(idxs)==0), uncontrollable_eigs=unordered, tol=tol)

def pbh_report_observability(L: np.ndarray, sensors_0: Iterable[int], tol: float = 1e-10) -> PBHReport:
    return pbh_report_controllability(L, sensors_0, tol)

# ---------------------- Demo ----------------------
if __name__ == "__main__":
    n1, n2 = 7, 15  # grid size from the paper's example
    L = grid_laplacian(n1, n2)
    # Choose nodes using 1-based indexing (paper's convention)
    nodes = [Node(1, 2), Node(4, 1)]
    # Convert to 0-based flat indices for PBH
    actuator_idx0 = [grid_index_1based_to_0based(n1, n2, nd.i, nd.j) for nd in nodes]
    rep = pbh_report_controllability(L, actuator_idx0, tol=1e-10)
    print(f"PBH controllable from {nodes}? {rep.controllable}")
    if rep.uncontrollable_eigs:
        print("Uncontrollable eigenvalues (value, multiplicity via eigenbasis zeros at actuators):")
        for val, mult in rep.uncontrollable_eigs:
            print(f"  {val:.12f} (mult {mult})")
    # If simple, also test Proposition 3.7
    if is_simple_grid(n1, n2):
        ok = prop37_is_controllable_from_set(n1, n2, nodes)
        print(f"Prop. 3.7 says controllable? {ok}")
        # Show predicted uncontrollable eigenvalues for the intersection, if not controllable
        if not ok:
            # Build intersection
            node_pairs = simple_grid_uncontrollable_nodes(n1, n2)
            sets = [set(node_pairs.get((nd.i, nd.j), set())) for nd in nodes]
            inter = set.intersection(*sets) if sets else set()
            vals = uncontrollable_eigs_from_pairs(n1, n2, inter)
            if vals:
                print("Prop. 3.7 predicted uncontrollable eigenvalues (rounded):")
                for v in vals:
                    print(f"  {v:.12f}")
    else:
        print("Grid is non-simple; use PBH numeric report (Prop. 3.7 not applicable).")
