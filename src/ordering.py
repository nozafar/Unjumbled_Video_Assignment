import numpy as np
from tqdm import tqdm

def compute_similarity_matrix(features):
    n = len(features)
    sim = np.zeros((n, n))

    for i in tqdm(range(n), desc="Calculating similarity", ncols=80):
        for j in range(n):
            if i != j:
                sim[i, j] = np.linalg.norm(features[i] - features[j])

    return sim


def reconstruct_sequence(sim):
    start = np.argmax(np.sum(sim, axis=1))
    n = sim.shape[0]
    visited = {start}
    seq = [start]
    current = start

    for _ in tqdm(range(n - 1), desc="Ordering frames", ncols=80):
        d = sim[current].copy()
        d[list(visited)] = float('inf')
        nxt = np.argmin(d)
        seq.append(nxt)
        visited.add(nxt)
        current = nxt

    return seq
