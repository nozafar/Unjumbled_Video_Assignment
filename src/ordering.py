import numpy as np


def find_start_frame(diff):
    sums = np.sum(diff, axis=1)
    return np.argmax(sums)


def reconstruct_sequence(diff, start):
    print("Reconstructing optimal sequence...")

    n = diff.shape[0]
    visited = {start}
    seq = [start]
    current = start

    for _ in range(n - 1):
        d = diff[current].copy()
        for v in visited:
            d[v] = float('inf')

        nxt = np.argmin(d)
        seq.append(nxt)
        visited.add(nxt)
        current = nxt

    return seq
