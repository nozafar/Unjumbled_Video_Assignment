import numpy as np
from tqdm import tqdm

def find_start_frame(diff):
    sums = np.sum(diff, axis=1)
    return np.argmax(sums)


def reconstruct_sequence(diff, start):
    print("\n🧠 Reconstructing optimal sequence...")

    n = diff.shape[0]
    visited = {start}
    seq = [start]
    current = start

    for _ in tqdm(range(n - 1), desc="🔄 Reordering frames", ncols=70):
        d = diff[current].copy()

        d[list(visited)] = float('inf')

        nxt = np.argmin(d)
        seq.append(nxt)
        visited.add(nxt)
        current = nxt

    return seq
