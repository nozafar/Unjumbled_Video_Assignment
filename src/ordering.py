# src/ordering.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def combined_similarity(feat1, feat2, w1=0.8, w2=0.2):
    # feat1, feat2: numpy arrays (N x d1), (N x d2)
    # compute cosine similarity matrices and weighted sum
    f1 = feat1.copy().astype(np.float32)
    f2 = feat2.copy().astype(np.float32)
    # normalize
    def norm_rows(x):
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        return x / n
    nf1 = norm_rows(f1)
    nf2 = norm_rows(f2)
    s1 = nf1.dot(nf1.T)
    s2 = nf2.dot(nf2.T)
    return w1 * s1 + w2 * s2

def greedy_beam_order(sim_matrix, beam_width=8):
    N = sim_matrix.shape[0]
    best_path = None
    best_score = -1e9
    # try every frame as start OR choose few seeds
    seeds = list(range(N))
    for s in seeds:
        beam = [ ( -0.0, [s], {s} ) ]  # (negative cumulative score, path, used_set)
        for _ in range(N-1):
            newbeam = []
            for neg_score, path, used in beam:
                last = path[-1]
                # candidate successors
                cands = [(sim_matrix[last, j], j) for j in range(N) if j not in used]
                cands.sort(reverse=True)
                topk = cands[:beam_width]
                for sim, j in topk:
                    newpath = path + [j]
                    newused = set(used)
                    newused.add(j)
                    newneg = neg_score - sim  # negative since we store -score
                    newbeam.append((newneg, newpath, newused))
            # prune
            if not newbeam:
                break
            newbeam.sort(key=lambda x: x[0])
            beam = newbeam[:beam_width]
        # evaluate beams
        for neg_score, path, used in beam:
            total = -neg_score
            if total > best_score:
                best_score = total
                best_path = path
    return best_path
