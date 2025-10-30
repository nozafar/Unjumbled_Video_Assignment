import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def combined_similarity(color_feats, orb_feats, cnn_feats, wc=0.2, wo=0.3, wd=0.5):
    def normalize(x):
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        return x / n

    c = normalize(color_feats)
    o = normalize(orb_feats)
    d = normalize(cnn_feats)

    sim_color = c @ c.T
    sim_orb   = o @ o.T
    sim_cnn   = d @ d.T  

    return (wc * sim_color) + (wo * sim_orb) + (wd * sim_cnn)


def greedy_beam_order(sim_matrix, beam_width=8):
    N = sim_matrix.shape[0]
    best_path = None
    best_score = -1e9
    seeds = list(range(N))
    for s in seeds:
        beam = [ ( -0.0, [s], {s} ) ]
        for _ in range(N-1):
            newbeam = []
            for neg_score, path, used in beam:
                last = path[-1]

                cands = [(sim_matrix[last, j], j) for j in range(N) if j not in used]
                cands.sort(reverse=True)
                topk = cands[:beam_width]
                for sim, j in topk:
                    newpath = path + [j]
                    newused = set(used)
                    newused.add(j)
                    newneg = neg_score - sim 
                    newbeam.append((newneg, newpath, newused))

            if not newbeam:
                break
            newbeam.sort(key=lambda x: x[0])
            beam = newbeam[:beam_width]
        for neg_score, path, used in beam:
            total = -neg_score
            if total > best_score:
                best_score = total
                best_path = path
    return best_path
