import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def combined_similarity(color, orb, cnn, flow, wc=0.1, wo=0.25, wd=0.45, wf=0.20):
    def norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        return x / n

    c = norm(color)
    o = norm(orb)
    d = norm(cnn)
    f = norm(flow)

    sim_c = c @ c.T
    sim_o = o @ o.T
    sim_d = d @ d.T
    sim_f = f @ f.T      

    return wc * sim_c + wo * sim_o + wd * sim_d + wf * sim_f



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
