import numpy as np
from scipy.spatial.distance import cdist

def combined_similarity(color, orb, cnn, flow, edge=None, 
                       wc=0.03, wo=0.12, wd=0.75, wf=0.10, we=0.0):
    """
    Compute combined similarity - HEAVY emphasis on CNN
    """
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
    
    combined = wc * sim_c + wo * sim_o + wd * sim_d + wf * sim_f
    
    if edge is not None and we > 0:
        e = norm(edge)
        sim_e = e @ e.T
        combined += we * sim_e
    
    return combined


def nearest_neighbor_chain(sim_matrix, start_idx=None):
    """
    Build path by always picking the nearest unvisited neighbor.
    This is simple but often very effective for videos.
    """
    N = sim_matrix.shape[0]
    
    if start_idx is None:
        
        start_idx = np.argmax(sim_matrix.sum(axis=1))
    
    path = [start_idx]
    used = {start_idx}
    
    current = start_idx
    
    while len(path) < N:
        remaining = [i for i in range(N) if i not in used]
        if not remaining:
            break
        
        
        similarities = sim_matrix[current, remaining]
        best_idx = np.argmax(similarities)
        next_frame = remaining[best_idx]
        
        path.append(next_frame)
        used.add(next_frame)
        current = next_frame
    
    return path


def bidirectional_chain(sim_matrix, start_idx=None):
    """
    Build chain from both directions (forward and backward from start)
    """
    N = sim_matrix.shape[0]
    
    if start_idx is None:
       
        start_idx = np.argmax(sim_matrix.sum(axis=1))
    
    
    forward = [start_idx]
    used = {start_idx}
    current = start_idx
    
    while len(used) < N:
        remaining = [i for i in range(N) if i not in used]
        if not remaining:
            break
        
        similarities = sim_matrix[current, remaining]
        best_idx = np.argmax(similarities)
        next_frame = remaining[best_idx]
        
        forward.append(next_frame)
        used.add(next_frame)
        current = next_frame
    
    
    backward = []
    used_back = {start_idx}
    current = start_idx
    
    while len(used_back) < N:
        remaining = [i for i in range(N) if i not in used_back]
        if not remaining:
            break
        
        similarities = sim_matrix[current, remaining]
        best_idx = np.argmax(similarities)
        next_frame = remaining[best_idx]
        
        backward.append(next_frame)
        used_back.add(next_frame)
        current = next_frame
    
    
    if backward:
        full_path = backward[::-1] + forward
    else:
        full_path = forward
    
    return full_path


def find_endpoints(sim_matrix, num_candidates=5):
    """
    Find likely start/end frames.
    These typically have lower average similarity (they're unique).
    """
    N = sim_matrix.shape[0]
    
    
    avg_sim = sim_matrix.sum(axis=1) / (N - 1)
    
    sorted_indices = np.argsort(avg_sim)
    
    
    bottom_candidates = sorted_indices[:num_candidates].tolist()
    
    
    top_candidates = sorted_indices[-num_candidates:].tolist()
    
    return bottom_candidates + top_candidates


def tsp_nearest_insertion(sim_matrix):
    """
    Use nearest insertion heuristic (TSP-like approach)
    """
    N = sim_matrix.shape[0]
    
    
    max_sim = -1
    best_pair = (0, 1)
    
    for i in range(N):
        for j in range(i+1, N):
            if sim_matrix[i, j] > max_sim:
                max_sim = sim_matrix[i, j]
                best_pair = (i, j)
    
    path = list(best_pair)
    used = set(best_pair)
    
    
    while len(path) < N:
        remaining = [i for i in range(N) if i not in used]
        if not remaining:
            break
        
        best_frame = None
        best_position = None
        best_increase = float('inf')
        
        
        for frame in remaining:
            for pos in range(len(path) + 1):
                
                if pos == 0:
                   
                    increase = -sim_matrix[frame, path[0]]
                elif pos == len(path):
                   
                    increase = -sim_matrix[path[-1], frame]
                else:
                   
                    old_cost = sim_matrix[path[pos-1], path[pos]]
                    new_cost = sim_matrix[path[pos-1], frame] + sim_matrix[frame, path[pos]]
                    increase = -new_cost + old_cost
                
                if increase < best_increase:
                    best_increase = increase
                    best_frame = frame
                    best_position = pos
        
        
        path.insert(best_position, best_frame)
        used.add(best_frame)
    
    return path


def multi_start_greedy(sim_matrix, num_starts=15):
    """
    Try multiple starting points and pick the best path.
    """
    N = sim_matrix.shape[0]
    
    
    candidates = find_endpoints(sim_matrix, num_candidates=num_starts)
    
    best_path = None
    best_score = -1e9
    
    for start in candidates:
      
        path = nearest_neighbor_chain(sim_matrix, start)
        
        
        score = sum(sim_matrix[path[i], path[i+1]] for i in range(len(path)-1))
        
        if score > best_score:
            best_score = score
            best_path = path
    
    return best_path


def check_and_reverse(sim_matrix, order):
    """
    Check if order should be reversed based on temporal consistency
    Videos typically have smoother transitions in the correct direction
    """
    N = len(order)
    
   
    forward_score = 0
    for i in range(N-1):
        forward_score += sim_matrix[order[i], order[i+1]]
    
    reversed_order = order[::-1]
    reverse_score = 0
    for i in range(N-1):
        reverse_score += sim_matrix[reversed_order[i], reversed_order[i+1]]
    
    print(f"  → Forward score: {forward_score:.2f}")
    print(f"  → Reverse score: {reverse_score:.2f}")
    
    
    if reverse_score > forward_score:
        print(f"  → Reversing video direction")
        return reversed_order
    else:
        print(f"  → Keeping original direction")
        return order


def two_opt_improvement(sim_matrix, path, max_iterations=50):
    """
    2-opt local search to improve path.
    Repeatedly try reversing segments to improve total similarity.
    """
    N = len(path)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(N - 2):
            for j in range(i + 2, N):
                
                
                if j + 1 < N:
                    old_sim = sim_matrix[path[i], path[i+1]] + sim_matrix[path[j], path[j+1]]
                    new_sim = sim_matrix[path[i], path[j]] + sim_matrix[path[i+1], path[j+1]]
                    
                    if new_sim > old_sim:
                        
                        path[i+1:j+1] = path[i+1:j+1][::-1]
                        improved = True
    
    return path


def hybrid_ordering(sim_matrix, beam_width=12):
    """
    Hybrid approach: Multiple strategies
    """
    
    path1 = multi_start_greedy(sim_matrix, num_starts=20)
    score1 = sum(sim_matrix[path1[i], path1[i+1]] for i in range(len(path1)-1))
    
    
    path2 = tsp_nearest_insertion(sim_matrix)
    score2 = sum(sim_matrix[path2[i], path2[i+1]] for i in range(len(path2)-1))
    
   
    if score1 >= score2:
        best_path = path1
        print(f"  → Selected multi-start greedy (score: {score1:.2f})")
    else:
        best_path = path2
        print(f"  → Selected TSP insertion (score: {score2:.2f})")
    
    
    print(f"  → Applying 2-opt refinement...")
    best_path = two_opt_improvement(sim_matrix, best_path, max_iterations=100)
    
    
    print(f"  → Checking video direction...")
    best_path = check_and_reverse(sim_matrix, best_path)
    
    final_score = sum(sim_matrix[best_path[i], best_path[i+1]] for i in range(len(best_path)-1))
    print(f"  → Final score: {final_score:.2f}")
    
    return best_path