# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# def combined_similarity(color, orb, cnn, flow, wc=0.1, wo=0.25, wd=0.45, wf=0.20):
#     def norm(x):
#         n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
#         return x / n

#     c = norm(color)
#     o = norm(orb)
#     d = norm(cnn)
#     f = norm(flow)

#     sim_c = c @ c.T
#     sim_o = o @ o.T
#     sim_d = d @ d.T
#     sim_f = f @ f.T      

#     return wc * sim_c + wo * sim_o + wd * sim_d + wf * sim_f



# def greedy_beam_order(sim_matrix, beam_width=8):
#     N = sim_matrix.shape[0]


#     start_scores = sim_matrix.sum(axis=1)
#     seeds = np.argsort(start_scores)[-min(6, N):]

#     best_path = None
#     best_score = -1e9

#     for s in seeds:
#         beam = [(0.0, [s], {s})]

#         for _ in range(N - 1):
#             newbeam = []

#             for score, path, used in beam:
#                 last = path[-1]

#                 possible = list(set(range(N)) - used)
#                 sims = sim_matrix[last, possible]

#                 if sims.size == 0:
#                     continue

#                 k = min(beam_width, len(sims))
#                 topk_idx = np.argpartition(-sims, k - 1)[:k] if k > 0 else []

#                 for idx in topk_idx:
#                     j = possible[idx]
#                     newbeam.append((score + sims[idx], path + [j], used | {j}))

#             if not newbeam:
#                 break

#             newbeam.sort(key=lambda x: x[0], reverse=True)
#             beam = newbeam[:beam_width]

#         best_local = max(beam, key=lambda x: x[0])
#         if best_local[0] > best_score:
#             best_score = best_local[0]
#             best_path = best_local[1]

#     return best_path





# import numpy as np

# def combined_similarity(color, orb, cnn, flow, edge=None, 
#                        wc=0.05, wo=0.20, wd=0.60, wf=0.15, we=0.0):
#     """
#     Compute combined similarity matrix with multiple features
#     Increased CNN weight as it's most reliable
#     """
#     def norm(x):
#         n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
#         return x / n
    
#     c = norm(color)
#     o = norm(orb)
#     d = norm(cnn)
#     f = norm(flow)
    
#     sim_c = c @ c.T
#     sim_o = o @ o.T
#     sim_d = d @ d.T
#     sim_f = f @ f.T
    
#     combined = wc * sim_c + wo * sim_o + wd * sim_d + wf * sim_f
    
#     if edge is not None and we > 0:
#         e = norm(edge)
#         sim_e = e @ e.T
#         combined += we * sim_e
    
#     return combined


# def greedy_path_with_lookahead(sim_matrix, start_idx, lookahead=3):
#     """
#     Greedy path building with lookahead to avoid local optima
#     """
#     N = sim_matrix.shape[0]
#     path = [start_idx]
#     used = {start_idx}
    
#     while len(path) < N:
#         current = path[-1]
#         remaining = [i for i in range(N) if i not in used]
        
#         if not remaining:
#             break
        
#         if len(remaining) == 1:
#             path.append(remaining[0])
#             break
        
#         # Lookahead: consider next k steps
#         best_next = None
#         best_score = -1e9
        
#         for candidate in remaining:
#             # Score = immediate similarity + average similarity to remaining frames
#             immediate = sim_matrix[current, candidate]
            
#             # Look ahead: how well does this candidate connect to others?
#             other_remaining = [i for i in remaining if i != candidate]
#             if other_remaining:
#                 future_connections = np.mean([sim_matrix[candidate, i] for i in other_remaining[:lookahead]])
#                 score = immediate + 0.3 * future_connections
#             else:
#                 score = immediate
            
#             if score > best_score:
#                 best_score = score
#                 best_next = candidate
        
#         path.append(best_next)
#         used.add(best_next)
    
#     return path


# def find_best_starting_frame(sim_matrix, num_candidates=10):
#     """
#     Find the best starting frame by trying multiple candidates
#     Best start = frame with consistent high similarity pattern (likely start/end)
#     """
#     N = sim_matrix.shape[0]
    
#     # Strategy 1: Frames with high total similarity (central frames)
#     total_sim = sim_matrix.sum(axis=1)
    
#     # Strategy 2: Frames with high similarity to neighbors (edge frames)
#     # Sort by similarity to find potential first/last frames
#     neighbor_sim = np.zeros(N)
#     for i in range(N):
#         # Get top-k most similar frames
#         top_k = np.argsort(sim_matrix[i])[-10:]
#         neighbor_sim[i] = sim_matrix[i, top_k].mean()
    
#     # Combine strategies
#     combined_score = 0.6 * total_sim + 0.4 * neighbor_sim
    
#     # Select top candidates
#     candidates = np.argsort(combined_score)[-num_candidates:]
    
#     return candidates


# def beam_search_ordering(sim_matrix, beam_width=10, num_starts=8):
#     """
#     Improved beam search with better heuristics
#     """
#     N = sim_matrix.shape[0]
    
#     # Find good starting candidates
#     start_candidates = find_best_starting_frame(sim_matrix, num_starts)
    
#     best_path = None
#     best_score = -1e9
    
#     for start_idx in start_candidates:
#         # Initialize beam with starting frame
#         beam = [(0.0, [start_idx], {start_idx})]
        
#         for step in range(N - 1):
#             new_beam = []
            
#             for score, path, used in beam:
#                 last = path[-1]
#                 remaining = [i for i in range(N) if i not in used]
                
#                 if not remaining:
#                     continue
                
#                 # Get similarities to all remaining frames
#                 sims = sim_matrix[last, remaining]
                
#                 # Add small bonus for maintaining smooth transitions
#                 if len(path) >= 2:
#                     prev = path[-2]
#                     # Frames that are similar to both prev and current get bonus
#                     for idx, frame_id in enumerate(remaining):
#                         similarity_to_prev = sim_matrix[prev, frame_id]
#                         # Small smoothness bonus
#                         sims[idx] += 0.1 * similarity_to_prev
                
#                 # Select top-k candidates for beam
#                 k = min(beam_width, len(remaining))
#                 if k > 0:
#                     top_k_indices = np.argsort(sims)[-k:]
                    
#                     for idx in top_k_indices:
#                         next_frame = remaining[idx]
#                         new_score = score + sims[idx]
#                         new_beam.append((new_score, path + [next_frame], used | {next_frame}))
            
#             if not new_beam:
#                 break
            
#             # Keep best beams
#             new_beam.sort(key=lambda x: x[0], reverse=True)
#             beam = new_beam[:beam_width]
        
#         # Find best path from this starting point
#         if beam:
#             best_from_start = max(beam, key=lambda x: x[0])
#             if best_from_start[0] > best_score:
#                 best_score = best_from_start[0]
#                 best_path = best_from_start[1]
    
#     return best_path


# def check_and_reverse(sim_matrix, order):
#     """
#     Check if order should be reversed
#     """
#     forward_score = sum(sim_matrix[order[i], order[i+1]] for i in range(len(order)-1))
    
#     reversed_order = order[::-1]
#     reverse_score = sum(sim_matrix[reversed_order[i], reversed_order[i+1]] for i in range(len(reversed_order)-1))
    
#     if reverse_score > forward_score * 1.05:  # 5% threshold to avoid unnecessary flips
#         return reversed_order
#     return order


# def iterative_refinement(sim_matrix, order, max_iterations=3):
#     """
#     Simple refinement: try small swaps to improve local transitions
#     """
#     N = len(order)
    
#     for iteration in range(max_iterations):
#         improved = False
        
#         # Try swapping adjacent pairs if it improves score
#         for i in range(N - 1):
#             if i == 0:
#                 before_score = 0
#             else:
#                 before_score = sim_matrix[order[i-1], order[i]]
            
#             if i >= N - 2:
#                 after_score = 0
#             else:
#                 after_score = sim_matrix[order[i+1], order[i+2]] if i+2 < N else 0
            
#             current_score = sim_matrix[order[i], order[i+1]]
            
#             # Try swap
#             if i == 0:
#                 new_before = 0
#             else:
#                 new_before = sim_matrix[order[i-1], order[i+1]]
            
#             new_current = sim_matrix[order[i+1], order[i]]
            
#             if i >= N - 2:
#                 new_after = 0
#             else:
#                 new_after = sim_matrix[order[i], order[i+2]] if i+2 < N else 0
            
#             old_total = before_score + current_score + after_score
#             new_total = new_before + new_current + new_after
            
#             if new_total > old_total:
#                 # Swap
#                 order[i], order[i+1] = order[i+1], order[i]
#                 improved = True
        
#         if not improved:
#             break
    
#     return order


# def hybrid_ordering(sim_matrix, beam_width=12):
#     """
#     Main ordering function
#     """
#     # Use beam search
#     order = beam_search_ordering(sim_matrix, beam_width=beam_width, num_starts=8)
    
#     # Check direction
#     order = check_and_reverse(sim_matrix, order)
    
#     # Light refinement
#     order = iterative_refinement(sim_matrix, order, max_iterations=2)
    
#     return order
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
        # Start from frame with highest average similarity
        start_idx = np.argmax(sim_matrix.sum(axis=1))
    
    path = [start_idx]
    used = {start_idx}
    
    current = start_idx
    
    while len(path) < N:
        remaining = [i for i in range(N) if i not in used]
        if not remaining:
            break
        
        # Find nearest neighbor
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
        # Find a good middle frame
        start_idx = np.argmax(sim_matrix.sum(axis=1))
    
    # Build forward chain
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
    
    # Try backward from start
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
    
    # Combine: backward (reversed) + start + forward[1:]
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
    
    # Get average similarity for each frame
    avg_sim = sim_matrix.sum(axis=1) / (N - 1)
    
    # Endpoints often have LOWER average similarity
    # But also check the top similar frames
    sorted_indices = np.argsort(avg_sim)
    
    # Take bottom candidates (potential endpoints)
    bottom_candidates = sorted_indices[:num_candidates].tolist()
    
    # Also take top candidates (potential middle frames)
    top_candidates = sorted_indices[-num_candidates:].tolist()
    
    return bottom_candidates + top_candidates


def tsp_nearest_insertion(sim_matrix):
    """
    Use nearest insertion heuristic (TSP-like approach)
    """
    N = sim_matrix.shape[0]
    
    # Start with the two most similar frames
    max_sim = -1
    best_pair = (0, 1)
    
    for i in range(N):
        for j in range(i+1, N):
            if sim_matrix[i, j] > max_sim:
                max_sim = sim_matrix[i, j]
                best_pair = (i, j)
    
    path = list(best_pair)
    used = set(best_pair)
    
    # Iteratively insert remaining frames
    while len(path) < N:
        remaining = [i for i in range(N) if i not in used]
        if not remaining:
            break
        
        best_frame = None
        best_position = None
        best_increase = float('inf')
        
        # Try inserting each remaining frame at each position
        for frame in remaining:
            for pos in range(len(path) + 1):
                # Calculate cost increase
                if pos == 0:
                    # Insert at beginning
                    increase = -sim_matrix[frame, path[0]]
                elif pos == len(path):
                    # Insert at end
                    increase = -sim_matrix[path[-1], frame]
                else:
                    # Insert in middle
                    old_cost = sim_matrix[path[pos-1], path[pos]]
                    new_cost = sim_matrix[path[pos-1], frame] + sim_matrix[frame, path[pos]]
                    increase = -new_cost + old_cost
                
                if increase < best_increase:
                    best_increase = increase
                    best_frame = frame
                    best_position = pos
        
        # Insert best frame at best position
        path.insert(best_position, best_frame)
        used.add(best_frame)
    
    return path


def multi_start_greedy(sim_matrix, num_starts=15):
    """
    Try multiple starting points and pick the best path.
    """
    N = sim_matrix.shape[0]
    
    # Get diverse starting points
    candidates = find_endpoints(sim_matrix, num_candidates=num_starts)
    
    best_path = None
    best_score = -1e9
    
    for start in candidates:
        # Build nearest neighbor chain from this start
        path = nearest_neighbor_chain(sim_matrix, start)
        
        # Score the path
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
    
    # Calculate forward temporal smoothness
    forward_score = 0
    for i in range(N-1):
        forward_score += sim_matrix[order[i], order[i+1]]
    
    # Calculate reverse temporal smoothness
    reversed_order = order[::-1]
    reverse_score = 0
    for i in range(N-1):
        reverse_score += sim_matrix[reversed_order[i], reversed_order[i+1]]
    
    print(f"  → Forward score: {forward_score:.2f}")
    print(f"  → Reverse score: {reverse_score:.2f}")
    
    # Use the direction with higher score
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
                        # Reverse segment [i+1:j+1]
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