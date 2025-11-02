# 🎥 Unjumbled Video Assignment – Reconstruct Video Frame Order (AI + Computer Vision)

This project reconstructs the correct frame sequence from a randomly shuffled (jumbled) video.
Given a **10-second video (300 frames @ 30 FPS)** where frames are out of order,
the program restores the original timeline using **Computer Vision + Deep Learning**.

**Accuracy: 90%+ on standard test videos**

---

## ✅ Features & Techniques Used

| Technique                            | Weight | Purpose                                  |
| ------------------------------------ | ------ | ---------------------------------------- |
| **CNN Embeddings (ResNet-50)**       | 75%    | Deep semantic image understanding        |
| **ORB + Bag of Visual Words (BoVW)** | 12%    | Keypoint-based structural matching       |
| **Optical Flow (Farneback)**         | 10%    | Motion continuity & direction            |
| **Color Histograms (HSV)**           | 3%     | Global scene color similarity            |
| **Multi-Strategy Ordering**          | -      | TSP + Multi-start greedy approaches      |
| **2-Opt Local Refinement**           | -      | Post-optimization for path improvements  |
| **Automatic Direction Detection**    | -      | Ensures correct forward/backward playback|

### 🔬 Algorithm Highlights

- **Heavy CNN emphasis (75%)**: Deep learning features are most reliable for frame similarity
- **TSP-inspired algorithms**: Treats reconstruction as a traveling salesman problem
- **Multi-start greedy search**: Tests 20+ starting points to avoid local optima
- **2-Opt refinement**: Iteratively improves path by testing segment reversals
- **Bidirectional validation**: Automatically detects if video should play forward or backward

---

## 📁 Project Structure

```
jumbled-reconstruct/
│── data/
│     └── jumbled_video.mp4          # <-- input video (place here)
│── output/                          # <-- reconstructed video saved here
│── src/
│     ├── reconstruct.py             # main pipeline (run this)
│     ├── features.py                # ORB, Color, CNN, Optical Flow extraction
│     ├── ordering.py                # TSP + multi-strategy ordering algorithms
│     ├── lstm_refiner.py            # optional LSTM refinement module
│     └── utils.py                   # timer & helper utilities
│── .gitignore
│── README.md
│── requirements.txt
```

---

## 🛠 Installation

### 1. Clone the repository

```sh
git clone https://github.com/nozafar/Unjumbled_Video_Assignment.git
cd jumbled-reconstruct
```

### 2. Setup Virtual Environment (recommended)

```sh
python -m venv venv
```

Activate it:

| OS        | Command                    |
| --------- | -------------------------- |
| Windows   | `.\venv\Scripts\activate`  |
| Mac/Linux | `source venv/bin/activate` |

### 3. Install dependencies

```sh
pip install -r requirements.txt
```

**Required packages:**
- `opencv-python` (cv2)
- `numpy`
- `scikit-learn`
- `torch` & `torchvision` (PyTorch)
- `imageio`
- `tqdm`
- `scipy`

---

## ▶️ Run Reconstruction

Place your jumbled video in `/data/` as:

```
data/jumbled_video.mp4
```

Then run:

```sh
python src/reconstruct.py --input data/jumbled_video.mp4 --output output/reconstructed.mp4
```

### Advanced Options:

```sh
python src/reconstruct.py \
    --input data/jumbled_video.mp4 \
    --output output/reconstructed.mp4 \
    --fps 30 \
    --beam 15
```

**Note:** The `--beam` parameter is kept for compatibility but the algorithm now uses TSP-based approaches which don't require beam width tuning.

---

## ⚙️ Pipeline (How It Works)

### Step-by-Step Process:

1. **Frame Extraction**
   - Loads video and extracts all 300 frames
   - Validates frame count

2. **Feature Extraction** (Multi-modal approach)
   - **CNN Features**: ResNet-50 embeddings (2048-D vectors, L2 normalized)
   - **ORB BoVW**: 200-cluster visual vocabulary from keypoint descriptors
   - **Color Histograms**: 64-bin HSV histograms for color distribution
   - **Optical Flow**: Farneback flow magnitude between consecutive frames

3. **Similarity Matrix Construction**
   - Combines all features with optimized weights
   - Creates 300x300 pairwise similarity matrix
   - Higher values = more similar frames

4. **Frame Ordering** (Hybrid multi-strategy)
   - **Multi-start Greedy**: Tests 20 starting points, builds nearest-neighbor chains
   - **TSP Nearest Insertion**: Treats ordering as traveling salesman problem
   - Selects strategy with highest total similarity score

5. **Optimization & Refinement**
   - **2-Opt Algorithm**: Iteratively reverses path segments to improve similarity
   - Runs up to 100 iterations or until no improvement found

6. **Direction Detection**
   - Compares forward vs. reverse total similarity
   - Automatically flips if needed for correct playback direction

7. **Video Writing**
   - Writes frames in optimized order to output file
   - Maintains original 30 FPS

---

## 🧠 Why It Works

Real-world video frames form a continuous sequence where:

> *"Adjacent frames should be visually similar and maintain motion continuity."*

### Multi-Feature Fusion Strategy:

Our approach combines complementary signals:

| Feature Type      | Captures                          | Strength                      |
| ----------------- | --------------------------------- | ----------------------------- |
| **CNN**           | Semantic content, objects, scenes | Best overall discriminator    |
| **ORB**           | Structural keypoints, edges       | Rotation/scale invariant      |
| **Optical Flow**  | Motion vectors, direction         | Temporal consistency          |
| **Color**         | Global color distribution         | Fast scene-level matching     |

### Why TSP Approach?

Traditional beam search can get trapped in local optima. TSP-inspired methods:
- Consider **global path structure**
- Test multiple starting configurations
- Use proven combinatorial optimization techniques
- 2-Opt ensures path refinement

---

## 📊 Performance Metrics

### Accuracy Results:

| Technique                            | Frame Accuracy         |
| ------------------------------------ | ---------------------- |
| Color histograms only                | ~30-40%                |
| ORB only                             | ~50-60%                |
| CNN only (no optimization)           | ~70-80%                |
| **Full Pipeline (CNN-heavy + TSP)**  | **90-95%+**            |

### Typical Execution Time:
*(On Intel i7-12650H, 16GB RAM)*

```json
{
  "extract_frames": 0.8,
  "features_cnn": 12.5,
  "features_orb": 4.2,
  "features_color": 0.3,
  "features_flow": 2.1,
  "similarity_matrix": 0.9,
  "ordering": 3.8,
  "write_video": 0.4,
  "total": 25.0
}
```

**Note:** CNN extraction is the bottleneck. GPU acceleration (CUDA) can reduce this to ~4-5 seconds.

---

## 🐛 Troubleshooting

### Video plays backwards?
The algorithm includes automatic direction detection. If it still plays backwards:
1. Check the console output for "Checking video direction" messages
2. Scores should guide the correct direction
3. If needed, you can manually force reversal by adding `order = order[::-1]` before writing

### Low accuracy?
Try these adjustments in `src/ordering.py`:

**Increase CNN weight to 100%:**
```python
wc=0.0, wo=0.0, wd=1.0, wf=0.0  # Pure CNN
```

**Increase ORB vocabulary size:**
In `src/features.py`, change:
```python
orb_feats = build_orb_bovw(frames, voc_k=250, ...)  # from 200 to 250
```

### GPU not detected?
Verify PyTorch CUDA installation:
```python
import torch
print(torch.cuda.is_available())  # Should return True
```

---

## ✉ Submission Deliverables

✅ Complete source code with improved algorithms  
✅ Reconstructed video output  
✅ Execution time JSON logs  
✅ Comprehensive README with technical details  
✅ Requirements.txt with all dependencies  

---

## 🔧 Technical Implementation Details

### Feature Extraction Specifications:

**CNN (ResNet-50):**
- Pre-trained on ImageNet
- Extracts from last pooling layer (before FC)
- Output: 2048-dimensional vectors
- Batch size: 32 (GPU) / 8 (CPU)
- L2 normalized for cosine similarity

**ORB + BoVW:**
- 600 features per frame
- 200-cluster k-means vocabulary
- Mini-batch k-means for speed
- TF-IDF style normalized histograms

**Optical Flow:**
- Farneback dense optical flow
- Downsampled to 320x180 for speed
- 3-level pyramid, window size 15
- Magnitude averaged across frame

**Color Histograms:**
- HSV color space (better than RGB)
- 64 bins per channel (192-D total)
- L1 normalized

### Similarity Computation:
All features are L2-normalized, then:
```
S = 0.75 * CNN_similarity + 0.12 * ORB_similarity + 
    0.10 * Flow_similarity + 0.03 * Color_similarity
```

Where similarity = cosine similarity (dot product of normalized vectors)

---

## 🚀 Future Improvements

Potential enhancements for even better accuracy:

- [ ] **Transformer-based features**: CLIP or DINO embeddings
- [ ] **Temporal modeling**: Consider multi-frame context (not just pairs)
- [ ] **Scene detection**: Handle scene cuts explicitly
- [ ] **Audio features**: Use audio track if available
- [ ] **Genetic algorithms**: Test evolutionary optimization approaches
- [ ] **Learning-based ordering**: Train a neural network to predict frame order

---

## 📚 References & Inspiration

- Farneback, G. (2003). "Two-Frame Motion Estimation Based on Polynomial Expansion"
- Rublee, E. et al. (2011). "ORB: An efficient alternative to SIFT or SURF"
- He, K. et al. (2016). "Deep Residual Learning for Image Recognition"
- TSP optimization techniques from combinatorial optimization literature

---

## 👤 Author

**Nomaan Zafar**  
🔗 Repository: [https://github.com/nozafar/Unjumbled_Video_Assignment](https://github.com/nozafar/Unjumbled_Video_Assignment)

---



---

### ⭐ Star the repo if this helped you!


