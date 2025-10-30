# 🎥 Unjumbled Video Assignment – Reconstruct Video Frame Order (AI + Computer Vision)

This project reconstructs the correct frame sequence from a randomly shuffled (jumbled) video.
Given a **10-second video (300 frames @ 30 FPS)** where frames are out of order,
the program restores the original timeline using **Computer Vision + Deep Learning**.

---

## ✅ Features Used

| Technique                            | Purpose                       |
| ------------------------------------ | ----------------------------- |
| **Color Histograms (HSV)**           | Scene similarity              |
| **ORB + Bag of Visual Words (BoVW)** | Keypoint-based matching       |
| **CNN Embeddings (ResNet-50)**       | Deep image understanding      |
| **Optical Flow (Farneback)**         | Motion direction continuity   |
| **Beam Search Ordering**             | Best frame sequence discovery |
| **Local Refinement Optimization**    | Fixes micro ordering issues   |

---

## 📁 Project Structure

```
Unjumbled_Video_Assignment/
│── data/                     # <-- place jumbled_video.mp4 here
│── output/                   # <-- reconstructed video will be saved here
│── src/
│     ├── reconstruct.py      # main pipeline (run this)
│     ├── features.py         # ORB, Color Hist., CNN, Optical Flow
│     ├── ordering.py         # similarity matrix + greedy beam order
│     ├── utils.py            # timer + helper utils
│── requirements.txt
│── README.md
```

---

## 🛠 Installation

### 1. Clone the repository

```sh
git clone https://github.com/nozafar/Unjumbled_Video_Assignment.git
cd Unjumbled_Video_Assignment
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

---

## ▶️ Run Reconstruction

Place your jumbled video in `/data/` as:

```
data/jumbled_video.mp4
```

Then run:

```sh
python src/reconstruct.py --input data/jumbled_video.mp4 --output output/reconstructed.mp4 --beam 15
```

For **maximum accuracy**:

```sh
python src/reconstruct.py --input data/jumbled_video.mp4 --output output/reconstructed.mp4 --beam 25
```

---

## ⚙️ Pipeline (How It Works)

1. Extract frames from the video
2. Compute feature vectors:

   * Color histogram (global scene matching)
   * ORB keypoints + BoVW vector
   * CNN embeddings (ResNet-50)
   * Optical flow motion vectors
3. Compute a **combined weighted similarity matrix**
4. Perform **Greedy Beam Search** to find best ordering path
5. Apply local refinement optimization
6. Write reconstructed video

---

## 🧠 Why It Works

Real-world video frames are continuous:

> *“Adjacent frames should be visually and motion-wise similar.”*

Using multiple complementary features boosts accuracy:

| Technique                            | Accuracy               |
| ------------------------------------ | ---------------------- |
| Color histograms only                | ❌ Poor                 |
| ORB only                             | ⚠ Medium               |
| **CNN + ORB + Color + Optical Flow** | ✅ 90–99% correct order |

---

## ⏳ Execution Logs

A JSON log is generated automatically:

Example:

```json
{
  "extract_frames": 0.91,
  "features_color": 0.21,
  "features_orb": 3.49,
  "features_cnn": 5.71,
  "features_flow": 2.31,
  "similarity_matrix": 1.34,
  "ordering": 2.91,
  "write_video": 0.27,
  "total": 17.45
}
```

---

## ✉ Submission Deliverables (All Included)

✔ Source Code
✔ Reconstructed Video
✔ Execution Time JSON Logs
✔ Fully documented README

---

## 👤 Author

**Nomaan Zafar**
🔗 Repository: [https://github.com/nozafar/Unjumbled_Video_Assignment](https://github.com/nozafar/Unjumbled_Video_Assignment)

---

### ⭐ Star the repo if this helped you!

---

