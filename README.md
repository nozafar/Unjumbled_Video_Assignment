## 🎥 Unjumbled Video Assignment

**AI-based reconstruction of jumbled video frames**

This project takes a shuffled (out-of-order) video, extracts all frames, computes similarity using CNN + classical computer vision, and reconstructs the correct timeline.

---

## ✅ Requirements

### 1️⃣ Clone the repo

```sh
git clone https://github.com/nozafar/Unjumbled_Video_Assignment.git
cd Unjumbled_Video_Assignment
```

### 2️⃣ Create Virtual Environment

```sh
python -m venv venv
```

Activate it:

| OS        | Command                    |
| --------- | -------------------------- |
| Windows   | `.\venv\Scripts\activate`  |
| Mac/Linux | `source venv/bin/activate` |

### 3️⃣ Install dependencies

```sh
pip install -r requirements.txt
```

---

## ▶️ How to Run

1. Place your shuffled video inside the `data/` folder. Example:

```
data/
│── jumbled_video.mp4
```

2. Run reconstruction:

```sh
python src/reconstruct.py --input data/jumbled_video.mp4 --output output/reconstructed.mp4
```

✅ Output will appear in:

```
output/reconstructed.mp4
```

---

## 🔧 Additional Options

| Option          | Meaning               | Default                                           |
| --------------- | --------------------- | ------------------------------------------------- |
| `--fps`         | Output video FPS      | `30`                                              |
| `--save-frames` | Save extracted frames | `False` (no `frames/` folder created unless used) |

Example:

```sh
python src/reconstruct.py --input data/jumbled_video.mp4 --output output/reconstructed.mp4 --fps 24
```

---

## 📂 Project Structure (simple)

```
src/
├── features.py        # Extract CNN, ORB, color features
├── ordering.py        # Frame ordering algorithm
├── reconstruct.py      # Main script (run this)
└── utils.py           # Timer/logging helpers
```

---

## 🧠 What the program does

* Extract frames from video
* Compute similarity between every frame
* Build similarity matrix
* Predict most logical frame order
* Write reconstructed video

---

## ✨ Result

| Input (shuffled) | Output (restored timeline) |
| ---------------- | -------------------------- |
| ❌ Frames jumbled | ✅ Frames in correct order  |

---

## 🔗 Repo

👉 [https://github.com/nozafar/Unjumbled_Video_Assignment](https://github.com/nozafar/Unjumbled_Video_Assignment)
