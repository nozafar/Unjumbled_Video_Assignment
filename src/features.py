import cv2
import numpy as np
import os
import shutil


def extract_frames(video_path, output_dir="frames"):
    print("Extracting frames from video...")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        path = os.path.join(output_dir, f"frame_{count:06d}.jpg")
        cv2.imwrite(path, frame)
        frames.append(path)
        count += 1

    cap.release()
    print(f"Extracted {count} frames")
    return frames


def preprocess_frame(path, target_size=(120, 90)):
    img = cv2.imread(path)
    if img is None:
        return None

    img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return (blur / 255.0).flatten()


def compute_frame_differences(frames):
    print("Computing frame differences (fast mode)...")
    n = len(frames)
    
    print("Preprocessing frames...")
    processed = []
    for path in frames:
        p = preprocess_frame(path)
        processed.append(p if p is not None else np.zeros(80 * 60))
    
    X = np.array(processed, dtype=np.float32)
    
    print("Calculating pairwise distances...")
    # Efficient vectorized computation: ||a - b||^2 = a² + b² - 2ab
    X2 = np.sum(X ** 2, axis=1, keepdims=True)
    diff = np.sqrt(np.maximum(X2 + X2.T - 2 * np.dot(X, X.T), 0))
    
    print("✅ Similarity matrix computed fast")
    return diff

