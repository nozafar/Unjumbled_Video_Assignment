import cv2
import numpy as np
import time
import argparse


def extract_frames(video_path):
    print("Extracting frames from video into memory...")
    cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    print(f"✅ Extracted {len(frames)} frames (in memory, no disk usage)")
    return frames


def preprocess_frame(frame, target_size=(120, 90)):
    try:
        frame = cv2.resize(frame, target_size)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return (blur / 255.0).flatten()
    except Exception:
        return np.zeros(target_size[0] * target_size[1])


def compute_frame_differences(frames):
    print("\nComputing similarity matrix (this is the heavy part)...")
    n = len(frames)
    diff = np.zeros((n, n))

    processed = [preprocess_frame(f) for f in frames]

    for i in range(n):
        if i % 50 == 0:
            print(f"  Computing similarities: {i}/{n}")
        for j in range(n):
            if i != j:
                diff[i, j] = np.linalg.norm(processed[i] - processed[j])

    return diff


def find_start_frame(diff):
    return np.argmax(np.sum(diff, axis=1))


def reconstruct_sequence(diff, start):
    print("\nReconstructing best possible frame order...")
    n = diff.shape[0]
    visited = set([start])
    seq = [start]
    current = start

    for i in range(n - 1):
        d = diff[current].copy()
        d[list(visited)] = float('inf')
        nxt = np.argmin(d)
        seq.append(nxt)
        visited.add(nxt)
        current = nxt

    return seq


def create_video_from_sequence(frames, seq, output, fps=30):
    print("\nCreating output video...")

    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for i, idx in enumerate(seq):
        if i % 50 == 0:
            print(f"  Writing frame {i}/{len(seq)}")
        out.write(frames[idx])

    out.release()
    print(f"✅ Video saved: {output}")


def main():
    parser = argparse.ArgumentParser(description="Unjumbled Video Reconstruction — In Memory Version")
    parser.add_argument("--input", "-i", default="jumbled_video.mp4")
    parser.add_argument("--output", "-o", default="reconstructed_video.mp4")
    args = parser.parse_args()

    print("\n🎥 Unjumbled Video Reconstruction Started...\n")
    t_start = time.time()

    frames = extract_frames(args.input)
    diff = compute_frame_differences(frames)
    start_index = find_start_frame(diff)
    sequence = reconstruct_sequence(diff, start_index)
    create_video_from_sequence(frames, sequence, args.output)

    elapsed = time.time() - t_start

    print("\n✅ Reconstruction Complete!")
    print(f"⏱ Total Time: {elapsed:.2f} seconds")
    print(f"📌 Output File: {args.output}")


if __name__ == "__main__":
    main()
