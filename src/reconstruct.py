import cv2
import argparse
import time
import numpy as np
from tqdm import tqdm


from features import preprocess_frame, extract_features
from ordering import compute_similarity_matrix, reconstruct_sequence
from utils import Timer


def extract_frames(video_path):
    print("Extracting frames...")
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    print(f"Loaded {len(frames)} frames into memory")
    return frames


def create_video(frames, order, output, fps=30):
    print("\nWriting final video...")

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for idx in tqdm(order, desc="📽 Saving frames", ncols=80):
        writer.write(frames[idx])

    writer.release()
    print(f"Done → {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default="reconstructed.mp4")
    args = parser.parse_args()

    timer = Timer()
    timer.start("total")

    frames = extract_frames(args.input)

    timer.start("features")
    features = extract_features(frames) 
    timer.stop("features")

    timer.start("similarity")
    sim = compute_similarity_matrix(features)  
    timer.stop("similarity")

    timer.start("ordering")
    order = reconstruct_sequence(sim) 
    timer.stop("ordering")

    create_video(frames, order, args.output)

    timer.stop("total")
    timer.print_summary()
    timer.to_json("runtime.json")


if __name__ == "__main__":
    main()
