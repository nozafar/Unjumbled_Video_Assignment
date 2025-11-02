import time
import argparse
import os

from features import extract_frames, compute_frame_differences
from ordering import find_start_frame, reconstruct_sequence
from utils import create_video_from_sequence


def main():
    parser = argparse.ArgumentParser(description="Jumbled Frames Reconstruction")
    parser.add_argument("--input", "-i", default="jumbled_video.mp4")
    parser.add_argument("--output", "-o", default="reconstructed_video.mp4")
    args = parser.parse_args()

    print("=== Jumbled Frames Reconstruction ===")
    start_time = time.time()

    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' not found")
        return 1

    frames = extract_frames(args.input)
    diff = compute_frame_differences(frames)

    start_idx = find_start_frame(diff)
    print(f"▶ Starting frame: {start_idx}")

    seq = reconstruct_sequence(diff, start_idx)
    create_video_from_sequence(frames, seq, args.output)

    print("\n✅ Reconstruction complete")
    print(f"⏱ Execution Time: {time.time() - start_time:.2f} seconds")
    print(f"📤 Output saved: {args.output}")
    return 0


if __name__ == "__main__":
    main()
