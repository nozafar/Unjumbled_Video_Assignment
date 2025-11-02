import cv2, os, argparse, numpy as np, imageio
from tqdm import tqdm
import time
import json

from features import (
    extract_color_histograms,
    build_orb_bovw,
    extract_cnn_embeddings,
    compute_optical_flow
)

from ordering import (
    combined_similarity, 
    hybrid_ordering
)


class Timer:
    def __init__(self):
        self.times = {}
        self.starts = {}
    
    def start(self, name):
        self.starts[name] = time.time()
    
    def stop(self, name):
        if name in self.starts:
            elapsed = time.time() - self.starts[name]
            self.times[name] = elapsed
    
    def print_summary(self):
        for name, t in self.times.items():
            print(f"  {name}: {t:.2f}s")
    
    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.times, f, indent=2)


def extract_frames(video_path, expected_frames=300):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if expected_frames and len(frames) != expected_frames:
        print(f"⚠ Warning: expected {expected_frames} frames but got {len(frames)}")
    return frames


def write_video(frames, order, out_path='reconstructed.mp4', fps=30):
    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="mpeg4",
        quality=9,
        macro_block_size=1
    )
    
    for idx in order:
        frame = frames[idx][:, :, ::-1]
        writer.append_data(frame)
    
    writer.close()


def main(args):
    timer = Timer()
    timer.start("total")
    
    print("\n" + "="*60)
    print("VIDEO RECONSTRUCTION PIPELINE")
    print("="*60)
    
    print("\n[1/6] Extracting frames...")
    timer.start("extract_frames")
    frames = extract_frames(args.input, expected_frames=300)
    timer.stop("extract_frames")
    print(f"✓ Extracted {len(frames)} frames")
    
    print("\n[2/6] Computing CNN embeddings (ResNet50)...")
    timer.start("features_cnn")
    cnn_feats = extract_cnn_embeddings(frames)
    timer.stop("features_cnn")
    print(f"✓ CNN features: {cnn_feats.shape}")
    
    print("\n[3/6] Computing ORB visual words...")
    timer.start("features_orb")
    orb_feats = build_orb_bovw(frames, voc_k=200, sample_limit=25000, verbose=True)
    timer.stop("features_orb")
    print(f"✓ ORB features: {orb_feats.shape}")
    
    print("\n[4/6] Computing color histograms...")
    timer.start("features_color")
    color_feats = extract_color_histograms(frames, bins=64)
    timer.stop("features_color")
    print(f"✓ Color features: {color_feats.shape}")
    
    print("\n[5/6] Computing optical flow...")
    timer.start("features_flow")
    flow_feats = compute_optical_flow(frames)
    timer.stop("features_flow")
    print(f"✓ Flow features: {flow_feats.shape}")
    
    print("\n[6/6] Building similarity matrix...")
    timer.start("similarity_matrix")
    sim = combined_similarity(
        color_feats, orb_feats, cnn_feats, flow_feats,
        wc=0.03,
        wo=0.12,   
        wd=0.75,   
        wf=0.10    
    )
    timer.stop("similarity_matrix")
    print(f"✓ Similarity matrix: {sim.shape}")
    
    print("\n[7/7] Finding optimal frame ordering...")
    print(f"  → Trying multiple strategies...")
    timer.start("ordering")
    order = hybrid_ordering(sim, beam_width=args.beam)
    timer.stop("ordering")
    print(f"✓ Order computed: {len(order)} frames")
    
    print("\n[8/8] Writing output video...")
    timer.start("write_video")
    write_video(frames, order, out_path=args.output, fps=args.fps)
    timer.stop("write_video")
    
    timer.stop("total")
    
    print("\n" + "="*60)
    print("✅ RECONSTRUCTION COMPLETE")
    print("="*60)
    print(f"Output: {args.output}")
    print(f"\n⏱  Total time: {timer.times['total']:.2f}s")
    print("\nDetailed timing:")
    timer.print_summary()
    timer.to_json("execution_time.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Frame Reconstruction")
    parser.add_argument("--input", required=True, help="Path to jumbled_video.mp4")
    parser.add_argument("--output", default="reconstructed.mp4", help="Output file name")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS")
    parser.add_argument("--beam", type=int, default=15, help="Beam width (10-20 recommended)")
    
    args = parser.parse_args()
    
    if args.fps != 30:
        print("⚠ Overriding fps to 30 (required)")
        args.fps = 30
    
    main(args)