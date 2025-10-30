import cv2, os, argparse, numpy as np, imageio
from tqdm import tqdm


from features import (
    extract_color_histograms,
    build_orb_bovw,
    extract_cnn_embeddings
)

from ordering import combined_similarity, greedy_beam_order
from utils import Timer


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
        print(f"Warning: expected {expected_frames} frames but got {len(frames)}")
    return frames


def write_video(frames, order, out_path='reconstructed.mp4', fps=30):
    writer = imageio.get_writer(out_path, fps=fps, codec="mpeg4")  # ✅ works on Windows
    for idx in order:
        frame = frames[idx][:, :, ::-1]
        writer.append_data(frame)
    writer.close()


def main(args):
    timer = Timer()
    timer.start("total")

    timer.start("extract_frames")
    frames = extract_frames(args.input, expected_frames=300)
    timer.stop("extract_frames")

    timer.start("features_color")
    color_feats = extract_color_histograms(frames, bins=32)
    timer.stop("features_color")


    timer.start("features_orb")
    orb_feats = build_orb_bovw(frames, voc_k=128, sample_limit=15000, verbose=True)
    timer.stop("features_orb")
    timer.start("features_cnn")
    cnn_feats = extract_cnn_embeddings(frames)
    timer.stop("features_cnn")

    timer.start("similarity_matrix")
    sim = combined_similarity(
        color_feats, orb_feats, cnn_feats,
        wc=0.15, 
        wo=0.35, 
        wd=0.50  
    )
    timer.stop("similarity_matrix")


    timer.start("ordering")
    order = greedy_beam_order(sim, beam_width=args.beam)
    timer.stop("ordering")


    timer.start("write_video")
    write_video(frames, order, out_path=args.output, fps=args.fps)
    timer.stop("write_video")

    timer.stop("total")
    timer.print_summary()
    timer.to_json("execution_time.json")
    print("✅ Reconstruction completed:", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to jumbled_video.mp4")
    parser.add_argument("--output", default="reconstructed.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--beam", type=int, default=10, help="Beam width (higher = more accuracy)")
    args = parser.parse_args()

    if args.fps != 30:
        print("⚠ Overriding fps to 30")
        args.fps = 30

    main(args)
