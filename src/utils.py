import cv2
import json
import time
from datetime import datetime


class Timer:
    def __init__(self):
        self.start_time = None
        self.records = {}

    def start(self, label):
        self.records[label] = -time.time()  

    def stop(self, label):
        self.records[label] += time.time()  

    def to_json(self, filename="execution_log.json", extra=None):
        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "execution_summary": self.records,
        }

        if extra:
            log_data.update(extra)

        with open(filename, "w") as f:
            json.dump(log_data, f, indent=4)

        print(f"📝 Execution log saved → {filename}")


def create_video_from_sequence(frames, seq, output, fps=30):
    print("Creating output video...")

    sample = cv2.imread(frames[0])
    h, w = sample.shape[:2]
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for idx in seq:
        frame = cv2.imread(frames[idx])
        if frame is not None:
            out.write(frame)

    out.release()
    print(f"✅ Video saved: {output}")
