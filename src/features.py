import numpy as np
import cv2

def preprocess_frame(frame):
    frame = cv2.resize(frame, (120, 90))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return (blur / 255.0).flatten()


def extract_features(frames):
    return np.array([preprocess_frame(f) for f in frames])
