# src/features.py
import cv2, numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import torch
import torchvision.transforms as T
from torchvision.models import resnet18

def extract_cnn_embeddings(frames):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier
    model.to(device)
    model.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    feats = []
    with torch.no_grad():
        for frame in frames:
            x = transform(frame).unsqueeze(0).to(device)
            f = model(x).cpu().numpy().flatten()
            feats.append(f)

    return np.array(feats).astype(np.float32)

def extract_color_histograms(frames, bins=32):
    hists = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        h = []
        for ch in range(3):
            hist = cv2.calcHist([hsv], [ch], None, [bins], [0,256]).flatten()
            hist = hist / (hist.sum() + 1e-9)
            h.append(hist)
        hists.append(np.concatenate(h))
    return np.vstack(hists).astype(np.float32)

def build_orb_bovw(frames, voc_k=128, sample_limit=20000, verbose=False):
    # Extract ORB descriptors per frame
    orb = cv2.ORB_create(nfeatures=500)
    descs = []
    descs_per_frame = []
    for f in tqdm(frames, desc="ORB detect", disable=not verbose):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        kps, d = orb.detectAndCompute(gray, None)
        if d is None:
            d = np.zeros((0,32), dtype=np.uint8)
        descs_per_frame.append(d)
        if d.shape[0] > 0:
            descs.append(d)
    if len(descs) == 0:
        # fallback: zeros
        return np.zeros((len(frames), voc_k), dtype=np.float32)
    all_desc = np.vstack(descs).astype(np.float32)

    # sample if too many
    n = all_desc.shape[0]
    if n > sample_limit:
        idx = np.random.choice(n, sample_limit, replace=False)
        sample = all_desc[idx]
    else:
        sample = all_desc

    kmeans = MiniBatchKMeans(n_clusters=voc_k, random_state=0, batch_size=4096)
    kmeans.fit(sample)

    histograms = []
    for d in descs_per_frame:
        if d.shape[0] == 0:
            histograms.append(np.zeros(voc_k, dtype=np.float32))
            continue
        labels = kmeans.predict(d.astype(np.float32))
        hist, _ = np.histogram(labels, bins=np.arange(voc_k+1))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-9)
        histograms.append(hist)
    return np.vstack(histograms)
