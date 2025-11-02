import cv2, numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

def compute_optical_flow(frames):
    """Compute optical flow - keep it simple with just magnitude"""
    flows = []
    
    prev = cv2.resize(frames[0], (320, 180))
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    
    for i in range(1, len(frames)):
        frame = frames[i]
        if frame is None:
            flows.append(flows[-1] if flows else 0.0)
            continue
            
        gray = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
    
        mag = np.linalg.norm(flow, axis=2).mean()
        flows.append(mag)
        
        prev = gray
    
    flows.append(flows[-1])
    
    return np.array(flows).reshape(-1, 1).astype(np.float32)


def extract_cnn_embeddings(frames):
    """Extract CNN embeddings with batch processing"""
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    feats = []
    batch_size = 32 if torch.cuda.is_available() else 8
    
    for i in tqdm(range(0, len(frames), batch_size), desc="CNN", disable=False):
        batch = frames[i:i+batch_size]
        batch_tensor = torch.stack([preprocess(f) for f in batch]).to(device)
        
        with torch.no_grad():
            embeddings = model(batch_tensor).squeeze().cpu().numpy()
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            feats.append(embeddings)
    
    feats = np.vstack(feats).astype(np.float32)
    

    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    
    return feats


def extract_color_histograms(frames, bins=64):
    """Extract color histograms - increased bins for better discrimination"""
    hists = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        h = []
        for ch in range(3):
            hist = cv2.calcHist([hsv], [ch], None, [bins], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-9)
            h.append(hist)
        hists.append(np.concatenate(h))
    return np.vstack(hists).astype(np.float32)


def build_orb_bovw(frames, voc_k=200, sample_limit=25000, verbose=False):
    """Build ORB bag-of-visual-words - increased vocabulary size"""
    orb = cv2.ORB_create(nfeatures=600)
    descs = []
    descs_per_frame = []
    
    for f in tqdm(frames, desc="ORB", disable=not verbose):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        kps, d = orb.detectAndCompute(gray, None)
        if d is None:
            d = np.zeros((0, 32), dtype=np.uint8)
        descs_per_frame.append(d)
        if d.shape[0] > 0:
            descs.append(d)
    
    if len(descs) == 0:
        return np.zeros((len(frames), voc_k), dtype=np.float32)
    
    all_desc = np.vstack(descs).astype(np.float32)
    n = all_desc.shape[0]
    
    if n > sample_limit:
        idx = np.random.choice(n, sample_limit, replace=False)
        sample = all_desc[idx]
    else:
        sample = all_desc
    
    kmeans = MiniBatchKMeans(n_clusters=voc_k, random_state=42, batch_size=4096, max_iter=50)
    kmeans.fit(sample)
    
    histograms = []
    for d in descs_per_frame:
        if d.shape[0] == 0:
            histograms.append(np.zeros(voc_k, dtype=np.float32))
            continue
        labels = kmeans.predict(d.astype(np.float32))
        hist, _ = np.histogram(labels, bins=np.arange(voc_k + 1))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-9)
        histograms.append(hist)
    
    return np.vstack(histograms)