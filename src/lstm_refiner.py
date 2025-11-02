import torch
import torch.nn as nn
import numpy as np

class OrderRefiner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lstm = nn.LSTM(dim, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x → (1, T, D)
        out, _ = self.lstm(x)
        score = self.fc(out).squeeze(-1)   # shape: (T,)
        return score


def refine_order_lstm(cnn_feats, order, epochs=12, lr=0.001):
    """
    cnn_feats: (N, D)
    order: initial ordering list of frame indexes
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Arrange CNN embeddings in initial order
    seq = cnn_feats[order]
    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

    model = OrderRefiner(seq.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Target: smooth increasing sequence
    T = seq.shape[1]
    target = torch.linspace(0, 1, T).to(device).unsqueeze(0)

    for _ in range(epochs):  # lightweight training loop
        optimizer.zero_grad()
        pred = model(seq)
        loss = nn.MSELoss()(pred, target)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_scores = model(seq).cpu().numpy().flatten()

    refined = sorted(range(len(order)), key=lambda i: final_scores[i])
    refined_order = [order[i] for i in refined]
    return refined_order
