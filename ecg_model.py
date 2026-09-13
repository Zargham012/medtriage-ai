"""
ecg_model.py
Lightweight ECG beat classifier for MedTriage AI.

Two modes:
1. TRAINED MODE: If a trained model file (ecg_cnn.pt) exists, load and use it.
2. FALLBACK MODE: If no trained model, use a fast heuristic based on R-R interval
   variability (works reasonably for Normal vs AFib vs PVC-like irregularity),
   so the demo NEVER breaks even if training didn't finish in time.

Classes: ['Normal', 'AFib', 'PVC', 'Other']
"""

import numpy as np
import torch
import torch.nn as nn
import os

CLASSES = ["Normal", "AFib", "PVC", "Other"]
MODEL_PATH = "ecg_cnn.pt"


class TinyECGNet(nn.Module):
    """Very small 1D CNN — trains in a few minutes on a few hundred beats."""

    def __init__(self, n_classes=4, input_len=250):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            flat_dim = self.net(dummy).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.net(x)
        return self.classifier(x)


def load_model():
    """Load trained model if available, else return None (fallback mode)."""
    if os.path.exists(MODEL_PATH):
        model = TinyECGNet()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        return model
    return None


def heuristic_classify(signal: np.ndarray, fs: int = 250):
    """
    Fast, no-training-required fallback classifier.
    Uses R-peak detection + R-R interval variability to guess rhythm class.
    Good enough for a live demo if the CNN isn't trained in time.
    """
    from scipy.signal import find_peaks

    signal = np.asarray(signal).flatten()
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    peaks, _ = find_peaks(signal, distance=fs * 0.3, height=1.0)
    if len(peaks) < 3:
        return "Other", 0.4

    rr_intervals = np.diff(peaks) / fs
    rr_std = np.std(rr_intervals)
    rr_mean = np.mean(rr_intervals)

    if rr_std < 0.05:
        return "Normal", 0.85
    elif rr_std >= 0.05 and rr_std < 0.15:
        return "PVC", 0.65
    else:
        return "AFib", 0.70


def predict(signal: np.ndarray, fs: int = 250):
    """
    Main entry point used by the app.
    Returns (predicted_class, confidence, all_class_probs_dict)
    """
    model = load_model()

    if model is not None:
        seg = np.asarray(signal).flatten()
        if len(seg) < 250:
            seg = np.pad(seg, (0, 250 - len(seg)))
        else:
            seg = seg[:250]
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)
        x = torch.tensor(seg, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).numpy().flatten()
        pred_idx = int(np.argmax(probs))
        return CLASSES[pred_idx], float(probs[pred_idx]), dict(zip(CLASSES, probs.tolist()))

    # Fallback: heuristic mode, still returns a usable result
    label, conf = heuristic_classify(signal, fs)
    probs = {c: (conf if c == label else (1 - conf) / 3) for c in CLASSES}
    return label, conf, probs
