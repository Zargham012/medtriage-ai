"""
train_ecg_colab.py
Run this IN GOOGLE COLAB if you have 20-30 spare minutes to get a real trained model.
If you don't have time, SKIP THIS FILE ENTIRELY — ecg_model.py automatically falls
back to a heuristic classifier and the app still works end-to-end.

Steps this script does:
1. Downloads a small subset of MIT-BIH Arrhythmia Database (free, via wfdb)
2. Extracts beat segments around annotated R-peaks
3. Labels them into 4 classes: Normal, AFib-like, PVC, Other
4. Trains TinyECGNet for a few epochs (fast on Colab CPU, faster on GPU)
5. Saves ecg_cnn.pt for ecg_model.py to load

Usage in Colab:
    !pip install wfdb torch numpy scipy -q
    # upload ecg_model.py to the Colab session first
    !python train_ecg_colab.py
"""

import wfdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ecg_model import TinyECGNet, CLASSES

# A small set of MIT-BIH records (public, free via PhysioNet) — keep this SHORT for speed
RECORDS = ["100", "101", "106", "119", "200", "223"]
SEGMENT_LEN = 250  # ~1 second at 250Hz, matches TinyECGNet input

# Map MIT-BIH annotation symbols to our 4 simplified classes
SYMBOL_MAP = {
    "N": "Normal", "L": "Normal", "R": "Normal", "e": "Normal", "j": "Normal",
    "A": "AFib", "a": "AFib", "J": "AFib", "S": "AFib",
    "V": "PVC", "E": "PVC",
    "F": "Other", "/": "Other", "f": "Other", "Q": "Other",
}


def extract_beats(record_name, pn_dir="mitdb"):
    record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    ann = wfdb.rdann(record_name, "atr", pn_dir=pn_dir)
    signal = record.p_signal[:, 0]  # first channel

    X, y = [], []
    for sample, symbol in zip(ann.sample, ann.symbol):
        label = SYMBOL_MAP.get(symbol)
        if label is None:
            continue
        start = sample - SEGMENT_LEN // 2
        end = sample + SEGMENT_LEN // 2
        if start < 0 or end > len(signal):
            continue
        seg = signal[start:end]
        X.append(seg)
        y.append(CLASSES.index(label))
    return X, y


def main():
    print("Downloading and extracting beats from MIT-BIH subset (this may take a few minutes)...")
    all_X, all_y = [], []
    for rec in RECORDS:
        try:
            X, y = extract_beats(rec)
            all_X.extend(X)
            all_y.extend(y)
            print(f"Record {rec}: {len(X)} beats extracted")
        except Exception as e:
            print(f"Skipping record {rec} due to error: {e}")

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    print(f"Total beats: {len(X)}")

    # normalize each beat
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    X_tensor = torch.tensor(X).unsqueeze(1)  # (N, 1, SEGMENT_LEN)
    y_tensor = torch.tensor(y)

    model = TinyECGNet(n_classes=len(CLASSES), input_len=SEGMENT_LEN)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 5  # kept low for speed — good enough for a demo-quality model
    BATCH_SIZE = 64
    n = len(X_tensor)

    print("Training...")
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = X_tensor[idx], y_tensor[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "ecg_cnn.pt")
    print("Saved ecg_cnn.pt — copy this file next to ecg_model.py in your Streamlit app.")


if __name__ == "__main__":
    main()
