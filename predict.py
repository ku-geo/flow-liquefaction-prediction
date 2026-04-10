import json, sys
from pathlib import Path

import numpy as np
import torch
import yaml

from preprocess import load_dataset, build_windows
from models.transformer import NoPatchTransformer

HERE = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")


def load_ensemble(checkpoints_dir, cfg):
    mc = cfg["model"]
    L = cfg["L"]
    n_thr = len(cfg["loss"]["thresholds"])
    models = []
    for pt in sorted(Path(checkpoints_dir).glob("model_*.pt")):
        model = NoPatchTransformer(
            n_in=mc["n_in"], L=L, d_model=mc["d_model"], n_heads=mc["n_heads"],
            n_layers=mc["n_layers"], dim_ff=mc["dim_ff"], dropout=mc["dropout"],
            n_thr=n_thr).to(DEVICE)
        model.load_state_dict(torch.load(pt, map_location=DEVICE, weights_only=True))
        model.eval()
        models.append(model)
    print(f"Loaded {len(models)} models from {checkpoints_dir}")
    return models


def predict_ensemble(models, X):
    all_probs = []
    all_mu = []
    for model in models:
        with torch.no_grad():
            reg, thr_logits = model(torch.FloatTensor(X).to(DEVICE))
        all_mu.append(np.maximum(reg.cpu().numpy(), 0.0))
        all_probs.append([torch.sigmoid(l).cpu().numpy() for l in thr_logits])
    mu = np.mean(all_mu, axis=0)
    n_thr = len(all_probs[0])
    probs = [np.mean([p[k] for p in all_probs], axis=0) for k in range(n_thr)]
    return mu, probs


def compute_f1(y_true_bin, y_pred_bin):
    tp = ((y_true_bin == 1) & (y_pred_bin == 1)).sum()
    fp = ((y_true_bin == 0) & (y_pred_bin == 1)).sum()
    fn = ((y_true_bin == 1) & (y_pred_bin == 0)).sum()
    r = tp / (tp + fn + 1e-8)
    p = tp / (tp + fp + 1e-8)
    return float(2 * r * p / (r + p + 1e-8)), float(r), float(p)


def main():
    with open(HERE / "configs" / "transformer.yaml") as f:
        cfg = yaml.safe_load(f)

    L, step = cfg["L"], cfg["step"]
    data_path = str((HERE / "data" / "liquefaction_dataset.npz").resolve())
    all_exps = load_dataset(data_path)
    train_exps = [e for e in all_exps if e["id"].startswith("SJT-") and int(e["id"].split("-")[1]) <= 30]
    test_exps = [e for e in all_exps if e["id"].startswith("SJT-") and int(e["id"].split("-")[1]) > 30]

    models = load_ensemble(HERE / "checkpoints", cfg)

    for name, exps in [("Training set", train_exps), ("Independent test", test_exps)]:
        X, y, ids = build_windows(exps, L, step)
        if len(X) == 0:
            continue
        mu, probs = predict_ensemble(models, X)
        y_bin = (y <= 1.0).astype(int)
        pred_bin = (probs[0] > 0.5).astype(int)
        f1, r, p = compute_f1(y_bin, pred_bin)
        print(f"\n{name}: {len(X)} windows")
        print(f"  P1-F1={f1:.4f}  Precision={p:.4f}  Recall={r:.4f}")

        if ids is not None and len(set(ids)) > 1:
            for eid in sorted(set(ids)):
                mask = ids == eid
                y_e = (y[mask] <= 1.0).astype(int)
                p_e = (probs[0][mask] > 0.5).astype(int)
                f1_e, _, _ = compute_f1(y_e, p_e)
                print(f"    {eid}: F1={f1_e:.4f} ({mask.sum()} windows)")


if __name__ == "__main__":
    main()
