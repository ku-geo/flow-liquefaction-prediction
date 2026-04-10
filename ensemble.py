import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def compute_f1(y_true_bin, y_pred_bin):
    tp = ((y_true_bin == 1) & (y_pred_bin == 1)).sum()
    fp = ((y_true_bin == 0) & (y_pred_bin == 1)).sum()
    fn = ((y_true_bin == 1) & (y_pred_bin == 0)).sum()
    r = tp / (tp + fn + 1e-8)
    p = tp / (tp + fp + 1e-8)
    return float(2 * r * p / (r + p + 1e-8))


def select_midband(results_dir, K=15, search_start=0.3, search_end=0.8):
    results_dir = Path(results_dir)
    seeds = []
    for d in sorted(results_dir.glob("seed_*")):
        sj = d / "summary.json"
        if not sj.exists():
            continue
        with open(sj) as f:
            s = json.load(f)
        seeds.append((int(d.name.split("_")[1]), s["loocv"]["p1_f1"], d))

    seeds.sort(key=lambda x: x[1], reverse=True)
    n = len(seeds)
    lo = int(n * search_start)
    hi = int(n * search_end)

    loocv_probs = {}
    y_bin = None
    for seed_id, _, d in seeds:
        p = d / "loocv.npz"
        if not p.exists():
            continue
        data = np.load(p)
        loocv_probs[seed_id] = data["prob_t0"]
        if y_bin is None:
            y_bin = (data["y_true"] <= 1.0).astype(np.int8)

    best_f1, best_start = -1, lo
    for start in range(lo, hi - K + 1):
        idx = [seeds[i][0] for i in range(start, start + K)]
        avg = np.mean([loocv_probs[s] for s in idx], axis=0)
        f1 = compute_f1(y_bin, (avg > 0.5).astype(np.int8))
        if f1 > best_f1:
            best_f1, best_start = f1, start

    selected = [seeds[i] for i in range(best_start, best_start + K)]
    print(f"Mid-band ensemble: K={K}, rank {best_start}-{best_start+K-1}")
    print(f"LOOCV F1 (ensemble): {best_f1:.4f}")
    for seed_id, f1, d in selected:
        print(f"  seed {seed_id:3d}  LOOCV F1={f1:.4f}")

    return [d / "model.pt" for _, _, d in selected]


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "transformer"
    results_dir = HERE / "results" / model
    select_midband(results_dir)
