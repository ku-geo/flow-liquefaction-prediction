import json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

from preprocess import load_dataset, build_windows, PTS_PER_CYCLE
from models.transformer import NoPatchTransformer
from models.lstm import LSTMModel
from models.fft_mlp import FFTMLPModel, extract_fft
from models.xgboost_model import train_xgb, predict_xgb

HERE = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")


def make_model(name, cfg):
    n_thr = len(cfg["loss"]["thresholds"])
    L = cfg["L"]
    if name == "transformer":
        mc = cfg["model"]
        return NoPatchTransformer(
            n_in=mc["n_in"], L=L, d_model=mc["d_model"], n_heads=mc["n_heads"],
            n_layers=mc["n_layers"], dim_ff=mc["dim_ff"], dropout=mc["dropout"],
            n_thr=n_thr).to(DEVICE)
    elif name == "lstm":
        mc = cfg["lstm"]
        return LSTMModel(n_in=5, hidden_size=mc["hidden_size"],
                         num_layers=mc["num_layers"], dropout=mc["dropout"],
                         n_thr=n_thr).to(DEVICE)
    elif name == "fft_mlp":
        mc = cfg["fft_mlp"]
        return FFTMLPModel(input_dim=5 * mc["n_fft"],
                           hidden_sizes=mc["hidden_sizes"],
                           dropout=mc["dropout"], n_thr=n_thr).to(DEVICE)
    raise ValueError(f"Unknown model: {name}")


def nn_input(name, X, cfg):
    if name == "fft_mlp":
        return extract_fft(X, cfg["fft_mlp"]["n_fft"])
    return X


def train_nn(name, X_tr, y_tr, cfg, seed):
    tc, lc = cfg["training"], cfg["loss"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_in = nn_input(name, X_tr, cfg)
    model = make_model(name, cfg)
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_in), torch.FloatTensor(y_tr)),
        batch_size=tc["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"], eta_min=tc["lr"] * 0.01)

    model.train()
    for _ in range(tc["epochs"]):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            reg, thr_logits = model(xb)
            mask = yb < lc["reg_cutoff"]
            loss_reg = torch.tensor(0.0, device=DEVICE)
            if mask.any():
                w = 1.0 + lc["alpha_reg"] / torch.clamp(yb[mask], min=0.5)
                loss_reg = (w * (reg[mask] - yb[mask]) ** 2).mean()
            loss_thr = sum(
                wk * F.binary_cross_entropy_with_logits(logit, (yb <= thr).float())
                for thr, wk, logit in zip(lc["thresholds"], lc["thr_weights"], thr_logits))
            (loss_reg + loss_thr).backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
        sch.step()
    return model


def predict_nn(name, model, X, cfg):
    model.eval()
    X_in = nn_input(name, X, cfg)
    with torch.no_grad():
        reg, thr_logits = model(torch.FloatTensor(X_in).to(DEVICE))
    mu = np.maximum(reg.cpu().numpy(), 0.0)
    probs = [torch.sigmoid(l).cpu().numpy() for l in thr_logits]
    return mu, probs


def compute_f1(y_true_bin, y_pred_bin):
    tp = ((y_true_bin == 1) & (y_pred_bin == 1)).sum()
    fp = ((y_true_bin == 0) & (y_pred_bin == 1)).sum()
    fn = ((y_true_bin == 1) & (y_pred_bin == 0)).sum()
    r = tp / (tp + fn + 1e-8)
    p = tp / (tp + fp + 1e-8)
    return float(2 * r * p / (r + p + 1e-8)), float(r), float(p)


def run_seed(model_name, seed, cfg, train_exps, test_exps):
    L, step = cfg["L"], cfg["step"]
    thresholds = cfg["loss"]["thresholds"]
    n_thr = len(thresholds)
    is_xgb = (model_name == "xgboost")

    out_dir = HERE / "results" / model_name / f"seed_{seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    all_yt, all_mu, all_probs = [], [], []
    for hold in train_exps:
        others = [e for e in train_exps if e["id"] != hold["id"]]
        X_tr, y_tr, _ = build_windows(others, L, step)
        X_val, y_val, _ = build_windows([hold], L, step)
        if len(X_val) == 0:
            continue
        if is_xgb:
            m = train_xgb(X_tr, y_tr, cfg, seed)
            mu, probs = predict_xgb(m, X_val, thresholds)
        else:
            m = train_nn(model_name, X_tr, y_tr, cfg, seed)
            mu, probs = predict_nn(model_name, m, X_val, cfg)
        all_yt.append(y_val)
        all_mu.append(mu)
        all_probs.append(probs)

    yt = np.concatenate(all_yt)
    mu_all = np.concatenate(all_mu)
    merged_probs = [np.concatenate([p[k] for p in all_probs]) for k in range(n_thr)]

    np.savez_compressed(out_dir / "loocv.npz",
        y_true=yt, mu=mu_all,
        **{f"prob_t{k}": merged_probs[k] for k in range(n_thr)})

    p1_true = (yt <= 1.0).astype(int)
    p1_f1, p1_r, p1_p = compute_f1(p1_true, (merged_probs[0] > 0.5).astype(int))
    near = yt < 5.0
    near_mae = float(np.mean(np.abs(yt[near] - mu_all[near]))) if near.any() else float("nan")

    X_full, y_full, _ = build_windows(train_exps, L, step)
    if is_xgb:
        model_full = train_xgb(X_full, y_full, cfg, seed)
        model_full.save_model(str(out_dir / "model.json"))
    else:
        model_full = train_nn(model_name, X_full, y_full, cfg, seed)
        torch.save(model_full.state_dict(), out_dir / "model.pt")

    X_test, y_test, ids_test = build_windows(test_exps, L, step)
    if len(X_test) > 0:
        if is_xgb:
            mu_test, probs_test = predict_xgb(model_full, X_test, thresholds)
        else:
            mu_test, probs_test = predict_nn(model_name, model_full, X_test, cfg)
        np.savez_compressed(out_dir / "test.npz",
            y_true=y_test, mu=mu_test, ids=ids_test,
            **{f"prob_t{k}": probs_test[k] for k in range(n_thr)})
        t_p1 = (y_test <= 1.0).astype(int)
        t_f1, t_r, t_p = compute_f1(t_p1, (probs_test[0] > 0.5).astype(int))
    else:
        t_f1, t_r, t_p = 0, 0, 0

    elapsed = time.time() - t0
    summary = {
        "model": model_name, "seed": seed, "L": L,
        "loocv": {"p1_f1": round(p1_f1, 4), "p1_recall": round(p1_r, 4),
                  "p1_precision": round(p1_p, 4), "near_mae": round(near_mae, 4),
                  "n_folds": len(all_yt), "n_windows": len(yt)},
        "test": {"p1_f1": round(t_f1, 4), "p1_recall": round(t_r, 4),
                 "p1_precision": round(t_p, 4), "n_windows": len(X_test)},
        "elapsed_sec": round(elapsed, 1),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{model_name} seed {seed:3d}  LOOCV={p1_f1:.4f}  Test={t_f1:.4f}  MAE={near_mae:.4f}  {elapsed:.0f}s")
    return summary


def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <model> [seed_start] [seed_end]")
        print("  model: transformer | lstm | fft_mlp | xgboost")
        sys.exit(1)

    model_name = sys.argv[1]
    assert model_name in ("transformer", "lstm", "fft_mlp", "xgboost")

    if model_name == "transformer":
        cfg_path = HERE / "configs" / "transformer.yaml"
    else:
        cfg_path = HERE / "configs" / "baselines.yaml"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_path = str((HERE / "data" / "liquefaction_dataset.npz").resolve())
    all_exps = load_dataset(data_path)
    train_exps = [e for e in all_exps if e["id"].startswith("SJT-") and int(e["id"].split("-")[1]) <= 30]
    test_exps = [e for e in all_exps if e["id"].startswith("SJT-") and int(e["id"].split("-")[1]) > 30]

    args = sys.argv[2:]
    if len(args) == 1:
        seeds = [int(args[0])]
    elif len(args) >= 2:
        seeds = list(range(int(args[0]), int(args[1]) + 1))
    else:
        seeds = list(range(100))

    print(f"{model_name} | L={cfg['L']}, device={DEVICE}, train={len(train_exps)}, test={len(test_exps)}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} total)\n")

    for seed in seeds:
        run_seed(model_name, seed, cfg, train_exps, test_exps)


if __name__ == "__main__":
    main()
