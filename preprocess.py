import json
from pathlib import Path

import numpy as np

Q, DU, PP, EA, RU, CYC = 0, 1, 2, 3, 4, 5
PTS_PER_CYCLE = 80


def load_dataset(path):
    ds = np.load(path, allow_pickle=True)
    meta = json.loads(str(ds["__meta__"]))
    experiments = []
    for m in meta:
        arr = np.array(ds[m["id"]], dtype=np.float32)
        experiments.append({**m, "arr": arr})
    return experiments


def normalize(arr, q_cyc, p0, qs):
    q_norm = (arr[:, Q] - qs) / q_cyc
    p_norm = arr[:, PP] / p0
    ru = arr[:, RU]
    ea = arr[:, EA]
    cyc = arr[:, CYC]
    dru = np.zeros(len(ru), dtype=np.float32)
    dcyc = np.diff(cyc)
    dcyc[dcyc == 0] = 1e-6
    dru[1:] = np.diff(ru) / dcyc
    dru[0] = dru[1] if len(dru) > 1 else 0
    return np.column_stack([q_norm, p_norm, ru, ea, dru]).astype(np.float32)


def build_windows(experiments, L, step):
    X_list, y_list, ids = [], [], []
    for exp in experiments:
        arr = exp["arr"]
        inst_idx = exp["instability_idx"]
        data = normalize(arr, exp["q_cyc"], exp["sigma_c"], exp.get("qs", 0))
        max_start = inst_idx - L + 1
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, step):
            end = start + L
            X_list.append(data[start:end])
            y_list.append((inst_idx - (end - 1)) / PTS_PER_CYCLE)
            ids.append(exp["id"])
    if not X_list:
        return np.empty((0, L, 5), dtype=np.float32), np.empty(0, dtype=np.float32), np.array([])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), np.array(ids)
