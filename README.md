# Real-Time Prediction of the Flow-Liquefaction Instability Point in Cyclic Triaxial Tests

This repository provides the code and data for the paper:

> **Real-Time Prediction of the Flow-Liquefaction Instability Point in Cyclic Triaxial Tests**

A sliding-window Transformer framework with a dual-head loss is proposed for real-time prediction of the instability point of flow liquefaction. The framework uses only the real-time stress, strain, and pore-pressure signals recorded during the test, without relying on specimen parameters such as relative density or CSR. Validation on 42 cyclic triaxial tests shows that the proposed method achieves an F1 score of 0.946 under leave-one-out cross-validation and 0.914 on the independent test set, outperforming all baseline models.

## Quick Start

```bash
pip install -r requirements.txt
python predict.py
```

## Dataset

`data/liquefaction_dataset.npz` contains 42 undrained stress-controlled cyclic triaxial tests (SJT-01 to SJT-35, SJF-01 to SJF-03, ZNF-01 to ZNF-04). See `data/README.md` for details.

## Code Structure

```
├── data/                   Dataset (42 experiments, NPZ format)
├── models/
│   ├── transformer.py      NoPatchTransformer (proposed model)
│   ├── lstm.py             LSTM baseline
│   ├── fft_mlp.py          FFT+MLP baseline
│   └── xgboost_model.py    XGBoost baseline
├── configs/
│   ├── transformer.yaml    Transformer hyperparameters
│   └── baselines.yaml      Baseline hyperparameters
├── preprocess.py           Condition normalization and sliding window
├── train.py                Unified training entry (LOOCV + test evaluation)
├── ensemble.py             Mid-band ensemble selection
├── predict.py              Inference with pre-trained ensemble
├── checkpoints/            15 pre-trained model weights
└── results/                Training output directory
```

`preprocess.py` loads the NPZ dataset, applies condition normalization to produce 5 input channels (normalized deviatoric stress, normalized effective confining pressure, pore pressure ratio, axial strain, per-cycle EPWP increment), and extracts sliding windows of length L=80 (one loading cycle).

`models/transformer.py` implements the NoPatchTransformer: pointwise linear projection with learnable positional embeddings, followed by a Transformer encoder, mean pooling, and dual output heads (regression + multi-threshold classification).

`models/lstm.py`, `models/fft_mlp.py`, and `models/xgboost_model.py` implement the three baseline models described in the paper.

`train.py` runs LOOCV on the 30 training experiments and evaluates on the 5 independent test experiments. It supports all four models through a unified interface.

`ensemble.py` implements the mid-band ensemble strategy: train multiple models with different random seeds, rank by LOOCV F1, discard top-ranked (potential overfit) and bottom-ranked (underfit) models, and select a contiguous block from the middle band.

`predict.py` loads the 15 pre-trained checkpoints and runs ensemble inference.

## Training

```bash
# Transformer (single seed)
python train.py transformer 42

# Transformer (100 seeds for ensemble selection)
python train.py transformer 0 99

# Baselines
python train.py lstm 0 99
python train.py fft_mlp 0 99
python train.py xgboost 0 99
```

## Mid-band Ensemble

After training 100 seeds, select the mid-band ensemble:

```bash
python ensemble.py transformer
```

The strategy trains N models with different random seeds, ranks them by LOOCV F1, discards the top (overfit) and bottom (underfit) portions, and selects K consecutive models from the middle band whose ensemble probability averaging maximizes LOOCV F1.

## Pre-trained Weights

`checkpoints/` contains 15 pre-trained NoPatchTransformer weights selected by the mid-band strategy. Run `python predict.py` to evaluate them directly.

## Citation

```bibtex
[To be added upon acceptance]
```

## License

[To be determined]
