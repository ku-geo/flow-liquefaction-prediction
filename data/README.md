# Data

Dataset for: **Data-driven Real-time Prediction of Flow Liquefaction Instability in Cyclic Triaxial Tests**

## File

`liquefaction_dataset.npz` — 42 undrained stress-controlled cyclic triaxial tests (40,260 data points total)

## Experiment ID Convention

```
{Institution}{Material}-{Number}
```

| Code | Institution / Material |
|------|------------------------|
| **SJ** | Shanghai Jiao Tong University (GDS cyclic triaxial system) |
| **ZN** | Central South University (CSU cyclic triaxial system) |
| **T** | Toyoura sand (Gs=2.64, D50=0.162 mm) |
| **F** | Fujian sand (Gs=2.65, D50=0.35 mm) |

| Prefix | Count | Institution + Material |
|--------|-------|------------------------|
| **SJT** | 35 | Shanghai Jiao Tong + Toyoura sand |
| **SJF** | 3 | Shanghai Jiao Tong + Fujian sand |
| **ZNF** | 4 | Central South + Fujian sand (with clay fines) |

## Experimental Conditions

### SJT-01 ~ SJT-35 (SJTU + Toyoura)

- **Apparatus:** GDS cyclic triaxial system, max axial force 10 kN, max confining/back pressure 2 MPa
- **Specimen:** phi50 mm x 100 mm, moist tamping (5 layers, 5% water content)
- **Saturation:** CO2 circulation -> deaired water injection -> back pressure 200 kPa (B > 0.98)
- **Consolidation:** Isotropic, sigma'c ~ 150 kPa
- **Loading:** Sinusoidal, 0.05 Hz, stress-controlled, undrained
- **Sampling:** 80 points per loading cycle

### SJF-01 ~ SJF-03 (SJTU + Fujian)

- **Apparatus:** Same GDS system as SJT series
- **Material:** Fujian standard sand
- **Consolidation:** Isotropic, sigma'c ~ 100-150 kPa
- **Loading:** Sinusoidal, stress-controlled, undrained
- **Sampling:** 80 points per loading cycle

### ZNF-01 ~ ZNF-04 (CSU + Fujian with clay)

- **Apparatus:** CSU cyclic triaxial system (Central South University)
- **Material:** Fujian sand mixed with clay fines (CC = 10-25%)
- **Specimen:** Medium-dense (Dr = 55-60%)
- **Consolidation:** Isotropic, sigma'c ~ 100 kPa
- **Loading:** Sinusoidal, stress-controlled, undrained
- **Sampling:** Originally 200 points/cycle, resampled to 80 points/cycle

## Experiment List

### SJT: Shanghai Jiao Tong + Toyoura Sand (35 experiments)

| ID | Dr (%) | qs (kPa) | q_cyc (kPa) | CSR | sigma'c (kPa) | Nf (cycles) | Data points |
|----|--------|----------|-------------|------|-----------|-------------|-------------|
| SJT-01 | 10 | 0 | 25 | 0.083 | 150 | 13.8 | 1131 |
| SJT-02 | 30 | 9 | 25 | 0.083 | 150 | 34.6 | 2772 |
| SJT-03 | 10 | 9 | 25 | 0.083 | 151 | 18.3 | 1469 |
| SJT-04 | 30 | 18 | 25 | 0.083 | 150 | 28.4 | 2275 |
| SJT-05 | 10 | 18 | 25 | 0.083 | 150 | 11.3 | 910 |
| SJT-06 | 30 | 30 | 25 | 0.083 | 151 | 18.3 | 1470 |
| SJT-07 | 10 | 30 | 25 | 0.083 | 150 | 4.2 | 341 |
| SJT-08 | 20 | 30 | 25 | 0.083 | 150 | 7.2 | 582 |
| SJT-09 | 20 | 9 | 25 | 0.083 | 151 | 26.4 | 2111 |
| SJT-10 | 20 | 0 | 25 | 0.083 | 151 | 22.8 | 1829 |
| SJT-11 | 20 | 18 | 25 | 0.083 | 151 | 22.4 | 1796 |
| SJT-12 | 10 | 0 | 30 | 0.100 | 151 | 5.7 | 456 |
| SJT-13 | 10 | 9 | 30 | 0.100 | 151 | 6.7 | 542 |
| SJT-14 | 30 | 0 | 30 | 0.100 | 133 | 9.8 | 786 |
| SJT-15 | 20 | 9 | 30 | 0.100 | 151 | 7.7 | 619 |
| SJT-16 | 20 | 0 | 30 | 0.100 | 151 | 7.8 | 624 |
| SJT-17 | 10 | 18 | 30 | 0.100 | 151 | 4.2 | 342 |
| SJT-18 | 20 | 18 | 30 | 0.100 | 138 | 9.8 | 787 |
| SJT-19 | 30 | 18 | 30 | 0.100 | 151 | 15.3 | 1227 |
| SJT-20 | 10 | 30 | 30 | 0.100 | 151 | 2.3 | 184 |
| SJT-21 | 20 | 30 | 30 | 0.100 | 151 | 5.2 | 420 |
| SJT-22 | 30 | 30 | 30 | 0.100 | 152 | 9.2 | 743 |
| SJT-23 | 10 | 18 | 36 | 0.120 | 150 | 2.2 | 180 |
| SJT-24 | 20 | 0 | 36 | 0.120 | 151 | 4.8 | 385 |
| SJT-25 | 30 | 12 | 36 | 0.120 | 151 | 9.8 | 784 |
| SJT-26 | 30 | 18 | 36 | 0.120 | 151 | 7.2 | 579 |
| SJT-27 | 10 | 0 | 36 | 0.120 | 151 | 2.7 | 220 |
| SJT-28 | 20 | 30 | 36 | 0.120 | 151 | 1.2 | 99 |
| SJT-29 | 20 | 9 | 36 | 0.120 | 151 | 4.7 | 377 |
| SJT-30 | 30 | 30 | 36 | 0.120 | 152 | 3.2 | 260 |
| SJT-31 | 30 | 0 | 36 | 0.120 | 150 | 6.8 | 558 |
| SJT-32 | 30 | 36 | 36 | 0.119 | 152 | 1.2 | 102 |
| SJT-33 | 10 | 9 | 36 | 0.120 | 150 | 2.8 | 225 |
| SJT-34 | 20 | 0 | 25 | 0.083 | 150 | 22.7 | 1894 |
| SJT-35 | 20 | 18 | 25 | 0.083 | 152 | 26.4 | 2114 |

### SJF: Shanghai Jiao Tong + Fujian Sand (3 experiments)

| ID | Dr (%) | qs (kPa) | q_cyc (kPa) | CSR | sigma'c (kPa) | Nf (cycles) | Data points |
|----|--------|----------|-------------|------|-----------|-------------|-------------|
| SJF-01 | 10 | 0 | 15 | 0.074 | 101 | 24.9 | 2001 |
| SJF-02 | 30 | 0 | 15 | 0.075 | 100 | 59.2 | 4741 |
| SJF-03 | 30 | 0 | 35 | 0.117 | 150 | 8.8 | 808 |

### ZNF: Central South + Fujian Sand with Clay (4 experiments)

| ID | Dr (%) | q_cyc (kPa) | CSR | sigma'c (kPa) | CC (%) | Nf (cycles) | Data points |
|----|--------|-------------|------|-----------|--------|-------------|-------------|
| ZNF-01 | 60 | 20.3 | 0.100 | 102 | 20 | 14.7 | 1207 |
| ZNF-02 | 60 | 18.7 | 0.100 | 93 | 25 | 3.7 | 321 |
| ZNF-03 | 60 | 40.8 | 0.200 | 102 | 15 | 2.6 | 247 |
| ZNF-04 | 55 | 29.1 | 0.150 | 97 | 10 | 9.7 | 807 |

**Parameter ranges (full dataset):**
- Relative density Dr: 10 - 60%
- Static shear stress qs: 0 - 36 kPa
- Cyclic stress amplitude q_cyc: 15 - 40.8 kPa
- Cyclic stress ratio CSR: 0.074 - 0.200
- Effective confining pressure sigma'c: 93 - 152 kPa
- Cycles to liquefaction Nf: 1.2 - 59.2
- Clay content CC: 0 - 25% (ZNF series only)

## Signal Channels

6-channel time series, sampled at 80 points per loading cycle:

| Index | Name | Unit | Physical meaning |
|-------|------|------|------------------|
| 0 | q | kPa | Deviatoric stress |
| 1 | delta_u | kPa | Excess pore water pressure |
| 2 | p_prime | kPa | Mean effective stress |
| 3 | epsilon_a | % | Axial strain |
| 4 | delta_u_p0 | — | Pore pressure ratio delta_u/p0 |
| 5 | cycle | — | Loading cycle number |

## NPZ Format

| Key | Type | Description |
|-----|------|-------------|
| `__meta__` | JSON string | List of metadata dicts (one per experiment) |
| `__col_names__` | JSON string | `["q", "delta_u", "p_prime", "epsilon_a", "delta_u_p0", "cycle"]` |
| `SJT-01` ... `ZNF-04` | float32 [T, 6] | Time series for each experiment |

### Loading example

```python
import json
import numpy as np

ds = np.load("liquefaction_dataset.npz", allow_pickle=True)
meta = json.loads(str(ds["__meta__"]))

# Access one experiment
arr = ds["SJT-01"]  # shape (1131, 6)
print(arr.shape)

# List all experiment IDs
print([m["id"] for m in meta])
```

