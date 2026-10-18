"""
Computes the Resilience-Accuracy Score (RAS) per dataset via AHP-Entropy
combined weighting of F1, MTTD, and MTTR (Zhou and Chen, 2025). Implements
Section 3.7.3's equations 3.1-3.8; output backs Tables 3.7-3.9 and Figure 3.2.

RAS is computed separately per dataset, not pooled, since MTTD/MTTR sit on
different absolute time scales (BATADAL: hours, from 24h/12h windowing;
SWaT/WaDi: seconds) - confirmed on real data (BATADAL LSTM-AE MTTD=100,800s
vs SWaT LSTM-AE MTTD=21.8s). Pooling would make BATADAL look
catastrophically slow purely from sampling rate, not model performance.
This also matches RQ4 (Section 3.7.4): "does the most accurate model rank
most resilient" is a within-dataset question, not a cross-dataset one.

Miss rate (attacks never detected) is a multiplicative penalty on the final
score, RAS_final = RAS_raw * (1 - miss_rate), rather than folded into MTTD -
keeping MTTD interpretable alone ("how fast, when detection succeeds")
while still capping a model that misses most attacks.
"""

import numpy as np

#----Real data, confirmed this session----
#F1 from results/<dataset>_<model>/metrics.json (tracker Section 4)
#MTTD/MTTR/miss_rate from compute_resilience_metrics.py --all output

DATA = {
    "BATADAL": {
        "RF":      {"f1": 0.200, "mttd": 8400.0,   "mttr": 6685.7, "miss": 0.571},
        "IF":      {"f1": 0.067, "mttd": 27600.0,  "mttr": 0.0,    "miss": 0.571},
        "CNN":     {"f1": 0.400, "mttd": 43200.0,  "mttr": 25920.0,"miss": 0.400},
        #LSTM-AE's MTTD is the RAS-vs-F1 divergence case in Section 4.4:
        #2nd by F1 here, but this MTTD dwarfs the other three BATADAL
        #models and drives it to last by RAS - see the ranking check below
        "LSTM-AE": {"f1": 0.222, "mttd": 100800.0, "mttr": 8640.0, "miss": 0.400},
    },
    "WaDi": {
        "RF":      {"f1": 0.075, "mttd": 2.0,    "mttr": 0.0, "miss": 0.833},
        "IF":      {"f1": 0.033, "mttd": 218.5,  "mttr": 0.0, "miss": 0.667},
        #mttd=None: zero attacks ever flagged (F1=0.000, Section 4.2), so
        #no detection latency exists to measure - miss=1.000 already
        #drives RAS_final to zero regardless of what mttd normalizes to
        "CNN":     {"f1": 0.000, "mttd": None,   "mttr": 0.0, "miss": 1.000},
        "LSTM-AE": {"f1": 0.335, "mttd": 45.0,   "mttr": 0.0, "miss": 0.333},
    },
    "SWaT": {
        "RF":      {"f1": 0.105, "mttd": 211.2, "mttr": 3.6,   "miss": 0.571},
        "IF":      {"f1": 0.326, "mttd": 22.5,  "mttr": 22.6,  "miss": 0.286},
        "CNN":     {"f1": 0.112, "mttd": 16.7,  "mttr": 228.5, "miss": 0.308},
        "LSTM-AE": {"f1": 0.294, "mttd": 21.8,  "mttr": 214.6, "miss": 0.154},
    },
}

MODELS = ["RF", "IF", "CNN", "LSTM-AE"]
CRITERIA = ["f1", "mttd", "mttr"]  #f1=benefit (higher better); mttd/mttr=cost (lower better)

#----Step 1: AHP pairwise comparison matrix (subjective weights)----
#Safety-Availability-Control priority ordering (Section 3.7.3): missing or
#slow-catching a real attack outranks a slow-clearing false alarm, so
#MTTD > F1 (classification quality, drives alarm fatigue) > MTTR
#(alert-clearance speed, secondary to detection happening at all).
#Saaty 1-9 scale. This ordering is one defensible judgement among several
#plausible ones, not a derived fact - Section 5.4 (Gap 2) discusses that
#a different ordering could shift BATADAL's LSTM-AE ranking further, or
#partially recover it
#            F1     MTTD   MTTR
AHP_MATRIX = [
    [1,     1/2,   3],   #F1
    [2,     1,     4],   #MTTD
    [1/3,   1/4,   1],   #MTTR
]

def ahp_weights(matrix):
    #principal eigenvector method for AHP weight derivation (equation 3.3)
    M = np.array(matrix, dtype=float)
    eigvals, eigvecs = np.linalg.eig(M)
    max_idx = np.argmax(eigvals.real)        #largest eigenvalue = lambda_max
    weights = eigvecs[:, max_idx].real
    weights = weights / weights.sum()        #normalize to sum=1
    lambda_max = eigvals[max_idx].real
    n = M.shape[0]
    CI = (lambda_max - n) / (n - 1)          #consistency index
    RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12}  #Saaty's random index
    RI = RI_TABLE[n]
    CR = CI / RI if RI > 0 else 0.0          #consistency ratio; <0.1 is acceptable
    return weights, lambda_max, CI, CR

ahp_w, lambda_max, CI, CR = ahp_weights(AHP_MATRIX)
print("=" * 70)
print("AHP subjective weights (from pairwise comparison matrix)")
print("=" * 70)
for c, w in zip(["F1", "MTTD", "MTTR"], ahp_w):
    print(f"  {c:6s}: {w:.4f}")
print(f"  lambda_max={lambda_max:.4f}, CI={CI:.4f}, CR={CR:.4f} "
      f"({'CONSISTENT, CR<0.1' if CR < 0.1 else 'INCONSISTENT, revise matrix'})")

#----Step 2 & 3: per-dataset normalization, entropy weights, RAS----
def minmax_normalize(values, higher_is_better):
    #equations 3.1 (benefit) and 3.2 (cost)
    values = np.array(values, dtype=float)
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return np.ones_like(values)                      #no spread: all equally best
    if higher_is_better:
        return (values - vmin) / (vmax - vmin)            #benefit criterion
    else:
        return (vmax - values) / (vmax - vmin)             #cost criterion, inverted

def entropy_weights(norm_matrix):
    #objective, data-derived weights (equations 3.4-3.5): a criterion where
    #every model scores about the same carries little information for
    #telling models apart, so it gets a low weight here regardless of how
    #the AHP matrix above prioritizes it subjectively - this is exactly
    #why WaDi's MTTR gets zero entropy weight (every WaDi model recorded
    #MTTR=0.0, so there is no variation left for entropy to reward)
    #norm_matrix: rows=alternatives, cols=criteria, values in [0,1]
    eps = 1e-12
    P = norm_matrix + eps
    P = P / P.sum(axis=0, keepdims=True)      #column-proportion normalization
    n = P.shape[0]
    k = 1.0 / np.log(n)
    entropy = -k * (P * np.log(P)).sum(axis=0)
    diversity = 1 - entropy                    #degree of differentiation
    weights = diversity / diversity.sum()
    return weights

results = {}
for dataset, models_data in DATA.items():
    print("\n" + "=" * 70)
    print(f"{dataset}")
    print("=" * 70)

    f1_vals = [models_data[m]["f1"] for m in MODELS]
    mttd_vals = [models_data[m]["mttd"] for m in MODELS]
    mttr_vals = [models_data[m]["mttr"] for m in MODELS]
    miss_vals = [models_data[m]["miss"] for m in MODELS]

    #MTTD=None (100% miss, e.g. WaDi CNN): substitute the group's worst MTTD
    #for normalization only. Final score is unaffected, since miss-rate
    #already zeroes a 100%-miss model; this just avoids a crash/skewed range.
    valid_mttd = [v for v in mttd_vals if v is not None]
    worst_mttd = max(valid_mttd) if valid_mttd else 0.0
    mttd_vals = [v if v is not None else worst_mttd for v in mttd_vals]

    f1_norm = minmax_normalize(f1_vals, higher_is_better=True)
    mttd_norm = minmax_normalize(mttd_vals, higher_is_better=False)
    mttr_norm = minmax_normalize(mttr_vals, higher_is_better=False)

    norm_matrix = np.column_stack([f1_norm, mttd_norm, mttr_norm])
    ent_w = entropy_weights(norm_matrix)

    #combine AHP (subjective) and entropy (objective) weights: multiply then
    #renormalize (Zhou and Chen, 2025, equation 3.6), honouring both expert
    #priority and this dataset's own discriminative structure
    combined_w = (ahp_w * ent_w)
    combined_w = combined_w / combined_w.sum()

    print(f"  Entropy weights (this dataset): F1={ent_w[0]:.4f}, MTTD={ent_w[1]:.4f}, MTTR={ent_w[2]:.4f}")
    print(f"  Combined weights:               F1={combined_w[0]:.4f}, MTTD={combined_w[1]:.4f}, MTTR={combined_w[2]:.4f}")
    print(f"\n  {'Model':<10}{'F1':>7}{'MTTD_n':>8}{'MTTR_n':>8}{'RAS_raw':>9}{'Miss%':>8}{'RAS_final':>11}")

    dataset_results = []
    for i, m in enumerate(MODELS):
        ras_raw = float(np.dot(combined_w, norm_matrix[i]))          #equation 3.7
        ras_final = ras_raw * (1 - miss_vals[i])                     #equation 3.8, miss-rate penalty
        dataset_results.append((m, f1_vals[i], mttd_norm[i], mttr_norm[i], ras_raw, miss_vals[i], ras_final))
        print(f"  {m:<10}{f1_vals[i]:>7.3f}{mttd_norm[i]:>8.3f}{mttr_norm[i]:>8.3f}"
              f"{ras_raw:>9.4f}{miss_vals[i]*100:>7.1f}%{ras_final:>11.4f}")

    ranked = sorted(dataset_results, key=lambda r: -r[6])
    print(f"\n  RAS ranking: {' > '.join(r[0] for r in ranked)}")
    best_f1_model = MODELS[int(np.argmax(f1_vals))]
    best_ras_model = ranked[0][0]
    #NOTE: this only compares the #1-ranked model, not the full order, so
    #it prints AGREES for BATADAL even though LSTM-AE drops from 2nd by F1
    #to last by RAS there - that divergence is the RQ4 finding (Section
    #4.4) and has to be read from the ranking string above, not this line
    agree = "AGREES with F1 ranking" if best_f1_model == best_ras_model else \
            f"DIFFERS from F1 ranking (F1 favours {best_f1_model})"
    print(f"  Best by RAS: {best_ras_model}  |  Best by raw F1: {best_f1_model}  -> {agree}")

    results[dataset] = dataset_results