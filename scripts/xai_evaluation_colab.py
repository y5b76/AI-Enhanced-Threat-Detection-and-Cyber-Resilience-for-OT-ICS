"""
Run this in Google Colab (GPU runtime recommended: Runtime > Change runtime
type > GPU), after uploading and unzipping the export(s) produced locally
by export_for_shap.py.

For each dataset/model combination, this:
  1. Loads the trained model and the stratified test sample exported
     locally (never re-derives sampling or preprocessing here - reuses
     exactly what export_for_shap.py already produced, so there is no risk
     of drift between what was trained and what gets explained)
  2. Applies the appropriate SHAP explainer (Section 3.8.1, Figure 3.3):
     TreeExplainer for RF/IF (exact, fast), GradientExplainer for CNN, and
     a reconstruction-error wrapper + GradientExplainer for LSTM-AE (since
     it is an autoencoder, not a classifier - there is no single class
     output to explain, so the scalar reconstruction error itself is the
     explained quantity)
  3. Computes three quantitative XAI metrics (Section 3.8.2), following the
     definitions used in Kalakoti et al. (2025) and implemented in the
     Quantus toolkit (Hedstrom et al., 2023) - implemented here directly
     rather than via the Quantus library itself, since Quantus is built
     primarily for single-framework (PyTorch/TensorFlow) differentiable
     models and this pipeline mixes sklearn tree models with Keras models:
       - Faithfulness (Bhatt et al., 2020-style correlation): does masking
         the features an explanation says matter actually change the
         model's output by a proportional amount?
       - Robustness (Yeh et al., 2019-style max-sensitivity): how much
         does the explanation itself change under small input
         perturbations? Lower = more stable.
       - Complexity (entropy of normalised |attribution|): how
         concentrated is the explanation on a few features versus spread
         thinly across many? Lower = more parsimonious, easier for an
         operator to act on.
     Sampling sizes and background-set construction follow Section 3.8.3.
  4. OPTIONALLY (PLOT_DIAGNOSTICS below) saves two illustrative figures for
     the dissertation (Figures 4.2 and 4.3): for Isolation Forest, a
     scatter plot of the single worst (most negative) faithfulness
     instance, since IF is the only model with negative faithfulness at
     all (Section 4.5); for LSTM-AE, a bar chart of per-feature
     attribution stability for the single most stable instance found,
     since LSTM-AE's low robustness values are this study's headline
     stability finding. Both plots are built from data the existing loops
     below already compute - no extra SHAP calls, no extra model
     inference, just capturing and visualising numbers that were being
     computed anyway.

Usage: set DATASET below to "batadal", "wadi", or "swat", then Run All.
Re-run once per dataset (after uploading that dataset's export).
"""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

#============================================================
#CONFIGURATION - edit before each run
#============================================================
DATASET = "batadal"        #"batadal", "wadi", or "swat"
EXPORT_ROOT = f"/content/{DATASET}"   #where you unzipped <dataset>_xai_export.zip

N_TRIALS_FAITHFULNESS = 15    #random feature-subsets tested per instance
N_PERTURBATIONS_ROBUSTNESS = 10   #perturbed copies tested per instance
PERTURBATION_EPSILON = 0.05   #relative to [0,1]-scaled feature range (CNN/LSTM-AE only;
                               #RF/IF operate on unscaled features, see ROBUSTNESS_EPSILON_RF below)
ROBUSTNESS_EPSILON_RF = None  #set automatically per-dataset below (5% of each feature's std)

PLOT_DIAGNOSTICS = True   #set False to skip the two illustrative figures below entirely

#============================================================
#Setup
#============================================================
try:
    import shap
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "shap"], check=True)
    import shap

import tensorflow as tf
import joblib
import sys
sys.path.insert(0, EXPORT_ROOT)
from models import build_cnn, build_lstm_ae

print(f"shap version: {shap.__version__}")
print(f"tensorflow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

with open(f"{EXPORT_ROOT}/environment_versions.json") as f:
    local_versions = json.load(f)
print(f"\nLocal (training-machine) versions were: {local_versions}")
print("If these differ substantially from the Colab versions above (especially "
      "scikit-learn for RF/IF, or TensorFlow for CNN/LSTM-AE), loading the saved "
      "model files below may raise a version-mismatch error - if so, the fix is "
      "usually `pip install scikit-learn==<local version>` before loading.\n")

with open(f"{EXPORT_ROOT}/feature_names.json") as f:
    feature_names = json.load(f)


#============================================================
#Model loading (Section 3.6.6/Appendix A workaround): CNN/LSTM-AE are
#never saved as a loadable .keras file for cross-environment use - only
#weights.h5 plus best_params.json are exported (the documented fix for
#the Keras version-mismatch issue between the training machine and
#Colab). Loading therefore reconstructs the architecture from models.py
#and loads weights only. Identical to extract_shap_attributions.py's
#function of the same name - copied verbatim so both scripts load a
#given model's weights the exact same way.
#============================================================
def load_windowed_model(model_dir, input_shape, is_autoencoder):
    with open(f"{model_dir}/best_params.json") as f:
        params = json.load(f)
    build_fn = build_lstm_ae if is_autoencoder else build_cnn
    model = build_fn(input_shape=input_shape, params=params)
    model.load_weights(f"{model_dir}/model.weights.h5")
    return model


#============================================================
#Metric implementations (UNCHANGED from the original script)
#============================================================
def faithfulness_correlation_flat(x, attribution, predict_fn, background_value,
                                   n_trials=N_TRIALS_FAITHFULNESS, seed=0):
    """For RF/IF: x is a 1D feature row. Masks random feature subsets with
    the background (mean training) value and checks whether the resulting
    prediction change correlates with the attribution assigned to the
    masked features - a faithful explanation should show a real drop in
    the model's output when the features it says matter are removed."""
    rng = np.random.default_rng(seed)
    d = len(x)
    f_x = predict_fn(x.reshape(1, -1))[0]
    deltas, attr_sums = [], []
    for _ in range(n_trials):
        subset_size = rng.integers(1, d) if d > 1 else 1
        subset = rng.choice(d, size=subset_size, replace=False)
        x_masked = x.copy()
        x_masked[subset] = background_value[subset]
        f_masked = predict_fn(x_masked.reshape(1, -1))[0]
        deltas.append(f_x - f_masked)
        attr_sums.append(attribution[subset].sum())
    if np.std(deltas) < 1e-12 or np.std(attr_sums) < 1e-12:
        return np.nan
    return float(np.corrcoef(deltas, attr_sums)[0, 1])


def faithfulness_correlation_windowed(x, attribution_per_feature, predict_fn,
                                       background_value_per_feature,
                                       n_trials=N_TRIALS_FAITHFULNESS, seed=0):
    """For CNN/LSTM-AE: x is a (window, features) tensor. Masking operates
    on whole FEATURES (a column across the entire window), not individual
    cells - keeps perturbation at a semantically meaningful level ('this
    sensor', not 'this exact second') and keeps the search space
    tractable (n_features candidates, not window_size*n_features)."""
    rng = np.random.default_rng(seed)
    d = x.shape[-1]
    f_x = predict_fn(x[np.newaxis])[0]
    deltas, attr_sums = [], []
    for _ in range(n_trials):
        subset_size = rng.integers(1, d) if d > 1 else 1
        subset = rng.choice(d, size=subset_size, replace=False)
        x_masked = x.copy()
        x_masked[:, subset] = background_value_per_feature[subset]
        f_masked = predict_fn(x_masked[np.newaxis])[0]
        deltas.append(f_x - f_masked)
        attr_sums.append(attribution_per_feature[subset].sum())
    if np.std(deltas) < 1e-12 or np.std(attr_sums) < 1e-12:
        return np.nan
    return float(np.corrcoef(deltas, attr_sums)[0, 1])


def complexity_entropy(attribution, eps=1e-12):
    """Entropy of normalised |attribution| - lower means fewer features
    dominate the explanation (more parsimonious, easier for an operator
    to act on)."""
    abs_vals = np.abs(attribution).ravel() + eps
    p = abs_vals / abs_vals.sum()
    return float(-np.sum(p * np.log(p)))


#============================================================
#Plotting helpers (NEW - additive only, no change to any metric above)
#============================================================
def plot_faithfulness_scatter(deltas, attr_sums, correlation, model_name, dataset_name, save_path):
    """One instance's masking trials, visualised (Figure 4.2 for IF).
    Rebuilt for a single, already-identified instance using the SAME seed
    that produced its reported faithfulness value
    (faithfulness_correlation_flat seeds each instance independently via
    seed=i), so this is an exact replay of that instance's trials, not a
    fresh, different sample."""
    plt.figure(figsize=(6, 5))
    plt.scatter(attr_sums, deltas, alpha=0.65, color="#B23A2E", edgecolor="none")
    if np.std(attr_sums) > 1e-12:
        z = np.polyfit(attr_sums, deltas, 1)
        xs = np.linspace(min(attr_sums), max(attr_sums), 50)
        plt.plot(xs, np.poly1d(z)(xs), "--", color="#333333", linewidth=1.2)
    plt.axhline(0, color="#999999", linewidth=0.8)
    plt.xlabel("Sum of SHAP attribution, masked feature subset")
    plt.ylabel("Actual change in model output")
    plt.title(f"{model_name}, {dataset_name.upper()}: worst-instance faithfulness\n"
              f"correlation = {correlation:.3f}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def plot_stability_bars(base_attr, perturbed_attrs, feature_names, model_name, dataset_name, save_path, top_k=10):
    """Per-feature attribution, mean and spread across the perturbation
    trials already run for this instance inside the robustness loop below
    (Figure 4.3 for LSTM-AE). base_attr and perturbed_attrs are captured
    live during that loop, not recomputed, so this is exactly the data
    behind the instance's reported robustness value, not an approximation
    of it."""
    perturbed_attrs = np.array(perturbed_attrs)
    mean_attr = perturbed_attrs.mean(axis=0)
    std_attr = perturbed_attrs.std(axis=0)
    top_idx = np.argsort(-np.abs(base_attr))[:top_k]

    plt.figure(figsize=(8, 5))
    x = np.arange(len(top_idx))
    plt.bar(x, mean_attr[top_idx], yerr=std_attr[top_idx], color="#8C6E4F", capsize=4)
    plt.xticks(x, [feature_names[i] for i in top_idx], rotation=45, ha="right")
    plt.ylabel(f"SHAP attribution (mean \u00b1 std, {perturbed_attrs.shape[0]} perturbations)")
    plt.title(f"{model_name}, {dataset_name.upper()}: most-stable-instance attribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


#============================================================
#Per-model evaluation functions
#============================================================
def evaluate_tree_model(model, X_sample, X_background, is_classifier, model_name):
    """RF and IF: TreeExplainer (exact, fast - no need to subsample further
    for the explainer itself, only for the earlier row sampling already
    done in export_for_shap.py)."""
    print(f"  Building TreeExplainer for {model_name}...")
    explainer = shap.TreeExplainer(model)

    if is_classifier:
        predict_fn = lambda x: model.predict_proba(x)[:, 1]
    else:
        predict_fn = lambda x: -model.decision_function(x)  #higher = more anomalous, matches training pipeline convention

    def get_shap(x_batch):
        sv = explainer.shap_values(x_batch)
        sv = np.array(sv)
        if sv.ndim == 3:  #(n, features, classes) - RF binary classifier
            sv = sv[:, :, 1]
        return sv

    background_value = X_background.mean(axis=0)
    n = len(X_sample)
    print(f"  Computing SHAP values for {n} instances...")
    t0 = time.time()
    shap_values = get_shap(X_sample)
    print(f"    done in {time.time()-t0:.1f}s")

    print(f"  Computing faithfulness ({N_TRIALS_FAITHFULNESS} trials/instance)...")
    t0 = time.time()
    faith = [faithfulness_correlation_flat(X_sample[i], shap_values[i], predict_fn, background_value, seed=i)
             for i in range(n)]
    print(f"    done in {time.time()-t0:.1f}s")

    print(f"  Computing robustness ({N_PERTURBATIONS_ROBUSTNESS} perturbations/instance)...")
    t0 = time.time()
    feature_std = X_background.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    robust = []
    rng = np.random.default_rng(1)
    for i in range(n):
        base = shap_values[i]
        perturbed_batch = np.stack([
            X_sample[i] + rng.normal(0, 0.05 * feature_std)
            for _ in range(N_PERTURBATIONS_ROBUSTNESS)
        ])
        pert_shap = get_shap(perturbed_batch)
        changes = np.linalg.norm(pert_shap - base, axis=1)
        robust.append(float(changes.max()))
    print(f"    done in {time.time()-t0:.1f}s")

    complexity = [complexity_entropy(shap_values[i]) for i in range(n)]

    #IF negative-faithfulness plot only (Figure 4.2, Section 4.5) - no change to results above
    if PLOT_DIAGNOSTICS and model_name == "IF":
        finite_faith = [f for f in faith if not np.isnan(f)]
        if finite_faith:
            worst_idx = int(np.nanargmin(faith))
            print(f"  Plotting worst-faithfulness instance for IF (index {worst_idx}, "
                  f"correlation={faith[worst_idx]:.3f})...")
            #exact replay of that instance's own trials: same seed=worst_idx
            #faithfulness_correlation_flat used internally, so these deltas
            #and attr_sums are identical to what produced faith[worst_idx].
            #this is the single most negative instance, not the dataset
            #mean - Table 3.12/Section 4.5 report the mean separately
            rng_replay = np.random.default_rng(worst_idx)
            x = X_sample[worst_idx]
            attribution = shap_values[worst_idx]
            d = len(x)
            f_x = predict_fn(x.reshape(1, -1))[0]
            deltas, attr_sums = [], []
            for _ in range(N_TRIALS_FAITHFULNESS):
                subset_size = rng_replay.integers(1, d) if d > 1 else 1
                subset = rng_replay.choice(d, size=subset_size, replace=False)
                x_masked = x.copy()
                x_masked[subset] = background_value[subset]
                f_masked = predict_fn(x_masked.reshape(1, -1))[0]
                deltas.append(f_x - f_masked)
                attr_sums.append(attribution[subset].sum())
            plot_faithfulness_scatter(
                deltas, attr_sums, faith[worst_idx], model_name, DATASET,
                save_path=f"{EXPORT_ROOT}/if_faithfulness_{DATASET}.png",
            )

    return {
        "faithfulness_mean": float(np.nanmean(faith)),
        "faithfulness_nan_count": int(np.sum(np.isnan(faith))),
        "robustness_mean": float(np.mean(robust)),
        "complexity_mean": float(np.mean(complexity)),
        "n_instances": n,
    }


def evaluate_windowed_model(model, Xw_sample, Xw_background, model_name, is_autoencoder):
    """CNN: GradientExplainer directly on the classifier.
    LSTM-AE: wrapped so SHAP explains the scalar reconstruction error,
    since there is no classification output to explain otherwise."""
    if is_autoencoder:
        print(f"  Building reconstruction-error wrapper for {model_name}...")
        inp = tf.keras.Input(shape=Xw_sample.shape[1:])
        recon = model(inp)
        error = tf.keras.layers.Lambda(
            lambda t: tf.reshape(tf.reduce_mean(tf.square(t[0] - t[1]), axis=[1, 2]), (-1, 1))
        )([inp, recon])
        explain_model = tf.keras.Model(inp, error)
        predict_fn = lambda x: explain_model.predict(x, verbose=0).ravel()
    else:
        explain_model = model
        predict_fn = lambda x: model.predict(x, verbose=0).ravel()

    print(f"  Building GradientExplainer for {model_name}...")
    explainer = shap.GradientExplainer(explain_model, Xw_background)

    def get_shap(x_batch):
        sv = np.array(explainer.shap_values(x_batch))
        if sv.ndim == 4:
            sv = sv[..., 0]
        return sv  #(n, window, features)

    n = len(Xw_sample)
    print(f"  Computing SHAP values for {n} instances...")
    t0 = time.time()
    shap_values = get_shap(Xw_sample)               #(n, window, features)
    shap_per_feature = shap_values.sum(axis=1)       #(n, features) - summed over time
    print(f"    done in {time.time()-t0:.1f}s")

    background_value_per_feature = Xw_background.mean(axis=(0, 1))

    print(f"  Computing faithfulness ({N_TRIALS_FAITHFULNESS} trials/instance)...")
    t0 = time.time()
    faith = [faithfulness_correlation_windowed(
                Xw_sample[i], shap_per_feature[i], predict_fn, background_value_per_feature, seed=i)
             for i in range(n)]
    print(f"    done in {time.time()-t0:.1f}s")

    print(f"  Computing robustness ({N_PERTURBATIONS_ROBUSTNESS} perturbations/instance) "
          f"- this is the slow step for windowed models...")
    t0 = time.time()
    robust = []
    #only populated/used when plotting LSTM-AE - captures each instance's
    #actual perturbed per-feature attributions LIVE, inside this loop, as
    #they're computed. The robustness loop below uses one shared rng
    #across all instances sequentially, so replaying "just instance i"
    #afterwards with a fresh rng would NOT reproduce the same
    #perturbations - capturing here is the only way to guarantee the plot
    #matches the instance's actually-reported robustness value exactly.
    capture_stability = PLOT_DIAGNOSTICS and is_autoencoder
    all_perturbed_attrs = [] if capture_stability else None
    rng = np.random.default_rng(1)
    for i in range(n):
        base = shap_per_feature[i]
        perturbed_batch = np.stack([
            Xw_sample[i] + rng.normal(0, PERTURBATION_EPSILON, size=Xw_sample[i].shape)
            for _ in range(N_PERTURBATIONS_ROBUSTNESS)
        ]).astype(np.float32)
        pert_sv = get_shap(perturbed_batch).sum(axis=1)  #(n_pert, features)
        changes = np.linalg.norm(pert_sv - base, axis=1)
        robust.append(float(changes.max()))
        if capture_stability:
            all_perturbed_attrs.append(pert_sv)
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{n} instances, {time.time()-t0:.1f}s elapsed")
    print(f"    done in {time.time()-t0:.1f}s total")

    complexity = [complexity_entropy(shap_per_feature[i]) for i in range(n)]

    #LSTM-AE stability plot only (Figure 4.3, Section 4.5) - no change to results above
    if capture_stability and all_perturbed_attrs:
        most_stable_idx = int(np.argmin(robust))
        print(f"  Plotting most-stable instance for LSTM-AE (index {most_stable_idx}, "
              f"max change={robust[most_stable_idx]:.4f})...")
        plot_stability_bars(
            shap_per_feature[most_stable_idx],
            all_perturbed_attrs[most_stable_idx],
            feature_names, model_name, DATASET,
            save_path=f"{EXPORT_ROOT}/lstm_ae_stability_{DATASET}.png",
        )

    return {
        "faithfulness_mean": float(np.nanmean(faith)),
        "faithfulness_nan_count": int(np.sum(np.isnan(faith))),
        "robustness_mean": float(np.mean(robust)),
        "complexity_mean": float(np.mean(complexity)),
        "n_instances": n,
    }


#============================================================
#Main
#============================================================
results = {}

#----RF----
print("\n" + "=" * 60)
print(f"{DATASET.upper()} / RF")
print("=" * 60)
rf_model = joblib.load(f"{EXPORT_ROOT}/rf/model.joblib")
d1 = np.load(f"{EXPORT_ROOT}/unwindowed_export.npz")
results["rf"] = evaluate_tree_model(rf_model, d1["X_sample"], d1["X_background"],
                                     is_classifier=True, model_name="RF")
print(json.dumps(results["rf"], indent=2))

#----IF----
print("\n" + "=" * 60)
print(f"{DATASET.upper()} / IF")
print("=" * 60)
if_model = joblib.load(f"{EXPORT_ROOT}/if/model.joblib")
results["if"] = evaluate_tree_model(if_model, d1["X_sample"], d1["X_background"],
                                     is_classifier=False, model_name="IF")
print(json.dumps(results["if"], indent=2))

#----CNN----
print("\n" + "=" * 60)
print(f"{DATASET.upper()} / CNN")
print("=" * 60)
d2 = np.load(f"{EXPORT_ROOT}/windowed_export.npz")
cnn_model = load_windowed_model(f"{EXPORT_ROOT}/cnn", input_shape=d2["Xw_sample"].shape[1:], is_autoencoder=False)
results["cnn"] = evaluate_windowed_model(cnn_model, d2["Xw_sample"], d2["Xw_background"],
                                          model_name="CNN", is_autoencoder=False)
print(json.dumps(results["cnn"], indent=2))

#----LSTM-AE----
print("\n" + "=" * 60)
print(f"{DATASET.upper()} / LSTM-AE")
print("=" * 60)
lstm_ae_model = load_windowed_model(f"{EXPORT_ROOT}/lstm_ae", input_shape=d2["Xw_sample"].shape[1:], is_autoencoder=True)
results["lstm_ae"] = evaluate_windowed_model(lstm_ae_model, d2["Xw_sample"], d2["Xw_background"],
                                              model_name="LSTM-AE", is_autoencoder=True)
print(json.dumps(results["lstm_ae"], indent=2))

#----Summary----
print("\n" + "=" * 60)
print(f"SUMMARY: {DATASET.upper()}")
print("=" * 60)
print(f"{'Model':<10}{'Faithfulness':>14}{'Robustness':>13}{'Complexity':>13}{'N':>6}")
for model_name, r in results.items():
    print(f"{model_name:<10}{r['faithfulness_mean']:>14.4f}{r['robustness_mean']:>13.4f}"
          f"{r['complexity_mean']:>13.4f}{r['n_instances']:>6}")

with open(f"{EXPORT_ROOT}/xai_metrics_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {EXPORT_ROOT}/xai_metrics_results.json - download this file "
      f"and send it back for the Section 3.8/Chapter 4 write-up.")

if PLOT_DIAGNOSTICS:
    print(f"Also saved: {EXPORT_ROOT}/if_faithfulness_{DATASET}.png, "
          f"{EXPORT_ROOT}/lstm_ae_stability_{DATASET}.png")