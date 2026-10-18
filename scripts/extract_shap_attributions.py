"""
Run this in Google Colab (GPU runtime), after uploading and unzipping the
SAME export(s) already produced by export_for_shap.py and used for
xai_evaluation_colab.py.

This is DELIBERATELY LIGHTER than xai_evaluation_colab.py: it computes SHAP
values once per instance and extracts the TOP-ATTRIBUTED FEATURE for each,
which is what Section 3.9.1's MITRE ATT&CK mapping methodology actually
needs. It does NOT repeat the expensive faithfulness/robustness loops
(which recompute the explanation many times per instance) - those already
exist in xai_metrics_results.json from the earlier run and are not
recomputed here. Extracting top-feature-per-instance only needs ONE SHAP
computation per instance, so this should run considerably faster than the
full quantitative evaluation.

Output: attributions_<dataset>.csv, one row per evaluated instance, with
columns: dataset, model, instance_index, true_label, predicted_label,
predicted_score, top_feature, top_feature_shap_value, top_feature_abs_shap.

This CSV is the direct input to mitre_attack_stix_mapping.py (Section 3.9,
Figure 4.4), which runs locally (no GPU needed) and does not depend on
this script's internals.

Usage: set DATASET below, then Run All. Re-run once per dataset.
"""

import json
from pathlib import Path

import numpy as np

DATASET = "swat"
EXPORT_ROOT = f"/content/{DATASET}"

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

with open(f"{EXPORT_ROOT}/feature_names.json") as f:
    feature_names = json.load(f)
print(f"Features ({len(feature_names)}): {feature_names}")


def load_windowed_model(model_dir, input_shape, is_autoencoder):
    """Identical to the function of the same name in xai_evaluation_colab.py -
    rebuilds the architecture from models.py + best_params.json and loads
    weights only, for cross-environment compatibility (Section 3.6.6/
    Appendix A's Keras version-mismatch workaround). See that script's
    docstring for the full rationale and verification already performed."""
    with open(f"{model_dir}/best_params.json") as f:
        params = json.load(f)
    build_fn = build_lstm_ae if is_autoencoder else build_cnn
    model = build_fn(input_shape=input_shape, params=params)
    model.load_weights(f"{model_dir}/model.weights.h5")
    return model


def extract_tree_model_attributions(model, X_sample, y_sample, is_classifier, model_name, decision_threshold=None):
    print(f"  Building TreeExplainer for {model_name}...")
    explainer = shap.TreeExplainer(model)
    sv = np.array(explainer.shap_values(X_sample))
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    if is_classifier:
        scores = model.predict_proba(X_sample)[:, 1]
        #RF uses the same flag-fraction decision threshold as CNN/LSTM-AE
        #(Section 3.6.6) - use the real, saved decision_threshold_on_test
        #rather than sklearn's raw .predict() default, so "flagged" here
        #matches what this study's own reported evaluation actually
        #flagged, not an arbitrary generic cutoff.
        if decision_threshold is not None:
            preds = (scores >= decision_threshold).astype(int)
        else:
            print(f"  [WARNING] No decision_threshold_on_test available for {model_name} - "
                  f"falling back to sklearn's default .predict(), which will NOT match "
                  f"this study's actual reported flag-fraction threshold (Section 3.6.6).")
            preds = model.predict(X_sample)
    else:
        #IF is unchanged: Section 3.6.6 confirms IF does not use flag-fraction
        #thresholding at all - its prediction is already governed by the
        #contamination parameter fitted directly into the model.
        scores = -model.decision_function(X_sample)
        preds = (model.predict(X_sample) == -1).astype(int)  #sklearn IF: -1 = anomaly

    rows = []
    for i in range(len(X_sample)):
        top_idx = int(np.argmax(np.abs(sv[i])))
        rows.append({
            "dataset": DATASET, "model": model_name, "instance_index": i,
            "true_label": int(y_sample[i]), "predicted_label": int(preds[i]),
            "predicted_score": float(scores[i]),
            "top_feature": feature_names[top_idx],
            "top_feature_shap_value": float(sv[i, top_idx]),
            "top_feature_abs_shap": float(abs(sv[i, top_idx])),
        })
    return rows


def extract_windowed_model_attributions(model, Xw_sample, yw_sample, model_name, is_autoencoder, decision_threshold=None):
    if is_autoencoder:
        inp = tf.keras.Input(shape=Xw_sample.shape[1:])
        recon = model(inp)
        error = tf.keras.layers.Lambda(
            lambda t: tf.reshape(tf.reduce_mean(tf.square(t[0] - t[1]), axis=[1, 2]), (-1, 1))
        )([inp, recon])
        explain_model = tf.keras.Model(inp, error)
        scores = explain_model.predict(Xw_sample, verbose=0).ravel()
    else:
        explain_model = model
        scores = model.predict(Xw_sample, verbose=0).ravel()

    #both CNN and LSTM-AE use the real, saved decision_threshold_on_test
    #(Section 3.6.6) rather than a generic 0.5 cutoff or a rank-based
    #fallback. This matters most for SWaT CNN specifically, whose real
    #threshold sits near 0.999 due to the documented saturation finding
    #(Section 4.3) - a naive 0.5 cutoff would flag nearly every instance,
    #which does not reflect what this study's actual pipeline flags.
    if decision_threshold is not None:
        preds = (scores >= decision_threshold).astype(int)
    else:
        print(f"  [WARNING] No decision_threshold_on_test available for {model_name} - "
              f"predicted_label will be left unset for this run.")
        preds = None

    print(f"  Building GradientExplainer for {model_name}...")
    explainer = shap.GradientExplainer(explain_model, Xw_sample[:min(50, len(Xw_sample))])
    sv = np.array(explainer.shap_values(Xw_sample))
    if sv.ndim == 4:
        sv = sv[..., 0]
    sv_per_feature = sv.sum(axis=1)  #(n, features) - summed over the window, as in Section 3.8.1

    rows = []
    for i in range(len(Xw_sample)):
        top_idx = int(np.argmax(np.abs(sv_per_feature[i])))
        rows.append({
            "dataset": DATASET, "model": model_name, "instance_index": i,
            "true_label": int(yw_sample[i]),
            "predicted_label": int(preds[i]) if preds is not None else None,
            "predicted_score": float(scores[i]),
            "top_feature": feature_names[top_idx],
            "top_feature_shap_value": float(sv_per_feature[i, top_idx]),
            "top_feature_abs_shap": float(abs(sv_per_feature[i, top_idx])),
        })
    return rows


def load_decision_threshold(model_dir):
    """Loads decision_threshold_on_test from a model's best_params.json,
    if present. Returns None (with the caller responsible for warning) if
    the file or key is missing, rather than silently defaulting to
    something that would not match this study's own reported methodology."""
    try:
        with open(f"{model_dir}/best_params.json") as f:
            params = json.load(f)
        return params.get("decision_threshold_on_test")
    except FileNotFoundError:
        return None


#top_feature per row here is exactly what feeds Section 3.9.1's tag-type
#classification (sensor/actuator/downstream) and, downstream of that,
#mitre_attack_stix_mapping.py's technique lookup - the CSV written at the
#end of this script is the sole handoff point between SHAP explanation and
#CTI mapping, nothing about the mapping logic lives in this file
all_rows = []

print("\n" + "=" * 60 + f"\n{DATASET.upper()} / RF\n" + "=" * 60)
rf_model = joblib.load(f"{EXPORT_ROOT}/rf/model.joblib")
rf_threshold = load_decision_threshold(f"{EXPORT_ROOT}/rf")
print(f"  RF decision_threshold_on_test: {rf_threshold}")
d1 = np.load(f"{EXPORT_ROOT}/unwindowed_export.npz")
all_rows += extract_tree_model_attributions(rf_model, d1["X_sample"], d1["y_sample"], True, "RF", decision_threshold=rf_threshold)

print("\n" + "=" * 60 + f"\n{DATASET.upper()} / IF\n" + "=" * 60)
if_model = joblib.load(f"{EXPORT_ROOT}/if/model.joblib")
all_rows += extract_tree_model_attributions(if_model, d1["X_sample"], d1["y_sample"], False, "IF")

print("\n" + "=" * 60 + f"\n{DATASET.upper()} / CNN\n" + "=" * 60)
d2 = np.load(f"{EXPORT_ROOT}/windowed_export.npz")
cnn_threshold = load_decision_threshold(f"{EXPORT_ROOT}/cnn")
print(f"  CNN decision_threshold_on_test: {cnn_threshold}")
cnn_model = load_windowed_model(f"{EXPORT_ROOT}/cnn", input_shape=d2["Xw_sample"].shape[1:], is_autoencoder=False)
all_rows += extract_windowed_model_attributions(cnn_model, d2["Xw_sample"], d2["yw_sample"], "CNN", False, decision_threshold=cnn_threshold)

print("\n" + "=" * 60 + f"\n{DATASET.upper()} / LSTM-AE\n" + "=" * 60)
lstm_ae_threshold = load_decision_threshold(f"{EXPORT_ROOT}/lstm_ae")
print(f"  LSTM-AE decision_threshold_on_test: {lstm_ae_threshold}")
lstm_ae_model = load_windowed_model(f"{EXPORT_ROOT}/lstm_ae", input_shape=d2["Xw_sample"].shape[1:], is_autoencoder=True)
all_rows += extract_windowed_model_attributions(lstm_ae_model, d2["Xw_sample"], d2["yw_sample"], "LSTM-AE", True, decision_threshold=lstm_ae_threshold)

import csv
out_path = f"{EXPORT_ROOT}/attributions_{DATASET}.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    writer.writeheader()
    writer.writerows(all_rows)

print(f"\n{len(all_rows)} rows written to {out_path}")
print("Download this file and send it back, or use it directly with mitre_attack_stix_mapping.py.")
