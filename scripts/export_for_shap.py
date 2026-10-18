"""
I ran this LOCALLY (same directory as train.py), once per dataset. Packages
everything needed for SHAP + quantitative XAI evaluation (Section 3.8) into
a folder ready to zip and upload to Colab:
  - each model's saved file (model.joblib / model.keras)
  - correctly preprocessed test data for that model - RAW feature rows for
    RF/IF, WINDOWED+SCALED tensors for CNN/LSTM-AE - produced by reusing
    this pipeline's own load_dataset / get_or_select_features / windowing
    code directly, not a reimplementation, so there is no risk of the
    exported data drifting from what the models were actually trained and
    evaluated on
  - a STRATIFIED sample of test instances (not pure random), guaranteeing
    meaningful attack representation even on BATADAL (0.57% attack rate)
    and WaDi (0.74%) where a naive random sample of even 2,000 rows would
    contain only a handful of real attacks - the same lesson already
    learned and fixed for SHAP feature selection earlier in this pipeline
    (feature_selection.py's stratified subsampling)
  - environment version info (scikit-learn, tensorflow, shap, numpy), so
    version mismatches between this machine and Colab can be caught before
    they cause an obscure model-loading error

Usage: python export_for_shap.py --dataset swat
       (repeat for wadi, batadal)

Sample sizes (fixed, not re-derived per dataset - Section 3.8.3):
  RF, IF:        2,000 instances
  CNN, LSTM-AE:  1,500 instances
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data_loading import load_dataset
from feature_selection import get_or_select_features
from windowing import make_windows, get_window_config

SAMPLE_SIZE_UNWINDOWED = 2000   #RF, IF
SAMPLE_SIZE_WINDOWED = 1500     #CNN, LSTM-AE
MIN_MINORITY_FRACTION = 0.10    #guarantee at least this share of attacks in the sample


def stratified_sample_indices(y, sample_size, min_minority_fraction=MIN_MINORITY_FRACTION, seed=42):
    """Same principle as feature_selection.py's stratified subsampling:
    guarantee a minimum share of the minority (attack) class in the sample
    rather than pure random draws, which would under-represent attacks on
    low-attack-rate datasets. Takes all available attacks if there are
    fewer than the target minority count.

    If the OTHER class (whichever is more common overall) runs short of
    rows before the sample reaches sample_size - possible on a windowed
    test set with very few windows total - tops up from the minority
    class's remaining rows rather than silently returning a smaller
    sample than requested. Caught during testing on a small synthetic
    windowed set where attacks unexpectedly outnumbered normal windows."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    sample_size = min(sample_size, n)

    minority_idx = np.where(y == 1)[0]
    majority_idx = np.where(y == 0)[0]

    target_minority = max(int(sample_size * min_minority_fraction), 1)
    n_minority_take = min(target_minority, len(minority_idx))
    n_majority_take = min(sample_size - n_minority_take, len(majority_idx))

    minority_sample = rng.choice(minority_idx, size=n_minority_take, replace=False) if n_minority_take > 0 else np.array([], dtype=int)
    majority_sample = rng.choice(majority_idx, size=n_majority_take, replace=False) if n_majority_take > 0 else np.array([], dtype=int)

    #top up from whichever class still has unused rows, if the sample is
    #still short of sample_size (the other class ran out first).
    taken_minority = set(minority_sample.tolist())
    taken_majority = set(majority_sample.tolist())
    shortfall = sample_size - (len(minority_sample) + len(majority_sample))
    if shortfall > 0:
        remaining_minority = np.array([i for i in minority_idx if i not in taken_minority])
        remaining_majority = np.array([i for i in majority_idx if i not in taken_majority])
        remaining_pool = np.concatenate([remaining_minority, remaining_majority])
        if len(remaining_pool) > 0:
            top_up = rng.choice(remaining_pool, size=min(shortfall, len(remaining_pool)), replace=False)
            minority_sample = np.concatenate([minority_sample, np.intersect1d(top_up, minority_idx)])
            majority_sample = np.concatenate([majority_sample, np.intersect1d(top_up, majority_idx)])

    idx = np.concatenate([minority_sample, majority_sample]).astype(int)
    rng.shuffle(idx)
    print(f"    Stratified sample: {len(idx)} rows drawn from {n:,} "
          f"({len(minority_sample)} minority / {len(majority_sample)} majority, "
          f"{100*len(minority_sample)/max(len(idx),1):.1f}% minority - vs. "
          f"{100*y.mean():.2f}% in the full population)")
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["swat", "wadi", "batadal"])
    parser.add_argument("--out", default="xai_export")
    args = parser.parse_args()

    out_dir = Path(args.out) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Exporting {args.dataset.upper()} for SHAP/XAI evaluation ===\n")

    #--- Environment versions, for Colab compatibility checking ---
    versions = {}
    for pkg in ["sklearn", "tensorflow", "numpy", "pandas", "shap"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "NOT INSTALLED LOCALLY (fine - only needed in Colab for shap)"
    with open(out_dir / "environment_versions.json", "w") as f:
        json.dump(versions, f, indent=2)
    print("Local environment versions (match these in Colab where possible):")
    for k, v in versions.items():
        print(f"  {k}: {v}")
    print()

    #--- Load data through the REAL pipeline, exactly as train.py does ---
    print("Loading dataset and applying cached feature selection...")
    data = load_dataset(args.dataset)
    selected_features, _ = get_or_select_features(
        args.dataset, data.X_train.values, data.y_train, data.feature_names)
    data.X_train = data.X_train[selected_features]
    data.X_test = data.X_test[selected_features]
    print(f"  {len(selected_features)} features: {selected_features}\n")

    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(selected_features, f, indent=2)

    #---------------- RF / IF: raw (unwindowed) test rows ----------------
    print("Preparing RF/IF export (raw feature rows)...")
    idx = stratified_sample_indices(data.y_test, SAMPLE_SIZE_UNWINDOWED)
    X_sample = data.X_test.values[idx]
    y_sample = data.y_test[idx]

    #background/reference set for SHAP (a modest random slice of TRAIN,
    #standard practice - never test, to avoid any leakage into the
    #explanation baseline itself, Section 3.8.3)
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(data.X_train), size=min(200, len(data.X_train)), replace=False)
    X_background = data.X_train.values[bg_idx]

    np.savez(out_dir / "unwindowed_export.npz",
             X_sample=X_sample, y_sample=y_sample, X_background=X_background)
    print(f"  Saved unwindowed_export.npz: X_sample={X_sample.shape}, "
          f"y_sample={y_sample.shape}, X_background={X_background.shape}\n")

    #---------------- CNN / LSTM-AE: windowed, scaled tensors ----------------
    print("Preparing CNN/LSTM-AE export (windowed, scaled tensors)...")
    ws, st = get_window_config(args.dataset)

    #scaler fit on FULL train, exactly as train.py's run_cnn/run_lstm_ae do
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(data.X_train.values)
    X_test_scaled = scaler.transform(data.X_test.values)

    Xw_test, yw_test = make_windows(X_test_scaled, data.y_test, window_size=ws, stride=st)
    print(f"  Windowed test set: {Xw_test.shape} ({int(yw_test.sum())} attack windows "
          f"of {len(yw_test)}, {100*yw_test.mean():.2f}%)")

    idx_w = stratified_sample_indices(yw_test, SAMPLE_SIZE_WINDOWED)
    Xw_sample = Xw_test[idx_w]
    yw_sample = yw_test[idx_w]

    Xw_train, yw_train = make_windows(X_train_scaled, data.y_train, window_size=ws, stride=st)
    bg_idx_w = rng.choice(len(Xw_train), size=min(100, len(Xw_train)), replace=False)
    Xw_background = Xw_train[bg_idx_w]

    np.savez(out_dir / "windowed_export.npz",
             Xw_sample=Xw_sample, yw_sample=yw_sample, Xw_background=Xw_background)
    print(f"  Saved windowed_export.npz: Xw_sample={Xw_sample.shape}, "
          f"yw_sample={yw_sample.shape}, Xw_background={Xw_background.shape}\n")

    #---------------- Copy model files ----------------
    print("Copying model files...")
    import shutil

    #RF / IF: joblib files are version-tolerant enough with a warning
    #(confirmed on real Colab output - InconsistentVersionWarning, not a
    #hard failure), copied as-is. best_params.json is ALSO copied here -
    #RF uses the same flag-fraction decision threshold mechanism as
    #CNN/LSTM-AE (Section 3.6.6), so its real decision_threshold_on_test
    #is needed downstream by extract_shap_attributions.py to determine
    #which instances the model actually flags, rather than falling back
    #to a generic default that would not match this study's own reported
    #methodology. IF's best_params.json is copied too for consistency,
    #though IF does not use this mechanism (Section 3.6.6): its
    #prediction is already governed by the contamination parameter
    #fitted directly into the model itself.
    for model_name in ["rf", "if"]:
        src_dir = Path("results") / f"{args.dataset}_{model_name}"
        src_model = src_dir / "model.joblib"
        src_params = src_dir / "best_params.json"
        if src_model.exists():
            dst_dir = out_dir / model_name
            dst_dir.mkdir(exist_ok=True)
            shutil.copy(src_model, dst_dir / "model.joblib")
            print(f"  {model_name}: copied {src_model} -> {dst_dir / 'model.joblib'}")
            if src_params.exists():
                shutil.copy(src_params, dst_dir / "best_params.json")
                print(f"  {model_name}: copied {src_params} -> {dst_dir / 'best_params.json'}")
            else:
                print(f"  {model_name}: [WARNING] {src_params} not found - "
                      f"decision_threshold_on_test will be unavailable downstream")
        else:
            print(f"  {model_name}: [WARNING] {src_model} not found - run train.py --model {model_name} first")

    #CNN / LSTM-AE: the full .keras file's serialized layer config is NOT
    #reliably portable across TensorFlow/Keras versions - confirmed
    #directly: a model trained locally on TF 2.21.0 failed to load on
    #Colab's TF 2.20.0 with a GlorotUniform deserialization error, since
    #the newer Keras added input_axes/output_axes fields to the
    #initializer's saved config that the older Keras does not recognise
    #(Section 3.6.6/Appendix A). Raw WEIGHT VALUES have no such
    #version-specific formatting, so the fix is to export weights-only,
    #plus the winning hyperparameters (best_params.json, already produced
    #by training) and models.py itself, so Colab can rebuild the
    #identical architecture from code and load only the numbers into it.
    #xai_evaluation_colab.py and extract_shap_attributions.py both
    #reconstruct from these same three artefacts via load_windowed_model.
    shutil.copy("models.py", out_dir / "models.py")
    print(f"  copied models.py -> {out_dir / 'models.py'} (for architecture reconstruction)")

    try:
        import tensorflow as tf
    except ImportError:
        print(f"\n  [WARNING] TensorFlow is not installed in this Python environment, so the "
              f"CNN/LSTM-AE weights cannot be exported right now. This does NOT affect the "
              f"RF/IF export above, which is already saved successfully.")
        print(f"  This is a genuine missing dependency, not a false alarm - confirmed by this "
              f"direct ImportError, not just the earlier version-check warning.")
        print(f"  If train.py has successfully trained CNN/LSTM-AE models in THIS SAME terminal "
              f"session before, this likely means a different Python environment is active now "
              f"than when those training runs happened (e.g. the venv was recreated, or a "
              f"different terminal/interpreter is in use). Run 'pip show tensorflow' to check, "
              f"or simply 'pip install tensorflow' to install it into the currently active "
              f"environment, then re-run this command - the RF/IF files already exported will "
              f"simply be overwritten with identical data, nothing is lost by re-running.")
        tf = None

    if tf is not None:
        for model_name in ["cnn", "lstm_ae"]:
            src_dir = Path("results") / f"{args.dataset}_{model_name}"
            src_model = src_dir / "model.keras"
            src_params = src_dir / "best_params.json"
            if src_model.exists() and src_params.exists():
                dst_dir = out_dir / model_name
                dst_dir.mkdir(exist_ok=True)
                loaded = tf.keras.models.load_model(src_model)
                loaded.save_weights(dst_dir / "model.weights.h5")
                shutil.copy(src_params, dst_dir / "best_params.json")
                print(f"  {model_name}: loaded {src_model} locally (this machine's own TF can read "
                      f"its own recent save), exported weights -> {dst_dir / 'model.weights.h5'} "
                      f"and hyperparameters -> {dst_dir / 'best_params.json'}")
            else:
                print(f"  {model_name}: [WARNING] {src_model} or {src_params} not found - "
                      f"run train.py --model {model_name} first")

    print(f"\nDone. Zip and upload the whole '{out_dir}' folder to Colab:")
    print(f"  cd {args.out} && zip -r {args.dataset}_xai_export.zip {args.dataset}/")


if __name__ == "__main__":
    main()