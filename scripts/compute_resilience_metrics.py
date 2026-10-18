"""
compute_resilience_metrics.py
===============================
Computes MTTD (Mean Time to Detect) and MTTR (Mean Time to Respond) from
each run's predictions.csv, for use in Section 3.7 (RAS).

DEFINITIONS (methodological choices, not arbitrary - see dissertation text):
  MTTD: for each discrete attack EVENT (a contiguous run of true-positive-
  labelled rows in the timeline), the elapsed time from the event's first
  row to the first row within that event where the model also predicted
  positive (a true positive). Events never detected within their own
  span are reported separately as a MISS RATE, not silently averaged in
  or silently dropped - both would misrepresent the result.

  MTTR: for each event, the elapsed time from the event's LAST row to the
  LAST row of lingering false-positive predictions immediately following
  it (i.e. how long the detector keeps alarming after the attack has
  actually ended, before returning to a clean run of correct negative
  predictions). This is a proxy - the pipeline has no automated response
  action to time - chosen because it connects to the alarm-fatigue/
  operator-trust theme already in the literature review (Section 2.6),
  not because it's the only defensible choice. If the alert clears
  immediately (no false positives follow the event), MTTR for that event
  is 0. Events with no trailing false positives are NOT the same as
  events never detected - both are tracked and reported distinctly.

Usage: python compute_resilience_metrics.py <path_to_predictions.csv>
   or: python compute_resilience_metrics.py --all results
       (walks every results/<dataset>_<model>/predictions.csv it finds)
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _find_column(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    for cand in candidates:
        for lc, orig in cols_lower.items():
            if cand in lc:
                return orig
    raise ValueError(f"None of {candidates} found in columns {list(df.columns)}")


def _find_events(y, ts_seconds):
    """Returns list of (start_idx, end_idx) inclusive, for contiguous runs of y==1."""
    events = []
    in_event = False
    start = None
    for i in range(len(y)):
        if y[i] == 1 and not in_event:
            in_event = True
            start = i
        elif y[i] == 0 and in_event:
            in_event = False
            events.append((start, i - 1))
    if in_event:
        events.append((start, len(y) - 1))
    return events


def compute_mttd_mttr(csv_path):
    df = pd.read_csv(csv_path)
    ts_col = _find_column(df, ["timestamp", "ts", "time", "date"])
    true_col = _find_column(df, ["y_true", "true", "label", "actual"])
    pred_col = _find_column(df, ["y_pred", "pred", "predicted"])

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    y_true = df[true_col].astype(int).values
    y_pred = df[pred_col].astype(int).values
    ts = df[ts_col].values

    n = len(df)
    if n < 2:
        raise ValueError(f"{csv_path}: fewer than 2 rows after parsing, cannot compute intervals.")

    # median row-to-row spacing, used to convert row-index gaps into elapsed time
    # robust to the file being windows (CNN/LSTM-AE) or raw rows (RF/IF)
    deltas = np.diff(ts).astype("timedelta64[s]").astype(float)
    deltas = deltas[deltas > 0]
    row_spacing_s = float(np.median(deltas)) if len(deltas) else 1.0

    events = _find_events(y_true, ts)
    if not events:
        return {"n_events": 0, "n_detected": 0, "miss_rate": None,
                "mttd_seconds": None, "mttr_seconds": None, "row_spacing_s": row_spacing_s}

    detect_latencies = []
    respond_latencies = []
    n_detected = 0

    for (start, end) in events:
        # MTTD: first predicted-positive row within [start, end]
        detected_at = None
        for i in range(start, end + 1):
            if y_pred[i] == 1:
                detected_at = i
                break
        if detected_at is not None:
            n_detected += 1
            latency_s = (ts[detected_at] - ts[start]) / np.timedelta64(1, "s")
            detect_latencies.append(max(latency_s, 0.0))

        # MTTR: trailing false positives immediately after event end
        j = end + 1
        while j < n and y_pred[j] == 1:
            j += 1
        last_fp = j - 1
        if last_fp >= end + 1:
            resp_s = (ts[last_fp] - ts[end]) / np.timedelta64(1, "s")
            respond_latencies.append(max(resp_s, 0.0))
        else:
            respond_latencies.append(0.0)  # cleared immediately, genuinely 0 - not missing

    return {
        "n_events": len(events),
        "n_detected": n_detected,
        "miss_rate": 1.0 - (n_detected / len(events)),
        "mttd_seconds": float(np.mean(detect_latencies)) if detect_latencies else None,
        "mttd_row_spacing_units": (float(np.mean(detect_latencies)) / row_spacing_s) if detect_latencies else None,
        "mttr_seconds": float(np.mean(respond_latencies)),
        "mttr_row_spacing_units": float(np.mean(respond_latencies)) / row_spacing_s,
        "row_spacing_s": row_spacing_s,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="predictions.csv path, or results/ root with --all")
    parser.add_argument("--all", action="store_true", help="walk results/<dataset>_<model>/predictions.csv")
    args = parser.parse_args()

    if args.all:
        root = Path(args.path)
        found = sorted(root.glob("*/predictions.csv"))
        if not found:
            print(f"No predictions.csv found under {root}/*/")
            return
        print(f"{'Run':<20} {'Events':>7} {'Detected':>9} {'MissRate':>9} "
              f"{'MTTD(s)':>10} {'MTTR(s)':>10} {'RowSpc(s)':>10}")
        for p in found:
            run_name = p.parent.name
            try:
                r = compute_mttd_mttr(p)
                mttd = f"{r['mttd_seconds']:.1f}" if r["mttd_seconds"] is not None else "N/A"
                miss = f"{r['miss_rate']:.1%}" if r["miss_rate"] is not None else "N/A"
                print(f"{run_name:<20} {r['n_events']:>7} {r['n_detected']:>9} {miss:>9} "
                      f"{mttd:>10} {r['mttr_seconds']:>10.1f} {r['row_spacing_s']:>10.2f}")
            except Exception as e:
                print(f"{run_name:<20} ERROR: {e}")
    else:
        r = compute_mttd_mttr(args.path)
        print(f"Events: {r['n_events']}, Detected: {r['n_detected']}, "
              f"Miss rate: {r['miss_rate']:.1%}" if r['miss_rate'] is not None else "N/A")
        print(f"MTTD: {r['mttd_seconds']:.2f} seconds "
              f"({r['mttd_row_spacing_units']:.2f} rows/windows)" if r['mttd_seconds'] is not None
              else "MTTD: N/A (no events detected)")
        print(f"MTTR: {r['mttr_seconds']:.2f} seconds ({r['mttr_row_spacing_units']:.2f} rows/windows)")
        print(f"Row/window spacing used for conversion: {r['row_spacing_s']:.2f}s")


if __name__ == "__main__":
    main()
