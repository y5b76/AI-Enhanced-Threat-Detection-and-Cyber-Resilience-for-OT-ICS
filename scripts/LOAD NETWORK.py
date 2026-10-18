import os
import re
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# File reading helper (extension-aware: .csv or .xlsx)
# ---------------------------------------------------------------------------

def read_any(path: str, header: int = 0) -> pd.DataFrame:
    """Read a .csv or .xlsx file with the correct pandas function based on
    its actual extension, so we stop guessing.

    For CSVs, malformed rows (e.g. an unescaped comma inside a free-text
    field like Modbus_Function_Description, which produces one extra column
    for that row only) are skipped rather than allowed to crash the entire
    read. The number skipped is counted and reported, since silently losing
    rows without knowing how many is its own kind of bug.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, header=header)

    # Cheap raw line count first, so we can report exactly how many data
    # rows got dropped by on_bad_lines='skip' below, without needing the
    # much slower python parser engine just to get a count.
    with open(path, "rb") as f:
        raw_line_count = sum(1 for _ in f)
    expected_data_rows = raw_line_count - (header + 1)

    df = pd.read_csv(path, header=header, on_bad_lines="skip", engine="c")

    dropped = expected_data_rows - len(df)
    if dropped > 0:
        print(f"[read] WARNING: {dropped} malformed row(s) skipped while "
              f"reading {path} (likely an unescaped comma inside a free-text "
              f"field). Kept {len(df):,} of {expected_data_rows:,} data rows.")

    return df


def read_any_fix_double_header(path: str) -> pd.DataFrame:
    """Some SWaT physical/historian files have TWO header rows: row 1 groups
    columns by process stage (P1, P2, ... P6, mostly blank between group
    labels), and row 2 has the real tag names (Timestamp, LIT101, MV101...).
    Reading with the default header=0 makes pandas treat row 1 as the
    header, leaving most columns named 'Unnamed: X'.

    This detects that situation (most columns start with 'Unnamed') and, if
    found, re-reads the file using row 2 as the real header instead.
    """
    df = read_any(path, header=0)
    unnamed_fraction = sum(str(c).startswith("Unnamed") for c in df.columns) / len(df.columns)

    if unnamed_fraction > 0.5:
        print(f"[read] {unnamed_fraction:.0%} of columns came back as 'Unnamed' "
              f"with header=0 — this looks like a two-row header file "
              f"(e.g. process-stage labels P1-P6 on row 1, real tag names on "
              f"row 2). Re-reading with the second row as the header.")
        df = read_any(path, header=1)
        print(f"[read] Re-read with header=1. Columns now: {list(df.columns)}")

    return df


# ---------------------------------------------------------------------------
# Column-name helpers
# ---------------------------------------------------------------------------

def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find the first column in df whose name contains any of the candidate
    substrings (case-insensitive). Raises a clear error if none match, rather
    than silently picking the wrong column.
    """
    cols_lower = {c: c.lower().strip() for c in df.columns}
    for candidate in candidates:
        for col, low in cols_lower.items():
            if candidate.lower() in low:
                return col
    raise KeyError(
        f"Could not find a column matching any of {candidates}. "
        f"Actual columns in this file are: {list(df.columns)}"
    )


def parse_datetime_flexible(series: pd.Series, label: str) -> pd.Series:
    """Try a list of known iTrust/SWaT date-time formats in turn, report
    which one worked, and fall back to pandas' general inference (slower,
    but informative) if none of the explicit formats match. This exists so
    we get a clear diagnosis on the first run instead of 100% NaT with no
    explanation.
    """
    cleaned = series.astype(str).str.strip()

    sample = cleaned.dropna().head(5).tolist()
    print(f"[{label}] Raw sample values before parsing (after whitespace strip): {sample}")

    candidate_formats = [
        "%d%b%Y %H:%M:%S",       # 28Dec2015 18:08:00
        "%d-%b-%Y %H:%M:%S",     # 28-Dec-2015 18:08:00
        "%d/%m/%Y %H:%M:%S",     # 28/12/2015 18:08:00
        "%d/%m/%Y %I:%M:%S %p",  # 28/12/2015 06:08:00 PM
        "%Y-%m-%d %H:%M:%S",     # 2015-12-28 18:08:00
        "%m/%d/%Y %H:%M:%S",     # 12/28/2015 18:08:00
    ]
    for fmt in candidate_formats:
        parsed = pd.to_datetime(cleaned, format=fmt, errors="coerce")
        success_rate = parsed.notna().mean()
        if success_rate > 0.95:
            print(f"[{label}] Matched format '{fmt}' ({success_rate:.0%} parsed successfully)")
            return parsed

    print(f"[{label}] No fixed format matched >95% of rows even after "
          f"stripping whitespace. Falling back to pandas' general date "
          f"inference with dayfirst=True. Still slower than a fixed format, "
          f"so if you see this on your real files, tell Claude the printed "
          f"sample above so an explicit format can be added.")
    parsed = pd.to_datetime(cleaned, errors="coerce", dayfirst=True)
    print(f"[{label}] General inference parsed {parsed.notna().mean():.0%} of rows successfully")
    return parsed


# ---------------------------------------------------------------------------
# Step 1: Load and inspect a parsed network file
# ---------------------------------------------------------------------------

def load_network_file(path: str) -> pd.DataFrame:
    """Load one parsed iTrust SWaT network log file and standardise it.

    Produces a dataframe with an added 'timestamp' column (pandas datetime)
    built from the file's date + time fields, plus a cleaned numeric
    Modbus/CIP value column where parseable.
    """
    df = read_any(path)
    print(f"[network] Loaded {path}: {df.shape[0]} rows, {df.shape[1]} columns")

    date_col = find_column(df, ["date"])
    time_col = find_column(df, ["time"])
    tag_col = find_column(df, ["scada_tag", "tag"])
    value_col = find_column(df, ["modbus_value", "value"])
    func_desc_col = find_column(df, ["function_descriptio", "function_description"])
    txn_col = find_column(df, ["transaction_id", "transaction"])

    candidates = [c for c in df.columns if c.lower().strip() == "tag"]
    net_label_col = candidates[0] if candidates else None

    combined = df[date_col].astype(str) + " " + df[time_col].astype(str)
    df["timestamp"] = parse_datetime_flexible(combined, label="network")

    df["is_response"] = df[func_desc_col].astype(str).str.contains(
        "response", case=False, na=False
    )
    df["value_numeric"] = df.apply(
        lambda r: _decode_hex_bytes(r[value_col]) if r["is_response"] else np.nan,
        axis=1,
    )

    df = df.rename(columns={tag_col: "scada_tag_std", txn_col: "transaction_id"})
    if net_label_col is not None:
        df = df.rename(columns={net_label_col: "net_label_std"})
        df["net_label_std"] = pd.to_numeric(df["net_label_std"], errors="coerce")

    return df


def _decode_hex_bytes(value) -> float:
    """Decode a space-separated hex byte string like '0xbb 0xc2 0x00' into a
    single integer (little-endian). Returns NaN if it doesn't look like hex.
    """
    if not isinstance(value, str):
        return np.nan
    tokens = value.strip().split()
    if not tokens or not all(re.fullmatch(r"0x[0-9a-fA-F]{1,2}", t) for t in tokens):
        return np.nan
    total = 0
    for i, t in enumerate(tokens):
        total += int(t, 16) << (8 * i)
    return float(total)


# ---------------------------------------------------------------------------
# Step 2: Load and inspect the physical (historian) file
# ---------------------------------------------------------------------------

def load_physical_file(path: str) -> pd.DataFrame:
    """Load one SWaT physical/historian file (Normal_v0, Normal_v1, or
    Attack_v0) and standardise its timestamp and label columns.
    """
    df = read_any_fix_double_header(path)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"[physical] Loaded {path}: {df.shape[0]} rows, {df.shape[1]} columns")

    ts_col = find_column(df, ["timestamp"])
    df["timestamp"] = parse_datetime_flexible(df[ts_col], label="physical")

    try:
        label_col = find_column(df, ["normal/attack", "attack", "label"])
        df = df.rename(columns={label_col: "label_std"})
    except KeyError:
        print("[physical] No label column found (expected for Normal_v0/v1, "
              "which have no attacks). Proceeding without a label column.")

    return df


# ---------------------------------------------------------------------------
# Step 2.5: Aggregate network packets into one row per second, per tag
# ---------------------------------------------------------------------------

def _transaction_id_anomaly_flags(group: pd.DataFrame) -> pd.Series:
    txn = pd.to_numeric(group["transaction_id"], errors="coerce").dropna()
    if len(txn) < 2:
        return pd.Series({"txn_repeat_count": 0, "txn_max_gap": np.nan})

    repeat_count = int(txn.duplicated().sum())

    MODBUS_TXN_MODULUS = 65536
    raw_diffs = txn.diff().dropna().abs()
    wrapped_diffs = raw_diffs.apply(lambda d: min(d, MODBUS_TXN_MODULUS - d))
    max_gap = float(wrapped_diffs.max()) if not wrapped_diffs.empty else np.nan

    return pd.Series({"txn_repeat_count": repeat_count, "txn_max_gap": max_gap})


def aggregate_network_features(net_df: pd.DataFrame) -> pd.DataFrame:
    net_df = net_df.copy()
    net_df["second"] = net_df["timestamp"].dt.floor("s")

    base_agg = net_df.groupby(["second", "scada_tag_std"]).agg(
        packet_count=("is_response", "size"),
        response_count=("is_response", "sum"),
        avg_value=("value_numeric", "mean"),
    )
    base_agg["request_count"] = base_agg["packet_count"] - base_agg["response_count"]
    base_agg["read_write_ratio"] = (
        base_agg["response_count"] / base_agg["request_count"].replace(0, np.nan)
    )

    txn_agg = (
        net_df.groupby(["second", "scada_tag_std"])
        .apply(_transaction_id_anomaly_flags, include_groups=False)
    )

    long_df = base_agg.join(txn_agg).reset_index()

    if "net_label_std" in net_df.columns:
        net_label_agg = (
            net_df.groupby(["second", "scada_tag_std"])["net_label_std"]
            .mean()
            .reset_index()
        )
        long_df = long_df.merge(net_label_agg, on=["second", "scada_tag_std"], how="left")

    value_cols = ["packet_count", "request_count", "response_count",
                  "read_write_ratio", "avg_value", "txn_repeat_count", "txn_max_gap"]
    wide = long_df.pivot_table(index="second", columns="scada_tag_std", values=value_cols)
    wide.columns = [f"{tag}_{stat}" for stat, tag in wide.columns]

    count_like = [c for c in wide.columns if c.endswith(
        ("_packet_count", "_request_count", "_response_count", "_txn_repeat_count")
    )]
    wide[count_like] = wide[count_like].fillna(0)

    if "net_label_std" in long_df.columns:
        net_second_label = (
            long_df.groupby("second")["net_label_std"].max().rename("net_label_second")
        )
        wide = wide.join(net_second_label)

    wide = wide.reset_index().rename(columns={"second": "timestamp"})

    print(f"[aggregate] Collapsed {len(net_df):,} packet rows into "
          f"{len(wide):,} per-second rows across "
          f"{net_df['scada_tag_std'].nunique()} tags "
          f"({len(wide.columns) - 1} feature columns)")

    return wide


# ---------------------------------------------------------------------------
# Step 3/4: Join aggregated network features and physical data on timestamp
# ---------------------------------------------------------------------------

def join_network_physical(
    net_agg_df: pd.DataFrame, phys_df: pd.DataFrame, tolerance: str = "1s"
) -> pd.DataFrame:
    net_sorted = net_agg_df.sort_values("timestamp").dropna(subset=["timestamp"])
    phys_sorted = phys_df.sort_values("timestamp").dropna(subset=["timestamp"])

    fused = pd.merge_asof(
        net_sorted,
        phys_sorted,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
        suffixes=("_net", "_phys"),
    )

    unmatched_physical_cols = fused.filter(like="LIT").isna().all(axis=1).sum() \
        if any("LIT" in c for c in fused.columns) else None
    print(f"[join] Fused table: {fused.shape[0]} rows, {fused.shape[1]} columns")
    if unmatched_physical_cols is not None:
        print(f"[join] Rows with no matching physical reading within "
              f"tolerance: {unmatched_physical_cols}")

    if "net_label_second" in fused.columns and "label_std" in fused.columns:
        phys_is_attack = fused["label_std"].astype(str).str.strip().str.lower().eq("attack")
        net_is_attack = fused["net_label_second"].fillna(0) >= 0.5
        agreement = (phys_is_attack == net_is_attack).mean()
        print(f"[join] Dual ground-truth agreement rate "
              f"(network Tag vs physical label_std): {agreement:.2%}")
        disagreements = fused.loc[phys_is_attack != net_is_attack, "timestamp"]
        if len(disagreements) > 0:
            print(f"[join] {len(disagreements)} disagreeing seconds. "
                  f"First few: {disagreements.head(5).tolist()}")

    return fused


# ---------------------------------------------------------------------------
# Batch processing with per-folder checkpointing.
#
# The crash that lost 22-24 Dec's processed output happened because nothing
# was written to disk until the ENTIRE multi-day batch finished. Given we've
# now seen a real crash partway through, that's not a safe assumption to
# keep: any single bad row, anywhere in ~35 million rows a day, can wipe out
# hours of prior work. Each date folder's aggregated result is now saved to
# its own small file the moment it's computed, and a folder already saved is
# skipped on the next run rather than reprocessed from scratch.
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "network_agg_checkpoints"


def process_date_folder(folder_path: str) -> pd.DataFrame:
    """Load and concatenate every parsed network CSV in one date folder,
    then aggregate ONCE across the whole folder (not per-file).
    """
    csv_files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".csv") or f.lower().endswith(".xlsx")
    )
    if not csv_files:
        print(f"[batch] {folder_path}: no CSV/XLSX files found, skipping")
        return pd.DataFrame()

    print(f"[batch] {folder_path}: {len(csv_files)} files to load")
    frames = []
    for fname in csv_files:
        fpath = os.path.join(folder_path, fname)
        frames.append(load_network_file(fpath))

    combined_raw = pd.concat(frames, ignore_index=True, sort=False)
    del frames

    print(f"[batch] {folder_path}: {len(combined_raw):,} total raw packet "
          f"rows across all files in this folder, aggregating now")
    folder_agg = aggregate_network_features(combined_raw)
    del combined_raw

    return folder_agg


def _folder_sort_key(folder_name: str):
    cleaned = folder_name.split("$")[0]
    try:
        return pd.to_datetime(cleaned, format="%d-%m-%Y")
    except ValueError:
        return pd.Timestamp.max


def process_all_dates(root_dir: str, force_reprocess: bool = False) -> pd.DataFrame:
    """Walk every date subfolder under root_dir, aggregate each one
    independently, checkpointing each folder's result to disk immediately so
    a crash partway through only costs the current folder's work, not
    everything processed before it.

    force_reprocess: if True, ignores any existing checkpoint and
    reprocesses every folder from the raw files (still overwriting the
    checkpoint afterward, so a later crash-recovery run can still resume
    from these fresh checkpoints). If False (default), a folder whose
    checkpoint already exists is loaded from disk instead of reprocessed.

    The checkpoint directory itself is excluded from the folders to process
    — it lives inside root_dir (since CHECKPOINT_DIR is a relative path
    resolved against the current working directory, which is root_dir when
    the script is run from inside the Network folder), and without this
    exclusion it gets mistaken for a 14th date folder full of raw files on
    the very next run, crashing when load_network_file() looks for raw
    packet columns ('date', 'time'...) in what are actually already-
    aggregated checkpoint files.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir_abs = os.path.abspath(CHECKPOINT_DIR)

    subfolders = sorted(
        (d for d in os.listdir(root_dir)
         if os.path.isdir(os.path.join(root_dir, d))
         and os.path.abspath(os.path.join(root_dir, d)) != checkpoint_dir_abs),
        key=_folder_sort_key,
    )
    print(f"[batch] Found {len(subfolders)} date folders under {root_dir}: {subfolders}")
    if force_reprocess:
        print(f"[batch] force_reprocess=True — reprocessing every folder from "
              f"raw files even where a checkpoint already exists.")

    all_aggs = []
    for folder_name in subfolders:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{folder_name.replace('$', '_')}.csv")

        if os.path.exists(checkpoint_path) and not force_reprocess:
            print(f"[batch] {folder_name}: checkpoint already exists at "
                  f"{checkpoint_path}, loading from disk instead of reprocessing")
            folder_agg = pd.read_csv(checkpoint_path, parse_dates=["timestamp"])
        else:
            folder_path = os.path.join(root_dir, folder_name)
            folder_agg = process_date_folder(folder_path)
            if not folder_agg.empty:
                folder_agg.to_csv(checkpoint_path, index=False)
                print(f"[batch] {folder_name}: checkpoint saved to {checkpoint_path} "
                      f"({len(folder_agg):,} rows) — a crash on a later folder "
                      f"will no longer lose this folder's work")

        if not folder_agg.empty:
            all_aggs.append(folder_agg)

    full_agg = pd.concat(all_aggs, ignore_index=True, sort=False)

    count_like = [c for c in full_agg.columns if c.endswith(
        ("_packet_count", "_request_count", "_response_count", "_txn_repeat_count")
    )]
    full_agg[count_like] = full_agg[count_like].fillna(0)

    dup_seconds = full_agg["timestamp"].duplicated(keep=False).sum()
    if dup_seconds > 0:
        n_distinct_dup_ts = full_agg.loc[full_agg["timestamp"].duplicated(keep=False), "timestamp"].nunique()
        print(f"[batch] Merging {dup_seconds} rows across {n_distinct_dup_ts} "
                f"boundary-second timestamp(s) shared between adjacent date "
                f"folders (each holds a partial slice of that second's "
                f"packets, so these are summed rather than dropped).")

        ratio_cols = [c for c in full_agg.columns if c.endswith("_read_write_ratio")]
        max_cols = [c for c in full_agg.columns if c.endswith("_txn_max_gap")] + \
                    (["net_label_second"] if "net_label_second" in full_agg.columns else [])
        avg_value_cols = [c for c in full_agg.columns if c.endswith("_avg_value")]
        sum_cols = count_like  # packet/request/response/txn_repeat counts

        agg_rules = {c: "sum" for c in sum_cols}
        agg_rules.update({c: "max" for c in max_cols})
        agg_rules.update({c: "mean" for c in avg_value_cols})  # best available combination without raw packets
        # ratio columns are dropped and recomputed below rather than merged directly
        other_cols = [c for c in full_agg.columns
                        if c not in sum_cols + max_cols + avg_value_cols + ratio_cols + ["timestamp"]]
        agg_rules.update({c: "first" for c in other_cols})

        full_agg = full_agg.groupby("timestamp", as_index=False).agg(agg_rules)

        for ratio_col in ratio_cols:
            tag_prefix = ratio_col.replace("_read_write_ratio", "")
            req_col, resp_col = f"{tag_prefix}_request_count", f"{tag_prefix}_response_count"
            if req_col in full_agg.columns and resp_col in full_agg.columns:
                full_agg[ratio_col] = full_agg[resp_col] / full_agg[req_col].replace(0, np.nan)

        print(f"[batch] After merge: {full_agg['timestamp'].duplicated().sum()} "
                f"duplicate timestamp(s) remain (should be 0).")
    else:
        print(f"[batch] No duplicate timestamps across date folders — each "
              f"second appears exactly once in the combined table.")

    print(f"[batch] Full aggregated table: {len(full_agg):,} rows spanning "
          f"{full_agg['timestamp'].min()} to {full_agg['timestamp'].max()}")

    return full_agg


def build_combined_physical(normal_path: str, attack_path: str) -> pd.DataFrame:
    normal_df = load_physical_file(normal_path)
    normal_df["label_std"] = normal_df.get("label_std", "Normal")

    attack_df = load_physical_file(attack_path)

    combined = pd.concat([normal_df, attack_df], ignore_index=True, sort=False)
    before = len(combined)
    combined = combined.sort_values("timestamp").drop_duplicates(
        subset="timestamp", keep="last"
    )
    after = len(combined)
    if before != after:
        print(f"[physical] Combined Normal_v1 + Attack_v0: {before:,} rows "
              f"before de-duplication, {after:,} after "
              f"({before - after} overlapping timestamps resolved in "
              f"favour of Attack_v0's label)")
    else:
        print(f"[physical] Combined Normal_v1 + Attack_v0: {after:,} rows, "
              f"no overlapping timestamps found")

    return combined


def safe_to_csv(df: pd.DataFrame, path: str):
    try:
        df.to_csv(path, index=False)
        print(f"Saved output to {path}")
    except PermissionError:
        import datetime
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(path)
        fallback_path = f"{base}_{stamp}{ext}"
        print(f"[warning] '{path}' is locked (likely still open in Excel from "
              f"a previous inspection) — writing to '{fallback_path}' instead.")
        df.to_csv(fallback_path, index=False)
        print(f"Saved output to {fallback_path}")


def run_full_batch():
    NETWORK_ROOT = r"C:\Users\ajoku\OneDrive\Documents\DISSERTATION WORKS\DATASETS\SWaT\Network"
    NORMAL_PATH = r"C:\Users\ajoku\OneDrive\Documents\DISSERTATION WORKS\DATASETS\SWaT\Physical\SWaT_Dataset_Normal_v1.xlsx"
    ATTACK_PATH = r"C:\Users\ajoku\OneDrive\Documents\DISSERTATION WORKS\DATASETS\SWaT\Physical\SWaT_Dataset_Attack_v0.xlsx"

    print("=" * 70)
    print("BATCH STEP 1: Aggregate every date folder's network files")
    print("=" * 70)
    # force_reprocess=True: reprocess every folder from raw files this run,
    # rather than loading the existing checkpoints. Checkpoints are still
    # written/overwritten as each folder completes, so crash-safety for
    # THIS run is preserved — a crash partway through still only costs the
    # current folder, and a future run can set this back to False to resume
    # from these freshly-written checkpoints instead of reprocessing again.
    full_network_agg = process_all_dates(NETWORK_ROOT, force_reprocess=False)

    print()
    print("=" * 70)
    print("BATCH STEP 2: Build combined physical timeline (Normal_v1 + Attack_v0)")
    print("=" * 70)
    full_physical = build_combined_physical(NORMAL_PATH, ATTACK_PATH)

    print()
    print("=" * 70)
    print("BATCH STEP 3: Join full network features to full physical timeline")
    print("=" * 70)
    full_fused = join_network_physical(full_network_agg, full_physical, tolerance="1s")

    safe_to_csv(full_fused, "fused_full_span_output.csv")


if __name__ == "__main__":
    run_full_batch()