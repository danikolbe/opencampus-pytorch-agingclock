"""
UKB Aging Clock — Feature Selection
-------------------------------------
Applies data-quality-based feature selection within each modality.
Produces per-modality feature manifests (CSV) listing selected fields,
and logs dropped features with reasons.

Selection steps (in order):
    1. Missingness filter     : drop fields with >40% missing within modality
    2. Near-zero variance     : drop fields with variance < 1e-6
    3. Redundancy removal     : for clusters with |r| > 0.90, keep the field
                                with lowest missingness; log the full cluster

Olink: steps 1 and 2 only — no redundancy removal (too many features,
       let the encoder handle compression).

Outputs (to --out_dir):
    {modality}_feature_manifest.csv   one per modality — selected fields
    {modality}_dropped.csv            dropped fields with reason
    {modality}_corr_clusters.csv      redundancy clusters (non-olink only)
    feature_selection_summary.txt     human-readable report
    feature_selection_summary.png     before/after bar chart

Usage:
    python ukb_feature_selection.py --data_root /path/to/ukb_data --out_dir ./feature_selection
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
MISS_THRESH   = 0.40   # drop if > 40% missing within modality participants
VAR_THRESH    = 1e-6   # drop if variance below this
CORR_THRESH   = 0.90   # cluster threshold for redundancy removal

MODALITIES = {
    "physical_measures":  {"subdir": "1_phenotypic",    "data": "Physical_measures_data.parquet", "fmt": "parquet"},
    "blood_biochemistry": {"subdir": "2_blood_measures","data": "BloodBiochemistry_data.tsv",     "fmt": "tsv"},
    "blood_count":        {"subdir": "2_blood_measures","data": "BloodCount_data.tsv",            "fmt": "tsv"},
    "metabolomics":       {"subdir": "2_blood_measures","data": "metabolomics_data.tsv",          "fmt": "tsv"},
    "olink":              {"subdir": "2_blood_measures","data": "olink_data.tsv",                 "fmt": "tsv"},
}

EXCLUDE_FIELDS = {
    21, 36, 37, 39, 40, 41, 43, 44,
    96, 3077, 4081,
    20041, 20046, 20047, 20048,
}

PHYS_FIELD_NAMES = {
    48:"Waist circumference", 49:"Hip circumference",
    50:"Standing height", 51:"Seated height",
    93:"Systolic BP (manual)", 94:"Diastolic BP (manual)",
    95:"Pulse rate (BP measurement)", 102:"Pulse rate (automated)",
    3160:"Weight (manual entry)", 4079:"Diastolic BP (automated)",
    4080:"Systolic BP (automated)", 12143:"Weight (pre-imaging)",
    12144:"Height", 20015:"Sitting height",
    21001:"BMI", 21002:"Weight", 23098:"Weight",
    23099:"Body fat %", 23100:"Whole body fat mass",
    23101:"Whole body fat-free mass", 23102:"Whole body water mass",
    23104:"BMI", 23106:"Impedance (whole body)",
    23107:"Impedance leg (R)", 23108:"Impedance leg (L)",
    23109:"Impedance arm (R)", 23110:"Impedance arm (L)",
    23111:"Leg fat % (R)", 23112:"Leg fat mass (R)",
    23113:"Leg fat-free mass (R)", 23114:"Leg predicted mass (R)",
    23115:"Leg fat % (L)", 23116:"Leg fat mass (L)",
    23117:"Leg fat-free mass (L)", 23118:"Leg predicted mass (L)",
    23119:"Arm fat % (R)", 23120:"Arm fat mass (R)",
    23121:"Arm fat-free mass (R)", 23122:"Arm predicted mass (R)",
    23123:"Arm fat % (L)", 23124:"Arm fat mass (L)",
    23125:"Arm fat-free mass (L)", 23126:"Arm predicted mass (L)",
    23127:"Trunk fat %", 23128:"Trunk fat mass",
    23129:"Trunk fat-free mass", 23130:"Trunk predicted mass",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_col(col):
    m = re.match(r"^f_(\d+)_(\d+)", col)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def is_baseline(col):
    if col == "eid":
        return True
    fid, inst = parse_col(col)
    if fid is None or fid in EXCLUDE_FIELDS:
        return False
    return inst == 0


def load_baseline(data_root, mod_name, info):
    path = data_root / info["subdir"] / info["data"]
    if not path.exists():
        print(f"  [MISSING] {path}")
        return None
    print(f"  Loading {mod_name}...")
    if info["fmt"] == "tsv":
        df = pd.read_csv(path, sep="\t", low_memory=False, encoding="latin-1")
    else:
        df = pd.read_parquet(path)
    return df[[c for c in df.columns if is_baseline(c)]].copy()


def eids_with_data(df):
    data_cols = [c for c in df.columns if c != "eid"]
    return set(df.loc[df[data_cols].notna().any(axis=1), "eid"].astype(int))


def restrict(df, eid_set):
    return df[df["eid"].isin(np.array(list(eid_set), dtype=int))].copy()


def load_field_names(data_root, mod_name):
    """Load FieldID -> name mapping."""
    if mod_name == "physical_measures":
        return PHYS_FIELD_NAMES

    chars_map = {
        "blood_biochemistry": ("2_blood_measures", "BloodBiochemistry_chars.tsv"),
        "blood_count":        ("2_blood_measures", "BloodCount_chars.tsv"),
        "metabolomics":       ("2_blood_measures", "metabolomics_chars.tsv"),
        "olink":              ("2_blood_measures", "olink_chars.tsv"),
    }
    if mod_name not in chars_map:
        return {}
    subdir, fname = chars_map[mod_name]
    for sd in [subdir, "1_phenotypic"]:
        path = data_root / sd / fname
        if path.exists():
            try:
                df = pd.read_csv(path, sep="\t", comment="#",
                                 low_memory=False, encoding="latin-1")
                if "FieldID" in df.columns and "Field" in df.columns:
                    return dict(zip(df["FieldID"].astype(int), df["Field"]))
            except Exception:
                pass
    return {}


def col_to_name(col, field_names):
    fid, _ = parse_col(col)
    return field_names.get(fid, col) if fid else col


# ── Selection steps ───────────────────────────────────────────────────────────

def step1_missingness(df_sub, log):
    """Drop columns with >MISS_THRESH missing within modality participants."""
    data_cols = [c for c in df_sub.columns if c != "eid"]
    miss      = df_sub[data_cols].isna().mean()
    drop      = miss[miss > MISS_THRESH].index.tolist()
    keep      = miss[miss <= MISS_THRESH].index.tolist()
    log["missingness"] = {
        "dropped": drop,
        "reason":  f">{MISS_THRESH*100:.0f}% missing within modality participants",
    }
    return df_sub[["eid"] + keep], drop


def step2_variance(df_sub, log):
    """Drop near-zero variance columns."""
    data_cols = [c for c in df_sub.columns if c != "eid"]
    data      = df_sub[data_cols].apply(pd.to_numeric, errors="coerce")
    variance  = data.var()
    drop      = variance[variance < VAR_THRESH].index.tolist()
    keep      = variance[variance >= VAR_THRESH].index.tolist()
    log["variance"] = {
        "dropped": drop,
        "reason":  f"variance < {VAR_THRESH}",
    }
    return df_sub[["eid"] + keep], drop


def step3_redundancy(df_sub, field_names, log):
    """
    Identify correlated clusters (|r| > CORR_THRESH).
    Within each cluster, keep the field with lowest missingness.
    Log the full cluster for reference.
    Returns filtered df, list of dropped cols, cluster records.
    """
    data_cols = [c for c in df_sub.columns if c != "eid"]
    data      = df_sub[data_cols].apply(pd.to_numeric, errors="coerce")

    print(f"    Computing {len(data_cols)}×{len(data_cols)} correlation matrix...")
    corr = data.corr(method="pearson").fillna(0)

    # Build distance matrix from |r|
    dist_mat = 1 - np.abs(corr.values)
    np.fill_diagonal(dist_mat, 0)
    dist_mat = np.clip(dist_mat, 0, None)

    # Hierarchical clustering
    linkage  = sch.linkage(squareform(dist_mat), method="complete")
    # Cut at distance = 1 - CORR_THRESH to get clusters where all |r| > threshold
    labels   = sch.fcluster(linkage, t=1 - CORR_THRESH, criterion="distance")

    # Group columns by cluster
    clusters = {}
    for col, label in zip(data_cols, labels):
        clusters.setdefault(label, []).append(col)

    miss_rates = df_sub[data_cols].isna().mean()

    to_drop    = []
    to_keep    = []
    cluster_records = []

    for cluster_id, cols in clusters.items():
        if len(cols) == 1:
            to_keep.append(cols[0])
            continue

        # Pick representative: lowest missingness
        miss_in_cluster = miss_rates[cols]
        representative  = miss_in_cluster.idxmin()
        dropped_here    = [c for c in cols if c != representative]

        to_keep.append(representative)
        to_drop.extend(dropped_here)

        # Log cluster
        cluster_records.append({
            "cluster_id":      cluster_id,
            "n_members":       len(cols),
            "representative":  representative,
            "representative_name": col_to_name(representative, field_names),
            "representative_miss": round(miss_rates[representative] * 100, 2),
            "dropped_cols":    "|".join(dropped_here),
            "dropped_names":   "|".join(col_to_name(c, field_names) for c in dropped_here),
            "all_cols":        "|".join(cols),
            "all_names":       "|".join(col_to_name(c, field_names) for c in cols),
        })

    log["redundancy"] = {
        "dropped": to_drop,
        "reason":  f"|r| > {CORR_THRESH} cluster — kept lowest-missing representative",
        "n_clusters_reduced": len(cluster_records),
    }

    return df_sub[["eid"] + to_keep], to_drop, cluster_records


# ── Main selection pipeline per modality ──────────────────────────────────────

def select_modality(mod_name, df, eid_set, field_names, out_dir, do_redundancy=True):
    print(f"\n  {'─'*55}")
    print(f"  {mod_name.upper()}")
    print(f"  {'─'*55}")

    df_sub    = restrict(df, eid_set)
    data_cols = [c for c in df_sub.columns if c != "eid"]
    n_start   = len(data_cols)
    print(f"    Start : {n_start} features, {len(df_sub):,} participants")

    log           = {}
    all_dropped   = []
    cluster_recs  = []

    # Step 1 — missingness
    df_sub, dropped = step1_missingness(df_sub, log)
    all_dropped.extend([(c, "missingness") for c in dropped])
    print(f"    Step 1 missingness  : dropped {len(dropped):>4}  → {len([c for c in df_sub.columns if c != 'eid'])} remain")

    # Step 2 — variance
    df_sub, dropped = step2_variance(df_sub, log)
    all_dropped.extend([(c, "near_zero_variance") for c in dropped])
    print(f"    Step 2 variance     : dropped {len(dropped):>4}  → {len([c for c in df_sub.columns if c != 'eid'])} remain")

    # Step 3 — redundancy (skip for olink)
    if do_redundancy:
        df_sub, dropped, cluster_recs = step3_redundancy(df_sub, field_names, log)
        all_dropped.extend([(c, "redundancy_cluster") for c in dropped])
        n_reduced = log["redundancy"]["n_clusters_reduced"]
        print(f"    Step 3 redundancy   : dropped {len(dropped):>4}  → {len([c for c in df_sub.columns if c != 'eid'])} remain  ({n_reduced} clusters reduced)")
    else:
        print(f"    Step 3 redundancy   : skipped (olink — encoder handles compression)")

    final_cols = [c for c in df_sub.columns if c != "eid"]
    n_final    = len(final_cols)
    print(f"    Final               : {n_final} features  (removed {n_start - n_final} = {(n_start-n_final)/n_start*100:.1f}%)")

    # ── Build manifest ────────────────────────────────────────────────────────
    manifest_rows = []
    for col in final_cols:
        fid, inst = parse_col(col)
        manifest_rows.append({
            "column":     col,
            "field_id":   fid,
            "instance":   inst,
            "field_name": col_to_name(col, field_names),
            "pct_missing": round(df_sub[col].isna().mean() * 100, 2),
        })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(out_dir / f"{mod_name}_feature_manifest.csv", index=False)

    # ── Dropped log ───────────────────────────────────────────────────────────
    dropped_rows = []
    for col, reason in all_dropped:
        fid, _ = parse_col(col)
        dropped_rows.append({
            "column":     col,
            "field_id":   fid,
            "field_name": col_to_name(col, field_names),
            "reason":     reason,
        })
    pd.DataFrame(dropped_rows).to_csv(
        out_dir / f"{mod_name}_dropped.csv", index=False)

    # ── Cluster log ───────────────────────────────────────────────────────────
    if cluster_recs:
        pd.DataFrame(cluster_recs).to_csv(
            out_dir / f"{mod_name}_corr_clusters.csv", index=False)

    return {
        "modality": mod_name,
        "n_start":  n_start,
        "n_final":  n_final,
        "n_dropped_miss":   len(log.get("missingness", {}).get("dropped", [])),
        "n_dropped_var":    len(log.get("variance",    {}).get("dropped", [])),
        "n_dropped_redund": len(log.get("redundancy",  {}).get("dropped", [])),
        "n_clusters":       len(cluster_recs),
    }


# ── Summary figure ────────────────────────────────────────────────────────────

def plot_summary(results, out_dir):
    PALETTE = {
        "physical_measures": "#DD8452",
        "blood_biochemistry":"#55A868",
        "blood_count":       "#C44E52",
        "metabolomics":      "#8172B2",
        "olink":             "#937860",
    }
    LABELS = {
        "physical_measures": "Physical Measures",
        "blood_biochemistry":"Blood Biochemistry",
        "blood_count":       "Blood Count",
        "metabolomics":      "Metabolomics (NMR)",
        "olink":             "Proteomics (Olink)",
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    mods    = [r["modality"] for r in results]
    starts  = [r["n_start"]  for r in results]
    finals  = [r["n_final"]  for r in results]
    labels  = [LABELS.get(m, m) for m in mods]
    colors  = [PALETTE.get(m, "#999") for m in mods]

    x      = np.arange(len(mods))
    width  = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Before / after grouped bar
    bars1 = axes[0].bar(x - width/2, starts, width, label="Before",
                        color=[c + "88" for c in colors],  # translucent
                        edgecolor=colors, linewidth=1.5)
    bars2 = axes[0].bar(x + width/2, finals, width, label="After",
                        color=colors, edgecolor="white")

    for bar, n in zip(bars2, finals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     str(n), ha="center", va="bottom", fontsize=9)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("Number of features")
    axes[0].set_title("Feature Count Before vs After Selection")
    axes[0].legend(frameon=False)

    # Stacked bar of what was dropped and why
    miss_d   = [r["n_dropped_miss"]   for r in results]
    var_d    = [r["n_dropped_var"]    for r in results]
    redund_d = [r["n_dropped_redund"] for r in results]

    axes[1].bar(x, miss_d,   label="Missingness >40%",      color="#E07B8A")
    axes[1].bar(x, var_d,    bottom=miss_d, label="Near-zero variance", color="#F0C040")
    bottom2 = [m + v for m, v in zip(miss_d, var_d)]
    axes[1].bar(x, redund_d, bottom=bottom2, label="Redundancy (|r|>0.90)", color="#7FB3D3")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("Features dropped")
    axes[1].set_title("Features Dropped by Reason")
    axes[1].legend(frameon=False)

    fig.suptitle(f"Feature Selection Summary  "
                 f"(missingness >{MISS_THRESH*100:.0f}%, "
                 f"var <{VAR_THRESH}, |r| >{CORR_THRESH})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_selection_summary.png")
    plt.close(fig)
    print("\n  [saved] feature_selection_summary.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_root: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines   = []
    results = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 60)
    log("  UKB AGING CLOCK — FEATURE SELECTION")
    log(f"  Missingness threshold  : >{MISS_THRESH*100:.0f}%")
    log(f"  Variance threshold     : <{VAR_THRESH}")
    log(f"  Redundancy threshold   : |r| >{CORR_THRESH}")
    log("=" * 60)

    for mod_name, info in MODALITIES.items():
        df = load_baseline(data_root, mod_name, info)
        if df is None:
            continue
        eid_set     = eids_with_data(df)
        field_names = load_field_names(data_root, mod_name)

        # Olink: skip redundancy step
        do_redund = (mod_name != "olink")

        result = select_modality(
            mod_name, df, eid_set, field_names, out_dir,
            do_redundancy=do_redund,
        )
        results.append(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("  SUMMARY")
    log("=" * 60)
    log(f"\n  {'Modality':<25} {'Start':>6} {'Final':>6} {'Dropped':>8}  "
        f"{'Miss':>6} {'Var':>5} {'Redund':>7} {'Clusters':>9}")
    log("  " + "─" * 75)
    for r in results:
        log(f"  {r['modality']:<25} {r['n_start']:>6} {r['n_final']:>6} "
            f"{r['n_start']-r['n_final']:>8}  "
            f"{r['n_dropped_miss']:>6} {r['n_dropped_var']:>5} "
            f"{r['n_dropped_redund']:>7} {r['n_clusters']:>9}")

    total_start = sum(r["n_start"] for r in results)
    total_final = sum(r["n_final"] for r in results)
    log(f"\n  Total features : {total_start} → {total_final} "
        f"({total_start - total_final} removed, "
        f"{(total_start-total_final)/total_start*100:.1f}%)")

    log(f"\n  Manifests saved to: {out_dir.resolve()}")
    log("  Load with: pd.read_csv(out_dir / '{modality}_feature_manifest.csv')")

    with open(out_dir / "feature_selection_summary.txt", "w") as f:
        f.write("\n".join(lines))
    log("\n  [saved] feature_selection_summary.txt")

    plot_summary(results, out_dir)
    log("Done.")


def eids_with_data(df):
    data_cols = [c for c in df.columns if c != "eid"]
    return set(df.loc[df[data_cols].notna().any(axis=1), "eid"].astype(int))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True,
                        help="Root dir with 1_phenotypic/ and 2_blood_measures/")
    parser.add_argument("--out_dir",   type=Path,
                        default=Path("./feature_selection"),
                        help="Output directory (default: ./feature_selection)")
    args = parser.parse_args()
    main(args.data_root, args.out_dir)
