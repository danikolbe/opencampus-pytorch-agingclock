"""
UKB Aging Clock — Classical ML Baseline
-----------------------------------------
Trains classical ML baselines for age prediction using selected features.

Models:
    - Null model (mean prediction)
    - ElasticNet (linear clock baseline)
    - XGBoost (main classical baseline)

Cohorts:
    - Model 1: all modalities excluding olink (~274k participants)
    - Model 2: all modalities including olink (~19-53k participants)

Outputs:
    results_summary.txt
    results_summary.csv
    predicted_vs_actual_{model}_{cohort}.png
    age_acceleration_{model}_{cohort}.png
    feature_importance_{model}_{cohort}.png
    {model}_{cohort}.pkl                     saved model

Usage:
    python ukb_baseline_models.py \
        --data_root /superscratch/dkolbe/aDNA/UKB/data \
        --manifest_dir /superscratch/dkolbe/aging_clock/modelling/outputs/feature_selection \
        --out_dir /superscratch/dkolbe/aging_clock/modelling/outputs/baseline_models
"""

import argparse
import pickle
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
MODALITIES_NO_OLINK = [
    "physical_measures",
    "blood_biochemistry",
    "blood_count",
    "metabolomics",
]
MODALITIES_WITH_OLINK = MODALITIES_NO_OLINK + ["olink"]

MODALITY_FILES = {
    "physical_measures":  ("1_phenotypic",    "Physical_measures_data.parquet", "parquet"),
    "blood_biochemistry": ("2_blood_measures", "BloodBiochemistry_data.tsv",     "tsv"),
    "blood_count":        ("2_blood_measures", "BloodCount_data.tsv",            "tsv"),
    "metabolomics":       ("2_blood_measures", "metabolomics_data.tsv",          "tsv"),
    "olink":              ("2_blood_measures", "olink_data.tsv",                 "tsv"),
}

EXCLUDE_FIELDS = {
    21, 36, 37, 39, 40, 41, 43, 44,
    96, 3077, 4081,
    20041, 20046, 20047, 20048,
}

N_CV_FOLDS  = 5
RANDOM_SEED = 42

PALETTE = {
    "null":       "#999999",
    "elasticnet": "#4C72B0",
    "xgboost":    "#DD8452",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})


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


def load_manifest(manifest_dir, mod_name):
    path = manifest_dir / f"{mod_name}_feature_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return pd.read_csv(path)["column"].tolist()


def load_modality(data_root, mod_name, selected_cols):
    subdir, fname, fmt = MODALITY_FILES[mod_name]
    path = data_root / subdir / fname
    print(f"  Loading {mod_name}...")
    if fmt == "tsv":
        df = pd.read_csv(path, sep="\t", low_memory=False, encoding="latin-1")
    else:
        df = pd.read_parquet(path)
    # Keep eid + selected baseline cols that exist in this file
    keep = ["eid"] + [c for c in selected_cols if c in df.columns]
    return df[keep].copy()


def load_age_sex(data_root):
    path = data_root / "1_phenotypic" / "popchar_data.tsv"
    df   = pd.read_csv(path, sep="\t", low_memory=False, encoding="latin-1",
                       usecols=["eid", "f_21022_0_0", "f_31_0_0"])
    # Use unambiguous names that cannot clash with any feature column
    df = df.rename(columns={"f_21022_0_0": "__age__", "f_31_0_0": "__sex__"})
    return df.dropna(subset=["__age__"])


PHYS_FIELD_NAMES = {
    48:"Waist circumference", 49:"Hip circumference",
    50:"Standing height", 51:"Seated height",
    93:"Systolic BP (manual)", 94:"Diastolic BP (manual)",
    95:"Pulse rate (BP)", 102:"Pulse rate (automated)",
    3160:"Weight (manual)", 4079:"Diastolic BP (automated)",
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

def load_all_field_names(data_root):
    """Load FieldID -> human name for all modalities."""
    chars_map = {
        "blood_biochemistry": ("2_blood_measures", "BloodBiochemistry_chars.tsv"),
        "blood_count":        ("2_blood_measures", "BloodCount_chars.tsv"),
        "metabolomics":       ("2_blood_measures", "metabolomics_chars.tsv"),
        "olink":              ("2_blood_measures", "olink_chars.tsv"),
    }
    field_names = {"physical_measures": PHYS_FIELD_NAMES}
    for mod, (subdir, fname) in chars_map.items():
        path = data_root / subdir / fname
        if path.exists():
            try:
                df = pd.read_csv(path, sep="\t", comment="#",
                                 low_memory=False, encoding="latin-1")
                if "FieldID" in df.columns and "Field" in df.columns:
                    field_names[mod] = dict(zip(df["FieldID"].astype(int), df["Field"]))
            except Exception:
                field_names[mod] = {}
        else:
            field_names[mod] = {}
    return field_names


def resolve_feature_name(prefixed_col, field_names, max_len=35):
    """Convert e.g. 'blood_count__f_30000_0_0' -> human readable name."""
    parts  = prefixed_col.split("__", 1)
    mod    = parts[0]
    col    = parts[1] if len(parts) > 1 else prefixed_col
    m      = re.match(r"^f_(\d+)_", col)
    fid    = int(m.group(1)) if m else None
    fnames = field_names.get(mod, {})
    name   = fnames.get(fid, col) if fid else col
    return name[:max_len-1] + "…" if len(name) > max_len else name


def build_cohort(data_root, manifest_dir, modality_list):
    """
    Load and merge all modalities for a given cohort definition.
    Returns X (features), y (age), eid array, feature names.
    """
    # Load age + sex — use dunder names to avoid any column clash with features
    pop = load_age_sex(data_root)

    # Keep only eid + target — do NOT include sex as a feature here
    # Sex can be added explicitly as a covariate later if needed
    target_df    = pop[["eid", "__age__"]].copy()
    feature_cols = []
    feat_dfs     = []

    for mod in modality_list:
        selected = load_manifest(manifest_dir, mod)
        df_mod   = load_modality(data_root, mod, selected)
        # Prefix every feature column with modality name — guaranteed no clashes
        rename   = {c: f"{mod}__{c}" for c in df_mod.columns if c != "eid"}
        df_mod   = df_mod.rename(columns=rename)
        feature_cols.extend(list(rename.values()))
        feat_dfs.append(df_mod)

    # Merge features onto target — inner join
    merged = target_df
    for df in feat_dfs:
        merged = merged.merge(df, on="eid", how="inner")

    # Drop participants missing age (should be none after inner join)
    merged = merged.dropna(subset=["__age__"])

    print(f"  Cohort size: {len(merged):,} participants, "
          f"{len(feature_cols)} features")

    # Verify age looks right
    age_vals = merged["__age__"].values.astype(np.float32)
    print(f"  Age range  : {age_vals.min():.1f} – {age_vals.max():.1f}  "
          f"mean={age_vals.mean():.1f}  std={age_vals.std():.1f}")

    X          = merged[feature_cols].values.astype(np.float32)
    y          = age_vals
    eids       = merged["eid"].values
    feat_names = feature_cols

    return X, y, eids, feat_names


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(X_train, X_test):
    """Impute (median) then standardise. Fit on train, apply to test."""
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    X_train = imputer.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)

    X_test  = imputer.transform(X_test)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, imputer, scaler


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, model_name):
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    r, p = pearsonr(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    return {
        "model": model_name,
        "mae":   round(mae, 3),
        "r2":    round(r2,  4),
        "pearson_r": round(r, 4),
        "bias":  round(bias, 3),
        "n":     len(y_true),
    }


# ── Null model ────────────────────────────────────────────────────────────────

def null_model_cv(y, n_splits=N_CV_FOLDS):
    """Predict mean of training set for all test samples."""
    kf      = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    y_pred  = np.zeros_like(y)
    for train_idx, test_idx in kf.split(y):
        y_pred[test_idx] = y[train_idx].mean()
    return y_pred


# ── ElasticNet ────────────────────────────────────────────────────────────────

def elasticnet_cv(X, y, n_splits=N_CV_FOLDS):
    """
    ElasticNetCV with nested cross-validation.
    Outer loop: predict out-of-fold. Inner loop: tune alpha/l1_ratio.
    """
    kf     = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    y_pred = np.zeros_like(y, dtype=np.float32)
    models = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"    Fold {fold+1}/{n_splits}...", end=" ", flush=True)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train         = y[train_idx]

        X_train, X_test, _, _ = preprocess(X_train, X_test)

        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
            cv=3, max_iter=5000, random_state=RANDOM_SEED, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred[test_idx] = model.predict(X_test)
        models.append(model)
        print(f"alpha={model.alpha_:.4f} l1={model.l1_ratio_:.2f}")

    # Refit on full data for saving + feature importance
    X_all, _, imp, scl = preprocess(X, X[:1])
    X_all = imp.transform(X)
    X_all = scl.transform(X_all)
    final = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
        cv=5, max_iter=5000, random_state=RANDOM_SEED, n_jobs=-1,
    )
    final.fit(X_all, y)

    return y_pred, final


# ── XGBoost ───────────────────────────────────────────────────────────────────

def xgboost_cv(X, y, n_splits=N_CV_FOLDS):
    """XGBoost with cross-validation."""
    kf     = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    y_pred = np.zeros_like(y, dtype=np.float32)
    models = []

    xgb_params = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="mae",
    )

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"    Fold {fold+1}/{n_splits}...", end=" ", flush=True)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train, X_test, _, _ = preprocess(X_train, X_test)

        # Split train into train/val for early stopping
        val_size  = int(len(X_train) * 0.1)
        X_val     = X_train[:val_size]
        y_val     = y_train[:val_size]
        X_tr      = X_train[val_size:]
        y_tr      = y_train[val_size:]

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        y_pred[test_idx] = model.predict(X_test)
        models.append(model)
        fold_mae = mean_absolute_error(y_test, y_pred[test_idx])
        print(f"MAE={fold_mae:.3f}")

    # Refit on full data
    X_all, _, imp, scl = preprocess(X, X[:1])
    X_all = imp.transform(X)
    X_all = scl.transform(X_all)

    val_size  = int(len(X_all) * 0.1)
    final = xgb.XGBRegressor(**{**xgb_params, "early_stopping_rounds": None,
                                "n_estimators": 500})
    final.fit(X_all, y, verbose=False)

    return y_pred, final


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(y_true, predictions, cohort_name, out_dir):
    """Overlay predicted vs actual age for all models."""
    fig, axes = plt.subplots(1, len(predictions), figsize=(6 * len(predictions), 5))
    if len(predictions) == 1:
        axes = [axes]

    for ax, (model_name, y_pred) in zip(axes, predictions.items()):
        metrics = compute_metrics(y_true, y_pred, model_name)
        ax.scatter(y_true, y_pred, alpha=0.05, s=1,
                   color=PALETTE.get(model_name, "#999"))
        lims = [y_true.min() - 1, y_true.max() + 1]
        ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual age (years)")
        ax.set_ylabel("Predicted age (years)")
        ax.set_title(
            f"{model_name.capitalize()}\n"
            f"MAE={metrics['mae']:.2f}y  R²={metrics['r2']:.3f}  "
            f"r={metrics['pearson_r']:.3f}"
        )
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(f"Predicted vs Actual Age — {cohort_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"predicted_vs_actual_{cohort_name}.png")
    plt.close(fig)
    print(f"  [saved] predicted_vs_actual_{cohort_name}.png")


def plot_age_acceleration(y_true, predictions, cohort_name, out_dir):
    """
    Age acceleration = predicted - actual.
    Positive = biologically older than chronological age.
    """
    fig, axes = plt.subplots(1, len(predictions), figsize=(6 * len(predictions), 5))
    if len(predictions) == 1:
        axes = [axes]

    for ax, (model_name, y_pred) in zip(axes, predictions.items()):
        accel = y_pred - y_true
        ax.hist(accel, bins=60, color=PALETTE.get(model_name, "#999"),
                alpha=0.8, edgecolor="white", linewidth=0.3, density=True)
        ax.axvline(0,           color="black", linewidth=1.2, linestyle="--")
        ax.axvline(accel.mean(), color="red",  linewidth=1,   linestyle=":",
                   label=f"Mean={accel.mean():.2f}y")
        ax.set_xlabel("Age acceleration (years)")
        ax.set_ylabel("Density")
        ax.set_title(f"{model_name.capitalize()}\n"
                     f"SD={accel.std():.2f}y  "
                     f"Bias={accel.mean():.2f}y")
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(f"Age Acceleration Distribution — {cohort_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"age_acceleration_{cohort_name}.png")
    plt.close(fig)
    print(f"  [saved] age_acceleration_{cohort_name}.png")


def plot_feature_importance(model, feat_names, model_name, cohort_name,
                            out_dir, field_names=None, top_n=30):
    """Top N feature importances — coefficients for ElasticNet, gain for XGBoost."""
    if model_name == "elasticnet":
        importance = np.abs(model.coef_)
        title_sfx  = "|Coefficient|"
    else:
        importance = model.feature_importances_
        title_sfx  = "Gain"

    # Sort and take top N
    idx    = np.argsort(importance)[::-1][:top_n]
    vals   = importance[idx]
    modals = [feat_names[i].split("__", 1)[0] for i in idx]
    if field_names:
        names = [resolve_feature_name(feat_names[i], field_names) for i in idx]
    else:
        names = [feat_names[i].split("__", 1)[-1] for i in idx]

    PALETTE_MOD = {
        "physical_measures": "#DD8452",
        "blood_biochemistry":"#55A868",
        "blood_count":       "#C44E52",
        "metabolomics":      "#8172B2",
        "olink":             "#937860",
    }
    colors = [PALETTE_MOD.get(m, "#999") for m in modals]

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.28)))
    ax.barh(range(len(vals)), vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(title_sfx)
    ax.set_title(f"Top {top_n} Features — {model_name.capitalize()} ({cohort_name})")

    # Legend for modalities
    from matplotlib.patches import Patch
    seen = []
    handles = []
    for m, c in zip(modals, colors):
        if m not in seen:
            handles.append(Patch(facecolor=c, label=m.replace("_", " ")))
            seen.append(m)
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / f"feature_importance_{model_name}_{cohort_name}.png")
    plt.close(fig)
    print(f"  [saved] feature_importance_{model_name}_{cohort_name}.png")


# ── Run one cohort ────────────────────────────────────────────────────────────

def run_cohort(cohort_name, modality_list, data_root, manifest_dir, out_dir):
    print(f"\n{'='*60}")
    print(f"  COHORT: {cohort_name.upper()}")
    print(f"{'='*60}")

    print("\nBuilding cohort...")
    X, y, eids, feat_names = build_cohort(data_root, manifest_dir, modality_list)

    all_metrics  = []
    all_preds    = {}

    # ── Null model ────────────────────────────────────────────────────────────
    print("\nNull model...")
    y_pred_null = null_model_cv(y)
    m = compute_metrics(y, y_pred_null, "null")
    all_metrics.append(m)
    all_preds["null"] = y_pred_null
    print(f"  MAE={m['mae']:.3f}  R²={m['r2']:.4f}  r={m['pearson_r']:.4f}")

    # ── ElasticNet ────────────────────────────────────────────────────────────
    print("\nElasticNet (cross-validated)...")
    y_pred_en, en_model = elasticnet_cv(X, y)
    m = compute_metrics(y, y_pred_en, "elasticnet")
    all_metrics.append(m)
    all_preds["elasticnet"] = y_pred_en
    print(f"  MAE={m['mae']:.3f}  R²={m['r2']:.4f}  r={m['pearson_r']:.4f}")

    with open(out_dir / f"elasticnet_{cohort_name}.pkl", "wb") as f:
        pickle.dump(en_model, f)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\nXGBoost (cross-validated)...")
    y_pred_xgb, xgb_model = xgboost_cv(X, y)
    m = compute_metrics(y, y_pred_xgb, "xgboost")
    all_metrics.append(m)
    all_preds["xgboost"] = y_pred_xgb
    print(f"  MAE={m['mae']:.3f}  R²={m['r2']:.4f}  r={m['pearson_r']:.4f}")

    with open(out_dir / f"xgboost_{cohort_name}.pkl", "wb") as f:
        pickle.dump(xgb_model, f)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    field_names = load_all_field_names(data_root)
    plot_predicted_vs_actual(y, all_preds, cohort_name, out_dir)
    plot_age_acceleration(y, all_preds, cohort_name, out_dir)
    plot_feature_importance(en_model,  feat_names, "elasticnet", cohort_name, out_dir, field_names)
    plot_feature_importance(xgb_model, feat_names, "xgboost",    cohort_name, out_dir, field_names)

    # Save predictions + age acceleration
    pred_df = pd.DataFrame({
        "eid":           eids,
        "age_actual":    y,
        "age_pred_null": y_pred_null,
        "age_pred_en":   y_pred_en,
        "age_pred_xgb":  y_pred_xgb,
        "accel_null":    y_pred_null - y,
        "accel_en":      y_pred_en   - y,
        "accel_xgb":     y_pred_xgb  - y,
    })
    # Sanity check — age should be in years (37-73 for UKB)
    assert y.min() > 30 and y.max() < 80,         f"Age values look wrong: min={y.min()}, max={y.max()}"
    pred_df.to_csv(out_dir / f"predictions_{cohort_name}.csv", index=False)
    print(f"  [saved] predictions_{cohort_name}.csv")

    return all_metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_root, manifest_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Cohort 1 — no olink
    metrics_1 = run_cohort(
        "no_olink", MODALITIES_NO_OLINK,
        data_root, manifest_dir, out_dir,
    )
    for m in metrics_1:
        m["cohort"] = "no_olink"
    all_results.extend(metrics_1)

    # Cohort 2 — with olink
    metrics_2 = run_cohort(
        "with_olink", MODALITIES_WITH_OLINK,
        data_root, manifest_dir, out_dir,
    )
    for m in metrics_2:
        m["cohort"] = "with_olink"
    all_results.extend(metrics_2)

    # ── Summary ───────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)[
        ["cohort", "model", "n", "mae", "r2", "pearson_r", "bias"]
    ]
    results_df.to_csv(out_dir / "results_summary.csv", index=False)

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))

    with open(out_dir / "results_summary.txt", "w") as f:
        f.write(results_df.to_string(index=False))

    print(f"\nDone. Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    type=Path, required=True)
    parser.add_argument("--manifest_dir", type=Path, required=True)
    parser.add_argument("--out_dir",      type=Path,
                        default=Path("./outputs/baseline_models"))
    args = parser.parse_args()
    main(args.data_root, args.manifest_dir, args.out_dir)