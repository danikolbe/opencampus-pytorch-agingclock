# Baseline Model

## Baseline Model Results

### Model Selection
- **Baseline Model Types:**
  - **Null model** — predicts the mean age of the training set for all participants
  - **ElasticNet** — regularised linear regression (L1 + L2 penalty)
  - **XGBoost** — gradient boosted decision trees

- **Rationale:** Three models are evaluated to establish a performance hierarchy.
  The null model defines the floor — any useful model must beat it. ElasticNet
  serves as a linear clock baseline, directly comparable to published aging clocks
  (PhenoAge, BioAge) which are also penalised linear models. XGBoost captures
  non-linear interactions between features that a linear model cannot, and
  represents the practical ceiling for classical ML on this task. Both are
  evaluated with 5-fold cross-validation to produce honest out-of-fold estimates.

### Model Performance

Two cohorts are evaluated: one without Olink proteomics (~502k participants,
~160 features) and one including Olink (~53k participants, ~2,100 features).

| Cohort | Model | MAE (years) | R² | Pearson r |
|---|---|---|---|---|
| No Olink | Null | 6.927 | 0.000 | -0.002 |
| No Olink | ElasticNet | 4.545 | 0.507 | 0.712 |
| No Olink | XGBoost | 3.444 | 0.710 | 0.845 |
| With Olink | Null | 7.022 | 0.000 | -0.010 |
| With Olink | ElasticNet | 2.426 | 0.859 | 0.927 |
| With Olink | XGBoost | 2.603 | 0.837 | 0.915 |

### Evaluation Methodology
- **Evaluation strategy:** 5-fold cross-validation with out-of-fold predictions.
  No separate test set is held out at this stage — all reported metrics are
  out-of-fold estimates, meaning no participant's label was ever seen by the
  model that predicted it.
- **Preprocessing:** Median imputation for remaining missing values, followed
  by StandardScaler (z-score normalisation). Both fit on training folds only
  to prevent data leakage.
- **ElasticNet:** Nested CV — inner 3-fold tunes alpha and l1_ratio across a
  grid of values; outer 5-fold produces evaluation predictions.
- **XGBoost:** Early stopping on a 10% validation split carved from each
  training fold; outer 5-fold produces evaluation predictions.

### Evaluation Metrics

- **MAE (Mean Absolute Error):** Primary metric. Directly interpretable —
  "on average, the model predicts age to within X years." Appropriate for a
  regression task where the scale of errors is clinically meaningful. An MAE
  of 3.4 years means the model's predicted age is within 3.4 years of actual
  age on average.

- **R²:** Proportion of variance in chronological age explained by the model.
  R² = 0 is equivalent to the null model; R² = 1 is perfect prediction.
  Useful for comparing across cohorts and against published clocks.

- **Pearson r:** Correlation between predicted and actual age. Standard
  reporting metric in the aging clock literature — allows direct comparison
  with published clocks (e.g. PhenoAge reports r ≈ 0.94 on UKB).

- **Bias:** Mean of (predicted − actual). Should be near zero — a systematic
  positive bias would mean the model consistently overestimates age.

### Metric Practical Relevance

The primary output of an aging clock is not just age prediction accuracy, but
the **age acceleration residual** (predicted age − actual age). A participant
predicted to be 65 when they are 58 has a +7 year acceleration — they are
biologically older than their chronological age, which is associated with
increased disease risk and mortality.

MAE directly bounds the reliability of this residual: a model with MAE of 3.4
years produces acceleration scores precise enough to stratify individuals into
biologically older/younger groups, but not precise enough to detect subtle
1-year differences. The with-Olink ElasticNet (MAE 2.4 years) approaches the
precision needed for individual-level clinical interpretation.

The ElasticNet > XGBoost reversal in the with-Olink cohort is notable — Olink
NPX values are already log-normalised, and at N ≈ 53k, linear relationships
appear to dominate. This is consistent with findings in the proteomics aging
clock literature.

## Next Steps
This baseline serves as the reference point for a multimodal deep learning
aging clock implemented in PyTorch, featuring per-modality encoders, an
intermediate fusion architecture, and a shared regression head. The neural
model is expected to outperform XGBoost by capturing cross-modality
interactions that classical models cannot represent.
