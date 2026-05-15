```
## Literature Review

Approaches or solutions that have been tried before on similar projects.

### Summary of Each Work

**Source 1: Proteomic aging clock predicts mortality and risk of common age-related diseases in diverse populations**

- Link: https://www.nature.com/articles/s41591-024-03164-7
- Objective: Develop a proteomic aging clock using plasma proteins to predict biological age, validated across genetically and geographically diverse populations.
- Methods: Trained LightGBM gradient boosting on 2,897 plasma proteins (Olink Explore 3072) from 45,441 UK Biobank participants (age 39-71). Compared six ML methods (LASSO, elastic net, gradient boosting, three neural networks). Used Boruta feature selection and SHAP values to identify 204 age-predictive proteins. 70/30 train/test split with external validation in China Kadoorie Biobank (n=3,977) and FinnGen (n=1,990).
- Outcomes: Achieved Pearson r = 0.94, R² = 0.88 in UK Biobank. Cross-population validation showed r = 0.92 (China) and r = 0.94 (Finland). Proteomic age gap predicted 18 chronic diseases and all-cause mortality (HR = 1.77 per SD). A reduced 20-protein model retained 91% accuracy. Top 5% vs bottom 5% of proteomic age gap differed by 12.3 years of biological aging.
- Relation to the Project: Provides validated methodology for proteomic feature selection and model training applicable to multimodal integration. Demonstrates LightGBM outperforms other methods for generalizability. External validation framework across diverse populations sets benchmark for robustness testing.

---

**Source 2: Metabolomic age (MileAge) predicts health and life span: A comparison of multiple machine learning algorithms**

- Link: https://www.science.org/doi/10.1126/sciadv.adp3743
- Objective: Benchmark 17 machine learning algorithms for metabolomic aging clock development and identify which best predicts health outcomes and mortality.
- Methods: Compared regularized regression, kernel methods (SVM), tree-based models, and ensemble methods using 168 NMR-quantified plasma metabolites from 225,212 UK Biobank participants (mean age 56.97). Implemented 10x5 nested cross-validation. Evaluated associations with frailty index, telomere length, chronic illness, self-rated health, and all-cause mortality.
- Outcomes: MAE ranged 5.31-6.36 years across algorithms (R² = 0.13-0.35). SVM-radial achieved lowest MAE. Cubist rule-based regression showed strongest associations with health markers. Individuals with accelerated MileAge had higher mortality (HR = 1.51, 95% CI 1.43-1.59), shorter telomeres, higher frailty, and worse self-rated health. 31.76% of participants showed age acceleration or deceleration of at least 3.75 years.
- Outcomes: Validation in additional 129,877 baseline participants and 10,618 repeat-assessment participants showed consistent performance.
- Relation to the Project: Provides comprehensive algorithm benchmarking for metabolomic data. Suggests SVM or Cubist as strong candidates for metabolomic modality. Demonstrates nonlinear models better capture aging signals than linear approaches. Large sample size (225k) sets expectations for metabolomic clock development.

---

**Source 3: Multimodal clocks of human aging**

- Link: https://pubmed.ncbi.nlm.nih.gov/42105758/
- Objective: Develop a comprehensive multimodal aging framework integrating clinical, physiological, and molecular data to quantify biological age heterogeneity and identify translational aging drivers.
- Methods: Developed mCAS (multicentric Chinese aging standardized) cohort from 2,019 individuals aged 18-91. Integrated high-dimensional clinical measurements, physiological assessments, and molecular profiling (plasma proteomics). Constructed three-tiered framework: (1) Core Capacity Clock (CC-clock) for clinical physiological decline, (2) Multimodal Clock (MM-clock) for enhanced precision across parameters, (3) organ-associated aging clocks for tissue-specific assessment.
- Outcomes: Demonstrated plasma protein clocks serve as efficient proxies for systemic physiological capacity. Cross-layer analysis validated molecular signatures against functional decline. Identified age-dependent accumulation of coagulation factors as novel driver of multi-organ senescence and systemic inflammatory activation. Framework bridges molecular biomarkers with clinical function.
- Relation to the Project: Most directly relevant multimodal integration framework. Three-tiered architecture (organismal, multimodal, organ-specific) provides template for UK Biobank implementation. Demonstrates value of cross-layer validation between molecular and clinical measures. Coagulation pathway findings suggest mechanistic targets worth investigating.

---

## Identified Gaps

Based on the literature review, this project addresses the following gaps:

1. **Limited multimodal integration on UK Biobank**: The mCAS multimodal framework was developed on a Chinese cohort. No equivalent comprehensive multimodal clock exists for UK Biobank despite availability of clinical, proteomic, metabolomic, and imaging data on the same individuals.

2. **Single-modality dominance**: Existing UK Biobank clocks operate on proteomics OR metabolomics OR imaging separately. Integration of multiple modalities from the same participants remains unexplored.

3. **No pretrained multimodal models**: Published studies provide methodology but not deployable model files. A reproducible multimodal pipeline with accessible weights would benefit the research community.

4. **Training target optimization**: Most clocks predict chronological age. Training directly on mortality or disease outcomes improves clinical utility but has not been applied to multimodal UK Biobank data.
```
