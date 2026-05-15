## Literature Review

Summary of prior approaches and solutions used in similar projects.

---

### Summary of Each Work

---

## Source 1: Proteomic Aging Clock Predicts Mortality and Risk of Age-Related Diseases

**Link:** https://www.nature.com/articles/s41591-024-03164-7  

**Objective:**  
Develop a proteomic aging clock using plasma protein data to predict biological age and validate it across genetically and geographically diverse populations.

**Methods:**  
- Trained LightGBM gradient boosting models on 2,897 plasma proteins (Olink Explore 3072) from 45,441 UK Biobank participants (ages 39–71).  
- Compared six ML methods: LASSO, elastic net, gradient boosting, and three neural network variants.  
- Used Boruta feature selection and SHAP analysis to identify 204 age-predictive proteins.  
- Train/test split: 70/30.  
- External validation datasets:
  - China Kadoorie Biobank (n = 3,977)  
  - FinnGen (n = 1,990)  

**Outcomes:**  
- Performance (UK Biobank): Pearson r = 0.94, R² = 0.88  
- External validation:
  - China: r = 0.92  
  - Finland: r = 0.94  
- Proteomic age gap associated with 18 chronic diseases and all-cause mortality (HR = 1.77 per SD).  
- Reduced model (20 proteins) retained ~91% predictive accuracy.  
- Top vs bottom 5% of proteomic age gap differed by 12.3 years of biological age.

**Relevance to Project:**  
- Demonstrates strong generalization of LightGBM for proteomic aging signals.  
- Provides a validated feature selection + interpretability pipeline (SHAP, Boruta).  
- Establishes robust multi-population validation as a key benchmark for biological age models.

---

## Source 2: Metabolomic Age (MileAge) Predicts Health and Lifespan

**Link:** https://www.science.org/doi/10.1126/sciadv.adp3743  

**Objective:**  
Benchmark machine learning models for metabolomic aging clocks and evaluate their association with health outcomes and mortality.

**Methods:**  
- Compared 17 ML algorithms (regularized regression, SVM, tree-based models, ensembles).  
- Input: 168 NMR-quantified plasma metabolites from 225,212 UK Biobank participants (mean age 56.97).  
- Used 10×5 nested cross-validation.  
- Evaluated outcomes including frailty index, telomere length, chronic disease burden, self-rated health, and mortality.

**Outcomes:**  
- MAE range: 5.31–6.36 years (R² = 0.13–0.35).  
- Best predictive model: SVM (radial kernel) for lowest MAE.  
- Strongest health associations: Cubist rule-based regression.  
- Mortality risk: HR = 1.51 (95% CI 1.43–1.59) for accelerated aging.  
- 31.76% of participants showed ≥ ±3.75 years age acceleration/deceleration.  
- Validation across:
  - 129,877 baseline participants  
  - 10,618 repeat-assessment participants  

**Relevance to Project:**  
- Provides large-scale benchmarking of metabolomic aging models.  
- Supports use of nonlinear models (SVM, rule-based ensembles) over linear regression.  
- Establishes expected performance bounds for metabolomic clocks in large cohorts.

---

## Source 3: Multimodal Clocks of Human Aging

**Link:** https://pubmed.ncbi.nlm.nih.gov/42105758/  

**Objective:**  
Develop a multimodal aging framework integrating clinical, physiological, and molecular data to capture heterogeneous aging trajectories.

**Methods:**  
- Built mCAS cohort (Chinese aging standardized cohort) with 2,019 individuals (ages 18–91).  
- Integrated:
  - Clinical measurements  
  - Physiological assessments  
  - Plasma proteomics  
- Developed three-tier aging framework:
  1. Core Capacity Clock (CC-clock)  
  2. Multimodal Clock (MM-clock)  
  3. Organ-specific aging clocks  

**Outcomes:**  
- Plasma protein signatures strongly reflect systemic physiological decline.  
- Cross-modal validation linked molecular markers with functional aging phenotypes.  
- Identified coagulation factor accumulation as a potential driver of multi-organ aging and inflammation.  
- Demonstrated feasibility of bridging molecular biomarkers with clinical aging measures.

**Relevance to Project:**  
- Most directly relevant multimodal integration framework.  
- Provides architectural blueprint for combining organism-level, multimodal, and organ-specific clocks.  
- Highlights importance of cross-layer validation between molecular and clinical domains.  
- Suggests coagulation pathways as potential mechanistic targets.

---
