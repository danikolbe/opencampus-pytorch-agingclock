# Dataset Characteristics

**[Notebook](exploratory_data_analysis.ipynb)**

## Dataset Information

### Dataset Source
- **Dataset Link:** https://www.ukbiobank.ac.uk/about-our-data/types-of-data/
- **Dataset Owner/Contact:** Data has restricted access.

### Dataset Characteristics
- **Number of Observations:** Different for different types of data – since this is a multimodal study.
- **Number of Features:** 

### Target Variable/Label
- **Label Name:** Age at recruitment
- **Label Type:** Regression
- **Label Description:** Biological age - predict this. Then ultimately, assess whether predicted is below or above age at recruitment (i.e., is this person aging quicker or slower than their actual age?)
- **Label Values:** A range from 40 to 70.
- **Label Distribution:** There is a peak around the 60 to 65. Figure is attached.

### Feature Description
After analysis, I decided to I will be using metabolomic, blood count, blood biochemistry, anthropometry and proteomic data.
Given that proteomic data is only available for a smaller subset, I will try and create 2 clocks, one featuring proteomic data and one without (but the full set of samples).

**Example format:**
- **Feature Group — Metabolomics (NMR):** 253 plasma metabolite measures from the
  Nightingale Health NMR panel, including lipoprotein subfractions, fatty acids,
  amino acids, glycolysis metabolites, ketone bodies, and the inflammatory marker
  GlycA. Available for ~274,000 participants.

- **Feature Group — Blood Count:** 31 haematological measures covering red blood
  cell indices, platelet parameters, and white blood cell differential counts.
  Available for ~478,000 participants with <5% missingness across all fields.

- **Feature Group — Blood Biochemistry:** 30 serum assays covering hepatic function,
  renal function, metabolic markers, and endocrine markers (IGF-1, testosterone,
  HbA1c). Available for ~490,000 participants.

- **Feature Group — Anthropometry:** 56 measures from clinical assessment including
  core anthropometrics (BMI, waist/hip circumference), segmental body composition
  from bioelectrical impedance (fat mass, fat-free mass, trunk/arm/leg), and blood
  pressure and pulse rate. Available for ~501,000 participants.

- **Feature Group — Proteomics (Olink Explore):** 1,983 plasma proteins measured
  via proximity extension assay, expressed as normalised log2 NPX values. Used
  exclusively in Model 2 (proteomics clock). Available for ~53,000 participants.



## Exploratory Data Analysis

The exploratory data analysis is conducted in the [exploratory_data_analysis.ipynb](exploratory_data_analysis.ipynb) notebook, which includes:

- Data loading and initial inspection
- Statistical summaries and distributions
- Missing value analysis
- Feature correlation analysis
- Data visualization and insights
- Data quality assessment

The output can be viewed in [exploratory_data_analysis.html](exploratory_data_analysis.html)
