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
- **Feature Group Metabolomic:** [Description of a group of related features]



## Exploratory Data Analysis

The exploratory data analysis is conducted in the [exploratory_data_analysis.ipynb](exploratory_data_analysis.ipynb) notebook, which includes:

- Data loading and initial inspection
- Statistical summaries and distributions
- Missing value analysis
- Feature correlation analysis
- Data visualization and insights
- Data quality assessment
