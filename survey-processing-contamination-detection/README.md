# Survey Processing Contamination Detection

A practical lab for detecting personal names accidentally entered in business name fields using machine learning and rule-based approaches.

## Problem Statement

Survey respondents sometimes mistakenly enter personal names (e.g., "John Smith") in fields intended for business names (e.g., "Smith & Associates LLC"). This lab demonstrates automated detection of such contamination using various approaches including spaCy NER, rule-based patterns, and custom ML models.

## Learning Objectives

- Build realistic contamination datasets using official government data sources
- Implement and compare multiple detection approaches
- Evaluate model performance at low contamination rates (0.1-1%)
- Create reproducible pipelines for data quality assessment

## Setup Instructions

### 1. Create Conda Environment

```bash
# Create new environment with Python 3.11
conda create -n spacy-detect python=3.11

# Activate environment
conda activate spacy-detect
```

### 2. Install Dependencies

```bash
# Install core packages
pip install pandas numpy matplotlib seaborn scikit-learn

# Install spaCy and download model
pip install spacy
python -m spacy download en_core_web_lg

# Install optional ML packages
pip install transformers torch  # For BERT experiments (optional)
```

### 3. Data Setup

Download the SSA baby names dataset:
1. Visit: https://www.ssa.gov/oact/babynames/limits.html
2. Download "National data" or "State-specific data" 
3. Extract to `data/namebystate/` directory

```bash
# Create data directory
mkdir -p data/namebystate
# Extract your downloaded files here
```

### 4. Verify Installation

```python
import spacy
nlp = spacy.load("en_core_web_lg")
print("spaCy model loaded successfully!")

# Test on sample text
doc = nlp("John Smith Inc")
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")
```

## Usage

1. **Generate contaminated dataset**: Create realistic business name contamination
2. **Run detection pipeline**: Apply spaCy NER and other detection methods
3. **Evaluate performance**: Calculate precision, recall, and F1 scores
4. **Compare approaches**: Analyze trade-offs between different detection methods

## Lab Structure

- `01_data_preparation.py` - Load and prepare name datasets
- `02_contamination_pipeline.py` - Generate realistic contaminated data
- `03_detection_methods.py` - Implement various detection approaches
- `04_evaluation.py` - Performance analysis and visualization

## Expected Outcomes

Learn to build production-ready contamination detection systems applicable to data quality problems across government and business survey processing workflows.