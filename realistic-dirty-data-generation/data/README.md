# Data Sources for Realistic Dirty Data Generation

This directory contains the source datasets needed to generate realistic contaminated survey data. Download the files below and place them in this `/data` directory.

## Required Datasets

### 1. SSA Baby Names by State
**Purpose**: Generate realistic first names for contamination  
**Source**: U.S. Social Security Administration  
**URL**: https://www.ssa.gov/oact/babynames/limits.html  
**File**: Download "State-specific Data" (all states)  
**Size**: ~23 MB compressed  
**Format**: CSV files with columns: State, Sex, Year, Name, Count  
**Threshold**: Names appearing 5+ times per state/year  
**Extract to**: `/namesbystate/` directory

**Instructions**:
1. Download the zip file from SSA
2. Extract all state files to `data/namesbystate/` directory
3. Files will be named like `AK.TXT`, `AL.TXT`, etc.

### 2. Census Surnames
**Purpose**: Generate realistic last names for contamination  
**Source**: U.S. Census Bureau  
**URL**: https://www.census.gov/topics/population/genealogy/data/2010_surnames.html  
**File**: Download "2010 Census Names Files"  
**Size**: ~5 MB  
**Format**: CSV with columns: Name, Count, Proportion  
**Threshold**: Names appearing 100+ times nationally  
**Extract to**: `/names/Names_2010Census.csv`

**Instructions**:
1. Download Names_2010Census.csv from Census Bureau
2. Extract to `data/names/Names_2010Census.csv`
3. Contains surnames with frequency data

### 3. National Address Database (NAD) - Pre-sampled
**Purpose**: Generate realistic addresses for contamination  
**Source**: Pre-processed sample from DOT National Address Database  
**Location**: `../lab-datasets/nad_addresses_sample_1M.csv`  
**Size**: ~100 MB (1M uniform sample from 84M records)  
**Format**: CSV with address information  

**Note**: Due to the large size of the full NAD (32GB uncompressed), we provide a pre-sampled 1M record uniform draw. This sample maintains geographic diversity while being manageable for lab use.

**For the brave**: Full NAD dataset available at https://www.transportation.gov/gis/national-address-database (6.4GB compressed, 84M records)

### 4. Business Names
**Purpose**: Base business names for the survey dataset  
**Source**: People Data Labs - Free 7 Million Company Dataset  
**URL**: https://www.kaggle.com/datasets/peopledatalabssf/free-7-million-company-dataset  
**File**: Download from Kaggle  
**Size**: ~200 MB compressed  
**Format**: CSV with company information including names  
**Coverage**: 7+ million company records  
**Extract to**: `/archive/companies_sorted.csv`

**Instructions**: 
1. Create free Kaggle account if needed
2. Download the dataset from Kaggle
3. Extract to `data/archive/companies_sorted.csv`
4. File contains company names and other business information

## File Organization

After downloading, your `/data` directory should look like:
```
data/
├── README.md                    # This file
├── names/
│   └── Names_2010Census.csv     # Census surnames data
├── namesbystate/
│   ├── AK.TXT                   # SSA data for Alaska
│   ├── AL.TXT                   # SSA data for Alabama
│   └── ...                      # All other state files
├── archive/
│   └── companies_sorted.csv     # Kaggle business names dataset
└── .gitignore                   # Prevents committing large data files
```

**Note**: NAD addresses are pre-sampled and located in `../lab-datasets/nad_addresses_sample_1M.csv`

## Data Processing Notes

- **SSA Names**: The lab aggregates state-level data to national frequencies for first names
- **Census Surnames**: Provides national frequency data for last names (100+ occurrences)
- **NAD Addresses**: The lab samples from the full dataset for memory efficiency  
- **Business Names**: Used as the base "clean" dataset before contamination
- **Geographic Diversity**: Address sampling ensures representation across regions
- **Frequency Weighting**: More common names/addresses appear more often in contamination

## Troubleshooting

**Large File Sizes**: The NAD dataset is very large. Consider downloading overnight or using a stable internet connection.

**File Formats**: If you download different formats (shapefile vs CSV), you may need to adjust the loading functions in the notebook.

**Missing Files**: The lab will generate synthetic alternatives if source files are missing, but real data provides better contamination realism.

## Data Usage Compliance

- SSA data is public domain
- NAD data is public domain (U.S. Government work)
- Follow any state-specific restrictions on business name data if used
- These datasets are for educational and research purposes as outlined in the lab