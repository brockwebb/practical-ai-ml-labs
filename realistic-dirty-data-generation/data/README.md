# Data Sources for Realistic Dirty Data Generation

This directory contains the source datasets needed to generate realistic contaminated survey data. Download the files below and place them in this `/data` directory.

## Required Datasets

### 1. SSA Baby Names by State
**Purpose**: Generate realistic personal names for contamination  
**Source**: U.S. Social Security Administration  
**URL**: https://www.ssa.gov/oact/babynames/limits.html  
**File**: Download "State-specific Data" (all states)  
**Size**: ~23 MB compressed  
**Format**: CSV files with columns: State, Sex, Year, Name, Count  
**Threshold**: Names appearing 5+ times per state/year  

**Instructions**:
1. Download the zip file from SSA
2. Extract all state files to this directory
3. Files will be named like `AK.TXT`, `AL.TXT`, etc.

### 2. National Address Database (NAD)
**Purpose**: Generate realistic addresses for contamination  
**Source**: U.S. Department of Transportation  
**URL**: https://www.transportation.gov/gis/national-address-database  
**File**: "National Address Database (NAD)" download  
**Size**: ~6.4 GB compressed, 84+ million records  
**Format**: Various (CSV, Shapefile, Geodatabase)  
**Coverage**: Nationwide address points  

**Instructions**:
1. Visit the DOT NAD download page
2. Download the full dataset (large file - plan accordingly)
3. Extract to this directory
4. The lab will sample from this dataset as needed

**Alternative**: For smaller experiments, you can use the address generation functions in the notebook that create realistic synthetic addresses without downloading the full NAD.

### 3. Business Names (Optional)
**Purpose**: Base business names for the survey dataset  
**Source**: Various options available  
**Options**:
- SEC EDGAR company filings
- State business registration databases  
- Synthetic generation (included in lab notebook)

**Note**: The lab includes business name generation functions, so external business name data is optional.

## File Organization

After downloading, your `/data` directory should look like:
```
data/
├── README.md           # This file
├── AK.TXT             # SSA data for Alaska
├── AL.TXT             # SSA data for Alabama
├── ...                # All other state files
├── nad_addresses.csv  # NAD address data (or whatever format you download)
└── .gitignore         # Prevents committing large data files
```

## Data Processing Notes

- **SSA Names**: The lab aggregates state-level data to national frequencies
- **NAD Addresses**: The lab samples from the full dataset for memory efficiency  
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