# K-Anonymity Implementation

A Python implementation of k-anonymity privacy-preserving technique, demonstrating defense against linkage attacks using the Massachusetts GIC case study.

## Overview

This project implements **k-anonymity**, a privacy model that protects against re-identification by ensuring each record is indistinguishable from at least k-1 other records based on quasi-identifying attributes.

### Features

- ✅ K-anonymity algorithm with generalization and suppression
- ✅ Privacy and utility metrics calculation
- ✅ Excel export with embedded charts and formatted sheets
- ✅ Comprehensive data transformation analysis
- ✅ Real-world medical dataset example

## Files

- `k_anonymity_implementation.py` - Core k-anonymity algorithm and analysis
- `k_anonymity_visualisations.py` - Visualization suite for privacy-utility tradeoffs
- `export_to_excel.py` - Excel export script with embedded charts (recommended)

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/k-anonymity-implementation.git
cd k-anonymity-implementation

# Install dependencies
pip install pandas numpy matplotlib seaborn openpyxl
```

## Usage

### Option 1: Excel Report (Recommended)

```bash
python3 export_to_excel.py
```

Generates `k_anonymity_report.xlsx` with:
- 5 sheets: Original data, K=3 anonymized, K=5 anonymized, metrics comparison, equivalence classes
- 4 embedded charts: Privacy protection, re-identification probability, information loss, precision loss
- Professional formatting and styling

Open the generated Excel file in Microsoft Excel, Google Sheets, or any spreadsheet application.

### Option 2: Python Scripts

```bash
# Run k-anonymity implementation (console output)
python3 k_anonymity_implementation.py

# Generate PNG visualizations
python3 k_anonymity_visualisations.py
```

## How It Works

### Generalization Hierarchy

**Age:**
- Level 0: Exact age (29)
- Level 1: 5-year ranges (25-29)
- Level 2: 10-year ranges (20-29)
- Level 3: 20-year ranges (20-39)

**ZIP Code:**
- Level 0: Full 5-digit (02138)
- Level 1: 4-digit prefix (0213*)
- Level 2: 3-digit prefix (021**)
- Level 3: 2-digit prefix (02***)

**Gender:**
- Original: Male/Female
- Generalized: Person

### Algorithm

1. Start with original dataset
2. Apply generalization to quasi-identifiers (Age, ZipCode, Gender)
3. Calculate equivalence classes
4. Check if minimum class size ≥ k
5. If not satisfied, increase generalization level
6. As last resort, suppress records in small classes

## Results

### Privacy Metrics

| k Value | Re-identification Probability | Privacy Gain |
|---------|------------------------------|--------------|
| k=3     | 33.3%                        | 66.7%        |
| k=5     | 20.0%                        | 80.0%        |

### Utility Cost

- Age precision loss: ~87%
- ZipCode precision loss: ~75%
- Information loss increases with higher k values

## Massachusetts GIC Case Study

This implementation is based on the famous privacy breach where Latanya Sweeney re-identified Governor William Weld's medical records by linking "anonymized" hospital data with public voter registration records using ZIP code, birth date, and gender.

**Key Learning:** Simple removal of direct identifiers (names, SSN) is insufficient for privacy protection.

## Privacy-Utility Tradeoff

Higher k values provide:
- ✅ **Better privacy** (lower re-identification risk)
- ❌ **Lower utility** (more information loss)

## Limitations of K-Anonymity

1. **Homogeneity Attack**: All records in equivalence class have same sensitive value
2. **Background Knowledge Attack**: Attacker knows victim is in dataset
3. **Composition Attack**: Multiple releases can be combined

**Stronger models:** ℓ-diversity, t-closeness, differential privacy

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- openpyxl (for Excel export)

## License

MIT License

## References

- Sweeney, L. (2002). "k-anonymity: A model for protecting privacy"
- Massachusetts GIC case study
- HIPAA Privacy Rule guidelines

## Author

Privacy Assignment Project - 2025
