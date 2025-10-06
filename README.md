# Privacy-Preserving Data Anonymization

Python implementations of **k-anonymity** and **ℓ-diversity** privacy-preserving techniques, demonstrating defense against linkage and homogeneity attacks using the Massachusetts GIC case study.

## Overview

This project implements two privacy models:

### K-Anonymity
Protects against **linkage attacks** by ensuring each record is indistinguishable from at least k-1 other records based on quasi-identifying attributes.

### ℓ-Diversity
Extends k-anonymity to prevent **homogeneity attacks** by ensuring each equivalence class contains at least ℓ distinct values for sensitive attributes.

## Features

### K-Anonymity
- ✅ K-anonymity algorithm with generalization and suppression
- ✅ Privacy and utility metrics calculation
- ✅ Excel export with embedded charts
- ✅ Defense against linkage attacks

### ℓ-Diversity
- ✅ Distinct and entropy ℓ-diversity implementations
- ✅ (k,ℓ)-anonymity combined protection
- ✅ Homogeneity attack demonstration
- ✅ Attribute disclosure probability analysis
- ✅ Excel export with comparative analysis

## Files

### K-Anonymity
- `k_anonymity_implementation.py` - Core k-anonymity algorithm and analysis
- `k_anonymity_visualisations.py` - Visualization suite for privacy-utility tradeoffs
- `export_to_excel.py` - Excel export with charts (recommended)

### ℓ-Diversity
- `ldiversity_implementation.py` - (k,ℓ)-anonymity algorithm with homogeneity attack demo
- `ldiversity_visualization.py` - Visualization suite for diversity metrics
- `export_ldiversity_to_excel.py` - Excel export with comparative analysis (recommended)

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/k-anonymity-implementation.git
cd k-anonymity-implementation

# Install dependencies
pip install pandas numpy matplotlib seaborn openpyxl
```

## Usage

### Excel Reports (Recommended)

#### K-Anonymity Report
```bash
python3 export_to_excel.py
```

Generates `k_anonymity_report.xlsx` with:
- 5 sheets: Original data, K=3/K=5 anonymized, metrics comparison, equivalence classes
- 4 embedded charts: Privacy protection, re-identification probability, information loss, precision loss

#### ℓ-Diversity Report
```bash
python3 export_ldiversity_to_excel.py
```

Generates `ldiversity_report.xlsx` with:
- 6 sheets: Original data, k-anon only (vulnerable), (k=3,ℓ=2)/(k=3,ℓ=3) anonymized, metrics, diversity analysis
- Charts: Attribute disclosure probability, privacy protection levels, diversity metrics
- Demonstrates homogeneity attack vulnerability and defense

### Python Scripts (Console Output & PNG Charts)

```bash
# K-Anonymity
python3 k_anonymity_implementation.py       # Console output
python3 k_anonymity_visualisations.py       # Generate PNG charts

# ℓ-Diversity
python3 ldiversity_implementation.py        # Console output with attack demo
python3 ldiversity_visualization.py         # Generate PNG charts
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

## Attack Models & Defenses

### Linkage Attack
**Attack:** Link anonymized data with external data using quasi-identifiers
**Defense:** K-anonymity (each record indistinguishable from k-1 others)
**Result:** Re-identification probability ≤ 1/k

### Homogeneity Attack
**Attack:** Infer sensitive value when all records in equivalence class have same value
**Defense:** ℓ-diversity (each class has ≥ ℓ distinct sensitive values)
**Result:** Attribute disclosure probability ≤ 1/ℓ

### Combined (k,ℓ)-Anonymity
Provides comprehensive protection against both identity disclosure and attribute disclosure.

## Limitations

**K-Anonymity:**
- Vulnerable to homogeneity attacks
- Vulnerable to background knowledge attacks
- Does not protect sensitive attribute diversity

**ℓ-Diversity:**
- Requires more aggressive generalization
- Higher information loss than k-anonymity alone
- Still vulnerable to skewness and similarity attacks

**Stronger models:** t-closeness, differential privacy

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
- Machanavajjhala, A., et al. (2007). "ℓ-diversity: Privacy beyond k-anonymity"
- Massachusetts GIC case study (Governor Weld re-identification)
- HIPAA Privacy Rule guidelines

## Author

Privacy Assignment Project - 2025
