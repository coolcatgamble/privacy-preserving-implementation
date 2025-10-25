## Large-Scale K-Anonymity & L-Diversity Analysis

**Assignment 2 - Privacy Preserving Project**

This project implements and validates k-anonymity and l-diversity privacy-preserving mechanisms on a realistic 5,000-record medical dataset modeled after the Massachusetts GIC case.

### Project Structure

```
Privacy_Assignment/
├── large_scale_privacy_analysis.py      # Core k-anonymity & l-diversity algorithms
├── large_scale_privacy_analysis_part2.py # Frontier analysis & statistical validation
├── visualizations.py                     # Publication-quality figures
├── main_analysis.py                      # Main execution pipeline
├── requirements.txt                      # Python dependencies
└── outputs/                              # Generated results
    ├── data/
    │   ├── medical_dataset_5000.csv
    │   ├── dataset_statistics.json
    │   └── distributions/
    ├── results/
    │   ├── attack_results.json
    │   ├── frontier_results.csv
    │   └── statistical_validation.json
    ├── figures/
    │   ├── figure8_attack_success_analysis.png
    │   ├── figure9_privacy_utility_frontier.png
    │   ├── figure10_attack_success_heatmap.png
    │   └── figure11_query_accuracy.png
    ├── tables/
    │   ├── table10_validation_summary.csv
    │   └── table11_attack_simulation.csv
    └── analysis_summary_report.md
```

### Installation

```bash
# Install required packages
pip install -r requirements.txt
```

### Usage

Run the complete analysis pipeline:

```bash
python main_analysis.py
```

Expected runtime: 15-25 minutes

### Output Files

#### Data
- `medical_dataset_5000.csv`: 5,000-record synthetic medical dataset
- `dataset_statistics.json`: Dataset characteristics and distributions
- Distribution plots: Age, ZIP code, and disease distributions

#### Results
- `attack_results.json`: Linkage attack simulation results (8+ configurations)
- `frontier_results.csv`: Privacy-utility metrics for 24 (k,l) configurations
- `statistical_validation.json`: Confidence intervals from 30 trials × 4 configurations

#### Visualizations
- **Figure 8**: Attack Success Analysis (2×2 grid)
- **Figure 9**: Privacy-Utility Frontier (scatter plot with Pareto frontier)
- **Figure 10**: Attack Success Heatmap ((k,l) configurations)
- **Figure 11**: Query Accuracy Analysis (5 query types)

#### Tables
- **Table 10**: Large-Scale Validation Summary
- **Table 11**: Attack Simulation Results

#### Report
- `analysis_summary_report.md`: Comprehensive analysis summary with:
  - Executive summary
  - Dataset characteristics
  - Attack simulation results
  - Privacy-utility tradeoffs
  - Query accuracy analysis
  - Statistical validation
  - Recommendations

### Key Findings

**Recommended Configuration: (k=5, l=3)**
- Re-identification rate: <20%
- Information loss: ~40-45%
- Query accuracy: <5% error on aggregates
- Pareto-optimal balance of privacy and utility

### Features

✅ Realistic 5,000-record Massachusetts medical dataset
✅ Complete k-anonymity and l-diversity implementations
✅ Linkage attack simulation (GIC-style)
✅ Privacy-utility frontier analysis (24 configurations)
✅ Statistical validation with confidence intervals
✅ Publication-quality visualizations (300 DPI)
✅ Comprehensive results tables
✅ Detailed analysis summary report

### Technical Details

**Privacy Mechanisms:**
- K-anonymity via hierarchical generalization
- L-diversity for protecting sensitive attributes
- Suppression for small equivalence classes

**Attack Model:**
- Auxiliary information: Voter registration (ZipCode, Age, Gender)
- Target: Anonymized medical records
- Success criterion: Unique re-identification

**Evaluation Metrics:**
- Privacy: Re-identification rate, attribute disclosure, diversity entropy
- Utility: Information loss, query accuracy, equivalence class metrics

### For Assignment Use

Include in your report (Section 3.3 - Large-Scale Validation):

1. Copy statistics from `dataset_statistics.json`
2. Reference Figures 8-11
3. Insert Tables 10 and 11
4. Quote key findings from `analysis_summary_report.md`

### License

Academic use only - Assignment 2, Privacy & Security Course
