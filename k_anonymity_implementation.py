import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Medical Dataset (Massachusetts GIC Case)
data = {
    'Name': ['Alice Johnson', 'Betty Smith', 'Carol Williams', 'David Brown', 
             'Edward Davis', 'Frank Miller', 'George Wilson', 'Helen Moore',
             'Ian Taylor', 'Jane Anderson', 'Kevin Thomas', 'Laura Jackson',
             'Michael White', 'Nancy Harris', 'Oliver Martin'],
    'ZipCode': ['02138', '02139', '02141', '02142', '02138', '02139', '02141', 
                '02142', '02138', '02139', '02141', '02142', '02138', '02139', '02141'],
    'Age': [29, 31, 28, 45, 47, 43, 52, 36, 41, 33, 55, 38, 49, 42, 58],
    'Gender': ['Female', 'Female', 'Female', 'Male', 'Male', 'Male', 'Male', 
               'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Disease': ['Ovarian Cancer', 'Breast Cancer', 'Ovarian Cancer', 'Heart Disease',
                'Heart Disease', 'Diabetes', 'Heart Disease', 'Diabetes', 
                'Prostate Cancer', 'Breast Cancer', 'Heart Disease', 'Diabetes',
                'Prostate Cancer', 'Breast Cancer', 'Diabetes']
}

df_original = pd.DataFrame(data)

# ============================================================================
# GENERALIZATION FUNCTIONS
# ============================================================================

def generalize_age(age: int, level: int) -> str:
    """
    Generalize age based on hierarchy level.
    Level 0: Exact age (e.g., 29)
    Level 1: 5-year ranges (e.g., 25-29)
    Level 2: 10-year ranges (e.g., 20-29)
    Level 3: 20-year ranges (e.g., 20-39)
    """
    if level == 0:
        return str(age)
    elif level == 1:
        lower = (age // 5) * 5
        return f"{lower}-{lower+4}"
    elif level == 2:
        lower = (age // 10) * 10
        return f"{lower}-{lower+9}"
    elif level == 3:
        if age < 40:
            return "20-39"
        elif age < 60:
            return "40-59"
        else:
            return "60+"
    return str(age)

def generalize_zipcode(zipcode: str, level: int) -> str:
    """
    Generalize ZIP code based on hierarchy level.
    Level 0: Full 5-digit (e.g., 02138)
    Level 1: 4-digit prefix (e.g., 0213*)
    Level 2: 3-digit prefix (e.g., 021**)
    Level 3: 2-digit prefix (e.g., 02***)
    """
    if level == 0:
        return zipcode
    elif level == 1:
        return zipcode[:4] + "*"
    elif level == 2:
        return zipcode[:3] + "**"
    elif level == 3:
        return zipcode[:2] + "***"
    return zipcode

# ============================================================================
# K-ANONYMITY CORE ALGORITHM
# ============================================================================

def calculate_equivalence_classes(df: pd.DataFrame, qi_columns: List[str]) -> Dict:
    """
    Calculate equivalence classes based on quasi-identifiers.
    Returns dictionary mapping QI combination to list of indices.
    """
    equivalence_classes = {}
    
    for idx, row in df.iterrows():
        qi_tuple = tuple(row[qi_columns].values)
        if qi_tuple not in equivalence_classes:
            equivalence_classes[qi_tuple] = []
        equivalence_classes[qi_tuple].append(idx)
    
    return equivalence_classes

def check_k_anonymity(df: pd.DataFrame, qi_columns: List[str], k: int) -> Tuple[bool, int]:
    """
    Check if dataset satisfies k-anonymity.
    Returns (is_k_anonymous, min_group_size).
    """
    eq_classes = calculate_equivalence_classes(df, qi_columns)
    min_size = min(len(indices) for indices in eq_classes.values())
    
    return min_size >= k, min_size

def k_anonymize(df: pd.DataFrame, k: int, qi_columns: List[str]) -> pd.DataFrame:
    """
    Apply k-anonymity through generalization and suppression.
    Uses a greedy approach that generalizes quasi-identifiers and
    suppresses records in small equivalence classes if needed.
    """
    df_anon = df.copy()

    # Generalization levels for each QI attribute
    age_level = 0
    zip_level = 0
    gender_generalized = False

    # Iteratively generalize until k-anonymity is achieved
    max_iterations = 20
    iteration = 0

    while iteration < max_iterations:
        # Apply current generalization levels
        df_anon['Age_Gen'] = df_anon['Age'].apply(lambda x: generalize_age(x, age_level))
        df_anon['ZipCode_Gen'] = df_anon['ZipCode'].apply(lambda x: generalize_zipcode(x, zip_level))

        # Generalize gender if needed
        if gender_generalized:
            df_anon['Gender_Gen'] = 'Person'
        else:
            df_anon['Gender_Gen'] = df_anon['Gender']

        # Check k-anonymity with generalized attributes
        qi_gen = ['Age_Gen', 'ZipCode_Gen', 'Gender_Gen']
        eq_classes = calculate_equivalence_classes(df_anon, qi_gen)

        # Find equivalence classes smaller than k
        small_classes = {qi: indices for qi, indices in eq_classes.items()
                        if len(indices) < k}

        if not small_classes:
            # k-anonymity achieved!
            df_final = df_anon.copy()
            df_final['Age'] = df_final['Age_Gen']
            df_final['ZipCode'] = df_final['ZipCode_Gen']
            df_final['Gender'] = df_final['Gender_Gen']
            df_final = df_final.drop(columns=['Age_Gen', 'ZipCode_Gen', 'Gender_Gen', 'Name'])
            return df_final

        # Strategy: Try more generalization first, then suppression as last resort
        if age_level < 3:
            age_level += 1
        elif zip_level < 3:
            zip_level += 1
        elif not gender_generalized:
            gender_generalized = True
        else:
            # Last resort: suppress records in small equivalence classes
            indices_to_suppress = []
            for indices in small_classes.values():
                indices_to_suppress.extend(indices)

            df_anon = df_anon.drop(index=indices_to_suppress).reset_index(drop=True)

            # Recheck with suppressed records
            df_anon['Age_Gen'] = df_anon['Age'].apply(lambda x: generalize_age(x, age_level))
            df_anon['ZipCode_Gen'] = df_anon['ZipCode'].apply(lambda x: generalize_zipcode(x, zip_level))
            if gender_generalized:
                df_anon['Gender_Gen'] = 'Person'
            else:
                df_anon['Gender_Gen'] = df_anon['Gender']

            df_final = df_anon.copy()
            df_final['Age'] = df_final['Age_Gen']
            df_final['ZipCode'] = df_final['ZipCode_Gen']
            df_final['Gender'] = df_final['Gender_Gen']
            df_final = df_final.drop(columns=['Age_Gen', 'ZipCode_Gen', 'Gender_Gen', 'Name'])
            return df_final

        iteration += 1

    # If max iterations reached (shouldn't happen with suppression), return what we have
    df_final = df_anon.copy()
    df_final['Age'] = df_final['Age_Gen']
    df_final['ZipCode'] = df_final['ZipCode_Gen']
    df_final['Gender'] = df_final['Gender_Gen']
    df_final = df_final.drop(columns=['Age_Gen', 'ZipCode_Gen', 'Gender_Gen', 'Name'])
    return df_final

# ============================================================================
# PRIVACY AND UTILITY METRICS
# ============================================================================

def calculate_discernibility_metric(df: pd.DataFrame, qi_columns: List[str]) -> float:
    """
    Calculate discernibility metric (lower is better for utility).
    Each record gets penalty equal to its equivalence class size.
    """
    eq_classes = calculate_equivalence_classes(df, qi_columns)
    total_cost = sum(len(indices) ** 2 for indices in eq_classes.values())
    return total_cost

def calculate_precision_loss(original_df: pd.DataFrame, anon_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate precision loss for each generalized attribute.
    """
    precision_loss = {}
    
    # Age precision loss
    if 'Age' in anon_df.columns:
        # Count how many unique values remain
        original_unique = original_df['Age'].nunique()
        anon_unique = anon_df['Age'].nunique()
        precision_loss['Age'] = 1 - (anon_unique / original_unique)
    
    # ZipCode precision loss
    if 'ZipCode' in anon_df.columns:
        original_unique = original_df['ZipCode'].nunique()
        anon_unique = anon_df['ZipCode'].nunique()
        precision_loss['ZipCode'] = 1 - (anon_unique / original_unique)
    
    return precision_loss

def calculate_reidentification_probability(k: int) -> float:
    """Calculate maximum re-identification probability for k-anonymous data."""
    return 1.0 / k

# ============================================================================
# DEMONSTRATION AND RESULTS
# ============================================================================

def run_k_anonymity_analysis(print_results=True, return_results=False):
    """
    Run k-anonymity analysis and optionally return results for visualization.

    Args:
        print_results: Whether to print results to console
        return_results: Whether to return results dictionary

    Returns:
        Dictionary containing analysis results if return_results=True
    """
    results = {
        'original_df': df_original.copy(),
        'k_values': [],
        'anonymized_dfs': {},
        'metrics': {},
        'equivalence_classes': {}
    }

    if print_results:
        print("=" * 80)
        print("K-ANONYMITY IMPLEMENTATION - MASSACHUSETTS GIC CASE STUDY")
        print("=" * 80)
        print()

        # Original dataset analysis
        print("ORIGINAL DATASET (Before Anonymization)")
        print("-" * 80)
        print(df_original.to_string(index=False))
        print()

    qi_original = ['ZipCode', 'Age', 'Gender']
    df_check = df_original[qi_original + ['Disease']].copy()
    is_anon_orig, min_size_orig = check_k_anonymity(df_check, qi_original, 2)

    if print_results:
        print(f"K-Anonymity Check (k=2): {'PASS' if is_anon_orig else 'FAIL'}")
        print(f"Minimum equivalence class size: {min_size_orig}")
        print()

    # Apply k-anonymity for different k values
    k_values = [3, 5]

    for k in k_values:
        results['k_values'].append(k)

        if print_results:
            print("=" * 80)
            print(f"K-ANONYMIZATION WITH k={k}")
            print("=" * 80)

        df_anonymized = k_anonymize(df_original, k, qi_original)
        results['anonymized_dfs'][k] = df_anonymized.copy()

        if print_results:
            print()
            print(f"ANONYMIZED DATASET (k={k})")
            print("-" * 80)
            print(df_anonymized.to_string(index=False))
            print()

        # Verify k-anonymity
        qi_anon = ['ZipCode', 'Age', 'Gender']
        is_anon, min_size = check_k_anonymity(df_anonymized, qi_anon, k)

        if print_results:
            print(f"K-Anonymity Verification: {'PASS ✓' if is_anon else 'FAIL ✗'}")
            print(f"Minimum equivalence class size: {min_size}")
            print()

        # Calculate metrics
        df_orig_check = df_original[qi_original + ['Disease']].copy()
        discern_orig = calculate_discernibility_metric(df_orig_check, qi_original)
        discern_anon = calculate_discernibility_metric(df_anonymized, qi_anon)
        precision = calculate_precision_loss(df_original, df_anonymized)
        reident_prob = calculate_reidentification_probability(k)

        # Store metrics
        results['metrics'][k] = {
            'is_anon': is_anon,
            'min_size': min_size,
            'reident_prob': reident_prob,
            'privacy_gain': 1 - reident_prob,
            'discern_orig': discern_orig,
            'discern_anon': discern_anon,
            'precision_loss': precision
        }

        if print_results:
            print("PRIVACY METRICS:")
            print(f"  • Re-identification Probability: {reident_prob:.2%} (1/{k})")
            print(f"  • Privacy Gain: {(1 - reident_prob):.2%}")
            print()

            print("UTILITY METRICS:")
            print(f"  • Discernibility Cost (Original): {discern_orig:.0f}")
            print(f"  • Discernibility Cost (Anonymized): {discern_anon:.0f}")
            print(f"  • Information Loss Increase: {((discern_anon - discern_orig) / discern_orig * 100):.1f}%")
            print(f"  • Age Precision Loss: {precision.get('Age', 0):.2%}")
            print(f"  • ZipCode Precision Loss: {precision.get('ZipCode', 0):.2%}")
            print()

        # Show equivalence classes
        eq_classes = calculate_equivalence_classes(df_anonymized, qi_anon)
        results['equivalence_classes'][k] = eq_classes

        if print_results:
            print(f"EQUIVALENCE CLASSES ({len(eq_classes)} groups):")
            for i, (qi_combo, indices) in enumerate(eq_classes.items(), 1):
                print(f"  Class {i}: {qi_combo} → {len(indices)} records")
            print()

    if print_results:
        print("=" * 80)
        print("SUMMARY: K-ANONYMITY DEFENSE EFFECTIVENESS")
        print("=" * 80)
        print()
        print("The k-anonymity algorithm successfully prevents linkage attacks by:")
        print("1. Generalizing quasi-identifiers (Age, ZipCode, Gender) to create indistinguishable groups")
        print("2. Ensuring each record is identical to at least k-1 other records on quasi-identifiers")
        print("3. Reducing maximum re-identification probability from potentially 100% to 1/k")
        print("4. Using suppression as a fallback when generalization alone is insufficient")
        print()
        print("Trade-off: Higher k values provide stronger privacy guarantees but reduce data utility")
        print("through more aggressive generalization and potential record suppression.")
        print()
        print("Note: This implementation generalizes Gender to 'Person' and uses 20-year age ranges")
        print("to achieve the required k-anonymity guarantee for this dataset.")
        print("=" * 80)

    if return_results:
        return results

# Run analysis when script is executed directly
if __name__ == "__main__":
    run_k_anonymity_analysis(print_results=True, return_results=False)