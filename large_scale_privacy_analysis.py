"""
Large-Scale K-Anonymity and L-Diversity Analysis
Assignment 2 - Privacy Preserving Project

Complete implementation for validating k-anonymity and l-diversity on 5,000-record dataset.
Includes attack simulation, privacy-utility frontier analysis, and statistical validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any
import json
import time
from datetime import datetime
from pathlib import Path
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# ============================================================================
# 1. GENERALIZATION HIERARCHIES
# ============================================================================

def generalize_age(age: int, level: int) -> str:
    """
    Generalize age based on hierarchy level (0-3).

    Levels:
    - 0: exact age (e.g., "25")
    - 1: 5-year ranges (e.g., "25-29")
    - 2: 10-year ranges (e.g., "20-29")
    - 3: 20-year ranges (e.g., "20-39", "40-59", "60+")
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
        if age < 20:
            return "18-19"
        elif age < 40:
            return "20-39"
        elif age < 60:
            return "40-59"
        else:
            return "60+"
    else:
        return "*"


def generalize_zipcode(zipcode: str, level: int) -> str:
    """
    Generalize ZIP code based on hierarchy level (0-3).

    Levels:
    - 0: 5-digit (e.g., "02138")
    - 1: 4-digit (e.g., "0213*")
    - 2: 3-digit (e.g., "021**")
    - 3: 2-digit (e.g., "02***")
    """
    if level == 0:
        return zipcode
    elif level == 1:
        return zipcode[:4] + "*"
    elif level == 2:
        return zipcode[:3] + "**"
    elif level == 3:
        return zipcode[:2] + "***"
    else:
        return "*****"


def matches_generalized_zip(generalized: str, original: str) -> bool:
    """Check if a generalized ZIP code matches an original ZIP code."""
    for i, char in enumerate(generalized):
        if char != '*' and char != original[i]:
            return False
    return True


def matches_generalized_age(generalized: str, original: int) -> bool:
    """Check if a generalized age range contains the original age."""
    if generalized == "*":
        return True

    if generalized == "60+":
        return original >= 60

    if '-' in generalized:
        parts = generalized.split('-')
        lower, upper = int(parts[0]), int(parts[1])
        return lower <= original <= upper
    else:
        try:
            return int(generalized) == original
        except ValueError:
            return False


# ============================================================================
# 2. K-ANONYMITY IMPLEMENTATION
# ============================================================================

def calculate_equivalence_classes(df: pd.DataFrame, qi_columns: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Group records into equivalence classes based on quasi-identifiers.
    Returns dictionary mapping class signature to dataframe of records.
    """
    classes = {}
    for _, group in df.groupby(qi_columns):
        key = tuple(group.iloc[0][qi_columns])
        classes[str(key)] = group
    return classes


def check_k_anonymity(df: pd.DataFrame, k: int, qi_columns: List[str]) -> Tuple[bool, int, int]:
    """
    Check if dataset satisfies k-anonymity.
    Returns (satisfied, num_classes, min_class_size).
    """
    classes = calculate_equivalence_classes(df, qi_columns)
    if len(classes) == 0:
        return False, 0, 0

    min_size = min(len(group) for group in classes.values())
    return min_size >= k, len(classes), min_size


def k_anonymize(df: pd.DataFrame, k: int, qi_columns: List[str] = None) -> pd.DataFrame:
    """
    Apply k-anonymity through iterative generalization and suppression.

    Parameters:
    -----------
    df : pd.DataFrame
        Original dataset
    k : int
        Privacy parameter k
    qi_columns : List[str]
        Quasi-identifier columns (default: ['ZipCode', 'Age', 'Gender'])

    Returns:
    --------
    pd.DataFrame : Anonymized dataset
    """
    if qi_columns is None:
        qi_columns = ['ZipCode', 'Age', 'Gender']

    df_anon = df.copy()

    # Generalization levels for each attribute
    age_level = 0
    zip_level = 0

    # Iteratively generalize until k-anonymity is achieved
    max_iterations = 10
    for iteration in range(max_iterations):
        satisfied, n_classes, min_size = check_k_anonymity(df_anon, k, qi_columns)

        if satisfied:
            break

        # Generalize the attribute that creates most diversity
        # Alternate between age and zipcode
        if iteration % 2 == 0 and age_level < 3:
            age_level += 1
            if 'Age' in qi_columns:
                df_anon['Age'] = df['Age'].apply(lambda x: generalize_age(x, age_level))
        elif zip_level < 3:
            zip_level += 1
            if 'ZipCode' in qi_columns:
                df_anon['ZipCode'] = df['ZipCode'].apply(lambda x: generalize_zipcode(x, zip_level))
        else:
            # If maximum generalization reached, suppress small groups
            classes = calculate_equivalence_classes(df_anon, qi_columns)
            indices_to_drop = []
            for class_key, group in classes.items():
                if len(group) < k:
                    indices_to_drop.extend(group.index.tolist())

            if indices_to_drop:
                df_anon = df_anon.drop(indices_to_drop)
            break

    return df_anon


def k_anonymize_with_metadata(df: pd.DataFrame, k: int, qi_columns: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply k-anonymity and return comprehensive metadata.

    Returns:
    --------
    (df_anonymized, metadata) where metadata contains:
    - execution_time: Time taken in seconds
    - n_equivalence_classes: Number of equivalence classes
    - min_class_size: Minimum class size
    - max_class_size: Maximum class size
    - avg_class_size: Average class size
    - information_loss: Information loss metric (0-1)
    - records_suppressed: Number of records removed
    - k_anonymity_satisfied: Boolean
    """
    if qi_columns is None:
        qi_columns = ['ZipCode', 'Age', 'Gender']

    start_time = time.time()
    original_size = len(df)

    # Apply k-anonymization
    df_anon = k_anonymize(df, k, qi_columns)

    execution_time = time.time() - start_time

    # Calculate equivalence classes
    classes = calculate_equivalence_classes(df_anon, qi_columns)
    class_sizes = [len(group) for group in classes.values()]

    # Calculate information loss
    info_loss = calculate_information_loss(df, df_anon)

    # Check k-anonymity
    satisfied, n_classes, min_size = check_k_anonymity(df_anon, k, qi_columns)

    metadata = {
        'execution_time': execution_time,
        'n_equivalence_classes': len(classes),
        'min_class_size': min(class_sizes) if class_sizes else 0,
        'max_class_size': max(class_sizes) if class_sizes else 0,
        'avg_class_size': np.mean(class_sizes) if class_sizes else 0,
        'information_loss': info_loss,
        'records_suppressed': original_size - len(df_anon),
        'k_anonymity_satisfied': satisfied
    }

    return df_anon, metadata


# ============================================================================
# 3. L-DIVERSITY IMPLEMENTATION
# ============================================================================

def check_l_diversity(df: pd.DataFrame, l: int, qi_columns: List[str], sensitive_col: str) -> bool:
    """Check if dataset satisfies l-diversity."""
    classes = calculate_equivalence_classes(df, qi_columns)

    for class_key, group in classes.items():
        unique_sensitive = group[sensitive_col].nunique()
        if unique_sensitive < l:
            return False

    return True


def kl_anonymize(df: pd.DataFrame, k: int, l: int,
                 qi_columns: List[str] = None,
                 sensitive_col: str = 'Disease') -> pd.DataFrame:
    """
    Apply (k,l)-anonymity (k-anonymity + l-diversity).

    Parameters:
    -----------
    df : pd.DataFrame
        Original dataset
    k : int
        Privacy parameter k
    l : int
        Diversity parameter l
    qi_columns : List[str]
        Quasi-identifier columns
    sensitive_col : str
        Sensitive attribute column

    Returns:
    --------
    pd.DataFrame : Anonymized dataset satisfying both k-anonymity and l-diversity
    """
    if qi_columns is None:
        qi_columns = ['ZipCode', 'Age', 'Gender']

    df_anon = df.copy()

    # First achieve k-anonymity
    df_anon = k_anonymize(df_anon, k, qi_columns)

    # Then enforce l-diversity by further generalization or suppression
    age_level = 1
    zip_level = 1

    max_iterations = 15
    for iteration in range(max_iterations):
        if check_l_diversity(df_anon, l, qi_columns, sensitive_col):
            break

        # Further generalize
        if iteration % 2 == 0 and age_level < 3:
            age_level += 1
            if 'Age' in qi_columns:
                df_anon['Age'] = df['Age'].apply(lambda x: generalize_age(x, age_level))
        elif zip_level < 3:
            zip_level += 1
            if 'ZipCode' in qi_columns:
                df_anon['ZipCode'] = df['ZipCode'].apply(lambda x: generalize_zipcode(x, zip_level))
        else:
            # Suppress equivalence classes that don't satisfy l-diversity
            classes = calculate_equivalence_classes(df_anon, qi_columns)
            indices_to_drop = []
            for class_key, group in classes.items():
                if group[sensitive_col].nunique() < l:
                    indices_to_drop.extend(group.index.tolist())

            if indices_to_drop:
                df_anon = df_anon.drop(indices_to_drop)
            break

    return df_anon


def kl_anonymize_with_metadata(df: pd.DataFrame, k: int, l: int,
                                qi_columns: List[str] = None,
                                sensitive_col: str = 'Disease') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply (k,l)-anonymity with comprehensive metadata."""
    if qi_columns is None:
        qi_columns = ['ZipCode', 'Age', 'Gender']

    start_time = time.time()
    original_size = len(df)

    df_anon = kl_anonymize(df, k, l, qi_columns, sensitive_col)

    execution_time = time.time() - start_time

    # Calculate metrics
    classes = calculate_equivalence_classes(df_anon, qi_columns)
    class_sizes = [len(group) for group in classes.values()]

    # Calculate diversity metrics
    min_diversity = min(group[sensitive_col].nunique() for group in classes.values()) if classes else 0

    info_loss = calculate_information_loss(df, df_anon)

    k_satisfied, n_classes, min_size = check_k_anonymity(df_anon, k, qi_columns)
    l_satisfied = check_l_diversity(df_anon, l, qi_columns, sensitive_col)

    metadata = {
        'execution_time': execution_time,
        'n_equivalence_classes': len(classes),
        'min_class_size': min(class_sizes) if class_sizes else 0,
        'max_class_size': max(class_sizes) if class_sizes else 0,
        'avg_class_size': np.mean(class_sizes) if class_sizes else 0,
        'information_loss': info_loss,
        'records_suppressed': original_size - len(df_anon),
        'k_anonymity_satisfied': k_satisfied,
        'l_diversity_satisfied': l_satisfied,
        'min_diversity': min_diversity
    }

    return df_anon, metadata


# ============================================================================
# 4. UTILITY METRICS
# ============================================================================

def calculate_information_loss(df_original: pd.DataFrame, df_anonymized: pd.DataFrame) -> float:
    """
    Calculate information loss as ratio of distinct values lost.
    Returns value between 0 (no loss) and 1 (complete loss).
    """
    qi_columns = ['ZipCode', 'Age', 'Gender']
    total_loss = 0

    for col in qi_columns:
        if col not in df_original.columns or col not in df_anonymized.columns:
            continue

        original_distinct = df_original[col].nunique()
        anonymized_distinct = df_anonymized[col].nunique()

        if original_distinct > 0:
            col_loss = 1 - (anonymized_distinct / original_distinct)
            total_loss += col_loss

    return total_loss / len(qi_columns)


def calculate_query_accuracy(df_original: pd.DataFrame, df_anonymized: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate relative error for common aggregate queries.

    Returns dictionary of query errors.
    """
    queries = {}

    # Query 1: Count of records in Boston area (021**)
    orig_q1 = len(df_original[df_original['ZipCode'].str.startswith('021')])
    anon_q1 = len(df_anonymized[df_anonymized['ZipCode'].str.contains('021')])
    queries['boston_count'] = abs(orig_q1 - anon_q1) / max(orig_q1, 1)

    # Query 2: Average age (overall)
    orig_q2 = df_original['Age'].mean()
    # For anonymized, need to handle ranges
    anon_ages = []
    for age_val in df_anonymized['Age']:
        if isinstance(age_val, str):
            if age_val == "60+":
                anon_ages.append(70)  # Approximate
            elif '-' in age_val:
                parts = age_val.split('-')
                anon_ages.append((int(parts[0]) + int(parts[1])) / 2)
            elif age_val == '*':
                anon_ages.append(50)  # Midpoint
            elif age_val.isdigit():
                anon_ages.append(int(age_val))
            else:
                anon_ages.append(50)  # Fallback
        else:
            # Already an integer
            anon_ages.append(int(age_val))
    anon_q2 = np.mean(anon_ages) if anon_ages else 0
    queries['avg_age'] = abs(orig_q2 - anon_q2) / max(orig_q2, 1)

    # Query 3: Count of Heart Disease patients
    orig_q3 = len(df_original[df_original['Disease'] == 'Heart Disease'])
    anon_q3 = len(df_anonymized[df_anonymized['Disease'] == 'Heart Disease'])
    queries['heart_disease_count'] = abs(orig_q3 - anon_q3) / max(orig_q3, 1)

    # Query 4: Count of ages 40-50
    orig_q4 = len(df_original[(df_original['Age'] >= 40) & (df_original['Age'] <= 50)])
    # For anonymized, approximate by checking if range overlaps 40-50
    anon_q4_count = 0
    for age_val in df_anonymized['Age']:
        if isinstance(age_val, str) and '-' in age_val:
            if age_val == "60+":
                continue
            parts = age_val.split('-')
            lower, upper = int(parts[0]), int(parts[1])
            if lower <= 50 and upper >= 40:
                anon_q4_count += 1
        elif isinstance(age_val, (int, str)) and str(age_val).isdigit():
            if 40 <= int(age_val) <= 50:
                anon_q4_count += 1
    queries['age_40_50_count'] = abs(orig_q4 - anon_q4_count) / max(orig_q4, 1)

    # Query 5: Median age (approximate)
    orig_q5 = df_original['Age'].median()
    anon_q5 = np.median(anon_ages) if anon_ages else 0
    queries['median_age'] = abs(orig_q5 - anon_q5) / max(orig_q5, 1)

    return queries


# ============================================================================
# 5. DATASET GENERATION
# ============================================================================

def generate_large_medical_dataset(n_records: int = 5000, save_path: str = None) -> pd.DataFrame:
    """
    Generate synthetic medical dataset mimicking Massachusetts GIC data.

    Parameters:
    -----------
    n_records : int
        Number of patient records (default: 5000)
    save_path : str
        Path to save dataset (optional)

    Returns:
    --------
    pd.DataFrame with columns: [PatientID, ZipCode, Age, Gender, Disease]
    """
    np.random.seed(42)

    print("="*80)
    print("GENERATING LARGE-SCALE MEDICAL DATASET")
    print("="*80)

    # Massachusetts ZIP codes (Cambridge + Boston)
    ma_zipcodes = ['02138', '02139', '02140', '02141', '02142', '02143',  # Cambridge
                   '02108', '02109', '02110', '02111', '02113', '02114',  # Boston
                   '02115', '02116', '02118', '02119', '02120', '02121']

    # Realistic distribution (more concentrated in Cambridge, reflecting GIC case)
    zip_probs = [0.10, 0.09, 0.08, 0.08, 0.07, 0.06,
                 0.08, 0.07, 0.06, 0.05, 0.05, 0.04,
                 0.05, 0.04, 0.03, 0.02, 0.02, 0.01]

    zipcodes = np.random.choice(ma_zipcodes, size=n_records, p=zip_probs)

    # Ages: Normal distribution (mean=45, std=15, range 18-90)
    ages = np.random.normal(loc=45, scale=15, size=n_records)
    ages = np.clip(ages, 18, 90).astype(int)

    # Gender: 48% Male, 52% Female
    genders = np.random.choice(['Male', 'Female'], size=n_records, p=[0.48, 0.52])

    # Diseases with realistic prevalence
    diseases = np.random.choice([
        'Heart Disease',    # 25%
        'Diabetes',        # 20%
        'Hypertension',    # 15%
        'Cancer',          # 12%
        'Asthma',          # 10%
        'Arthritis',       # 8%
        'Mental Health',   # 5%
        'Obesity',         # 3%
        'COPD'            # 2%
    ], size=n_records, p=[0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03, 0.02])

    # Create DataFrame
    df = pd.DataFrame({
        'PatientID': [f'P{i:05d}' for i in range(1, n_records + 1)],
        'ZipCode': zipcodes,
        'Age': ages,
        'Gender': genders,
        'Disease': diseases
    })

    # Print statistics
    print(f"\n✓ Generated {n_records:,} patient records")
    print(f"\nDataset Statistics:")
    print(f"  • ZIP Codes: {df['ZipCode'].nunique()} unique values")
    print(f"  • Age range: {df['Age'].min()}-{df['Age'].max()} years (mean: {df['Age'].mean():.1f})")
    print(f"  • Gender: Male {(df['Gender']=='Male').sum()} ({(df['Gender']=='Male').mean():.1%}), "
          f"Female {(df['Gender']=='Female').sum()} ({(df['Gender']=='Female').mean():.1%})")
    print(f"  • Diseases: {df['Disease'].nunique()} types")

    # Save statistics
    if save_path:
        stats = {
            'n_records': n_records,
            'generation_date': datetime.now().isoformat(),
            'zipcodes': {
                'unique': int(df['ZipCode'].nunique()),
                'distribution': df['ZipCode'].value_counts().to_dict()
            },
            'age': {
                'min': int(df['Age'].min()),
                'max': int(df['Age'].max()),
                'mean': float(df['Age'].mean()),
                'std': float(df['Age'].std())
            },
            'gender': {
                'male_count': int((df['Gender'] == 'Male').sum()),
                'female_count': int((df['Gender'] == 'Female').sum()),
                'male_percentage': float((df['Gender'] == 'Male').mean())
            },
            'disease': {
                'unique': int(df['Disease'].nunique()),
                'distribution': df['Disease'].value_counts().to_dict()
            }
        }

        stats_path = Path(save_path).parent / 'dataset_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n✓ Saved statistics to {stats_path}")

        # Save dataset
        df.to_csv(save_path, index=False)
        print(f"✓ Saved dataset to {save_path}")

    return df


# ============================================================================
# 6. LINKAGE ATTACK SIMULATION
# ============================================================================

def simulate_linkage_attack(df_original: pd.DataFrame,
                           df_anonymized: pd.DataFrame,
                           n_attempts: int = 1000,
                           qi_columns: List[str] = None) -> Dict[str, Any]:
    """
    Simulate Massachusetts GIC-style linkage attack.

    Attack model:
    - Attacker has voter registration data (ZipCode, Age, Gender)
    - Attempts unique re-identification in anonymized medical data
    - Success = finding exactly one matching record

    Returns attack success metrics.
    """
    if qi_columns is None:
        qi_columns = ['ZipCode', 'Age', 'Gender']

    unique_successes = 0
    partial_successes = 0
    failures = 0
    candidate_counts = []
    confidence_scores = []

    # Sample random targets
    np.random.seed(42)
    sample_size = min(n_attempts, len(df_original))
    targets = df_original.sample(n=sample_size, random_state=42)

    for idx, target in targets.iterrows():
        # Find matches in anonymized data
        matches = df_anonymized.copy()

        # Match on ZipCode
        if 'ZipCode' in qi_columns:
            matches = matches[matches['ZipCode'].apply(
                lambda x: matches_generalized_zip(x, target['ZipCode'])
            )]

        # Match on Age
        if 'Age' in qi_columns:
            matches = matches[matches['Age'].apply(
                lambda x: matches_generalized_age(str(x), target['Age'])
            )]

        # Match on Gender
        if 'Gender' in qi_columns and 'Gender' in matches.columns:
            matches = matches[
                (matches['Gender'] == target['Gender']) |
                (matches['Gender'] == 'Person')
            ]

        n_matches = len(matches)
        candidate_counts.append(n_matches)

        if n_matches == 1:
            unique_successes += 1
            confidence_scores.append(1.0)
        elif 2 <= n_matches <= 5:
            partial_successes += 1
            confidence_scores.append(1.0 / n_matches)
        else:
            failures += 1
            confidence_scores.append(0.0)

    total = len(targets)

    return {
        'total_attempts': total,
        'unique_successes': unique_successes,
        'partial_successes': partial_successes,
        'failures': failures,
        'success_rate': unique_successes / total if total > 0 else 0,
        'partial_rate': partial_successes / total if total > 0 else 0,
        'failure_rate': failures / total if total > 0 else 0,
        'avg_candidates': np.mean(candidate_counts) if candidate_counts else 0,
        'median_candidates': np.median(candidate_counts) if candidate_counts else 0,
        'confidence_scores': confidence_scores
    }


# ============================================================================
# Continued in next part due to length...
# ============================================================================
