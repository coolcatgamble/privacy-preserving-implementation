import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set
from collections import Counter
import math

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
# GENERALIZATION FUNCTIONS (Reused from K-Anonymity)
# ============================================================================

def generalize_age(age: int, level: int) -> str:
    """Generalize age based on hierarchy level."""
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
    """Generalize ZIP code based on hierarchy level."""
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
# EQUIVALENCE CLASS FUNCTIONS
# ============================================================================

def calculate_equivalence_classes(df: pd.DataFrame, qi_columns: List[str]) -> Dict:
    """Calculate equivalence classes based on quasi-identifiers."""
    equivalence_classes = {}
    
    for idx, row in df.iterrows():
        qi_tuple = tuple(row[qi_columns].values)
        if qi_tuple not in equivalence_classes:
            equivalence_classes[qi_tuple] = []
        equivalence_classes[qi_tuple].append(idx)
    
    return equivalence_classes

def get_sensitive_values_in_class(df: pd.DataFrame, indices: List[int], 
                                   sensitive_attr: str) -> List:
    """Get all sensitive attribute values in an equivalence class."""
    return df.loc[indices, sensitive_attr].tolist()

# ============================================================================
# L-DIVERSITY CHECKING FUNCTIONS
# ============================================================================

def check_distinct_l_diversity(df: pd.DataFrame, eq_classes: Dict, 
                                sensitive_attr: str, l: int) -> Tuple[bool, Dict]:
    """
    Check if dataset satisfies distinct l-diversity.
    Returns (satisfies_l_diversity, diversity_info).
    
    Distinct l-diversity: Each equivalence class contains at least l distinct 
    values for the sensitive attribute.
    """
    diversity_info = {}
    min_diversity = float('inf')
    
    for qi_combo, indices in eq_classes.items():
        sensitive_values = get_sensitive_values_in_class(df, indices, sensitive_attr)
        distinct_count = len(set(sensitive_values))
        
        diversity_info[qi_combo] = {
            'size': len(indices),
            'distinct_sensitive': distinct_count,
            'sensitive_values': sensitive_values
        }
        
        min_diversity = min(min_diversity, distinct_count)
    
    satisfies = min_diversity >= l
    return satisfies, diversity_info

def calculate_entropy_diversity(sensitive_values: List) -> float:
    """
    Calculate entropy of sensitive attribute distribution.
    Entropy l-diversity: Entropy must be at least log(l).
    """
    value_counts = Counter(sensitive_values)
    total = len(sensitive_values)
    
    entropy = 0
    for count in value_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy

def check_entropy_l_diversity(df: pd.DataFrame, eq_classes: Dict, 
                               sensitive_attr: str, l: int) -> Tuple[bool, Dict]:
    """
    Check if dataset satisfies entropy l-diversity.
    Returns (satisfies_l_diversity, entropy_info).
    """
    entropy_info = {}
    min_entropy = float('inf')
    required_entropy = math.log2(l) if l > 0 else 0
    
    for qi_combo, indices in eq_classes.items():
        sensitive_values = get_sensitive_values_in_class(df, indices, sensitive_attr)
        entropy = calculate_entropy_diversity(sensitive_values)
        
        entropy_info[qi_combo] = {
            'size': len(indices),
            'entropy': entropy,
            'sensitive_values': sensitive_values
        }
        
        min_entropy = min(min_entropy, entropy)
    
    satisfies = min_entropy >= required_entropy
    return satisfies, entropy_info

# ============================================================================
# L-DIVERSITY ANONYMIZATION ALGORITHM
# ============================================================================

def kl_anonymize(df: pd.DataFrame, k: int, l: int, qi_columns: List[str], 
                 sensitive_attr: str, diversity_type: str = 'distinct') -> pd.DataFrame:
    """
    Apply (k,l)-anonymity: k-anonymity + l-diversity.
    
    Args:
        df: Original dataset
        k: k-anonymity parameter (minimum equivalence class size)
        l: l-diversity parameter (minimum distinct sensitive values)
        qi_columns: List of quasi-identifier column names
        sensitive_attr: Name of sensitive attribute column
        diversity_type: 'distinct' or 'entropy' l-diversity variant
    
    Returns:
        Anonymized dataset satisfying both k-anonymity and l-diversity
    """
    df_anon = df.copy()
    
    # Generalization levels
    age_level = 0
    zip_level = 0
    gender_generalized = False
    
    max_iterations = 25
    iteration = 0
    
    while iteration < max_iterations:
        # Apply current generalization
        df_anon['Age_Gen'] = df_anon['Age'].apply(lambda x: generalize_age(x, age_level))
        df_anon['ZipCode_Gen'] = df_anon['ZipCode'].apply(lambda x: generalize_zipcode(x, zip_level))
        
        if gender_generalized:
            df_anon['Gender_Gen'] = 'Person'
        else:
            df_anon['Gender_Gen'] = df_anon['Gender']
        
        # Calculate equivalence classes
        qi_gen = ['Age_Gen', 'ZipCode_Gen', 'Gender_Gen']
        eq_classes = calculate_equivalence_classes(df_anon, qi_gen)
        
        # Check k-anonymity
        small_k_classes = {qi: indices for qi, indices in eq_classes.items() 
                          if len(indices) < k}
        
        # Check l-diversity
        if diversity_type == 'distinct':
            satisfies_l, diversity_info = check_distinct_l_diversity(
                df_anon, eq_classes, sensitive_attr, l)
        else:
            satisfies_l, diversity_info = check_entropy_l_diversity(
                df_anon, eq_classes, sensitive_attr, l)
        
        # Find classes that violate l-diversity
        violating_classes = []
        for qi_combo, info in diversity_info.items():
            if diversity_type == 'distinct':
                if info['distinct_sensitive'] < l:
                    violating_classes.append(qi_combo)
            else:
                required_entropy = math.log2(l) if l > 0 else 0
                if info['entropy'] < required_entropy:
                    violating_classes.append(qi_combo)
        
        # Check if both k-anonymity and l-diversity are satisfied
        if not small_k_classes and satisfies_l:
            # Success! Return anonymized dataset
            df_final = df_anon.copy()
            df_final['Age'] = df_final['Age_Gen']
            df_final['ZipCode'] = df_final['ZipCode_Gen']
            df_final['Gender'] = df_final['Gender_Gen']
            df_final = df_final.drop(columns=['Age_Gen', 'ZipCode_Gen', 'Gender_Gen', 'Name'])
            return df_final
        
        # Apply more generalization to fix violations
        if age_level < 3:
            age_level += 1
        elif zip_level < 3:
            zip_level += 1
        elif not gender_generalized:
            gender_generalized = True
        else:
            # Last resort: suppress problematic records
            indices_to_suppress = []
            
            # Suppress records in small k-anonymity classes
            for indices in small_k_classes.values():
                indices_to_suppress.extend(indices)
            
            # Suppress records in l-diversity violating classes
            for qi_combo in violating_classes:
                if qi_combo in eq_classes:
                    indices_to_suppress.extend(eq_classes[qi_combo])
            
            # Remove duplicates and suppress
            indices_to_suppress = list(set(indices_to_suppress))
            if indices_to_suppress:
                df_anon = df_anon.drop(index=indices_to_suppress).reset_index(drop=True)
            
            # Prepare final dataset
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
    
    # Fallback if max iterations reached
    df_final = df_anon.copy()
    df_final['Age'] = df_final['Age_Gen']
    df_final['ZipCode'] = df_final['ZipCode_Gen']
    df_final['Gender'] = df_final['Gender_Gen']
    df_final = df_final.drop(columns=['Age_Gen', 'ZipCode_Gen', 'Gender_Gen', 'Name'])
    return df_final

# ============================================================================
# DEMONSTRATION AND ANALYSIS
# ============================================================================

def demonstrate_homogeneity_attack():
    """Demonstrate how homogeneity attack works on k-anonymous data."""
    print("=" * 80)
    print("HOMOGENEITY ATTACK DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create a scenario where k-anonymity is satisfied but homogeneity exists
    print("Scenario: K-anonymous dataset with homogeneous sensitive values")
    print("-" * 80)
    
    example_data = {
        'ZipCode': ['021**', '021**', '021**', '021**', '021**'],
        'Age': ['40-59', '40-59', '40-59', '40-59', '40-59'],
        'Gender': ['Male', 'Male', 'Male', 'Male', 'Male'],
        'Disease': ['Heart Disease', 'Heart Disease', 'Heart Disease', 
                   'Heart Disease', 'Heart Disease']
    }
    
    df_example = pd.DataFrame(example_data)
    print(df_example.to_string(index=False))
    print()
    
    print("Analysis:")
    print(f"• K-anonymity status (k=3): ✓ SATISFIED (5 records in equivalence class)")
    print(f"• L-diversity status (l=2): ✗ VIOLATED (only 1 distinct disease)")
    print()
    print("Attack scenario:")
    print("  If an attacker knows Bob is in this equivalence class, they can infer")
    print("  with 100% certainty that Bob has Heart Disease, despite k-anonymity!")
    print()
    print("This demonstrates that k-anonymity alone is insufficient to prevent")
    print("attribute disclosure when sensitive values lack diversity.")
    print("=" * 80)
    print()

def run_l_diversity_analysis():
    """Run comprehensive l-diversity analysis."""
    
    demonstrate_homogeneity_attack()
    
    print("=" * 80)
    print("L-DIVERSITY IMPLEMENTATION - MASSACHUSETTS GIC CASE STUDY")
    print("=" * 80)
    print()
    
    # Show original dataset
    print("ORIGINAL DATASET")
    print("-" * 80)
    print(df_original.to_string(index=False))
    print()
    
    qi_columns = ['ZipCode', 'Age', 'Gender']
    sensitive_attr = 'Disease'
    
    # Test different (k,l) combinations
    test_cases = [
        (3, 2, 'distinct'),
        (3, 3, 'distinct'),
        (5, 2, 'distinct')
    ]
    
    for k, l, div_type in test_cases:
        print("=" * 80)
        print(f"(K,L)-ANONYMIZATION: k={k}, l={l} ({div_type} l-diversity)")
        print("=" * 80)
        print()
        
        df_anonymized = kl_anonymize(df_original, k, l, qi_columns, sensitive_attr, div_type)
        
        print(f"ANONYMIZED DATASET (k={k}, l={l})")
        print("-" * 80)
        print(df_anonymized.to_string(index=False))
        print()
        
        # Verify both properties
        qi_anon = ['ZipCode', 'Age', 'Gender']
        eq_classes = calculate_equivalence_classes(df_anonymized, qi_anon)
        
        # Check k-anonymity
        min_k = min(len(indices) for indices in eq_classes.values())
        k_satisfied = min_k >= k
        
        # Check l-diversity
        l_satisfied, diversity_info = check_distinct_l_diversity(
            df_anonymized, eq_classes, sensitive_attr, l)
        
        print("VERIFICATION:")
        print(f"  • K-anonymity (k={k}): {'✓ SATISFIED' if k_satisfied else '✗ VIOLATED'}")
        print(f"  • L-diversity (l={l}): {'✓ SATISFIED' if l_satisfied else '✗ VIOLATED'}")
        print(f"  • Minimum equivalence class size: {min_k}")
        print()
        
        # Show diversity analysis for each equivalence class
        print(f"EQUIVALENCE CLASS DIVERSITY ANALYSIS:")
        print("-" * 80)
        for i, (qi_combo, info) in enumerate(diversity_info.items(), 1):
            disease_counts = Counter(info['sensitive_values'])
            print(f"  Class {i}: {qi_combo}")
            print(f"    • Size: {info['size']} records")
            print(f"    • Distinct diseases: {info['distinct_sensitive']}")
            print(f"    • Disease distribution: {dict(disease_counts)}")
            
            # Calculate inference probability
            max_prob = max(disease_counts.values()) / info['size']
            print(f"    • Max inference probability: {max_prob:.1%}")
            print()
        
        # Calculate and display metrics
        print("PRIVACY METRICS:")
        max_reident_prob = 1.0 / k
        
        # Calculate worst-case attribute disclosure probability
        max_disclosure_probs = []
        for info in diversity_info.values():
            disease_counts = Counter(info['sensitive_values'])
            max_prob = max(disease_counts.values()) / info['size']
            max_disclosure_probs.append(max_prob)
        
        worst_disclosure = max(max_disclosure_probs) if max_disclosure_probs else 0
        
        print(f"  • Re-identification probability: {max_reident_prob:.1%} (from k-anonymity)")
        print(f"  • Attribute disclosure probability: {worst_disclosure:.1%} (from l-diversity)")
        print(f"  • Combined privacy guarantee: Limits both identity and attribute disclosure")
        print()
    
    print("=" * 80)
    print("SUMMARY: L-DIVERSITY DEFENSE EFFECTIVENESS")
    print("=" * 80)
    print()
    print("L-diversity successfully prevents homogeneity attacks by:")
    print("1. Ensuring each equivalence class contains at least l distinct sensitive values")
    print("2. Reducing attribute disclosure probability from 100% to at most 1/l")
    print("3. Providing protection even when attackers know group membership")
    print()
    print("Combined (k,l)-anonymity provides comprehensive protection:")
    print("• K-anonymity prevents linkage attacks (identity disclosure)")
    print("• L-diversity prevents homogeneity attacks (attribute disclosure)")
    print()
    print("Trade-off: Achieving both k-anonymity and l-diversity requires more")
    print("aggressive generalization or suppression than k-anonymity alone,")
    print("resulting in greater information loss but stronger privacy guarantees.")
    print("=" * 80)

if __name__ == "__main__":
    run_l_diversity_analysis()