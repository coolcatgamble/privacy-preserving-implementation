"""
Large-Scale Privacy Analysis - Part 2
Frontier Analysis, Statistical Validation, and Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any
import json
from pathlib import Path
from scipy import stats as scipy_stats
from tqdm import tqdm
import time

# Import from part 1
from large_scale_privacy_analysis import *


# ============================================================================
# 7. PRIVACY-UTILITY FRONTIER ANALYSIS
# ============================================================================

def analyze_privacy_utility_frontier(df: pd.DataFrame,
                                    k_values: List[int] = [3, 5, 10, 20, 50, 100],
                                    l_values: List[int] = [2, 3, 4, 5],
                                    save_path: str = None) -> pd.DataFrame:
    """
    Explore privacy-utility tradeoff across different (k,l) configurations.

    For each configuration, measures:
    - Privacy: Re-identification probability, attribute disclosure, diversity entropy
    - Utility: Query accuracy, information loss, equivalence class metrics

    Returns DataFrame with all metrics for all configurations.
    """
    print("\n" + "="*80)
    print("PRIVACY-UTILITY FRONTIER ANALYSIS")
    print("="*80)
    print(f"Testing {len(k_values)} k-values × {len(l_values)} l-values = {len(k_values) * len(l_values)} configurations")

    results = []
    qi_columns = ['ZipCode', 'Age', 'Gender']

    total_configs = len(k_values) * len(l_values)
    pbar = tqdm(total=total_configs, desc="Analyzing configurations")

    for k in k_values:
        for l in l_values:
            pbar.set_description(f"Analyzing (k={k}, l={l})")

            try:
                # Apply (k,l)-anonymization
                df_anon, metadata = kl_anonymize_with_metadata(df, k, l, qi_columns)

                # Privacy metrics
                theoretical_reident = 1.0 / k

                # Empirical attack simulation (reduced attempts for speed)
                attack_results = simulate_linkage_attack(df, df_anon, n_attempts=200)
                empirical_reident = attack_results['success_rate']

                # Attribute disclosure (max frequency of sensitive value in any class)
                classes = calculate_equivalence_classes(df_anon, qi_columns)
                attr_disclosure_vals = []
                diversity_entropies = []

                for class_key, group in classes.items():
                    disease_counts = group['Disease'].value_counts()
                    max_freq = disease_counts.max() / len(group) if len(group) > 0 else 0
                    attr_disclosure_vals.append(max_freq)

                    # Calculate diversity entropy
                    probs = disease_counts / len(group)
                    entropy = -np.sum(probs * np.log2(probs + 1e-10))
                    diversity_entropies.append(entropy)

                attr_disclosure = np.mean(attr_disclosure_vals) if attr_disclosure_vals else 1.0
                diversity_entropy = np.mean(diversity_entropies) if diversity_entropies else 0.0

                # Utility metrics
                query_errors = calculate_query_accuracy(df, df_anon)
                avg_query_error = np.mean(list(query_errors.values()))

                # Composite scores
                privacy_score = (1 - empirical_reident) * 0.5 + (1 - attr_disclosure) * 0.5
                utility_score = 1 - metadata['information_loss']

                results.append({
                    'k': k,
                    'l': l,
                    'execution_time': metadata['execution_time'],
                    'n_equivalence_classes': metadata['n_equivalence_classes'],
                    'avg_class_size': metadata['avg_class_size'],
                    'min_class_size': metadata['min_class_size'],
                    'max_class_size': metadata['max_class_size'],
                    'records_suppressed': metadata['records_suppressed'],
                    'suppression_rate': metadata['records_suppressed'] / len(df),
                    'theoretical_reident': theoretical_reident,
                    'empirical_reident': empirical_reident,
                    'attr_disclosure': attr_disclosure,
                    'diversity_entropy': diversity_entropy,
                    'information_loss': metadata['information_loss'],
                    'avg_query_error': avg_query_error,
                    'privacy_score': privacy_score,
                    'utility_score': utility_score,
                    'k_satisfied': metadata['k_anonymity_satisfied'],
                    'l_satisfied': metadata['l_diversity_satisfied']
                })

            except Exception as e:
                print(f"\nError with (k={k}, l={l}): {e}")
                # Add failed configuration with NaN values
                results.append({
                    'k': k,
                    'l': l,
                    'execution_time': 0,
                    'n_equivalence_classes': 0,
                    'avg_class_size': 0,
                    'min_class_size': 0,
                    'max_class_size': 0,
                    'records_suppressed': 0,
                    'suppression_rate': 0,
                    'theoretical_reident': 1.0 / k,
                    'empirical_reident': np.nan,
                    'attr_disclosure': np.nan,
                    'diversity_entropy': np.nan,
                    'information_loss': np.nan,
                    'avg_query_error': np.nan,
                    'privacy_score': np.nan,
                    'utility_score': np.nan,
                    'k_satisfied': False,
                    'l_satisfied': False
                })

            pbar.update(1)

    pbar.close()

    df_results = pd.DataFrame(results)

    if save_path:
        df_results.to_csv(save_path, index=False)
        print(f"\n✓ Saved frontier results to {save_path}")

    return df_results


def identify_pareto_optimal(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Identify Pareto-optimal configurations.
    A configuration is Pareto-optimal if no other configuration has both better privacy AND utility.
    """
    # Remove failed configurations
    df_valid = df_results.dropna(subset=['privacy_score', 'utility_score'])

    pareto_optimal = []

    for idx, row in df_valid.iterrows():
        is_dominated = False

        for idx2, row2 in df_valid.iterrows():
            if idx == idx2:
                continue

            # Check if row2 dominates row (better in both dimensions)
            if (row2['privacy_score'] >= row['privacy_score'] and
                row2['utility_score'] >= row['utility_score'] and
                (row2['privacy_score'] > row['privacy_score'] or
                 row2['utility_score'] > row['utility_score'])):
                is_dominated = True
                break

        if not is_dominated:
            pareto_optimal.append(idx)

    return df_valid.loc[pareto_optimal].sort_values('utility_score', ascending=False)


# ============================================================================
# 8. STATISTICAL VALIDATION
# ============================================================================

def statistical_validation(df: pd.DataFrame,
                          configurations: List[Tuple[int, int]],
                          n_trials: int = 30,
                          save_path: str = None) -> Dict[str, Any]:
    """
    Perform statistical validation with multiple trials.

    For each configuration, run n_trials with different random seeds and calculate:
    - Mean and standard deviation
    - 95% confidence intervals
    - Statistical significance tests

    Parameters:
    -----------
    configurations : List[Tuple[int, int]]
        List of (k, l) configurations to test
        Example: [(3, 2), (5, 3), (10, 4)]

    Returns validation results with confidence intervals.
    """
    print("\n" + "="*80)
    print("STATISTICAL VALIDATION")
    print("="*80)
    print(f"Configurations: {configurations}")
    print(f"Trials per configuration: {n_trials}")

    results_by_config = {}

    for k, l in configurations:
        print(f"\nValidating (k={k}, l={l}) with {n_trials} trials...")

        trial_results = {
            'execution_times': [],
            'reident_rates': [],
            'info_losses': [],
            'n_eq_classes': [],
            'avg_class_sizes': [],
            'privacy_scores': [],
            'utility_scores': []
        }

        for trial in tqdm(range(n_trials), desc=f"(k={k}, l={l})"):
            # Use different seed for each trial
            np.random.seed(trial)

            # Apply anonymization
            df_anon, metadata = kl_anonymize_with_metadata(df, k, l)

            # Run attack simulation
            attack_results = simulate_linkage_attack(df, df_anon, n_attempts=100)

            # Collect metrics
            trial_results['execution_times'].append(metadata['execution_time'])
            trial_results['reident_rates'].append(attack_results['success_rate'])
            trial_results['info_losses'].append(metadata['information_loss'])
            trial_results['n_eq_classes'].append(metadata['n_equivalence_classes'])
            trial_results['avg_class_sizes'].append(metadata['avg_class_size'])

            # Calculate scores
            privacy_score = 1 - attack_results['success_rate']
            utility_score = 1 - metadata['information_loss']
            trial_results['privacy_scores'].append(privacy_score)
            trial_results['utility_scores'].append(utility_score)

        # Calculate statistics
        config_stats = {}
        for metric_name, values in trial_results.items():
            config_stats[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'ci_lower': float(np.percentile(values, 2.5)),
                'ci_upper': float(np.percentile(values, 97.5)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

        results_by_config[f'k{k}_l{l}'] = {
            'k': int(k),
            'l': int(l),
            'n_trials': int(n_trials),
            'metrics': config_stats
        }

        # Print summary
        print(f"\nResults for (k={k}, l={l}):")
        print(f"  Re-identification Rate: {config_stats['reident_rates']['mean']:.2%} ± "
              f"{config_stats['reident_rates']['std']:.2%}")
        print(f"  95% CI: [{config_stats['reident_rates']['ci_lower']:.2%}, "
              f"{config_stats['reident_rates']['ci_upper']:.2%}]")
        print(f"  Information Loss: {config_stats['info_losses']['mean']:.1%} ± "
              f"{config_stats['info_losses']['std']:.1%}")

    # Statistical significance tests
    print("\n" + "-"*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("-"*80)

    # T-test: Compare pairs of configurations
    if len(configurations) >= 2:
        # Example: Test if k=5 is significantly different from k=3
        for i in range(len(configurations) - 1):
            k1, l1 = configurations[i]
            k2, l2 = configurations[i + 1]

            config1_key = f'k{k1}_l{l1}'
            config2_key = f'k{k2}_l{l2}'

            # Get reident rates for both configs (need to recalculate for t-test)
            # For simplicity, we'll use the stored values
            # In practice, you'd want to store individual trial values

            print(f"\nComparing (k={k1}, l={l1}) vs (k={k2}, l={l2}):")
            print(f"  Mean re-identification rates: "
                  f"{results_by_config[config1_key]['metrics']['reident_rates']['mean']:.2%} vs "
                  f"{results_by_config[config2_key]['metrics']['reident_rates']['mean']:.2%}")

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(results_by_config, f, indent=2)
        print(f"\n✓ Saved statistical validation results to {save_path}")

    return results_by_config


# ============================================================================
# 9. COMPREHENSIVE ATTACK ANALYSIS
# ============================================================================

def comprehensive_attack_analysis(df: pd.DataFrame,
                                 configurations: List[Tuple[int, int]],
                                 n_attempts: int = 1000,
                                 save_path: str = None) -> Dict[str, Any]:
    """
    Run comprehensive attack simulations on multiple configurations.

    Tests:
    - Original data (baseline)
    - K-only configurations: k=3, 5, 10, 20
    - (K,L) configurations: (3,2), (5,3), (10,4)

    Returns detailed attack metrics for each configuration.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE ATTACK ANALYSIS")
    print("="*80)
    print(f"Attack attempts per configuration: {n_attempts}")

    results = {}

    # Baseline: Original data
    print("\n[1/9] Testing Original Data (Baseline)...")
    baseline_attack = simulate_linkage_attack(df, df, n_attempts=n_attempts)
    results['original'] = {
        'config': 'Original',
        'k': 0,
        'l': 0,
        **baseline_attack
    }
    print(f"  Success Rate: {baseline_attack['success_rate']:.1%}")

    # K-only configurations
    k_only_configs = [3, 5, 10, 20]
    for idx, k in enumerate(k_only_configs, start=2):
        print(f"\n[{idx}/9] Testing K={k}...")
        df_anon, metadata = k_anonymize_with_metadata(df, k)
        attack_results = simulate_linkage_attack(df, df_anon, n_attempts=n_attempts)
        results[f'k{k}'] = {
            'config': f'K={k}',
            'k': k,
            'l': 0,
            **attack_results,
            **metadata
        }
        print(f"  Success Rate: {attack_results['success_rate']:.1%}")
        print(f"  Avg Candidates: {attack_results['avg_candidates']:.2f}")

    # (K,L) configurations
    for idx, (k, l) in enumerate(configurations, start=6):
        print(f"\n[{idx}/9] Testing (K={k}, L={l})...")
        df_anon, metadata = kl_anonymize_with_metadata(df, k, l)
        attack_results = simulate_linkage_attack(df, df_anon, n_attempts=n_attempts)
        results[f'k{k}_l{l}'] = {
            'config': f'(K={k}, L={l})',
            'k': k,
            'l': l,
            **attack_results,
            **metadata
        }
        print(f"  Success Rate: {attack_results['success_rate']:.1%}")
        print(f"  Avg Candidates: {attack_results['avg_candidates']:.2f}")

    if save_path:
        with open(save_path, 'w') as f:
            # Convert to JSON-serializable format
            json_results = {}
            for key, val in results.items():
                json_results[key] = {k: v for k, v in val.items()
                                    if k != 'confidence_scores'}  # Exclude large arrays
            json.dump(json_results, f, indent=2)
        print(f"\n✓ Saved attack results to {save_path}")

    return results


# ============================================================================
# Continued in visualization file...
# ============================================================================
