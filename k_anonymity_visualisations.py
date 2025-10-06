import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import sys
import os

# Import the k-anonymity implementation
from k_anonymity_implementation import run_k_anonymity_analysis

# Set style for professional visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Get actual results from implementation
print("Loading data from k-anonymity implementation...")
results = run_k_anonymity_analysis(print_results=False, return_results=True)

# Extract data for visualizations
k_values = results['k_values']
reidentification_prob = [results['metrics'][k]['reident_prob'] for k in k_values]
privacy_gain = [results['metrics'][k]['privacy_gain'] * 100 for k in k_values]
discernibility_costs = [results['metrics'][k]['discern_anon'] for k in k_values]
age_precision_loss = [results['metrics'][k]['precision_loss'].get('Age', 0) for k in k_values]
zipcode_precision_loss = [results['metrics'][k]['precision_loss'].get('ZipCode', 0) for k in k_values]

# Get original discernibility cost
discern_orig = results['metrics'][k_values[0]]['discern_orig']

# ============================================================================
# FIGURE 1: PRIVACY-UTILITY TRADEOFF
# ============================================================================

def plot_privacy_utility_tradeoff():
    """
    Create comprehensive privacy-utility tradeoff visualization.
    Shows how privacy improves while utility decreases as k increases.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Re-identification Probability
    ax1.plot(k_values, reidentification_prob, marker='o', linewidth=2, 
             markersize=8, color='#e74c3c', label='Re-identification Risk')
    ax1.fill_between(k_values, reidentification_prob, alpha=0.3, color='#e74c3c')
    ax1.set_xlabel('K Value', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Re-identification Probability', fontsize=12, fontweight='bold')
    ax1.set_title('Privacy Protection: Re-identification Probability vs K', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.6)
    
    # Add percentage labels
    for k, prob in zip(k_values, reidentification_prob):
        ax1.annotate(f'{prob:.1%}', (k, prob), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold')
    
    # Plot 2: Privacy Gain
    bars = ax2.bar(k_values, privacy_gain, color=['#3498db', '#2ecc71', '#f39c12', '#9b59b6'], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('K Value', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Privacy Gain (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Privacy Gain: Protection Level vs K', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, gain in zip(bars, privacy_gain):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{gain:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Information Loss (Discernibility)
    ax3.plot(k_values, discernibility_costs, marker='s', linewidth=2, 
             markersize=8, color='#e67e22', label='Discernibility Cost')
    ax3.fill_between(k_values, discernibility_costs, alpha=0.3, color='#e67e22')
    ax3.set_xlabel('K Value', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Discernibility Cost', fontsize=12, fontweight='bold')
    ax3.set_title('Data Utility: Information Loss vs K', 
                  fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Precision Loss Comparison
    x_pos = np.arange(len(k_values))
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, [p*100 for p in age_precision_loss], width,
                    label='Age Precision Loss', color='#8e44ad', alpha=0.7, edgecolor='black')
    bars2 = ax4.bar(x_pos + width/2, [p*100 for p in zipcode_precision_loss], width,
                    label='ZIP Code Precision Loss', color='#16a085', alpha=0.7, edgecolor='black')
    
    ax4.set_xlabel('K Value', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Precision Loss (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Attribute Precision Loss vs K', 
                  fontsize=14, fontweight='bold', pad=15)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(k_values)
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 100)
    
    plt.suptitle('K-Anonymity: Privacy-Utility Tradeoff Analysis', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('k_anonymity_privacy_utility_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# FIGURE 2: EQUIVALENCE CLASS DISTRIBUTION
# ============================================================================

def plot_equivalence_class_distribution():
    """
    Visualize how records are grouped into equivalence classes.
    Uses actual data from k-anonymity implementation.
    """
    # Get actual equivalence class sizes
    k3_eq_classes = results['equivalence_classes'][3]
    k5_eq_classes = results['equivalence_classes'][5]

    k3_classes = [len(indices) for indices in k3_eq_classes.values()]
    k5_classes = [len(indices) for indices in k5_eq_classes.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for k=3
    class_labels_k3 = [f'Class {i+1}' for i in range(len(k3_classes))]
    colors_k3 = plt.cm.Set3(np.linspace(0, 1, len(k3_classes)))

    ax1.bar(class_labels_k3, k3_classes, color=colors_k3, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=3, color='red', linestyle='--', linewidth=2, label='k=3 threshold')
    ax1.set_ylabel('Number of Records', fontsize=12, fontweight='bold')
    ax1.set_title('Equivalence Class Distribution (k=3)', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(k3_classes):
        ax1.text(i, v + 0.2, str(v), ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Plot for k=5
    class_labels_k5 = [f'Class {i+1}' for i in range(len(k5_classes))]
    colors_k5 = plt.cm.Set2(np.linspace(0, 1, len(k5_classes)))

    ax2.bar(class_labels_k5, k5_classes, color=colors_k5, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=5, color='red', linestyle='--', linewidth=2, label='k=5 threshold')
    ax2.set_ylabel('Number of Records', fontsize=12, fontweight='bold')
    ax2.set_title('Equivalence Class Distribution (k=5)', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(k5_classes):
        ax2.text(i, v + 0.2, str(v), ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.suptitle('Record Grouping into Equivalence Classes',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('equivalence_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# FIGURE 3: ATTACK SUCCESS RATE COMPARISON
# ============================================================================

def plot_attack_success_rate():
    """
    Compare attack success rates on original vs k-anonymized data.
    Uses actual data from k-anonymity implementation.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Build categories and success rates from actual data
    categories = ['Original\nData']
    linkage_attack_success = [100]  # Original data has 100% attack success

    for k in k_values:
        categories.append(f'k={k}')
        linkage_attack_success.append(results['metrics'][k]['reident_prob'] * 100)
    
    # Dynamically create colors based on actual number of categories
    color_palette = ['#e74c3c', '#f39c12', '#2ecc71']
    colors = color_palette[:len(categories)]
    bars = ax.bar(categories, linkage_attack_success, color=colors,
                  edgecolor='black', linewidth=2, alpha=0.8)
    
    ax.set_ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Linkage Attack Success Rate: Original vs K-Anonymized Data', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels and risk level indicators
    for i, (bar, value) in enumerate(zip(bars, linkage_attack_success)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{value:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
        
        # Add risk level text
        if value >= 50:
            risk = 'HIGH RISK'
            risk_color = 'red'
        elif value >= 25:
            risk = 'MEDIUM RISK'
            risk_color = 'orange'
        else:
            risk = 'LOW RISK'
            risk_color = 'green'
        
        ax.text(bar.get_x() + bar.get_width()/2., 5,
                risk, ha='center', va='bottom', 
                fontweight='bold', fontsize=9, color=risk_color)
    
    # Add annotation explaining the defense
    ax.annotate('K-anonymity defense reduces\nattack success rate from 100% to 1/k',
                xy=(3, 20), xytext=(3, 60),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'),
                fontsize=11, ha='center', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('attack_success_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# FIGURE 4: DATA TRANSFORMATION VISUALIZATION
# ============================================================================

def plot_data_transformation():
    """
    Show before/after comparison of data generalization.
    Uses actual data from k-anonymity implementation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Calculate actual unique values from original and anonymized datasets
    original_df = results['original_df']
    k3_df = results['anonymized_dfs'][3]
    k5_df = results['anonymized_dfs'][5]

    attributes = ['ZIP Code', 'Age', 'Gender']
    original_specificity = [
        original_df['ZipCode'].nunique(),
        original_df['Age'].nunique(),
        original_df['Gender'].nunique()
    ]
    anonymized_k3_specificity = [
        k3_df['ZipCode'].nunique(),
        k3_df['Age'].nunique(),
        k3_df['Gender'].nunique()
    ]
    anonymized_k5_specificity = [
        k5_df['ZipCode'].nunique(),
        k5_df['Age'].nunique(),
        k5_df['Gender'].nunique()
    ]
    
    x = np.arange(len(attributes))
    width = 0.25
    
    bars1 = ax1.bar(x - width, original_specificity, width, 
                    label='Original', color='#3498db', edgecolor='black')
    bars2 = ax1.bar(x, anonymized_k3_specificity, width, 
                    label='k=3 Anonymized', color='#e67e22', edgecolor='black')
    bars3 = ax1.bar(x + width, anonymized_k5_specificity, width, 
                    label='k=5 Anonymized', color='#e74c3c', edgecolor='black')
    
    ax1.set_ylabel('Number of Distinct Values', fontsize=12, fontweight='bold')
    ax1.set_title('Attribute Generalization Impact', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(attributes, fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Privacy vs Utility scatter plot using actual data
    privacy_scores = [results['metrics'][k]['privacy_gain'] * 100 for k in k_values]
    # Calculate utility as inverse of information loss (normalized)
    max_discern = max([results['metrics'][k]['discern_anon'] for k in k_values])
    utility_scores = [100 - ((results['metrics'][k]['discern_anon'] - discern_orig) /
                             (max_discern - discern_orig) * 50) for k in k_values]
    k_labels = [f'k={k}' for k in k_values]
    
    # Dynamically size and color points based on number of k values
    sizes = [200 + i * 100 for i in range(len(k_values))]
    colors_scatter = ['#3498db', '#2ecc71'][:len(k_values)]

    scatter = ax2.scatter(utility_scores, privacy_scores,
                         s=sizes,
                         c=colors_scatter,
                         alpha=0.6, edgecolors='black', linewidth=2)
    
    for i, label in enumerate(k_labels):
        ax2.annotate(label, (utility_scores[i], privacy_scores[i]), 
                    fontsize=11, fontweight='bold', ha='center', va='center')
    
    ax2.set_xlabel('Data Utility Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Privacy Score', fontsize=12, fontweight='bold')
    ax2.set_title('Privacy-Utility Space', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(40, 105)
    ax2.set_ylim(40, 95)
    
    # Add Pareto frontier line
    ax2.plot(utility_scores, privacy_scores, 'k--', alpha=0.3, linewidth=2)
    
    plt.suptitle('Data Transformation Under K-Anonymity', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('data_transformation_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# EXECUTE ALL VISUALIZATIONS
# ============================================================================

if __name__ == "__main__":
    print("Generating K-Anonymity Visualization Suite...")
    print("=" * 80)
    
    print("\n[1/4] Creating Privacy-Utility Tradeoff Analysis...")
    plot_privacy_utility_tradeoff()
    print("✓ Saved: k_anonymity_privacy_utility_tradeoff.png")
    
    print("\n[2/4] Creating Equivalence Class Distribution...")
    plot_equivalence_class_distribution()
    print("✓ Saved: equivalence_class_distribution.png")
    
    print("\n[3/4] Creating Attack Success Rate Comparison...")
    plot_attack_success_rate()
    print("✓ Saved: attack_success_rate_comparison.png")
    
    print("\n[4/4] Creating Data Transformation Visualization...")
    plot_data_transformation()
    print("✓ Saved: data_transformation_visualization.png")
    
    print("\n" + "=" * 80)
    print("All visualizations generated successfully!")
    print("=" * 80)
    print("\nVisualization Summary:")
    print("• Figure 1: Comprehensive privacy-utility tradeoff showing re-identification")
    print("           probability, privacy gain, information loss, and precision loss")
    print("• Figure 2: Equivalence class distributions demonstrating record grouping")
    print("• Figure 3: Attack success rate comparison across different k values")
    print("• Figure 4: Data transformation impact and privacy-utility space mapping")
    print("\nThese figures demonstrate that k-anonymity successfully defends against")
    print("linkage attacks by reducing re-identification probability from 100% to 1/k,")
    print("with the tradeoff of reduced data utility through attribute generalization.")