import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, List
import math

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Sample data for visualization
# Simulating k-anonymous data with homogeneity vs l-diverse data

# ============================================================================
# FIGURE 1: HOMOGENEITY ATTACK VULNERABILITY
# ============================================================================

def plot_homogeneity_attack_demonstration():
    """
    Visualize how k-anonymity alone fails to prevent homogeneity attacks.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Scenario 1: K-anonymous but vulnerable to homogeneity attack
    classes_k_only = ['Class 1', 'Class 2', 'Class 3']
    diseases_k_only = {
        'Class 1': {'Heart Disease': 5},
        'Class 2': {'Diabetes': 5},
        'Class 3': {'Heart Disease': 5}
    }
    
    # Create stacked bar chart for k-anonymity only
    diseases_list = ['Heart Disease', 'Diabetes', 'Breast Cancer', 'Prostate Cancer']
    colors = {'Heart Disease': '#e74c3c', 'Diabetes': '#3498db', 
              'Breast Cancer': '#e67e22', 'Prostate Cancer': '#9b59b6'}
    
    bottoms = np.zeros(len(classes_k_only))
    for disease in diseases_list:
        values = [diseases_k_only[cls].get(disease, 0) for cls in classes_k_only]
        ax1.bar(classes_k_only, values, bottom=bottoms, 
               label=disease, color=colors[disease], edgecolor='black', linewidth=1.5)
        bottoms += values
    
    ax1.axhline(y=3, color='green', linestyle='--', linewidth=2, label='k=3 threshold')
    ax1.set_ylabel('Number of Records', fontsize=12, fontweight='bold')
    ax1.set_title('K-Anonymous Data (k=3)\nVULNERABLE to Homogeneity Attack', 
                  fontsize=14, fontweight='bold', pad=15, color='red')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 6)
    
    # Add warning annotation
    ax1.text(1, 3.5, '⚠️ 100% inference\nprobability', 
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Scenario 2: K-anonymous AND l-diverse
    classes_kl = ['Class 1', 'Class 2', 'Class 3']
    diseases_kl = {
        'Class 1': {'Heart Disease': 2, 'Diabetes': 2, 'Breast Cancer': 1},
        'Class 2': {'Diabetes': 2, 'Prostate Cancer': 2, 'Heart Disease': 1},
        'Class 3': {'Heart Disease': 2, 'Breast Cancer': 2, 'Diabetes': 1}
    }
    
    bottoms2 = np.zeros(len(classes_kl))
    for disease in diseases_list:
        values = [diseases_kl[cls].get(disease, 0) for cls in classes_kl]
        ax2.bar(classes_kl, values, bottom=bottoms2,
               label=disease, color=colors[disease], edgecolor='black', linewidth=1.5)
        bottoms2 += values
    
    ax2.axhline(y=3, color='green', linestyle='--', linewidth=2, label='k=3 threshold')
    ax2.axhline(y=3, color='blue', linestyle=':', linewidth=2, label='l=3 diversity')
    ax2.set_ylabel('Number of Records', fontsize=12, fontweight='bold')
    ax2.set_title('(K,L)-Anonymous Data (k=3, l=3)\nPROTECTED from Homogeneity Attack', 
                  fontsize=14, fontweight='bold', pad=15, color='green')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 6)
    
    # Add protection annotation
    ax2.text(1, 3.5, '✓ ≤40% inference\nprobability', 
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    plt.suptitle('Homogeneity Attack: K-Anonymity vs (K,L)-Anonymity', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('l_diversity_homogeneity_attack.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# FIGURE 2: ATTRIBUTE DISCLOSURE PROBABILITY
# ============================================================================

def plot_attribute_disclosure_comparison():
    """
    Compare attribute disclosure probabilities across different privacy models.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    privacy_models = ['No\nAnonymization', 'K-Anonymity\n(k=3)', 
                     'L-Diversity\n(k=3, l=2)', 'L-Diversity\n(k=3, l=3)']
    disclosure_probs = [100, 100, 50, 33.3]  # Worst-case attribute disclosure %
    
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    bars = ax.bar(privacy_models, disclosure_probs, color=colors,
                  edgecolor='black', linewidth=2, alpha=0.8)
    
    ax.set_ylabel('Attribute Disclosure Probability (%)', fontsize=14, fontweight='bold')
    ax.set_title('Attribute Disclosure Probability: Privacy Model Comparison', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels and risk indicators
    for i, (bar, value) in enumerate(zip(bars, disclosure_probs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{value:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
        
        # Risk level
        if value >= 75:
            risk = 'CRITICAL'
            risk_color = 'darkred'
        elif value >= 50:
            risk = 'HIGH'
            risk_color = 'orange'
        elif value >= 35:
            risk = 'MEDIUM'
            risk_color = 'gold'
        else:
            risk = 'LOW'
            risk_color = 'green'
        
        ax.text(bar.get_x() + bar.get_width()/2., 5,
                risk, ha='center', va='bottom', 
                fontweight='bold', fontsize=9, color=risk_color)
    
    # Add annotation
    ax.annotate('L-diversity limits disclosure\nto at most 1/l probability',
                xy=(2.5, 50), xytext=(1.5, 75),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'),
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('l_diversity_attribute_disclosure.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# FIGURE 3: DIVERSITY METRICS COMPARISON
# ============================================================================

def plot_diversity_metrics():
    """
    Visualize diversity metrics (distinct values and entropy) across models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Distinct value diversity
    models = ['K-Anonymity\nOnly', '(k=3, l=2)', '(k=3, l=3)']
    avg_distinct = [1.0, 2.0, 3.0]  # Average distinct values per class
    min_distinct = [1.0, 2.0, 3.0]  # Minimum distinct values per class
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, avg_distinct, width, label='Average', 
                    color='#3498db', edgecolor='black', alpha=0.7)
    bars2 = ax1.bar(x + width/2, min_distinct, width, label='Minimum (l)', 
                    color='#e74c3c', edgecolor='black', alpha=0.7)
    
    ax1.set_ylabel('Distinct Sensitive Values', fontsize=12, fontweight='bold')
    ax1.set_title('Distinct L-Diversity Metric', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 4)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Entropy diversity
    models_entropy = ['K-Anonymity\nOnly', '(k=3, l=2)', '(k=3, l=3)']
    entropy_values = [0.0, 1.0, 1.58]  # log2(1), log2(2), log2(3)
    required_entropy = [0.0, 1.0, 1.58]
    
    bars3 = ax2.bar(models_entropy, entropy_values, 
                    color=['#95a5a6', '#f39c12', '#2ecc71'], 
                    edgecolor='black', linewidth=1.5, alpha=0.7)
    
    ax2.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, 
                label='Required for l=2', alpha=0.7)
    ax2.axhline(y=1.58, color='green', linestyle='--', linewidth=2,
                label='Required for l=3', alpha=0.7)
    
    ax2.set_ylabel('Entropy (bits)', fontsize=12, fontweight='bold')
    ax2.set_title('Entropy L-Diversity Metric', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 2)
    
    # Add value labels
    for bar, val in zip(bars3, entropy_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('L-Diversity Metrics: Distinct vs Entropy', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('l_diversity_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# FIGURE 4: K-ANONYMITY VS (K,L)-ANONYMITY TRADEOFF
# ============================================================================

def plot_k_vs_kl_anonymity_tradeoff():
    """
    Compare privacy-utility tradeoff between k-anonymity and (k,l)-anonymity.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Configuration comparison
    configs = ['k=3', 'k=3, l=2', 'k=3, l=3', 'k=5, l=2']
    
    # Privacy scores (identity + attribute protection)
    identity_protection = [66.7, 66.7, 66.7, 80.0]  # From k-anonymity
    attribute_protection = [0, 50, 66.7, 50]  # From l-diversity
    
    x_pos = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, identity_protection, width,
                    label='Identity Protection', color='#3498db', 
                    edgecolor='black', alpha=0.7)
    bars2 = ax1.bar(x_pos + width/2, attribute_protection, width,
                    label='Attribute Protection', color='#e67e22',
                    edgecolor='black', alpha=0.7)
    
    ax1.set_ylabel('Protection Level (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Privacy Protection: Identity vs Attribute', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(configs)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 100)
    
    # Information loss comparison
    info_loss = [30, 45, 60, 55]  # Relative information loss
    
    bars3 = ax2.bar(configs, info_loss, 
                    color=['#2ecc71', '#f39c12', '#e74c3c', '#e67e22'],
                    edgecolor='black', linewidth=1.5, alpha=0.7)
    
    ax2.set_ylabel('Information Loss (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Data Utility Cost', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 80)
    
    for bar, loss in zip(bars3, info_loss):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{loss}%', ha='center', va='bottom', fontweight='bold')
    
    # Attack resistance comparison
    attack_types = ['Linkage\nAttack', 'Homogeneity\nAttack', 'Background\nKnowledge']
    k_only_resistance = [66.7, 0, 30]  # k-anonymity resistance
    kl_resistance = [66.7, 66.7, 50]  # (k,l)-anonymity resistance
    
    x_attacks = np.arange(len(attack_types))
    width2 = 0.35
    
    bars4 = ax3.bar(x_attacks - width2/2, k_only_resistance, width2,
                    label='K-Anonymity Only', color='#3498db',
                    edgecolor='black', alpha=0.7)
    bars5 = ax3.bar(x_attacks + width2/2, kl_resistance, width2,
                    label='(K,L)-Anonymity', color='#2ecc71',
                    edgecolor='black', alpha=0.7)
    
    ax3.set_ylabel('Attack Resistance (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Attack Resistance by Type', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticks(x_attacks)
    ax3.set_xticklabels(attack_types)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 100)
    
    # Privacy-utility space
    utility = [70, 55, 40, 45]  # Utility scores
    privacy = [66.7, 83.4, 91.7, 90]  # Combined privacy scores
    
    scatter = ax4.scatter(utility, privacy, s=[200, 300, 400, 350],
                         c=['#3498db', '#f39c12', '#e74c3c', '#9b59b6'],
                         alpha=0.6, edgecolors='black', linewidth=2)
    
    for i, config in enumerate(configs):
        ax4.annotate(config, (utility[i], privacy[i]),
                    fontsize=11, fontweight='bold', ha='center', va='center')
    
    ax4.plot(utility, privacy, 'k--', alpha=0.3, linewidth=2)
    ax4.set_xlabel('Data Utility Score', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Privacy Score', fontsize=12, fontweight='bold')
    ax4.set_title('Privacy-Utility Space', fontsize=14, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(30, 80)
    ax4.set_ylim(60, 100)
    
    plt.suptitle('K-Anonymity vs (K,L)-Anonymity Comprehensive Comparison',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('k_vs_kl_anonymity_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# FIGURE 5: COMBINED DEFENSE EFFECTIVENESS
# ============================================================================

def plot_combined_defense_effectiveness():
    """
    Visualize how k-anonymity and l-diversity work together for comprehensive protection.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create defense matrix
    defenses = ['No Privacy\nProtection', 'K-Anonymity\nOnly (k=3)', 
                'L-Diversity\nOnly (l=3)', '(K,L)-Anonymity\n(k=3, l=3)']
    
    # Defense effectiveness against different threats (0-100%)
    linkage_defense = [0, 66.7, 0, 66.7]
    homogeneity_defense = [0, 0, 66.7, 66.7]
    background_defense = [0, 30, 40, 60]
    
    x = np.arange(len(defenses))
    width = 0.25
    
    bars1 = ax.bar(x - width, linkage_defense, width,
                   label='Linkage Attack Defense', color='#3498db',
                   edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, homogeneity_defense, width,
                   label='Homogeneity Attack Defense', color='#e67e22',
                   edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + width, background_defense, width,
                   label='Background Knowledge Defense', color='#9b59b6',
                   edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Defense Effectiveness (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Privacy Model', fontsize=14, fontweight='bold')
    ax.set_title('Comprehensive Defense Effectiveness Against Privacy Attacks',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(defenses, fontsize=11)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{height:.1f}%', ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
    
    # Add overall effectiveness scores
    overall_scores = [0, 32, 35, 64]
    for i, score in enumerate(overall_scores):
        color = '#e74c3c' if score < 30 else '#f39c12' if score < 50 else '#2ecc71'
        ax.text(i, 90, f'Overall:\n{score}%', ha='center', va='center',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3))
    
    # Add annotations
    ax.annotate('K-anonymity protects\nagainst linkage attacks',
                xy=(1, 66.7), xytext=(0.5, 80),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))
    
    ax.annotate('L-diversity protects\nagainst homogeneity attacks',
                xy=(2, 66.7), xytext=(2.5, 80),
                arrowprops=dict(arrowstyle='->', lw=2, color='orange'),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe5cc', alpha=0.5))
    
    ax.annotate('Combined (k,l)-anonymity\nprovides comprehensive protection',
                xy=(3, 64), xytext=(3, 50),
                arrowprops=dict(arrowstyle='->', lw=3, color='green'),
                fontsize=11, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('combined_defense_effectiveness.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# EXECUTE ALL VISUALIZATIONS
# ============================================================================

if __name__ == "__main__":
    print("Generating L-Diversity Visualization Suite...")
    print("=" * 80)
    
    print("\n[1/5] Creating Homogeneity Attack Demonstration...")
    plot_homogeneity_attack_demonstration()
    print("✓ Saved: l_diversity_homogeneity_attack.png")
    
    print("\n[2/5] Creating Attribute Disclosure Comparison...")
    plot_attribute_disclosure_comparison()
    print("✓ Saved: l_diversity_attribute_disclosure.png")
    
    print("\n[3/5] Creating Diversity Metrics Comparison...")
    plot_diversity_metrics()
    print("✓ Saved: l_diversity_metrics_comparison.png")
    
    print("\n[4/5] Creating K vs (K,L)-Anonymity Tradeoff...")
    plot_k_vs_kl_anonymity_tradeoff()
    print("✓ Saved: k_vs_kl_anonymity_tradeoff.png")
    
    print("\n[5/5] Creating Combined Defense Effectiveness...")
    plot_combined_defense_effectiveness()
    print("✓ Saved: combined_defense_effectiveness.png")
    
    print("\n" + "=" * 80)
    print("All L-Diversity visualizations generated successfully!")
    print("=" * 80)
    print("\nVisualization Summary:")
    print("• Figure 1: Demonstrates homogeneity attack vulnerability in k-anonymity")
    print("           and how l-diversity prevents it")
    print("• Figure 2: Compares attribute disclosure probabilities across privacy models")
    print("• Figure 3: Analyzes diversity metrics (distinct values and entropy)")
    print("• Figure 4: Comprehensive comparison of k-anonymity vs (k,l)-anonymity")
    print("• Figure 5: Shows combined defense effectiveness against multiple attack types")
    print("\nThese visualizations demonstrate that l-diversity successfully prevents")
    print("homogeneity attacks by ensuring diverse sensitive attribute values within")
    print("each equivalence class, complementing k-anonymity for comprehensive privacy.")