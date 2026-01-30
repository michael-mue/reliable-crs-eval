#!/usr/bin/env python3
"""
ICC Power Analysis & Visualization

This script performs power analysis for Inter-Rater Reliability (ICC) and 
generates visualization plots for the paper. It covers:
1. Power Analysis: Required N vs. Target ICC
2. Precision Analysis: Expected CI Width
3. Correlation Power Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

# --- Setup ---
# Ensure the output directory exists
output_dir = "img"
os.makedirs(output_dir, exist_ok=True)

# Use better styling for plots
sns.set_theme(style="whitegrid")


# ==========================================
# 1. Power Analysis Framework
# ==========================================

def calculate_icc_power_n(rho, rho0, k, alpha=0.05, power=0.8):
    """
    Walter et al. (1998) formula for required number of subjects N.
    """
    if rho <= rho0 or rho >= 1:
        return np.nan
    
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    
    # C parameter: ratio of variance ratios
    c_num = 1 + k * rho / (1 - rho)
    c_den = 1 + k * rho0 / (1 - rho0)
    C = c_num / c_den
    
    n = 1 + (2 * (z_alpha + z_beta)**2 * k) / ((k - 1) * (np.log(C)**2))
    return int(np.ceil(n))


def run_power_analysis_plot():
    print("Generating Power Analysis (Required N) plot...")
    
    # Parameters for simulation
    icc_targets = np.linspace(0.01, 0.95, 200)
    rater_counts = [2, 3, 5, 10]
    null_icc = 0.0
    chosen_N = 200
    chosen_k = 5

    plot_data = []
    min_detectable_rho = None

    for k in rater_counts:
        for rho in icc_targets:
            n_req = calculate_icc_power_n(rho, null_icc, k)
            plot_data.append({"Target ICC": rho, "Required Dialogues": n_req, "Raters (k)": k})
            
            # Capture the specific intersection
            if k == chosen_k and min_detectable_rho is None and n_req <= chosen_N:
                min_detectable_rho = rho

    df_plot = pd.DataFrame(plot_data)

    # --- ACM Two-Column Style Configuration ---
    plt.figure(figsize=(6, 4)) 

    # Set context to "paper" for appropriate scaling of elements
    sns.set_context("paper", font_scale=1.4) 
    sns.set_style("ticks") # "ticks" is cleaner than "whitegrid" for publications

    # Okabe-Ito Palette (Colorblind friendly)
    custom_colors = {
        2: "#E69F00", 
        3: "#56B4E9", 
        5: "#009E73", 
        10: "#CC79A7" 
    }

    # Main Plot
    sns.lineplot(
        data=df_plot, 
        x="Target ICC", 
        y="Required Dialogues", 
        hue="Raters (k)", 
        palette=custom_colors,
        linewidth=2.5,
        legend="full"
    )

    # 1. Mark chosen N = 200 (Thinner, dashed black line)
    plt.axhline(y=chosen_N, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    # Moved to right side (0.98), right-aligned (ha='right')
    plt.text(0.98, chosen_N + 8, f'Study Size ($N$={chosen_N})', color='black', fontsize=10, va='bottom', ha='right')

    # 2. Annotation for specific design
    if min_detectable_rho:
        plt.plot(min_detectable_rho, chosen_N, marker='o', color='black', markersize=6, zorder=10)
        
        # Compact annotation with a clearly visible, filled arrow
        plt.annotate(
            f"Detectable\nICC > {min_detectable_rho:.2f}",
            xy=(min_detectable_rho, chosen_N),
            xytext=(min_detectable_rho + 0.2, chosen_N + 100),
            arrowprops=dict(facecolor='black', edgecolor='black', shrink=0.05, width=1.5, headwidth=8),
            fontsize=11,
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.8)
        )

    # Formatting axes
    plt.ylim(0, 400)
    plt.xlim(0, 1.0)
    plt.ylabel("Sample Size ($N$)", fontsize=12)
    plt.xlabel(r"Target ICC ($\rho$)", fontsize=12)

    # Legend cleanup
    plt.legend(title="Raters ($k$)", title_fontsize=11, fontsize=10, loc='upper right', frameon=False)

    # Remove top and right spines (standard scientific style)
    sns.despine()

    plt.tight_layout()
    
    # Save as vector graphic
    save_path = os.path.join(output_dir, "icc_power_analysis_acm.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    # NOTE: Absolute path removed for anonymity
    print(f"Saved figure to {save_path}")
    
    # plt.show() # Commented out for script execution
    plt.close()

    print(f"For k={chosen_k} and N={chosen_N}, the minimum detectable ICC is {min_detectable_rho:.3f}")
    print("-" * 30)


# ==========================================
# 2. Precision Analysis: Expected CI Width
# ==========================================

def calculate_expected_ci_width(n, k, rho):
    """
    Approximate width of 95% CI for ICC(1) based on Donner & Eliasziw (1987).
    Standard Error approx: sqrt( 2(1-rho)^2 (1+(k-1)rho)^2 / (k(k-1)(n-1)) )
    """
    if n <= 1: return np.nan
    var_rho = (2 * ((1 - rho)**2) * ((1 + (k - 1) * rho)**2)) / (k * (k - 1) * (n - 1))
    se = np.sqrt(var_rho)
    # 95% CI width is approx 2 * 1.96 * SE
    return 2 * 1.96 * se


def run_precision_analysis_plot():
    print("Generating Precision Analysis (CI Width) plot...")

    # Parameters
    n_range = np.arange(5, 500, 10)
    target_rho = 0.6  # Assuming 'Moderate' agreement for planning
    k_values = [2, 3, 5, 10]
    
    # Re-define palette locally if needed, or reuse global
    custom_colors = {2: "#E69F00", 3: "#56B4E9", 5: "#009E73", 10: "#CC79A7"}

    precision_data = []
    for k in k_values:
        for n in n_range:
            width = calculate_expected_ci_width(n, k, target_rho)
            precision_data.append({"N": n, "CI Width": width, "Raters (k)": k})

    df_precision = pd.DataFrame(precision_data)

    # --- ACM Two-Column Style ---
    plt.figure(figsize=(6, 4))
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")

    # Plot
    sns.lineplot(
        data=df_precision,
        x="N",
        y="CI Width",
        hue="Raters (k)",
        palette=custom_colors,
        linewidth=2.5,
        legend="full"
    )

    # Mark Study Design
    chosen_N = 200
    chosen_k = 5
    width_at_200 = calculate_expected_ci_width(chosen_N, chosen_k, target_rho)

    plt.axvline(x=chosen_N, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.text(chosen_N + 10, 0.45, f'Study N={chosen_N}', color='black', fontsize=10, rotation=0)

    # Annotation for specific width
    plt.plot(chosen_N, width_at_200, marker='o', color='black', markersize=6, zorder=10)
    plt.annotate(
        f"Width $\\approx$ {width_at_200:.2f}",
        xy=(chosen_N, width_at_200),
        xytext=(chosen_N + 50, width_at_200 + 0.1),
        arrowprops=dict(facecolor='black', edgecolor='black', shrink=0.05, width=1.5, headwidth=8),
        fontsize=11,
        bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.8)
    )

    plt.ylabel("CI Width (Total range)", fontsize=12)
    plt.xlabel("Sample Size ($N$)", fontsize=12)
    plt.ylim(0, 0.5)  # Focus on practical precision range
    plt.xlim(0, 500)

    plt.legend(title="Raters ($k$)", title_fontsize=11, fontsize=10, loc='upper right', frameon=False)
    sns.despine()

    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "icc_precision_acm.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved figure to {save_path}")
    
    # plt.show()
    plt.close()

    print(f"At N={chosen_N}, k={chosen_k}, ICC={target_rho}, expected CI width is {width_at_200:.3f} (approx +/- {width_at_200/2:.3f})")
    print("-" * 30)


# ==========================================
# 3. Power Analysis Correlation
# ==========================================

def calculate_correlation_power(n, r, alpha=0.05, alternative='two-sided'):
    """
    Calculates power for a Pearson correlation test using Fisher's Z-transformation.
    """
    if r == 0:
        return alpha
        
    # Fisher's Z-transformation of the correlation coefficient
    z_r = 0.5 * np.log((1 + r) / (1 - r))
    
    # Standard error of the Z-transformed correlation
    se = 1 / np.sqrt(n - 3)
    
    # Critical value from standard normal distribution
    if alternative == 'two-sided':
        z_crit = stats.norm.ppf(1 - alpha / 2)
        # Power calculation
        power = (1 - stats.norm.cdf(z_crit - z_r / se)) + \
                stats.norm.cdf(-z_crit - z_r / se)
    else:
        z_crit = stats.norm.ppf(1 - alpha)
        power = 1 - stats.norm.cdf(z_crit - z_r / se)
        
    return power

def run_correlation_power_analysis():
    print("Calculating Correlation Power...")
    
    N_dialogues = 200
    alpha_level = 0.05

    # 1. Calculate specific points
    r_weak = 0.20
    r_medium = 0.30

    power_weak = calculate_correlation_power(N_dialogues, r_weak, alpha_level)
    power_medium = calculate_correlation_power(N_dialogues, r_medium, alpha_level)

    print(f"--- Power Analysis for N={N_dialogues}, alpha={alpha_level} ---")
    print(f"Power to detect weak correlation (r={r_weak}):   {power_weak:.4f} ({power_weak*100:.1f}%)")
    print(f"Power to detect medium correlation (r={r_medium}): {power_medium:.4f} ({power_medium*100:.1f}%)")

    # 2. (Optional) Generate a plot curve
    effect_sizes = np.linspace(0.05, 0.5, 50)
    powers = [calculate_correlation_power(N_dialogues, r, alpha_level) for r in effect_sizes]

    plt.figure(figsize=(8, 5))
    plt.plot(effect_sizes, powers, lw=2, label=f'N={N_dialogues}')
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% Power Threshold')
    plt.axvline(x=r_weak, color='gray', linestyle=':', alpha=0.5, label=f'Weak (r={r_weak})')
    plt.axvline(x=r_medium, color='gray', linestyle=':', alpha=0.5, label=f'Medium (r={r_medium})')

    plt.title('Statistical Power vs. Correlation Effect Size')
    plt.xlabel('Correlation Coefficient (r)')
    plt.ylabel('Power (1 - beta)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.savefig('img/power_curve_correlation.pdf') # Uncomment to save
    # plt.show()
    plt.close()


if __name__ == "__main__":
    run_power_analysis_plot()
    run_precision_analysis_plot()
    run_correlation_power_analysis()
