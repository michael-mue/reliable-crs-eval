#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

os.makedirs("output", exist_ok=True)

CHOSEN_N = 200
CHOSEN_K = 5
CUSTOM_COLORS = {2: "#E69F00", 3: "#56B4E9", 5: "#009E73", 10: "#CC79A7"}


def calculate_icc_power_n(rho, rho0, k, alpha=0.05, power=0.8):
    # Walter et al. (1998)
    if rho <= rho0 or rho >= 1:
        return np.nan
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    C = (1 + k * rho / (1 - rho)) / (1 + k * rho0 / (1 - rho0))
    n = 1 + (2 * (z_alpha + z_beta)**2 * k) / ((k - 1) * (np.log(C)**2))
    return int(np.ceil(n))


def calculate_expected_ci_width(n, k, rho):
    # Donner & Eliasziw (1987) approximation
    if n <= 1:
        return np.nan
    var_rho = (2 * (1 - rho)**2 * (1 + (k - 1) * rho)**2) / (k * (k - 1) * (n - 1))
    return 2 * 1.96 * np.sqrt(var_rho)


def calculate_correlation_power(n, r, alpha=0.05, alternative='two-sided'):
    if r == 0:
        return alpha
    z_r = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    if alternative == 'two-sided':
        z_crit = stats.norm.ppf(1 - alpha / 2)
        return (1 - stats.norm.cdf(z_crit - z_r / se)) + stats.norm.cdf(-z_crit - z_r / se)
    z_crit = stats.norm.ppf(1 - alpha)
    return 1 - stats.norm.cdf(z_crit - z_r / se)


def run_power_analysis_plot():
    print("Generating Power Analysis (Required N) plot...")
    icc_targets = np.linspace(0.01, 0.95, 200)
    rater_counts = [2, 3, 5, 10]

    plot_data = []
    min_detectable_rho = None
    for k in rater_counts:
        for rho in icc_targets:
            n_req = calculate_icc_power_n(rho, 0.0, k)
            plot_data.append({"Target ICC": rho, "Required Dialogues": n_req, "Raters (k)": k})
            if k == CHOSEN_K and min_detectable_rho is None and n_req <= CHOSEN_N:
                min_detectable_rho = rho

    plt.figure(figsize=(6, 4))
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")

    sns.lineplot(data=pd.DataFrame(plot_data), x="Target ICC", y="Required Dialogues",
                 hue="Raters (k)", palette=CUSTOM_COLORS, linewidth=2.5, legend="full")

    plt.axhline(y=CHOSEN_N, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.text(0.98, CHOSEN_N + 8, f'Study Size ($N$={CHOSEN_N})', color='black', fontsize=10, va='bottom', ha='right')

    if min_detectable_rho:
        plt.plot(min_detectable_rho, CHOSEN_N, marker='o', color='black', markersize=6, zorder=10)
        plt.annotate(
            f"Detectable\nICC > {min_detectable_rho:.2f}",
            xy=(min_detectable_rho, CHOSEN_N),
            xytext=(min_detectable_rho + 0.2, CHOSEN_N + 100),
            arrowprops=dict(facecolor='black', edgecolor='black', shrink=0.05, width=1.5, headwidth=8),
            fontsize=11,
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.8)
        )

    plt.ylim(0, 400)
    plt.xlim(0, 1.0)
    plt.ylabel("Sample Size ($N$)", fontsize=12)
    plt.xlabel(r"Target ICC ($\rho$)", fontsize=12)
    plt.legend(title="Raters ($k$)", title_fontsize=11, fontsize=10, loc='upper right', frameon=False)
    sns.despine()
    plt.tight_layout()

    save_path = os.path.join("output", "icc_power_analysis_acm.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {save_path}")
    print(f"For k={CHOSEN_K} and N={CHOSEN_N}, the minimum detectable ICC is {min_detectable_rho:.3f}")


def run_precision_analysis_plot():
    print("Generating Precision Analysis (CI Width) plot...")
    target_rho = 0.6

    precision_data = []
    for k in [2, 3, 5, 10]:
        for n in np.arange(5, 500, 10):
            precision_data.append({"N": n, "CI Width": calculate_expected_ci_width(n, k, target_rho), "Raters (k)": k})

    width_at_study = calculate_expected_ci_width(CHOSEN_N, CHOSEN_K, target_rho)

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=pd.DataFrame(precision_data), x="N", y="CI Width",
                 hue="Raters (k)", palette=CUSTOM_COLORS, linewidth=2.5, legend="full")

    plt.axvline(x=CHOSEN_N, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.text(CHOSEN_N + 10, 0.45, f'Study N={CHOSEN_N}', color='black', fontsize=10)
    plt.plot(CHOSEN_N, width_at_study, marker='o', color='black', markersize=6, zorder=10)
    plt.annotate(
        f"Width $\\approx$ {width_at_study:.2f}",
        xy=(CHOSEN_N, width_at_study),
        xytext=(CHOSEN_N + 50, width_at_study + 0.1),
        arrowprops=dict(facecolor='black', edgecolor='black', shrink=0.05, width=1.5, headwidth=8),
        fontsize=11,
        bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.8)
    )

    plt.ylabel("CI Width (Total range)", fontsize=12)
    plt.xlabel("Sample Size ($N$)", fontsize=12)
    plt.ylim(0, 0.5)
    plt.xlim(0, 500)
    plt.legend(title="Raters ($k$)", title_fontsize=11, fontsize=10, loc='upper right', frameon=False)
    sns.despine()
    plt.tight_layout()

    save_path = os.path.join("output", "icc_precision_acm.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {save_path}")
    print(f"At N={CHOSEN_N}, k={CHOSEN_K}, ICC={target_rho}, expected CI width is {width_at_study:.3f} (approx +/- {width_at_study/2:.3f})")


def run_correlation_power_analysis():
    print("Calculating Correlation Power...")
    r_weak, r_medium = 0.20, 0.30
    power_weak = calculate_correlation_power(CHOSEN_N, r_weak)
    power_medium = calculate_correlation_power(CHOSEN_N, r_medium)
    print(f"--- Power Analysis for N={CHOSEN_N}, alpha=0.05 ---")
    print(f"Power to detect weak correlation (r={r_weak}):   {power_weak:.4f} ({power_weak*100:.1f}%)")
    print(f"Power to detect medium correlation (r={r_medium}): {power_medium:.4f} ({power_medium*100:.1f}%)")


if __name__ == "__main__":
    run_power_analysis_plot()
    run_precision_analysis_plot()
    run_correlation_power_analysis()
