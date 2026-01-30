import os
import numpy as np
import pandas as pd
import krippendorff
import statsmodels.api as sm

# ==========================================
# Configuration / Constants
# ==========================================
# Path to the input CSV file
DATA_PATH = "data/exports/20260124_124114/annotations_with_demographics.csv"

# Output directory for tables
OUT_DIR = "."

# Dimensions to analyze
DIMENSIONS = [
    "accuracy", "novelty", "interaction_adequacy", "explainability",
    "cui_adaptability", "cui_understanding", "cui_response_quality",
    "cui_attentiveness", "perceived_ease_of_use", "perceived_usefulness",
    "user_control", "transparency", "cui_humanness", "cui_rapport",
    "trust_confidence", "satisfaction", "intention_to_use", "intention_to_purchase"
]

# ==========================================
# Helper Functions
# ==========================================

def alpha_ordinal(df_long, item_col="dialogue_id", rater_col="participant_id", value_col="rating"):
    """
    Computes Krippendorff's alpha (ordinal metric).
    """
    mat = df_long.pivot(index=rater_col, columns=item_col, values=value_col).to_numpy(dtype=float)
    return krippendorff.alpha(reliability_data=mat, level_of_measurement="ordinal")

def harmonic_mean(x):
    """
    Computes harmonic mean of a vector x.
    """
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    return len(x) / np.sum(1.0 / x) if len(x) else np.nan

def icc_oneway_mixedlm(df_long, target_col, rating_col):
    """
    One-way random effects ICC using MixedLM:
      y = mu + u_target + e
    Returns ICC(1), ICC(1,k) where k is harmonic mean of ratings per target.
    """
    d = df_long[[target_col, rating_col]].dropna().copy()
    if d.empty:
        return None

    counts = d.groupby(target_col)[rating_col].count()
    d = d[d[target_col].isin(counts[counts >= 2].index)]
    if d.empty:
        return None

    try:
        model = sm.MixedLM.from_formula(f"{rating_col} ~ 1", groups=target_col, data=d)
        res = model.fit(reml=True, method="lbfgs", disp=False)

        var_target = float(res.cov_re.iloc[0, 0])
        var_resid  = float(res.scale)

        denom = var_target + var_resid
        icc1 = var_target / denom if denom > 0 else np.nan

        per_target_n = d.groupby(target_col)[rating_col].count().values
        k_eff = harmonic_mean(per_target_n)
        icc1k = var_target / (var_target + var_resid / k_eff) if k_eff and k_eff > 0 else np.nan

        return {
            "ICC1": icc1,
            "ICC1k": icc1k,
            "k_eff_harmonic": k_eff,
            "var_target": var_target,
            "var_resid_oneway": var_resid,
        }
    except Exception as e:
        print(f"Error in ICC(1) for {rating_col}: {e}")
        return None

def crossed_reliability_mixedlm(df_long, dialogue_col, rater_col, rating_col, min_ratings_per_dialogue=2):
    """
    Crossed random-effects model:
        y = mu + u_dialogue + v_rater + e
    Returns reliability metrics derived from variance components.
    """
    d = df_long[[dialogue_col, rater_col, rating_col]].dropna().copy()
    if d.empty:
        return None

    counts = d.groupby(dialogue_col)[rating_col].count()
    keep = counts[counts >= min_ratings_per_dialogue].index
    d = d[d[dialogue_col].isin(keep)]
    if d.empty:
        return None

    per_dialogue_n = d.groupby(dialogue_col)[rating_col].count().values
    k_eff = harmonic_mean(per_dialogue_n)

    try:
        model = sm.MixedLM.from_formula(
            f"{rating_col} ~ 1",
            groups=d[dialogue_col],
            re_formula="1",
            vc_formula={"rater": f"0 + C({rater_col})"},
            data=d
        )
        res = model.fit(reml=True, method="lbfgs", disp=False)

        var_dial = float(res.cov_re.iloc[0, 0])
        var_resid = float(res.scale)
        var_rater = float(res.vcomp[0]) if hasattr(res, "vcomp") and len(res.vcomp) else np.nan

        # guard numerical issues
        var_dial  = max(var_dial, 0.0)
        var_resid = max(var_resid, 0.0)
        if np.isfinite(var_rater):
            var_rater = max(var_rater, 0.0)

        denom_single = var_dial + (var_rater if np.isfinite(var_rater) else 0.0) + var_resid
        rel_single = var_dial / denom_single if denom_single > 0 else np.nan

        if k_eff and k_eff > 0 and np.isfinite(var_rater):
            denom_k = var_dial + var_rater / k_eff + var_resid / k_eff
            rel_k = var_dial / denom_k if denom_k > 0 else np.nan
        else:
            rel_k = np.nan

        return {
            "Rel_dial_single": rel_single,
            "Rel_dial_k": rel_k,
            "k_eff_harmonic": k_eff,
            "var_dialogue": var_dial,
            "var_rater": var_rater,
            "var_resid_crossed": var_resid,
            "converged": bool(getattr(res, "converged", True)),
        }
    except Exception as e:
        print(f"Error in crossed model for {rating_col}: {e}")
        return None

# ==========================================
# Main Execution
# ==========================================

def main():
    print(f"Loading data from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print("Error: File not found.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Initial shape: {df.shape}")

    # Preprocessing / Filtering
    # Only filter for gold standard and excluded flag, removed time filter
    df_clean = df[
        (df["is_gold_standard"] == 0) &
        (df["should_exclude"] == False)
    ].copy()

    ratings_count = df_clean.groupby("dialogue_id").size()
    print(f"Dialogues: {df_clean['dialogue_id'].nunique()}")
    print(f"Raters: {df_clean['participant_id'].nunique()}")
    print(f"Total ratings rows: {len(df_clean)}")
    print(f"Average ratings per dialogue: {ratings_count.mean():.2f}")

    # 1. Krippendorff's Alpha
    print("\nComputing Krippendorff's Alpha...")
    alpha_rows = []
    for dim in DIMENSIONS:
        if dim not in df_clean.columns:
            print(f"Warning: dimension '{dim}' not found in dataframe columns.")
            continue
            
        d = df_clean[["dialogue_id", "participant_id", dim]].dropna()
        counts = d.groupby("dialogue_id")[dim].count()
        d = d[d["dialogue_id"].isin(counts[counts >= 2].index)]
        a = alpha_ordinal(d, value_col=dim) if len(d) else np.nan
        alpha_rows.append({"dimension": dim, "alpha_ordinal": a})

    alpha_df = pd.DataFrame(alpha_rows).round(3)

    # 2. ICC(1) One-Way
    print("\nComputing ICC(1) One-Way MixedLM...")
    icc_rows = []
    for dim in DIMENSIONS:
        if dim not in df_clean.columns: continue
        out = icc_oneway_mixedlm(df_clean, target_col="dialogue_id", rating_col=dim)
        icc_rows.append({"dimension": dim, **(out or {})})
    
    icc_df = pd.DataFrame(icc_rows).round(3)

    # 3. Crossed Random-Effects
    print("\nComputing Crossed Random-Effects Models...")
    crossed_rows = []
    for dim in DIMENSIONS:
        if dim not in df_clean.columns: continue
        out = crossed_reliability_mixedlm(
            df_clean,
            dialogue_col="dialogue_id",
            rater_col="participant_id",
            rating_col=dim,
            min_ratings_per_dialogue=2
        )
        crossed_rows.append({"dimension": dim, **(out or {})})

    crossed_df = pd.DataFrame(crossed_rows).round(3)

    # 4. Combine Summaries
    print("\nCreating Summary Table...")
    summary_df = (
        alpha_df
        .merge(icc_df, on="dimension", how="left")
        .merge(crossed_df, on="dimension", how="left")
    )

    summary_tbl = summary_df.copy()
    summary_tbl["Dimension"] = (
        summary_tbl["dimension"]
        .str.replace("_", " ")
        .str.title()
        .str.replace("Cui", "CUI", regex=False)
    )

    summary_tbl = summary_tbl.sort_values(
        by=["Rel_dial_k", "Rel_dial_single", "alpha_ordinal", "Dimension"],
        ascending=[False, False, False, True],
        na_position="last"
    )

    # 5. Export LaTeX
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        print(f"Created output directory: {OUT_DIR}")

    print(f"Saving LaTeX table to {OUT_DIR}/irr_summary.tex ...")
    
    latex_summary = summary_tbl.copy()
    k_eff_vals = latex_summary["k_eff_harmonic"].dropna().unique() if "k_eff_harmonic" in latex_summary.columns else np.array([])
    k_eff_caption = f"{float(k_eff_vals[0]):.2f}" if len(k_eff_vals) else "N/A"

    latex_summary = latex_summary.rename(columns={
        "alpha_ordinal": r"$\alpha_{\text{ord}}$",
        "ICC1": "ICC(1)",
        "ICC1k": r"ICC(1,$k$)",
        "Rel_dial_single": r"Rel$_{\text{dial}}^{\text{single}}$",
        "Rel_dial_k": r"Rel$_{\text{dial}}^{(k)}$",
    })

    cols_to_print = [
        "Dimension",
        r"Rel$_{\text{dial}}^{(k)}$",
        r"Rel$_{\text{dial}}^{\text{single}}$",
        r"$\alpha_{\text{ord}}$",
        "ICC(1)",
        r"ICC(1,$k$)",
    ]
    cols_to_print = [c for c in cols_to_print if c in latex_summary.columns]

    caption = (
        "Inter-Rater Reliability Metrics by Dimension. "
        f"Crossed-model aggregation uses an effective number of raters $k_\\text{{eff}}={k_eff_caption}$ "
        "(harmonic mean across dialogues). Dimensions are sorted by "
        r"Rel$_{\text{dial}}^{(k)}$."
    )

    latex_summary[cols_to_print].to_latex(
        f"{OUT_DIR}/irr_summary.tex",
        index=False,
        float_format="%.2f",
        escape=False,
        column_format="lccccc",
        caption=caption,
        label="tab:irr_summary"
    )

    print("Done.")

if __name__ == "__main__":
    main()
