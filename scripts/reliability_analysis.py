import os
import numpy as np
import pandas as pd
import krippendorff
import statsmodels.api as sm

DATA_PATH = "data/annotations.csv"
OUT_DIR = "output"

DIMENSIONS = [
    "accuracy", "novelty", "interaction_adequacy", "explainability",
    "cui_adaptability", "cui_understanding", "cui_response_quality",
    "cui_attentiveness", "perceived_ease_of_use", "perceived_usefulness",
    "user_control", "transparency", "cui_humanness", "cui_rapport",
    "trust_confidence", "satisfaction", "intention_to_use", "intention_to_purchase"
]


def alpha_ordinal(df_long, item_col="dialogue_id", rater_col="participant_id", value_col="rating"):
    mat = df_long.pivot(index=rater_col, columns=item_col, values=value_col).to_numpy(dtype=float)
    return krippendorff.alpha(reliability_data=mat, level_of_measurement="ordinal")


def harmonic_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    return len(x) / np.sum(1.0 / x) if len(x) else np.nan


def icc_oneway_mixedlm(df_long, target_col, rating_col):
    # y = mu + u_target + e
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
        var_resid = float(res.scale)
        denom = var_target + var_resid
        icc1 = var_target / denom if denom > 0 else np.nan

        k_eff = harmonic_mean(d.groupby(target_col)[rating_col].count().values)
        icc1k = var_target / (var_target + var_resid / k_eff) if k_eff > 0 else np.nan

        return {"ICC1": icc1, "ICC1k": icc1k, "k_eff_harmonic": k_eff,
                "var_target": var_target, "var_resid_oneway": var_resid}
    except Exception as e:
        print(f"Error in ICC(1) for {rating_col}: {e}")
        return None


def crossed_reliability_mixedlm(df_long, dialogue_col, rater_col, rating_col, min_ratings_per_dialogue=2):
    # y = mu + u_dialogue + v_rater + e
    d = df_long[[dialogue_col, rater_col, rating_col]].dropna().copy()
    if d.empty:
        return None

    counts = d.groupby(dialogue_col)[rating_col].count()
    d = d[d[dialogue_col].isin(counts[counts >= min_ratings_per_dialogue].index)]
    if d.empty:
        return None

    k_eff = harmonic_mean(d.groupby(dialogue_col)[rating_col].count().values)

    try:
        model = sm.MixedLM.from_formula(
            f"{rating_col} ~ 1",
            groups=d[dialogue_col],
            re_formula="1",
            vc_formula={"rater": f"0 + C({rater_col})"},
            data=d
        )
        res = model.fit(reml=True, method="lbfgs", disp=False)

        var_dial = max(float(res.cov_re.iloc[0, 0]), 0.0)
        var_resid = max(float(res.scale), 0.0)
        var_rater = float(res.vcomp[0]) if hasattr(res, "vcomp") and len(res.vcomp) else np.nan
        if np.isfinite(var_rater):
            var_rater = max(var_rater, 0.0)

        denom_single = var_dial + (var_rater if np.isfinite(var_rater) else 0.0) + var_resid
        rel_single = var_dial / denom_single if denom_single > 0 else np.nan

        if k_eff > 0 and np.isfinite(var_rater):
            denom_k = var_dial + var_rater / k_eff + var_resid / k_eff
            rel_k = var_dial / denom_k if denom_k > 0 else np.nan
        else:
            rel_k = np.nan

        return {"Rel_dial_single": rel_single, "Rel_dial_k": rel_k, "k_eff_harmonic": k_eff,
                "var_dialogue": var_dial, "var_rater": var_rater, "var_resid_crossed": var_resid,
                "converged": bool(getattr(res, "converged", True))}
    except Exception as e:
        print(f"Error in crossed model for {rating_col}: {e}")
        return None


def main():
    df = pd.read_csv(DATA_PATH)
    df_clean = df[df["is_gold_standard"] == 0].copy()

    ratings_count = df_clean.groupby("dialogue_id").size()
    print(f"Dialogues: {df_clean['dialogue_id'].nunique()}")
    print(f"Raters: {df_clean['participant_id'].nunique()}")
    print(f"Total ratings rows: {len(df_clean)}")
    print(f"Average ratings per dialogue: {ratings_count.mean():.2f}")

    print("\nComputing Krippendorff's Alpha...")
    alpha_rows = []
    for dim in DIMENSIONS:
        d = df_clean[["dialogue_id", "participant_id", dim]].dropna()
        counts = d.groupby("dialogue_id")[dim].count()
        d = d[d["dialogue_id"].isin(counts[counts >= 2].index)]
        a = alpha_ordinal(d, value_col=dim) if len(d) else np.nan
        alpha_rows.append({"dimension": dim, "alpha_ordinal": a})
    alpha_df = pd.DataFrame(alpha_rows).round(3)

    print("\nComputing ICC(1) One-Way MixedLM...")
    icc_rows = []
    for dim in DIMENSIONS:
        out = icc_oneway_mixedlm(df_clean, target_col="dialogue_id", rating_col=dim)
        icc_rows.append({"dimension": dim, **(out or {})})
    icc_df = pd.DataFrame(icc_rows).round(3)

    print("\nComputing Crossed Random-Effects Models...")
    crossed_rows = []
    for dim in DIMENSIONS:
        out = crossed_reliability_mixedlm(
            df_clean, dialogue_col="dialogue_id", rater_col="participant_id", rating_col=dim
        )
        crossed_rows.append({"dimension": dim, **(out or {})})
    crossed_df = pd.DataFrame(crossed_rows).round(3)

    print("\nCreating Summary Table...")
    summary_tbl = (
        alpha_df
        .merge(icc_df, on="dimension", how="left")
        .merge(crossed_df, on="dimension", how="left")
    )
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

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Saving LaTeX table to {OUT_DIR}/irr_summary.tex ...")

    k_eff_vals = summary_tbl["k_eff_harmonic"].dropna().unique() if "k_eff_harmonic" in summary_tbl.columns else np.array([])
    k_eff_caption = f"{float(k_eff_vals[0]):.2f}" if len(k_eff_vals) else "N/A"

    latex_tbl = summary_tbl.rename(columns={
        "alpha_ordinal": r"$\alpha_{\text{ord}}$",
        "ICC1": "ICC(1)",
        "ICC1k": r"ICC(1,$k$)",
        "Rel_dial_single": r"Rel$_{\text{dial}}^{\text{single}}$",
        "Rel_dial_k": r"Rel$_{\text{dial}}^{(k)}$",
    })

    cols = ["Dimension", r"Rel$_{\text{dial}}^{(k)}$", r"Rel$_{\text{dial}}^{\text{single}}$",
            r"$\alpha_{\text{ord}}$", "ICC(1)", r"ICC(1,$k$)"]
    cols = [c for c in cols if c in latex_tbl.columns]

    caption = (
        "Inter-Rater Reliability Metrics by Dimension. "
        f"Crossed-model aggregation uses an effective number of raters $k_\\text{{eff}}={k_eff_caption}$ "
        r"(harmonic mean across dialogues). Dimensions are sorted by Rel$_{\text{dial}}^{(k)}$."
    )

    latex_tbl[cols].to_latex(
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
