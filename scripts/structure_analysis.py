import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from matplotlib.patches import Rectangle

os.makedirs("output", exist_ok=True)

DATA_PATH = "data/annotations.csv"

DIMENSIONS = [
    "accuracy", "novelty", "interaction_adequacy", "explainability",
    "cui_adaptability", "cui_understanding", "cui_response_quality",
    "cui_attentiveness", "perceived_ease_of_use", "perceived_usefulness",
    "user_control", "transparency", "cui_humanness", "cui_rapport",
    "trust_confidence", "satisfaction", "intention_to_use", "intention_to_purchase"
]


def main():
    df = pd.read_csv(DATA_PATH)
    df_clean = df[df["is_gold_standard"] == 0].copy()

    ratings_count = df_clean.groupby("dialogue_id").size()
    print(f"Dialogues: {df_clean['dialogue_id'].nunique()}")
    print(f"Raters: {df_clean['participant_id'].nunique()}")
    print(f"Total ratings rows: {len(df_clean)}")
    print(f"Average ratings per dialogue: {ratings_count.mean():.2f}")

    sns.set_context("paper", font_scale=1.0)

    corr = df_clean[DIMENSIONS].corr(method='spearman')
    corr.index = corr.index.str.replace('_', ' ').str.title().str.replace('Cui', 'CUI', regex=False)
    corr.columns = corr.columns.str.replace('_', ' ').str.title().str.replace('Cui', 'CUI', regex=False)

    # APA style: drop leading zero (".85" not "0.85")
    annot_labels = corr.map(lambda v: f"{v:.2f}".replace("0.", ".").replace("-0.", "-.").replace("1.00", "1.0"))

    g = sns.clustermap(
        corr,
        method='ward',
        annot=annot_labels,
        annot_kws={"size": 8},
        fmt="",
        cmap='RdBu_r',
        vmin=0, vmax=1,
        figsize=(7.2, 7.2),
        linewidths=0.5,
        tree_kws=dict(linewidths=1.5, colors='#999999'),
        dendrogram_ratio=0.12,
        cbar_pos=None
    )

    # invisible anchor keeps rotated x-labels from being clipped
    g.ax_heatmap.text(1.2, 0.5, '.', alpha=0, transform=g.ax_heatmap.transAxes)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

    k = 8
    Z = g.dendrogram_row.linkage
    clusters = fcluster(Z, t=k, criterion='maxclust')
    clusters_reordered = clusters[g.dendrogram_row.reordered_ind]

    blocks = []
    start = 0
    for idx in range(1, len(clusters_reordered)):
        if clusters_reordered[idx] != clusters_reordered[idx - 1]:
            blocks.append((start, idx - start))
            start = idx
    blocks.append((start, len(clusters_reordered) - start))

    for s, h in blocks:
        g.ax_heatmap.add_patch(Rectangle((s, s), h, h, fill=False, lw=1.5, edgecolor='black'))

    plt.savefig("output/correlation_heatmap.pdf", format='pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()
