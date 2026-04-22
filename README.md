# Can Third-Party Annotators Reliably Evaluate Conversational Recommender Systems?

Reproducibility package for the paper:

> Michael Müller, Amir Reza Mohammadi, Andreas Peintner, Beatriz Barroso Gstrein, Günther Specht, Eva Zangerle.
> **Can Third-Party Annotators Reliably Evaluate Conversational Recommender Systems?**
> *34th ACM Conference on User Modeling, Adaptation and Personalization (UMAP '26)*, June 08–11, 2026, Gothenburg, Sweden.
> https://doi.org/10.1145/3774935.3806172

## Data

`data/annotations.csv` contains 1,053 crowdsourced annotations collected from 117 Prolific workers (after quality control) on 200 ReDial movie recommendation dialogues. Each row is one annotation with 18 quality dimensions from the [CRS-Que](https://dl.acm.org/doi/10.1145/3626772.3657901) framework, rated on a 1–5 Likert scale.

| Column | Description |
|---|---|
| `annotation_id` | Unique annotation identifier |
| `participant_id` | Anonymized worker ID |
| `dialogue_id` | ReDial dialogue identifier |
| `is_gold_standard` | 1 if quasi-gold control dialogue, 0 otherwise |
| `is_prolific_user` | 1 if recruited via Prolific |
| `accuracy` … `intention_to_purchase` | 18 CRS-Que dimension ratings (1–5) |
| `time_spent` | Time spent on annotation (seconds) |
| `timestamp` | Submission timestamp |

## Setup

Install [pixi](https://pixi.sh), then run:

```bash
pixi install
```

## Reproducing the Results

```bash
pixi run all
```

This runs all three analyses and writes outputs to `output/`.

| Task | Command | Output | Corresponds to |
|---|---|---|---|
| Inter-rater reliability | `pixi run reliability` | `output/irr_summary.tex` | Table 1 |
| ICC power analysis | `pixi run power` | `output/icc_power_analysis_acm.pdf`, `output/icc_precision_acm.pdf` | Figure 1 |
| Correlation heatmap | `pixi run structure` | `output/correlation_heatmap.pdf` | Figure 2 |

## Repository Structure

```
├── data/
│   └── annotations.csv       # public annotation dataset
├── scripts/
│   ├── reliability_analysis.py   # Krippendorff's α, ICC(1), crossed reliability → Table 1
│   ├── power_analysis.py         # ICC power and precision analysis → Figure 1
│   └── structure_analysis.py     # Spearman correlation clustermap → Figure 2
├── output/                   # generated outputs (gitignored)
├── pyproject.toml
└── pixi.lock
```
