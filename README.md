# Reproducibility Package

This repository contains the data and analysis scripts accompanying the paper *[Title]*.

## Data

`data/annotations.csv` — crowdsourced annotations for 200 CRS dialogues collected from 117 Prolific workers. Each row is one annotation with 18 quality dimensions rated on a 1–5 Likert scale (`accuracy`, `novelty`, `interaction_adequacy`, `explainability`, `cui_adaptability`, `cui_understanding`, `cui_response_quality`, `cui_attentiveness`, `perceived_ease_of_use`, `perceived_usefulness`, `user_control`, `transparency`, `cui_humanness`, `cui_rapport`, `trust_confidence`, `satisfaction`, `intention_to_use`, `intention_to_purchase`). Gold-standard annotation trials are marked via `is_gold_standard`.

## Setup

Install [pixi](https://pixi.sh), then run:

```bash
pixi install
```

## Reproducing the Results

```bash
pixi run all
```

This runs all three analyses sequentially and writes all outputs to `output/`.

| Task | Command | Output |
|---|---|---|
| Inter-rater reliability | `pixi run reliability` | `output/irr_summary.tex` |
| ICC power analysis | `pixi run power` | `output/icc_power_analysis_acm.pdf`, `output/icc_precision_acm.pdf` |
| Correlation heatmap | `pixi run structure` | `output/correlation_heatmap.pdf` |

## Repository Structure

```
├── data/
│   └── annotations.csv       # public dataset
├── scripts/
│   ├── reliability_analysis.py   # Krippendorff's alpha, ICC(1), crossed reliability
│   ├── power_analysis.py         # ICC power and precision analysis figures
│   └── structure_analysis.py     # Spearman correlation clustermap
├── output/                   # generated (gitignored)
├── pyproject.toml
└── pixi.lock
```
