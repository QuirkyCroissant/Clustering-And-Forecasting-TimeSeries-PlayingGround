# Leveraging Clustering for Large-Scale Time Series Forecasting

This repository contains our final code submission for the RDKDD project on household electricity-consumption forecasting.

The final submission is centered on three parts:

- clustering of the time series data,
- XGBoost forecasting,
- Approximate Gaussian Process (AGP) forecasting.

Our report and final presentation use two clustering cases most heavily:

- `Case 2`: best overall clustering result,
- `Case 5`: clustering configuration used for forecasting.

The final workflows are:

- [`notebooks/Clustering.ipynb`](notebooks/Clustering.ipynb)
- [`notebooks/XGB_Orchestration.ipynb`](notebooks/XGB_Orchestration.ipynb)
- [`notebooks/AGP.ipynb`](notebooks/AGP.ipynb)

## Final Submission Scope

We intentionally focused the final submission on:

- the full clustering pipeline,
- XGB forecasting,
- AGP forecasting.

## Repository Overview

Important folders and files:

- `data/raw/`
  - `sample_23.csv`: 2023 household consumption data used as training input
  - `sample_24.csv`: 2024 household consumption data used for evaluation / forecasting
- `data/processed/`
  - normalized data and exported forecast files
- `data/predictions/`
  - contains our final predictions using the Ensemble model
- `notebooks/`
  - final notebooks
- `archive/`
  - older notebooks of failed / dismissed experiments
- `notebooks/outputs/feature/`
  - saved clustering labels and search summaries for `case1` to `case6`
- `notebooks/outputs/shapelet/` and `notebooks/outputs/shapelet_experiments/`
  - saved shapelet features and related clustering artifacts
- `src/forecasting/`
  - reusable forecasting helpers used by the XGB orchestration notebook
- `outputs/`
  - tracked experiment outputs and metric summaries

## Environment Setup

We used notebooks as the main execution interface.

For a local setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- Keep the repository structure unchanged. The notebooks expect the checked-in relative paths.
- Some notebooks include extra install cells for optional packages such as `hdbscan`, `tslearn`, `pyarrow`, `dtaidistance`, or `gpytorch`.
- GPU is not required for every step, but it is strongly recommended for forecasting notebooks, especially XGB and AGP runs.

## How To Reproduce Our Results

### 1. Clustering

Primary notebook:

- [`notebooks/Clustering.ipynb`](notebooks/Clustering.ipynb)

What it covers:

- z-normalisation,
- shapelet generation,
- best clustering analysis (`Case 2`),
- forecasting-oriented clustering analysis (`Case 5`).

Important reproducibility note:

- The shapelet stage is not ideally rerun if the goal is to reproduce the exact final results from the report / presentation.
- Shapelet candidate sampling is stochastic, so rerunning it can change the final `Case 2` / `Case 6` outputs and therefore also affect downstream forecasting.

Recommended interpretation:

- If you want to inspect or reuse the final clustering outputs, use the committed artifacts already stored in:
  - `notebooks/outputs/shapelet/`
  - `notebooks/outputs/shapelet_experiments/`
  - `notebooks/outputs/feature/`
- If you want to rerun the full clustering notebook end to end, treat it as a fresh experiment and expect small differences.

Most important saved clustering files:

- `notebooks/outputs/feature/case2_clusters.csv`
- `notebooks/outputs/feature/case5_clusters.csv`

The final report logic is:

- `Case 2` is the best clustering overall,
- `Case 5` is the clustering used for forecasting.

Also note:

- all-zero households are assigned cluster label `-1` and kept in the exported outputs.

### 2. XGB Forecasting

Primary notebook:

- [`notebooks/XGB_Orchestration.ipynb`](notebooks/XGB_Orchestration.ipynb)

This notebook is the main orchestration layer for our XGB experiments. It calls helper code from `src/forecasting/`.

Historical execution setup:

- The notebook was written so it can be run locally,
- but in the actual project workflow it was run in Kaggle,
- by uploading the orchestration notebook,
- cloning the repo branch inside the Kaggle environment,
- and using a `P100` GPU accelerator for better runtime.

Practical recommendation for reproducing the final XGB run:

1. Open or upload `notebooks/XGB_Orchestration.ipynb` in Kaggle.
1. Use a GPU notebook, ideally `P100`.
1. Make sure the repository is available under the expected structure in `/kaggle/working/Clustering-And-Forecasting-TimeSeries-PlayingGround`.
1. Make sure the clustering inputs exist, especially `case5_clusters.csv`.
1. Run the notebook with the checked-in defaults or with the same branch setup used historically.

Important notebook detail:

The tracked `main` branch currently contains:

- the notebook itself,
- the reusable forecasting helpers under `src/forecasting/`,
- tracked metric summaries under `outputs/metrics/`,
- tracked `case5_base` experiment outputs under `outputs/experiments/case5_base/`.

Current tracked XGB result summary on `main`:

- `cluster_xgb` on `case5_base`: mean MAE `3.3684`
- `global_xgb` on `case5_base`: mean MAE `3.4135`

Relevant tracked output files:

- `outputs/metrics/all_experiments_overall_summary.csv`
- `outputs/metrics/all_experiments_recursive_validation_summary.csv`
- `outputs/experiments/case5_base/`

### 3. AGP Forecasting

Primary notebook:

- [`notebooks/AGP.ipynb`](notebooks/AGP.ipynb)

This notebook contains the Approximate Gaussian Process workflow used in the final project stage.

Current focus of the tracked AGP notebook:

- global forecasting export,
- cluster-specific combined forecasting for `Case 2`,
- cluster-specific combined forecasting for `Case 5`.

The notebook is Kaggle-oriented as well and expects the training and test datasets in Kaggle-style paths by default.

Typical exports from the notebook are:

- `agp_forecast_all_data.csv`
- `agp_forecast_case2_clusters_cluster_specific_combined.csv`
- `agp_forecast_case5_clusters_cluster_specific_combined.csv`

For a broader AGP sweep across all clustering cases, the older notebook
[`notebooks/archive/sparseGpForecasting.ipynb`](notebooks/archive/sparseGpForecasting.ipynb)
is still in the repo, but `AGP.ipynb` is the final notebook we point to first.

## What Is Already Reproducible From `main`

On the current `main` branch, the cleanest reproducible story is:

- clustering outputs are already saved,
- `Case 2` and `Case 5` cluster labels are already present,
- XGB forecasting is reproducible through the final orchestration notebook,
- AGP forecasting is reproducible through the AGP notebook,
- tracked XGB metric summaries are already committed.

## Important Limitations

### Shapelets

- Exact shapelet-based clustering results are sensitive to reruns.
- If exact report-aligned reproduction matters, reuse the committed shapelet artifacts instead of regenerating them.

### Branch History

- Some experimental work also lives in branch history, especially around AGP and extended XGB experiments.
- Historically relevant branch names include `agp-2` and `extended-xgb-experiments`.

### AGP + XGB Combination

- The ensemble experiments are implemented in the notebook [notebooks/ensembleExperiment.ipynb](notebooks/ensembleExperiment.ipynb#L1). That notebook contains the original ensemble logic (use XGB predictions for IDs in large clusters and AGP for smaller clusters) and exploration of variants.
- The helper script [src/forecasting/combine_predictions_case5.py](src/forecasting/combine_predictions_case5.py#L1) reproduces the simple "big-cluster XGB / small-cluster AGP" merging rule and can be used to create a Case-5 combined file (default `outputs/pred_combined_case5.csv`). Note: in our experiments this particular big-vs-small ensemble did not reliably improve performance over the weighted-average ensemble.
- Ensemble-ready predictions (final, weighted-average forecasts used for submission/analysis) are available as [data/predictions/weighted_average_predictions.csv](data/predictions/weighted_average_predictions.csv#L1).

## Minimal Reproduction Path

If you only want the shortest path to our final results:

1. Use the committed clustering outputs, especially `case2_clusters.csv` and `case5_clusters.csv`.
1. Do not rerun shapelet generation unless you intentionally want a fresh variant.
1. Run `notebooks/XGB_Orchestration.ipynb` in Kaggle with a `P100` GPU for the XGB workflow.
1. Run `notebooks/AGP.ipynb` for the AGP workflow.
1. Compare against the tracked outputs in `outputs/` and the saved report-aligned artifacts in `notebooks/outputs/`.
1. Use the final ensemble predictions for submission/analysis: [data/predictions/weighted_average_predictions.csv](data/predictions/weighted_average_predictions.csv#L1). If you prefer the AGP+XGB case-5 combined output, see [outputs/pred_combined_case5.csv](outputs/pred_combined_case5.csv#L1) (the default output of `src/forecasting/combine_predictions_case5.py`).



## Disclaimer:
The creation of the README was assisted by AI and was compiled using information from the repository code, notebooks, tracked outputs, and other submission artifacts such as the final report and presentation.
