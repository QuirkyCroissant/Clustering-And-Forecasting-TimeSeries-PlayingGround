import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import (
    ensure_experiment_dirs,
    load_cluster_labels,
    load_static_features,
    load_wide_csv,
)
from .evaluate import evaluate_global_and_cluster, plot_sample_households_by_group
from .features import make_training_frame
from .predict import forecast_by_group, forecast_global
from .train import fit_cluster_models, fit_global_model


def maybe_apply_debug_subset(train_23, test_24, cluster_labels, static_features, debug, debug_frac, random_state):
    if not debug:
        return train_23, test_24, cluster_labels, static_features

    sampled_ids = set(
        train_23["ID"].sample(frac=debug_frac, random_state=random_state).astype(str)
    )

    train_23 = train_23[train_23["ID"].astype(str).isin(sampled_ids)].reset_index(drop=True)
    test_24 = test_24[test_24["ID"].astype(str).isin(sampled_ids)].reset_index(drop=True)
    cluster_labels = cluster_labels[cluster_labels["ID"].astype(str).isin(sampled_ids)].reset_index(drop=True)

    if static_features is not None:
        static_features = static_features[static_features["ID"].astype(str).isin(sampled_ids)].reset_index(drop=True)

    return train_23, test_24, cluster_labels, static_features


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_experiment(repo_root: Path, exp_config: dict, settings: dict):
    import matplotlib.pyplot as plt

    repo_root = Path(repo_root)
    exp_dirs = ensure_experiment_dirs(repo_root, exp_config["experiment_name"])

    train_23 = load_wide_csv(repo_root / "data" / "raw" / "sample_23.csv")
    test_24 = load_wide_csv(repo_root / "data" / "raw" / "sample_24.csv")
    cluster_labels = load_cluster_labels(Path(exp_config["cluster_path"]))
    static_features = None
    if exp_config.get("static_features_path"):
        static_features = load_static_features(Path(exp_config["static_features_path"]))

    train_23, test_24, cluster_labels, static_features = maybe_apply_debug_subset(
        train_23=train_23,
        test_24=test_24,
        cluster_labels=cluster_labels,
        static_features=static_features,
        debug=settings["debug"],
        debug_frac=settings["debug_frac"],
        random_state=settings["random_state"],
    )

    train_df = make_training_frame(
        train_wide=train_23,
        cluster_labels=cluster_labels,
        static_features=static_features,
        show_progress=True,
    )
    train_frame_shape = list(train_df.shape)
    feature_cols = [
        col for col in train_df.columns
        if col not in ["ID", "ds", "ForecastGroup", "target"]
    ]
    future_dates = pd.to_datetime(test_24.columns[1:])

    global_fit = fit_global_model(
        train_df=train_df,
        feature_cols=feature_cols,
        model_name=settings["model_name"],
        use_gpu=settings["gpu_enabled"],
        verbose=50,
    )
    cluster_fits = fit_cluster_models(
        train_df=train_df,
        feature_cols=feature_cols,
        model_name=settings["model_name"],
        min_households=settings["min_cluster_households"],
        min_rows=settings["min_cluster_rows"],
        use_gpu=settings["gpu_enabled"],
        show_progress=True,
        verbose=False,
    )

    del train_df
    gc.collect()

    pred_global = forecast_global(
        train_23_wide=train_23,
        future_dates=future_dates,
        model=global_fit["model"],
        static_features=static_features,
        show_progress=True,
        feature_cols=feature_cols,
    )
    pred_cluster = forecast_by_group(
        train_23_wide=train_23,
        cluster_labels=cluster_labels,
        future_dates=future_dates,
        group_models=cluster_fits,
        fallback_model=global_fit["model"],
        static_features=static_features,
        show_progress=True,
        feature_cols=feature_cols,
    )

    eval_tables = evaluate_global_and_cluster(
        pred_global_wide=pred_global,
        pred_cluster_wide=pred_cluster,
        truth_wide=test_24,
        cluster_labels=cluster_labels,
        trained_groups=cluster_fits.keys(),
        global_model_label=f"global_{settings['model_name']}",
        cluster_model_label=f"cluster_{settings['model_name']}",
    )

    sampled_households, fig = plot_sample_households_by_group(
        train_23_wide=train_23,
        test_24_wide=test_24,
        pred_cluster_wide=pred_cluster,
        cluster_labels=cluster_labels,
        mae_cluster_df=eval_tables["mae_cluster_detail"],
        pred_global_wide=pred_global,
        bucket_col="SparsityBucket",
        cluster_col="ForecastGroup",
        n_per_group=settings["plot_sample_per_group"],
        random_state=settings["random_state"],
        max_groups=settings["plot_max_groups"],
    )

    pred_global.to_csv(exp_dirs["predictions"] / f"pred_2024_global_{settings['model_name']}.csv", index=False)
    pred_cluster.to_csv(exp_dirs["predictions"] / f"pred_2024_cluster_{settings['model_name']}.csv", index=False)
    sampled_households.to_csv(exp_dirs["plots"] / "sampled_households.csv", index=False)
    fig.savefig(exp_dirs["plots"] / "sample_households_by_group.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    metric_frames = {
        "overall_summary": eval_tables["overall_summary"],
        "mae_global_detail": eval_tables["mae_global_detail"],
        "mae_cluster_detail": eval_tables["mae_cluster_detail"],
        "route_summary": eval_tables["route_summary"],
        "cluster_mae_summary": eval_tables["cluster_mae_summary"],
        "compare_detail": eval_tables["compare_detail"],
        "cluster_compare_summary": eval_tables["cluster_compare_summary"],
    }
    for table_name, df in metric_frames.items():
        df.assign(experiment_name=exp_config["experiment_name"]).to_csv(
            exp_dirs["metrics"] / f"{table_name}.csv",
            index=False,
        )

    experiment_metadata = {
        "experiment_name": exp_config["experiment_name"],
        "case_name": exp_config["case_name"],
        "variant_name": exp_config["variant_name"],
        "cluster_path": str(exp_config["cluster_path"]),
        "static_features_path": exp_config.get("static_features_path"),
        "model_name": settings["model_name"],
        "gpu_enabled": settings["gpu_enabled"],
        "debug": settings["debug"],
        "debug_frac": settings["debug_frac"],
        "random_state": settings["random_state"],
        "train_shape": list(train_23.shape),
        "test_shape": list(test_24.shape),
        "train_frame_shape": train_frame_shape,
        "feature_cols": feature_cols,
        "n_feature_cols": len(feature_cols),
        "trained_groups": sorted(map(str, cluster_fits.keys())),
        "n_trained_cluster_models": len(cluster_fits),
        "global_fit_metadata": global_fit["metadata"],
        "cluster_fit_metadata": {
            str(group): fit["metadata"] for group, fit in cluster_fits.items()
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    write_json(exp_dirs["metadata"] / "experiment_metadata.json", experiment_metadata)
    write_json(exp_dirs["metadata"] / "global_eval_log.json", global_fit["eval_log"])
    write_json(
        exp_dirs["metadata"] / "cluster_eval_logs.json",
        {str(group): fit["eval_log"] for group, fit in cluster_fits.items()},
    )

    del global_fit
    del cluster_fits
    del train_23
    del test_24
    del cluster_labels
    del static_features
    del pred_global
    del pred_cluster
    del eval_tables
    del sampled_households
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Run one forecasting experiment in an isolated process.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--settings-json", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    exp_config = json.loads(args.config_json)
    settings = json.loads(args.settings_json)

    run_experiment(
        repo_root=repo_root,
        exp_config=exp_config,
        settings=settings,
    )


if __name__ == "__main__":
    main()
