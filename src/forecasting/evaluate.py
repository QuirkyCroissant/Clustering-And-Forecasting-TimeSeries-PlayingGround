import numpy as np
import pandas as pd


def mae_by_household(pred_wide: pd.DataFrame, truth_wide: pd.DataFrame) -> pd.DataFrame:
    pred = pred_wide.copy()
    truth = truth_wide.copy()

    pred = pred.rename(columns={pred.columns[0]: "ID"})
    truth = truth.rename(columns={truth.columns[0]: "ID"})

    pred = pred.set_index("ID").sort_index()
    truth = truth.set_index("ID").sort_index()

    common_cols = [c for c in truth.columns if c in pred.columns]
    mae = (truth[common_cols] - pred[common_cols]).abs().mean(axis=1)

    return mae.rename("MAE").reset_index()


def summarise_mae(mae_df: pd.DataFrame, metric_col: str = "MAE") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_households": [len(mae_df)],
            "mean_mae": [mae_df[metric_col].mean()],
            "median_mae": [mae_df[metric_col].median()],
            "std_mae": [mae_df[metric_col].std()],
        }
    )


def get_cluster_metadata_columns(cluster_labels: pd.DataFrame) -> list[str]:
    meta_cols = ["ID"]
    for col in ["ForecastGroup", "RefinedCluster", "SparsityBucket", "SparsityGroup"]:
        if col in cluster_labels.columns and col not in meta_cols:
            meta_cols.append(col)
    return meta_cols


def get_cluster_group_columns(cluster_labels: pd.DataFrame) -> list[str]:
    group_cols = ["ForecastGroup"]
    for col in ["RefinedCluster", "SparsityBucket", "SparsityGroup"]:
        if col in cluster_labels.columns and col not in group_cols:
            group_cols.append(col)
    return group_cols


def attach_cluster_metadata(mae_df: pd.DataFrame, cluster_labels: pd.DataFrame) -> pd.DataFrame:
    meta_cols = get_cluster_metadata_columns(cluster_labels)
    cluster_meta = cluster_labels[meta_cols].drop_duplicates().copy()
    cluster_meta["ID"] = cluster_meta["ID"].astype(str)

    detail = mae_df.copy()
    detail["ID"] = detail["ID"].astype(str)
    return detail.merge(cluster_meta, on="ID", how="left")


def assign_model_routes(
    mae_cluster_detail: pd.DataFrame,
    trained_groups,
    group_col: str = "ForecastGroup",
) -> pd.DataFrame:
    detail = mae_cluster_detail.copy()
    trained_groups = {str(group) for group in trained_groups}
    detail[group_col] = detail[group_col].astype(str)
    detail["model_route"] = np.where(
        detail[group_col].isin(trained_groups),
        "trained_cluster_model",
        np.where(
            detail[group_col].eq("inactive"),
            "inactive_zero",
            "global_fallback",
        ),
    )
    return detail


def summarise_routes(mae_cluster_detail: pd.DataFrame) -> pd.DataFrame:
    return (
        mae_cluster_detail.groupby("model_route")["ID"]
        .count()
        .rename("n_households")
        .reset_index()
        .sort_values("model_route")
        .reset_index(drop=True)
    )


def summarise_cluster_mae(
    mae_cluster_detail: pd.DataFrame,
    group_cols: list[str],
    route_value: str = "trained_cluster_model",
) -> pd.DataFrame:
    filtered = mae_cluster_detail[mae_cluster_detail["model_route"] == route_value]
    if filtered.empty:
        return pd.DataFrame(
            columns=group_cols + [
                "n_households",
                "mean_mae",
                "median_mae",
                "std_mae",
                "min_mae",
                "max_mae",
            ]
        )

    return (
        filtered.groupby(group_cols, dropna=False)["MAE"]
        .agg(
            n_households="size",
            mean_mae="mean",
            median_mae="median",
            std_mae="std",
            min_mae="min",
            max_mae="max",
        )
        .reset_index()
        .sort_values("mean_mae")
        .reset_index(drop=True)
    )


def compare_global_vs_cluster(
    mae_global_detail: pd.DataFrame,
    mae_cluster_detail: pd.DataFrame,
    group_cols: list[str],
    cluster_labels: pd.DataFrame,
    route_value: str = "trained_cluster_model",
):
    merge_cols = get_cluster_metadata_columns(cluster_labels)
    compare_detail = (
        mae_global_detail.rename(columns={"MAE": "MAE_global"})
        .merge(
            mae_cluster_detail.rename(columns={"MAE": "MAE_cluster"}),
            on=merge_cols,
            how="inner",
        )
    )
    compare_detail["delta_cluster_minus_global"] = (
        compare_detail["MAE_cluster"] - compare_detail["MAE_global"]
    )

    filtered = compare_detail[compare_detail["model_route"] == route_value]
    if filtered.empty:
        summary = pd.DataFrame(
            columns=group_cols + [
                "n_households",
                "mean_mae_global",
                "mean_mae_cluster",
                "mean_delta",
                "median_delta",
            ]
        )
    else:
        summary = (
            filtered.groupby(group_cols, dropna=False)
            .agg(
                n_households=("ID", "size"),
                mean_mae_global=("MAE_global", "mean"),
                mean_mae_cluster=("MAE_cluster", "mean"),
                mean_delta=("delta_cluster_minus_global", "mean"),
                median_delta=("delta_cluster_minus_global", "median"),
            )
            .reset_index()
            .sort_values("mean_delta")
            .reset_index(drop=True)
        )

    return compare_detail, summary


def evaluate_global_and_cluster(
    pred_global_wide: pd.DataFrame,
    pred_cluster_wide: pd.DataFrame,
    truth_wide: pd.DataFrame,
    cluster_labels: pd.DataFrame,
    trained_groups,
    global_model_label: str = "global_xgb",
    cluster_model_label: str = "cluster_xgb",
):
    mae_global = mae_by_household(pred_global_wide, truth_wide)
    mae_cluster = mae_by_household(pred_cluster_wide, truth_wide)

    overall_summary = pd.concat(
        [
            summarise_mae(mae_global).assign(model=global_model_label),
            summarise_mae(mae_cluster).assign(model=cluster_model_label),
        ],
        ignore_index=True,
    )

    mae_global_detail = attach_cluster_metadata(mae_global, cluster_labels)
    mae_cluster_detail = attach_cluster_metadata(mae_cluster, cluster_labels)
    mae_cluster_detail = assign_model_routes(mae_cluster_detail, trained_groups=trained_groups)

    route_summary = summarise_routes(mae_cluster_detail)
    group_cols = get_cluster_group_columns(cluster_labels)
    cluster_mae_summary = summarise_cluster_mae(
        mae_cluster_detail=mae_cluster_detail,
        group_cols=group_cols,
    )
    compare_detail, cluster_compare_summary = compare_global_vs_cluster(
        mae_global_detail=mae_global_detail,
        mae_cluster_detail=mae_cluster_detail,
        group_cols=group_cols,
        cluster_labels=cluster_labels,
    )

    return {
        "overall_summary": overall_summary,
        "mae_global_detail": mae_global_detail,
        "mae_cluster_detail": mae_cluster_detail,
        "route_summary": route_summary,
        "cluster_mae_summary": cluster_mae_summary,
        "compare_detail": compare_detail,
        "cluster_compare_summary": cluster_compare_summary,
        "group_cols": group_cols,
    }


def plot_sample_households_by_group(
    train_23_wide: pd.DataFrame,
    test_24_wide: pd.DataFrame,
    pred_cluster_wide: pd.DataFrame,
    cluster_labels: pd.DataFrame,
    mae_cluster_df: pd.DataFrame,
    pred_global_wide: pd.DataFrame | None = None,
    bucket_col: str = "SparsityBucket",
    cluster_col: str = "ForecastGroup",
    n_per_group: int = 2,
    random_state: int = 42,
    max_groups: int | None = None,
    figsize_per_row=(14, 4),
):
    import matplotlib.pyplot as plt

    train_df = train_23_wide.copy()
    test_df = test_24_wide.copy()
    pred_cluster_df = pred_cluster_wide.copy()
    meta_df = cluster_labels.copy()
    mae_df = mae_cluster_df.copy()

    for df in [train_df, test_df, pred_cluster_df, meta_df, mae_df]:
        df["ID"] = df["ID"].astype(str)

    if pred_global_wide is not None:
        pred_global_df = pred_global_wide.copy()
        pred_global_df["ID"] = pred_global_df["ID"].astype(str)
    else:
        pred_global_df = None

    if bucket_col not in meta_df.columns:
        meta_df[bucket_col] = "all"

    meta_df = meta_df[["ID", bucket_col, cluster_col]].drop_duplicates()

    sampled_parts = []
    for _, group_df in meta_df.groupby([bucket_col, cluster_col], sort=False):
        sampled_parts.append(
            group_df.sample(
                n=min(n_per_group, len(group_df)),
                random_state=random_state,
            )
        )

    sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else meta_df.iloc[0:0].copy()

    group_keys = sampled[[bucket_col, cluster_col]].drop_duplicates()
    if max_groups is not None:
        group_keys = group_keys.head(max_groups)
        sampled = sampled.merge(group_keys, on=[bucket_col, cluster_col], how="inner")

    sampled = sampled.merge(mae_df[["ID", "MAE"]], on="ID", how="left")

    train_lookup = train_df.set_index("ID")
    test_lookup = test_df.set_index("ID")
    pred_cluster_lookup = pred_cluster_df.set_index("ID")
    pred_global_lookup = pred_global_df.set_index("ID") if pred_global_df is not None else None

    train_dates = pd.to_datetime(train_df.columns[1:])
    test_dates = pd.to_datetime(test_df.columns[1:])

    unique_groups = sampled[[bucket_col, cluster_col]].drop_duplicates().to_records(index=False)
    n_rows = len(unique_groups)
    n_cols = n_per_group

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_row[0], figsize_per_row[1] * max(n_rows, 1)),
        squeeze=False,
    )

    for row_i, (bucket_value, cluster_value) in enumerate(unique_groups):
        group_sample = sampled[
            (sampled[bucket_col] == bucket_value)
            & (sampled[cluster_col] == cluster_value)
        ].reset_index(drop=True)

        for col_i in range(n_cols):
            ax = axes[row_i, col_i]

            if col_i >= len(group_sample):
                ax.axis("off")
                continue

            hh_id = group_sample.loc[col_i, "ID"]
            mae_val = group_sample.loc[col_i, "MAE"]

            train_vals = train_lookup.loc[hh_id].to_numpy(dtype=float)
            actual_vals = test_lookup.loc[hh_id].to_numpy(dtype=float)
            cluster_pred_vals = pred_cluster_lookup.loc[hh_id].to_numpy(dtype=float)

            ax.plot(train_dates, train_vals, label="2023 train")
            ax.plot(test_dates, actual_vals, label="2024 actual")
            ax.plot(test_dates, cluster_pred_vals, label="2024 cluster forecast")

            if pred_global_lookup is not None and hh_id in pred_global_lookup.index:
                global_pred_vals = pred_global_lookup.loc[hh_id].to_numpy(dtype=float)
                ax.plot(test_dates, global_pred_vals, label="2024 global forecast", linestyle="--")

            ax.axvline(test_dates[0], linestyle="--", linewidth=1)
            ax.set_title(
                f"ID={hh_id} | bucket={bucket_value} | cluster={cluster_value} | MAE={mae_val:.3f}"
            )
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return sampled, fig
