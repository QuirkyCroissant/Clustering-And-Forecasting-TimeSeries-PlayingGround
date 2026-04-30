from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from .features import LAGS, ROLL_WINDOWS, SEASONAL_PRIOR_FEATURES, TIME_FEATURES


TIME_FEATURE_ORDER = TIME_FEATURES

HISTORY_FEATURE_ORDER = (
    [f"lag_{lag}" for lag in LAGS]
    + [
        f"roll_mean_{w}"
        for w in ROLL_WINDOWS
    ]
    + [
        f"roll_std_{w}"
        for w in ROLL_WINDOWS
    ]
    + [
        f"roll_min_{w}"
        for w in ROLL_WINDOWS
    ]
    + [
        f"roll_max_{w}"
        for w in ROLL_WINDOWS
    ]
    + ["hist_mean", "hist_std", "hist_zero_fraction"]
)


def _prepare_base_arrays(train_23_wide: pd.DataFrame):
    """Extract household ids and the raw history matrix from the wide 2023 table."""

    date_cols = [c for c in train_23_wide.columns if c != "ID"]
    ids = train_23_wide["ID"].to_numpy()
    hist0 = train_23_wide[date_cols].to_numpy(dtype=np.float32, copy=True)
    return ids, date_cols, hist0


def _prepare_future_date_features(future_dates):
    """Precompute calendar features for each future forecast date."""

    future_dates = pd.to_datetime(pd.Index(future_dates))

    dow = future_dates.dayofweek.to_numpy()
    month = future_dates.month.to_numpy()
    doy = future_dates.dayofyear.to_numpy()

    return {
        "dow": dow,
        "month": month,
        "doy": doy,
        "dow_sin": np.sin(2 * np.pi * dow / 7).astype(np.float32),
        "dow_cos": np.cos(2 * np.pi * dow / 7).astype(np.float32),
        "doy_sin": np.sin(2 * np.pi * doy / 366).astype(np.float32),
        "doy_cos": np.cos(2 * np.pi * doy / 366).astype(np.float32),
        "quarter": future_dates.quarter.to_numpy(dtype=np.float32, copy=False),
        "weekofyear": future_dates.isocalendar().week.to_numpy(dtype=np.float32, copy=False),
        "day": future_dates.day.to_numpy(dtype=np.float32, copy=False),
        "is_weekend": (dow >= 5).astype(np.float32),
        "is_month_start": future_dates.is_month_start.astype(np.float32),
        "is_month_end": future_dates.is_month_end.astype(np.float32),
    }


def _prepare_static_df(ids: np.ndarray, static_features: pd.DataFrame | None):
    """Align optional static features to the household order used by the prediction arrays."""

    if static_features is None:
        return None, []

    if "ID" not in static_features.columns:
        raise ValueError("static_features must contain an 'ID' column")

    merged = pd.DataFrame({"ID": ids}).merge(static_features, on="ID", how="left")
    static_cols = [c for c in merged.columns if c != "ID"]

    if not static_cols:
        return None, []

    static_df = merged[static_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return static_df, static_cols


def _default_feature_cols(static_cols: list[str]) -> list[str]:
    """Build the default column order expected by the forecasting helpers."""

    return TIME_FEATURE_ORDER + HISTORY_FEATURE_ORDER + static_cols + SEASONAL_PRIOR_FEATURES


def _add_seasonal_prior_features(
    data: dict,
    ids: np.ndarray,
    group_arr: np.ndarray,
    row_idx: np.ndarray,
    step: int,
    date_feature_store: dict,
    seasonal_prior_store: dict | None,
):
    """Append seasonal prior values for one forecast day and one batch of households."""

    if seasonal_prior_store is None:
        return

    month = int(date_feature_store["month"][step])
    dow = int(date_feature_store["dow"][step])
    n_rows = len(row_idx)
    data["global_month_mean"] = np.full(
        n_rows,
        seasonal_prior_store["global_month_mean"][month],
        dtype=np.float32,
    )
    data["global_dow_mean"] = np.full(
        n_rows,
        seasonal_prior_store["global_dow_mean"][dow],
        dtype=np.float32,
    )
    data["global_month_dow_mean"] = np.full(
        n_rows,
        seasonal_prior_store["global_month_dow_mean"][month, dow],
        dtype=np.float32,
    )

    profile = seasonal_prior_store["profile"]
    global_mean = seasonal_prior_store["global_mean"]
    cluster_month = np.empty(n_rows, dtype=np.float32)
    cluster_dow = np.empty(n_rows, dtype=np.float32)
    cluster_month_dow = np.empty(n_rows, dtype=np.float32)
    hh_month = np.empty(n_rows, dtype=np.float32)
    hh_dow = np.empty(n_rows, dtype=np.float32)

    for out_idx, src_idx in enumerate(row_idx):
        group = str(group_arr[src_idx])
        cluster_month[out_idx] = seasonal_prior_store["cluster_month_mean"].get(
            group, seasonal_prior_store["global_month_mean"]
        )[month]
        cluster_dow[out_idx] = seasonal_prior_store["cluster_dow_mean"].get(
            group, seasonal_prior_store["global_dow_mean"]
        )[dow]
        cluster_month_dow[out_idx] = seasonal_prior_store["cluster_month_dow_mean"].get(
            group, seasonal_prior_store["global_month_dow_mean"]
        )[month, dow]

        hh_id = ids[src_idx]
        if hh_id in profile.index:
            hh_profile = profile.loc[hh_id]
            hh_month[out_idx] = hh_profile.get(f"hh_month_mean_{month}", global_mean)
            hh_dow[out_idx] = hh_profile.get(f"hh_dow_mean_{dow}", global_mean)
        else:
            hh_month[out_idx] = seasonal_prior_store["global_month_mean"][month]
            hh_dow[out_idx] = seasonal_prior_store["global_dow_mean"][dow]

    data["cluster_month_mean"] = cluster_month
    data["cluster_dow_mean"] = cluster_dow
    data["cluster_month_dow_mean"] = cluster_month_dow
    data["hh_current_month_mean"] = hh_month
    data["hh_current_dow_mean"] = hh_dow


def _build_feature_frame(
    history_buffer: np.ndarray,
    ids: np.ndarray,
    group_arr: np.ndarray,
    row_idx: np.ndarray,
    current_end: int,
    step: int,
    date_feature_store: dict,
    running_sum: np.ndarray,
    running_sum_sq: np.ndarray,
    running_zero_count: np.ndarray,
    static_df: pd.DataFrame | None = None,
    seasonal_prior_store: dict | None = None,
    feature_cols: list[str] | None = None,
):
    """Assemble the feature frame for one recursive forecast step."""

    row_idx = np.asarray(row_idx, dtype=np.int64)
    n_rows = len(row_idx)

    data = {}

    # date features: scalar for this day, repeated for all rows in the batch
    for name in TIME_FEATURE_ORDER:
        data[name] = np.full(n_rows, date_feature_store[name][step], dtype=np.float32)

    # lag features
    for lag in LAGS:
        data[f"lag_{lag}"] = history_buffer[row_idx, current_end - lag]

    # rolling features
    for w in ROLL_WINDOWS:
        window = history_buffer[row_idx, current_end - w:current_end]
        data[f"roll_mean_{w}"] = window.mean(axis=1)
        data[f"roll_std_{w}"] = window.std(axis=1)
        data[f"roll_min_{w}"] = window.min(axis=1)
        data[f"roll_max_{w}"] = window.max(axis=1)

    # running whole-history stats
    current_len = float(current_end)
    hist_mean = running_sum[row_idx] / current_len
    variance = np.maximum((running_sum_sq[row_idx] / current_len) - np.square(hist_mean), 0.0)

    data["hist_mean"] = hist_mean.astype(np.float32)
    data["hist_std"] = np.sqrt(variance).astype(np.float32)
    data["hist_zero_fraction"] = (running_zero_count[row_idx] / current_len).astype(np.float32)
    _add_seasonal_prior_features(
        data=data,
        ids=ids,
        group_arr=group_arr,
        row_idx=row_idx,
        step=step,
        date_feature_store=date_feature_store,
        seasonal_prior_store=seasonal_prior_store,
    )

    X = pd.DataFrame(data)

    if static_df is not None:
        X = pd.concat(
            [X.reset_index(drop=True), static_df.iloc[row_idx].reset_index(drop=True)],
            axis=1,
        )

    if feature_cols is not None:
        X = X.reindex(columns=feature_cols, fill_value=0.0)

    return X


def _to_output_df(ids: np.ndarray, preds: np.ndarray, future_dates) -> pd.DataFrame:
    """Convert the prediction matrix back into the wide forecast format."""

    date_cols = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in future_dates]
    out = pd.DataFrame(preds, columns=date_cols)
    out.insert(0, "ID", ids)
    return out


def forecast_global(
    train_23_wide,
    future_dates,
    model,
    cluster_labels=None,
    static_features=None,
    seasonal_prior_store=None,
    show_progress=True,
    feature_cols=None,
):
    """
    Run fast recursive forecasting with one global model.

    The function predicts one day at a time, but batches all active households
    into one model call per day so the recursion stays efficient.
    """

    ids, _, hist0 = _prepare_base_arrays(train_23_wide)
    n_households, n_hist = hist0.shape
    horizon = len(future_dates)

    static_df, static_cols = _prepare_static_df(ids, static_features)
    if feature_cols is None:
        feature_cols = _default_feature_cols(static_cols)

    date_feature_store = _prepare_future_date_features(future_dates)

    history_buffer = np.zeros((n_households, n_hist + horizon), dtype=np.float32)
    history_buffer[:, :n_hist] = hist0

    preds = np.zeros((n_households, horizon), dtype=np.float32)

    running_sum = hist0.sum(axis=1, dtype=np.float64)
    running_sum_sq = np.square(hist0, dtype=np.float64).sum(axis=1, dtype=np.float64)
    running_zero_count = (hist0 == 0.0).sum(axis=1).astype(np.float64)

    all_zero_mask = np.all(hist0 == 0.0, axis=1)
    active_idx = np.where(~all_zero_mask)[0]

    if len(active_idx) == 0:
        return _to_output_df(ids, preds, future_dates)

    if cluster_labels is None or "ForecastGroup" not in cluster_labels.columns:
        group_arr = np.full(n_households, "unknown", dtype=object)
    else:
        group_df = pd.DataFrame({"ID": ids}).merge(
            cluster_labels[["ID", "ForecastGroup"]],
            on="ID",
            how="left",
        )
        group_arr = group_df["ForecastGroup"].fillna("unknown").to_numpy()

    iterator = range(horizon)
    if show_progress:
        iterator = tqdm(iterator, total=horizon, desc="Forecasting global")

    for step in iterator:
        current_end = n_hist + step

        X_step = _build_feature_frame(
            history_buffer=history_buffer,
            ids=ids,
            group_arr=group_arr,
            row_idx=active_idx,
            current_end=current_end,
            step=step,
            date_feature_store=date_feature_store,
            running_sum=running_sum,
            running_sum_sq=running_sum_sq,
            running_zero_count=running_zero_count,
            static_df=static_df,
            seasonal_prior_store=seasonal_prior_store,
            feature_cols=feature_cols,
        )

        yhat = np.asarray(model.predict(X_step), dtype=np.float32)
        yhat = np.maximum(yhat, 0.0)

        preds[active_idx, step] = yhat
        history_buffer[active_idx, current_end] = yhat

        running_sum[active_idx] += yhat
        running_sum_sq[active_idx] += np.square(yhat, dtype=np.float64)
        running_zero_count[active_idx] += (yhat == 0.0)

    return _to_output_df(ids, preds, future_dates)


def forecast_by_group(
    train_23_wide,
    cluster_labels,
    future_dates,
    group_models,
    fallback_model=None,
    allowed_groups=None,
    static_features=None,
    seasonal_prior_store=None,
    show_progress=True,
    feature_cols=None,
):
    """
    Run fast recursive forecasting with cluster-specific models.

    Each forecast day is processed group by group, with an optional global
    fallback for groups that should not use their local model.
    """
    
    ids, _, hist0 = _prepare_base_arrays(train_23_wide)
    n_households, n_hist = hist0.shape
    horizon = len(future_dates)

    static_df, static_cols = _prepare_static_df(ids, static_features)
    if feature_cols is None:
        feature_cols = _default_feature_cols(static_cols)

    date_feature_store = _prepare_future_date_features(future_dates)

    history_buffer = np.zeros((n_households, n_hist + horizon), dtype=np.float32)
    history_buffer[:, :n_hist] = hist0

    preds = np.zeros((n_households, horizon), dtype=np.float32)

    running_sum = hist0.sum(axis=1, dtype=np.float64)
    running_sum_sq = np.square(hist0, dtype=np.float64).sum(axis=1, dtype=np.float64)
    running_zero_count = (hist0 == 0.0).sum(axis=1).astype(np.float64)

    all_zero_mask = np.all(hist0 == 0.0, axis=1)

    group_df = pd.DataFrame({"ID": ids}).merge(
        cluster_labels[["ID", "ForecastGroup"]],
        on="ID",
        how="left",
    )
    group_arr = group_df["ForecastGroup"].fillna("unknown").to_numpy()
    if allowed_groups is not None:
        allowed_groups = {str(group) for group in allowed_groups}

    routing = []
    for group in pd.unique(group_arr):
        idx = np.where(group_arr == group)[0]
        idx = idx[~all_zero_mask[idx]]

        if len(idx) == 0:
            continue
        if group == "inactive":
            continue

        group_key = str(group)
        if allowed_groups is not None and group_key not in allowed_groups:
            model = fallback_model
        else:
            model = group_models.get(group, {}).get("model", fallback_model)
        if model is None:
            continue

        routing.append((group, idx, model))

    iterator = range(horizon)
    if show_progress:
        iterator = tqdm(iterator, total=horizon, desc="Forecasting by cluster")

    for step in iterator:
        current_end = n_hist + step

        for _, idx, model in routing:
            X_step = _build_feature_frame(
                history_buffer=history_buffer,
                ids=ids,
                group_arr=group_arr,
                row_idx=idx,
                current_end=current_end,
                step=step,
                date_feature_store=date_feature_store,
                running_sum=running_sum,
                running_sum_sq=running_sum_sq,
                running_zero_count=running_zero_count,
                static_df=static_df,
                seasonal_prior_store=seasonal_prior_store,
                feature_cols=feature_cols,
            )

            yhat = np.asarray(model.predict(X_step), dtype=np.float32)
            yhat = np.maximum(yhat, 0.0)

            preds[idx, step] = yhat
            history_buffer[idx, current_end] = yhat

            running_sum[idx] += yhat
            running_sum_sq[idx] += np.square(yhat, dtype=np.float64)
            running_zero_count[idx] += (yhat == 0.0)

    return _to_output_df(ids, preds, future_dates)
