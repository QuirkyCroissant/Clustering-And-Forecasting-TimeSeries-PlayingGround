from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from .features import LAGS, ROLL_WINDOWS


TIME_FEATURE_ORDER = [
    "dow",
    "month",
    "doy",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
]

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
    date_cols = [c for c in train_23_wide.columns if c != "ID"]
    ids = train_23_wide["ID"].to_numpy()
    hist0 = train_23_wide[date_cols].to_numpy(dtype=np.float32, copy=True)
    return ids, date_cols, hist0


def _prepare_future_date_features(future_dates):
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
    }


def _prepare_static_df(ids: np.ndarray, static_features: pd.DataFrame | None):
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
    return TIME_FEATURE_ORDER + HISTORY_FEATURE_ORDER + static_cols


def _build_feature_frame(
    history_buffer: np.ndarray,
    row_idx: np.ndarray,
    current_end: int,
    step: int,
    date_feature_store: dict,
    running_sum: np.ndarray,
    running_sum_sq: np.ndarray,
    running_zero_count: np.ndarray,
    static_df: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
):
    row_idx = np.asarray(row_idx, dtype=np.int64)
    n_rows = len(row_idx)

    data = {}

    # date features: scalar for this day, repeated for all rows in the batch
    data["dow"] = np.full(n_rows, date_feature_store["dow"][step], dtype=np.float32)
    data["month"] = np.full(n_rows, date_feature_store["month"][step], dtype=np.float32)
    data["doy"] = np.full(n_rows, date_feature_store["doy"][step], dtype=np.float32)
    data["dow_sin"] = np.full(n_rows, date_feature_store["dow_sin"][step], dtype=np.float32)
    data["dow_cos"] = np.full(n_rows, date_feature_store["dow_cos"][step], dtype=np.float32)
    data["doy_sin"] = np.full(n_rows, date_feature_store["doy_sin"][step], dtype=np.float32)
    data["doy_cos"] = np.full(n_rows, date_feature_store["doy_cos"][step], dtype=np.float32)

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
    date_cols = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in future_dates]
    out = pd.DataFrame(preds, columns=date_cols)
    out.insert(0, "ID", ids)
    return out


def forecast_global(
    train_23_wide,
    future_dates,
    model,
    static_features=None,
    show_progress=True,
    feature_cols=None,
):
    """
    Fast batched recursive forecasting for the global model.
    One predict() call per forecast day instead of one per household-day.
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

    iterator = range(horizon)
    if show_progress:
        iterator = tqdm(iterator, total=horizon, desc="Forecasting global")

    for step in iterator:
        current_end = n_hist + step

        X_step = _build_feature_frame(
            history_buffer=history_buffer,
            row_idx=active_idx,
            current_end=current_end,
            step=step,
            date_feature_store=date_feature_store,
            running_sum=running_sum,
            running_sum_sq=running_sum_sq,
            running_zero_count=running_zero_count,
            static_df=static_df,
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
    static_features=None,
    show_progress=True,
    feature_cols=None,
):
    """
    Fast batched recursive forecasting for cluster-specific models.
    One predict() call per (group, day) batch.
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

    routing = []
    for group in pd.unique(group_arr):
        idx = np.where(group_arr == group)[0]
        idx = idx[~all_zero_mask[idx]]

        if len(idx) == 0:
            continue
        if group == "inactive":
            continue

        model = group_models.get(group, fallback_model)
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
                row_idx=idx,
                current_end=current_end,
                step=step,
                date_feature_store=date_feature_store,
                running_sum=running_sum,
                running_sum_sq=running_sum_sq,
                running_zero_count=running_zero_count,
                static_df=static_df,
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