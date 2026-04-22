from tqdm.auto import tqdm
import numpy as np
import pandas as pd

LAGS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 14, 28]
MAX_LAG = max(LAGS)
HOUSEHOLDS_PER_CHUNK = 256


def make_time_features(ds: pd.Timestamp) -> dict:
    dow = ds.dayofweek
    month = ds.month
    doy = ds.dayofyear
    return {
        "dow": dow,
        "month": month,
        "doy": doy,
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "doy_sin": np.sin(2 * np.pi * doy / 366),
        "doy_cos": np.cos(2 * np.pi * doy / 366),
    }


def make_history_features(history: np.ndarray, ds: pd.Timestamp) -> dict:
    history = np.asarray(history, dtype=float)
    feats = make_time_features(ds)

    for lag in LAGS:
        feats[f"lag_{lag}"] = history[-lag]

    for w in ROLL_WINDOWS:
        window = history[-w:]
        feats[f"roll_mean_{w}"] = float(np.mean(window))
        feats[f"roll_std_{w}"] = float(np.std(window))
        feats[f"roll_min_{w}"] = float(np.min(window))
        feats[f"roll_max_{w}"] = float(np.max(window))

    feats["hist_mean"] = float(np.mean(history))
    feats["hist_std"] = float(np.std(history))
    feats["hist_zero_fraction"] = float(np.mean(history == 0))
    return feats


def _precompute_time_feature_arrays(dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    target_dates = dates[MAX_LAG:]
    dow = target_dates.dayofweek.to_numpy(dtype=np.float32, copy=False)
    month = target_dates.month.to_numpy(dtype=np.float32, copy=False)
    doy = target_dates.dayofyear.to_numpy(dtype=np.float32, copy=False)
    return {
        "dow": dow,
        "month": month,
        "doy": doy,
        "dow_sin": np.sin(2 * np.pi * dow / 7).astype(np.float32),
        "dow_cos": np.cos(2 * np.pi * dow / 7).astype(np.float32),
        "doy_sin": np.sin(2 * np.pi * doy / 366).astype(np.float32),
        "doy_cos": np.cos(2 * np.pi * doy / 366).astype(np.float32),
    }


def _repeat_static_chunk(static_chunk: np.ndarray, rows_per_household: int, static_cols: list[str]) -> dict:
    repeated = {}
    for idx, col in enumerate(static_cols):
        repeated[col] = np.repeat(static_chunk[:, idx], rows_per_household).astype(np.float32, copy=False)
    return repeated


def _build_training_chunk(
    train_chunk: pd.DataFrame,
    date_cols: list[str],
    dates: pd.DatetimeIndex,
    group_map: dict,
    time_feature_arrays: dict[str, np.ndarray],
    static_features: pd.DataFrame | None,
):
    ids = train_chunk["ID"].to_numpy(copy=False)
    y_values = train_chunk[date_cols].to_numpy(dtype=np.float32, copy=True)
    n_households, n_days = y_values.shape
    rows_per_household = n_days - MAX_LAG
    n_rows = n_households * rows_per_household

    repeated_ids = np.repeat(ids, rows_per_household)
    repeated_dates = np.tile(dates[MAX_LAG:].to_numpy(copy=False), n_households)
    repeated_groups = np.repeat(
        [group_map.get(hh_id, "unknown") for hh_id in ids],
        rows_per_household,
    )

    data = {
        "ID": repeated_ids,
        "ds": repeated_dates,
        "ForecastGroup": pd.Categorical(repeated_groups),
        "target": np.empty(n_rows, dtype=np.float32),
    }

    for name, values in time_feature_arrays.items():
        data[name] = np.tile(values, n_households).astype(np.float32, copy=False)

    for lag in LAGS:
        data[f"lag_{lag}"] = np.empty(n_rows, dtype=np.float32)

    for window in ROLL_WINDOWS:
        data[f"roll_mean_{window}"] = np.empty(n_rows, dtype=np.float32)
        data[f"roll_std_{window}"] = np.empty(n_rows, dtype=np.float32)
        data[f"roll_min_{window}"] = np.empty(n_rows, dtype=np.float32)
        data[f"roll_max_{window}"] = np.empty(n_rows, dtype=np.float32)

    data["hist_mean"] = np.empty(n_rows, dtype=np.float32)
    data["hist_std"] = np.empty(n_rows, dtype=np.float32)
    data["hist_zero_fraction"] = np.empty(n_rows, dtype=np.float32)

    static_cols = []
    static_chunk = None
    if static_features is not None:
        static_chunk_df = static_features.reindex(ids).fillna(0.0)
        static_cols = list(static_chunk_df.columns)
        static_chunk = static_chunk_df.to_numpy(dtype=np.float32, copy=False)
        data.update(_repeat_static_chunk(static_chunk, rows_per_household, static_cols))

    hist_len = np.arange(MAX_LAG, n_days, dtype=np.float32)
    for hh_idx, y in enumerate(y_values):
        row_slice = slice(hh_idx * rows_per_household, (hh_idx + 1) * rows_per_household)

        data["target"][row_slice] = y[MAX_LAG:]

        for lag in LAGS:
            data[f"lag_{lag}"][row_slice] = y[MAX_LAG - lag:n_days - lag]

        for window in ROLL_WINDOWS:
            rolling = np.lib.stride_tricks.sliding_window_view(y, window)[MAX_LAG - window:n_days - window]
            data[f"roll_mean_{window}"][row_slice] = rolling.mean(axis=1, dtype=np.float32)
            data[f"roll_std_{window}"][row_slice] = rolling.std(axis=1, dtype=np.float32)
            data[f"roll_min_{window}"][row_slice] = rolling.min(axis=1)
            data[f"roll_max_{window}"][row_slice] = rolling.max(axis=1)

        cumulative_sum = np.cumsum(y, dtype=np.float64)
        cumulative_sum_sq = np.cumsum(np.square(y, dtype=np.float64), dtype=np.float64)
        cumulative_zero = np.cumsum(y == 0.0, dtype=np.int32)

        hist_sum = cumulative_sum[MAX_LAG - 1:n_days - 1]
        hist_sum_sq = cumulative_sum_sq[MAX_LAG - 1:n_days - 1]
        hist_zero = cumulative_zero[MAX_LAG - 1:n_days - 1]

        hist_mean = (hist_sum / hist_len).astype(np.float32)
        variance = np.maximum((hist_sum_sq / hist_len) - np.square(hist_mean, dtype=np.float32), 0.0)

        data["hist_mean"][row_slice] = hist_mean
        data["hist_std"][row_slice] = np.sqrt(variance, dtype=np.float32)
        data["hist_zero_fraction"][row_slice] = (hist_zero / hist_len).astype(np.float32)

    return pd.DataFrame(data)


def make_training_frame(
    train_wide: pd.DataFrame,
    cluster_labels: pd.DataFrame,
    static_features: pd.DataFrame | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    date_cols = [c for c in train_wide.columns if c != "ID"]
    dates = pd.DatetimeIndex(pd.to_datetime(date_cols))
    group_map = cluster_labels.set_index("ID")["ForecastGroup"].to_dict()

    if static_features is not None:
        static_features = static_features.set_index("ID")

    time_feature_arrays = _precompute_time_feature_arrays(dates)
    chunks = []

    progress = None
    if show_progress:
        progress = tqdm(total=len(train_wide), desc="Building training rows")

    try:
        for start in range(0, len(train_wide), HOUSEHOLDS_PER_CHUNK):
            stop = min(start + HOUSEHOLDS_PER_CHUNK, len(train_wide))
            train_chunk = train_wide.iloc[start:stop].copy()
            chunk_df = _build_training_chunk(
                train_chunk=train_chunk,
                date_cols=date_cols,
                dates=dates,
                group_map=group_map,
                time_feature_arrays=time_feature_arrays,
                static_features=static_features,
            )
            chunk_df["ForecastGroup"] = chunk_df["ForecastGroup"].astype("category")
            chunks.append(chunk_df)

            if progress is not None:
                progress.update(len(train_chunk))
    finally:
        if progress is not None:
            progress.close()

    if not chunks:
        return pd.DataFrame(columns=["ID", "ds", "ForecastGroup", "target"])

    train_df = pd.concat(chunks, ignore_index=True)
    train_df["ForecastGroup"] = train_df["ForecastGroup"].astype("category")
    return train_df
