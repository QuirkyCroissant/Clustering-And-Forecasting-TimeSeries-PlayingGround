from tqdm.auto import tqdm
import numpy as np
import pandas as pd

LAGS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 14, 28]
MAX_LAG = max(LAGS)
HOUSEHOLDS_PER_CHUNK = 256

BASE_TIME_FEATURES = [
    "dow",
    "month",
    "doy",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
]
EXTRA_TIME_FEATURES = [
    "quarter",
    "weekofyear",
    "day",
    "is_weekend",
    "is_month_start",
    "is_month_end",
]
TIME_FEATURES = BASE_TIME_FEATURES + EXTRA_TIME_FEATURES

PROFILE_FEATURES = [
    "hh_mean",
    "hh_std",
    "hh_min",
    "hh_max",
    "hh_median",
    "hh_p10",
    "hh_p90",
    "hh_zero_fraction",
    "hh_weekday_mean",
    "hh_weekend_mean",
    "hh_winter_mean",
    "hh_summer_mean",
    "hh_winter_summer_ratio",
    "hh_last_28_mean",
    "hh_last_28_std",
    "hh_last_56_mean",
    "hh_last_56_std",
    "hh_recent_28_vs_hist",
    "hh_trend",
] + [f"hh_month_mean_{month}" for month in range(1, 13)] + [f"hh_dow_mean_{dow}" for dow in range(7)]

CLUSTER_FEATURES = ["forecast_group_code", "is_inactive_cluster"]
SEASONAL_PRIOR_FEATURES = [
    "global_month_mean",
    "global_dow_mean",
    "global_month_dow_mean",
    "cluster_month_mean",
    "cluster_dow_mean",
    "cluster_month_dow_mean",
    "hh_current_month_mean",
    "hh_current_dow_mean",
]


def _safe_ratio(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Divide two arrays while protecting against near-zero denominators."""

    return num / np.where(np.abs(denom) < 1e-6, 1.0, denom)


def _mean_by_mask(values: np.ndarray, mask: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Average over a masked subset of days and fall back when the mask is empty."""

    if not np.any(mask):
        return fallback
    return values[:, mask].mean(axis=1)


def _trend(values: np.ndarray) -> np.ndarray:
    """Estimate a simple per-household linear trend over the available history."""
    n_days = values.shape[1]
    x = np.arange(n_days, dtype=np.float32)
    x = x - x.mean()
    denom = float(np.sum(np.square(x)))
    if denom <= 0:
        return np.zeros(values.shape[0], dtype=np.float32)
    centered = values - values.mean(axis=1, keepdims=True)
    return (centered @ x / denom).astype(np.float32)


def _forecast_group_code(value) -> float:
    """Map textual forecast groups to stable numeric codes for tree models."""

    if pd.isna(value) or value == "unknown":
        return -2.0
    if value == "inactive":
        return -1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(abs(hash(str(value))) % 10000)


def make_time_features(ds: pd.Timestamp) -> dict:
    """Build calendar and cyclic time features for one target day."""

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
        "quarter": ds.quarter,
        "weekofyear": ds.isocalendar().week,
        "day": ds.day,
        "is_weekend": int(dow >= 5),
        "is_month_start": int(ds.is_month_start),
        "is_month_end": int(ds.is_month_end),
    }


def make_history_features(history: np.ndarray, ds: pd.Timestamp) -> dict:
    """Build one forecast row from a single household history buffer."""

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
    """Precompute time features for every trainable day to avoid repeated work."""

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
        "quarter": target_dates.quarter.to_numpy(dtype=np.float32, copy=False),
        "weekofyear": target_dates.isocalendar().week.to_numpy(dtype=np.float32, copy=False),
        "day": target_dates.day.to_numpy(dtype=np.float32, copy=False),
        "is_weekend": (dow >= 5).astype(np.float32),
        "is_month_start": target_dates.is_month_start.astype(np.float32),
        "is_month_end": target_dates.is_month_end.astype(np.float32),
    }


def make_profile_features(train_wide: pd.DataFrame, cluster_labels: pd.DataFrame) -> pd.DataFrame:
    """Build household profile statistics and append the routing fields used later by forecasting."""

    date_cols = [c for c in train_wide.columns if c != "ID"]
    dates = pd.DatetimeIndex(pd.to_datetime(date_cols))
    values = train_wide[date_cols].to_numpy(dtype=np.float32, copy=True)
    ids = train_wide["ID"].to_numpy(copy=False)

    hist_mean = values.mean(axis=1)
    profile = pd.DataFrame({"ID": ids})
    profile["hh_mean"] = hist_mean
    profile["hh_std"] = values.std(axis=1)
    profile["hh_min"] = values.min(axis=1)
    profile["hh_max"] = values.max(axis=1)
    profile["hh_median"] = np.median(values, axis=1)
    profile["hh_p10"] = np.percentile(values, 10, axis=1)
    profile["hh_p90"] = np.percentile(values, 90, axis=1)
    profile["hh_zero_fraction"] = (values == 0.0).mean(axis=1)

    dow = dates.dayofweek.to_numpy()
    month = dates.month.to_numpy()
    profile["hh_weekday_mean"] = _mean_by_mask(values, dow < 5, hist_mean)
    profile["hh_weekend_mean"] = _mean_by_mask(values, dow >= 5, hist_mean)
    profile["hh_winter_mean"] = _mean_by_mask(values, np.isin(month, [1, 2, 12]), hist_mean)
    profile["hh_summer_mean"] = _mean_by_mask(values, np.isin(month, [6, 7, 8]), hist_mean)
    profile["hh_winter_summer_ratio"] = _safe_ratio(
        profile["hh_winter_mean"].to_numpy(dtype=np.float32),
        profile["hh_summer_mean"].to_numpy(dtype=np.float32),
    )

    last_28 = values[:, -min(28, values.shape[1]):]
    last_56 = values[:, -min(56, values.shape[1]):]
    profile["hh_last_28_mean"] = last_28.mean(axis=1)
    profile["hh_last_28_std"] = last_28.std(axis=1)
    profile["hh_last_56_mean"] = last_56.mean(axis=1)
    profile["hh_last_56_std"] = last_56.std(axis=1)
    profile["hh_recent_28_vs_hist"] = _safe_ratio(
        profile["hh_last_28_mean"].to_numpy(dtype=np.float32),
        hist_mean,
    )
    profile["hh_trend"] = _trend(values)

    for month_num in range(1, 13):
        profile[f"hh_month_mean_{month_num}"] = _mean_by_mask(values, month == month_num, hist_mean)
    for dow_num in range(7):
        profile[f"hh_dow_mean_{dow_num}"] = _mean_by_mask(values, dow == dow_num, hist_mean)

    group_lookup = cluster_labels[["ID", "ForecastGroup"]].drop_duplicates()
    profile = profile.merge(group_lookup, on="ID", how="left")
    profile["ForecastGroup"] = profile["ForecastGroup"].fillna("unknown")
    profile["forecast_group_code"] = profile["ForecastGroup"].map(_forecast_group_code).astype(np.float32)
    profile["is_inactive_cluster"] = profile["ForecastGroup"].eq("inactive").astype(np.float32)
    profile = profile.drop(columns=["ForecastGroup"])

    for col in PROFILE_FEATURES + CLUSTER_FEATURES:
        profile[col] = pd.to_numeric(profile[col], errors="coerce").fillna(0.0).astype(np.float32)
    return profile


def make_seasonal_prior_store(train_wide: pd.DataFrame, cluster_labels: pd.DataFrame) -> dict:
    """Precompute global, cluster, and household seasonal averages for reuse during forecasting."""

    date_cols = [c for c in train_wide.columns if c != "ID"]
    dates = pd.DatetimeIndex(pd.to_datetime(date_cols))
    values = train_wide[date_cols].to_numpy(dtype=np.float32, copy=True)
    ids = train_wide["ID"].to_numpy(copy=False)
    month = dates.month.to_numpy()
    dow = dates.dayofweek.to_numpy()
    global_mean = float(values.mean())

    def date_means(mask_values, size, fallback=global_mean):
        out = np.full(size, fallback, dtype=np.float32)
        for idx in range(size):
            mask = mask_values == idx
            if np.any(mask):
                out[idx] = float(values[:, mask].mean())
        return out

    global_month_mean = np.full(13, global_mean, dtype=np.float32)
    for month_num in range(1, 13):
        mask = month == month_num
        if np.any(mask):
            global_month_mean[month_num] = float(values[:, mask].mean())
    global_dow_mean = date_means(dow, 7)
    global_month_dow_mean = np.full((13, 7), global_mean, dtype=np.float32)
    for month_num in range(1, 13):
        for dow_num in range(7):
            mask = (month == month_num) & (dow == dow_num)
            if np.any(mask):
                global_month_dow_mean[month_num, dow_num] = float(values[:, mask].mean())

    group_map = cluster_labels.set_index("ID")["ForecastGroup"].to_dict()
    groups = np.array([str(group_map.get(hh_id, "unknown")) for hh_id in ids])
    cluster_month_mean = {}
    cluster_dow_mean = {}
    cluster_month_dow_mean = {}
    for group in pd.unique(groups):
        group_values = values[groups == group]
        if len(group_values) == 0:
            continue
        cluster_fallback = float(group_values.mean())
        month_arr = np.full(13, cluster_fallback, dtype=np.float32)
        dow_arr = np.full(7, cluster_fallback, dtype=np.float32)
        month_dow_arr = np.full((13, 7), cluster_fallback, dtype=np.float32)
        for month_num in range(1, 13):
            mask = month == month_num
            if np.any(mask):
                month_arr[month_num] = float(group_values[:, mask].mean())
        for dow_num in range(7):
            mask = dow == dow_num
            if np.any(mask):
                dow_arr[dow_num] = float(group_values[:, mask].mean())
        for month_num in range(1, 13):
            for dow_num in range(7):
                mask = (month == month_num) & (dow == dow_num)
                if np.any(mask):
                    month_dow_arr[month_num, dow_num] = float(group_values[:, mask].mean())
        cluster_month_mean[group] = month_arr
        cluster_dow_mean[group] = dow_arr
        cluster_month_dow_mean[group] = month_dow_arr

    profile = make_profile_features(train_wide, cluster_labels).set_index("ID")
    return {
        "global_month_mean": global_month_mean,
        "global_dow_mean": global_dow_mean,
        "global_month_dow_mean": global_month_dow_mean,
        "cluster_month_mean": cluster_month_mean,
        "cluster_dow_mean": cluster_dow_mean,
        "cluster_month_dow_mean": cluster_month_dow_mean,
        "global_mean": np.float32(global_mean),
        "profile": profile,
    }


def _seasonal_prior_arrays(
    ids: np.ndarray,
    groups: np.ndarray,
    dates: pd.DatetimeIndex,
    store: dict,
) -> dict[str, np.ndarray]:
    """Expand seasonal prior lookups into row-aligned arrays for one training chunk."""

    n_households = len(ids)
    n_dates = len(dates)
    month = dates.month.to_numpy()
    dow = dates.dayofweek.to_numpy()

    data = {
        "global_month_mean": np.tile(store["global_month_mean"][month], n_households),
        "global_dow_mean": np.tile(store["global_dow_mean"][dow], n_households),
        "global_month_dow_mean": np.tile(store["global_month_dow_mean"][month, dow], n_households),
    }

    cluster_month = np.empty(n_households * n_dates, dtype=np.float32)
    cluster_dow = np.empty(n_households * n_dates, dtype=np.float32)
    cluster_month_dow = np.empty(n_households * n_dates, dtype=np.float32)
    hh_month = np.empty(n_households * n_dates, dtype=np.float32)
    hh_dow = np.empty(n_households * n_dates, dtype=np.float32)
    profile = store["profile"]

    for hh_idx, (hh_id, group) in enumerate(zip(ids, groups)):
        row_slice = slice(hh_idx * n_dates, (hh_idx + 1) * n_dates)
        group = str(group)
        cluster_month[row_slice] = store["cluster_month_mean"].get(group, store["global_month_mean"])[month]
        cluster_dow[row_slice] = store["cluster_dow_mean"].get(group, store["global_dow_mean"])[dow]
        cluster_month_dow[row_slice] = store["cluster_month_dow_mean"].get(
            group, store["global_month_dow_mean"]
        )[month, dow]

        if hh_id in profile.index:
            hh_profile = profile.loc[hh_id]
            hh_month[row_slice] = [hh_profile.get(f"hh_month_mean_{m}", store["global_mean"]) for m in month]
            hh_dow[row_slice] = [hh_profile.get(f"hh_dow_mean_{d}", store["global_mean"]) for d in dow]
        else:
            hh_month[row_slice] = store["global_month_mean"][month]
            hh_dow[row_slice] = store["global_dow_mean"][dow]

    data["cluster_month_mean"] = cluster_month
    data["cluster_dow_mean"] = cluster_dow
    data["cluster_month_dow_mean"] = cluster_month_dow
    data["hh_current_month_mean"] = hh_month
    data["hh_current_dow_mean"] = hh_dow
    return {key: values.astype(np.float32, copy=False) for key, values in data.items()}


def merge_static_features(
    train_wide: pd.DataFrame,
    cluster_labels: pd.DataFrame,
    static_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge generated profile features with optional external static features into one table."""

    generated = make_profile_features(train_wide, cluster_labels)
    if static_features is None:
        return generated
    merged = generated.merge(static_features, on="ID", how="left")
    feature_cols = [col for col in merged.columns if col != "ID"]
    merged[feature_cols] = merged[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return merged


def _repeat_static_chunk(static_chunk: np.ndarray, rows_per_household: int, static_cols: list[str]) -> dict:
    """Repeat household-level static features so each daily row keeps the same household profile."""
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
    seasonal_prior_store: dict | None,
):
    """Convert one chunk of wide household histories into supervised daily training rows."""
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
    group_values = np.array([group_map.get(hh_id, "unknown") for hh_id in ids])

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

    if seasonal_prior_store is not None:
        target_dates = dates[MAX_LAG:]
        data.update(
            _seasonal_prior_arrays(
                ids=ids,
                groups=group_values,
                dates=target_dates,
                store=seasonal_prior_store,
            )
        )

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
    include_profile_features: bool = True,
    include_seasonal_priors: bool = True,
) -> pd.DataFrame:
    """Transform the full wide 2023 history into the tabular frame used by XGB-style models."""
    
    date_cols = [c for c in train_wide.columns if c != "ID"]
    dates = pd.DatetimeIndex(pd.to_datetime(date_cols))
    group_map = cluster_labels.set_index("ID")["ForecastGroup"].to_dict()

    if include_profile_features:
        static_features = merge_static_features(train_wide, cluster_labels, static_features)

    if static_features is not None:
        static_features = static_features.set_index("ID")

    time_feature_arrays = _precompute_time_feature_arrays(dates)
    seasonal_prior_store = (
        make_seasonal_prior_store(train_wide, cluster_labels)
        if include_seasonal_priors
        else None
    )
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
                seasonal_prior_store=seasonal_prior_store,
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
