from tqdm.auto import tqdm
import numpy as np
import pandas as pd

LAGS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 14, 28]
MAX_LAG = max(LAGS)

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

def make_training_frame(
    train_wide: pd.DataFrame,
    cluster_labels: pd.DataFrame,
    static_features: pd.DataFrame | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    date_cols = [c for c in train_wide.columns if c != "ID"]
    dates = pd.to_datetime(date_cols)

    group_map = cluster_labels.set_index("ID")["ForecastGroup"].to_dict()

    if static_features is not None:
        static_features = static_features.set_index("ID")

    rows = []
    iterator = train_wide.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(train_wide), desc="Building training rows")

    for _, row in iterator:
        hh_id = row["ID"]
        y = row[date_cols].to_numpy(dtype=float)
        forecast_group = group_map.get(hh_id, "unknown")

        static_vals = {}
        if static_features is not None and hh_id in static_features.index:
            static_vals = static_features.loc[hh_id].to_dict()

        for t in range(MAX_LAG, len(y)):
            history = y[:t]
            target = y[t]
            ds = dates[t]

            rec = {
                "ID": hh_id,
                "ds": ds,
                "ForecastGroup": forecast_group,
                "target": float(target),
            }
            rec.update(make_history_features(history, ds))
            rec.update(static_vals)
            rows.append(rec)

    return pd.DataFrame(rows)
