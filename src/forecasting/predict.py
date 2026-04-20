from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from .features import make_history_features

def _predict_one_series(model, history, future_dates, static_vals=None):
    history = list(map(float, history))
    static_vals = static_vals or {}
    preds = []

    for ds in future_dates:
        feats = make_history_features(np.array(history, dtype=float), pd.Timestamp(ds))
        feats.update(static_vals)
        X = pd.DataFrame([feats]).fillna(0.0)
        yhat = float(model.predict(X)[0])
        yhat = max(0.0, yhat)
        preds.append(yhat)
        history.append(yhat)

    return preds

def forecast_global(train_23_wide, future_dates, model, static_features=None, show_progress=True):
    date_cols = [c for c in train_23_wide.columns if c != "ID"]

    static_map = {}
    if static_features is not None:
        static_map = static_features.set_index("ID").to_dict(orient="index")

    rows = []
    iterator = train_23_wide.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(train_23_wide), desc="Forecasting global")

    for _, row in iterator:
        hh_id = row["ID"]
        history = row[date_cols].to_numpy(dtype=float)

        if np.allclose(history, 0.0):
            preds = [0.0] * len(future_dates)
        else:
            preds = _predict_one_series(
                model=model,
                history=history,
                future_dates=future_dates,
                static_vals=static_map.get(hh_id, {}),
            )

        rows.append([hh_id] + preds)

    cols = ["ID"] + [pd.Timestamp(d).strftime("%Y-%m-%d") for d in future_dates]
    return pd.DataFrame(rows, columns=cols)

def forecast_by_group(
    train_23_wide,
    cluster_labels,
    future_dates,
    group_models,
    fallback_model=None,
    static_features=None,
    show_progress=True,
):
    date_cols = [c for c in train_23_wide.columns if c != "ID"]
    group_map = cluster_labels.set_index("ID")["ForecastGroup"].to_dict()

    static_map = {}
    if static_features is not None:
        static_map = static_features.set_index("ID").to_dict(orient="index")

    rows = []
    iterator = train_23_wide.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(train_23_wide), desc="Forecasting by cluster")

    for _, row in iterator:
        hh_id = row["ID"]
        history = row[date_cols].to_numpy(dtype=float)
        group = group_map.get(hh_id, "unknown")

        if group == "inactive" or np.allclose(history, 0.0):
            preds = [0.0] * len(future_dates)
        else:
            model = group_models.get(group, fallback_model)
            if model is None:
                preds = [0.0] * len(future_dates)
            else:
                preds = _predict_one_series(
                    model=model,
                    history=history,
                    future_dates=future_dates,
                    static_vals=static_map.get(hh_id, {}),
                )

        rows.append([hh_id] + preds)

    cols = ["ID"] + [pd.Timestamp(d).strftime("%Y-%m-%d") for d in future_dates]
    return pd.DataFrame(rows, columns=cols)